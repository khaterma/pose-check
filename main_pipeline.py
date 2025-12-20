"""
Human Body 3D Reconstruction Pipeline
Handles depth estimation, pose detection, segmentation, and SMPL-X fitting.
"""

import torch
import cv2
import numpy as np
from PIL import Image
import os

from pose import draw_pose, yolo_pose_inference
from depth_anything_3.api import DepthAnything3
from fov_estimator import FOVEstimator
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from metric_smpl_fitter import MetricSMPLFitter
from smpl_visualization import (
    visualize_fitting_results,
    compare_phase1_phase2,
    export_colored_mesh
)
from utils import project_to_3d, lift2d_keypoints_to_3d
import open3d as o3d


BODY_PARTS = {
    0: "Nose", 1: "Left Eye", 2: "Right Eye", 3: "Left Ear", 4: "Right Ear",
    5: "Left Shoulder", 6: "Right Shoulder", 7: "Left Elbow", 8: "Right Elbow",
    9: "Left Wrist", 10: "Right Wrist", 11: "Left Hip", 12: "Right Hip",
    13: "Left Knee", 14: "Right Knee", 15: "Left Ankle", 16: "Right Ankle"
}


class BodyReconstructionPipeline:
    """Main pipeline for 3D body reconstruction from a single image."""
    
    def __init__(self, device=None, output_dir="output"):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Models will be loaded on demand
        self.depth_model = None
        self.fov_estimator = None
        self.sam_model = None
        self.sam_processor = None
        
    def run(self, image_path, gender="male", enable_visualization=True):
        """
        Run the complete reconstruction pipeline.
        
        Args:
            image_path: Path to input image
            gender: Gender for SMPL-X model ("male", "female", "neutral")
            enable_visualization: Whether to generate visualization outputs
            
        Returns:
            Dictionary with all pipeline results
        """
        print("="*60)
        print("Starting 3D Body Reconstruction Pipeline")
        print("="*60)
        
        # Step 1: Depth Estimation
        depth_map, processed_image, cam_intrinsics = self._estimate_depth_and_fov(image_path)
        self.cam_intrinsics = cam_intrinsics
        
        # Step 2: 2D Pose Detection
        keypoints_2d = self._detect_pose_2d(image_path, processed_image)
        
        # Step 3: Segmentation (SAM3)
        masks, boxes, scores = self._generate_segmentation_masks(processed_image)
        
        # Step 4: Create 3D Point Cloud
        point_cloud_array, pcd = self._create_point_cloud(depth_map, processed_image)
        
        # Clean up GPU memory before fitting
        self._cleanup_gpu()
        
        # Step 5: SMPL-X Fitting
        fitted_params = self._fit_smplx_model(
            processed_image,
            depth_map,
            keypoints_2d,
            cam_intrinsics,
            point_cloud_array,
            gender, 
            masks
        )
        
        # Step 6: Generate Visualizations (optional)
        if enable_visualization and fitted_params is not None:
            self._generate_visualizations(
                processed_image,
                cam_intrinsics,
                point_cloud_array,
                fitted_params
            )
        
        return {
            "depth_map": depth_map,
            "processed_image": processed_image,
            "cam_intrinsics": cam_intrinsics,
            "keypoints_2d": keypoints_2d,
            "masks": masks,
            "point_cloud": point_cloud_array,
            "fitted_params": fitted_params
        }
    
    def _estimate_depth_and_fov(self, image_path):
        """Step 1: Estimate depth map and camera FOV."""
        print("\n[1/5] Estimating depth and camera FOV...")
        
        # Load depth model
        if self.depth_model is None:
            self.depth_model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
            self.depth_model = self.depth_model.to(device=self.device)
        
        # Run depth inference
        prediction = self.depth_model.inference([image_path], export_dir=self.output_dir)
        depth_map = prediction.depth[0]
        processed_image = prediction.processed_images[0]
        
        # Estimate FOV
        if self.fov_estimator is None:
            self.fov_estimator = FOVEstimator(name="moge2", device=self.device)
        
        cam_intrinsics = self.fov_estimator.get_cam_intrinsics(img=processed_image)
        
        print(f"✓ Depth map shape: {depth_map.shape}")
        print(f"✓ Camera intrinsics estimated")
        
        return depth_map, processed_image, cam_intrinsics
    
    def _detect_pose_2d(self, original_image_path, processed_image):
        """Step 2: Detect 2D pose keypoints using YOLO."""
        print("\n[2/5] Detecting 2D pose keypoints...")
        
        # Run YOLO pose detection
        results = yolo_pose_inference([original_image_path])
        
        # Get original and processed image dimensions
        original_shape = cv2.imread(original_image_path).shape
        processed_shape = processed_image.shape
        
        ratio_h = processed_shape[0] / original_shape[0]
        ratio_w = processed_shape[1] / original_shape[1]
        
        # Extract keypoints
        keypoints_list = []
        for result in results:
            if result.keypoints is not None:
                for person_idx, person_kpts in enumerate(result.keypoints.xy.cpu().numpy()):
                    conf = result.keypoints.conf.cpu().numpy()[person_idx]
                    keypoints = np.concatenate(
                        [person_kpts * [ratio_w, ratio_h], conf[:, None]], axis=1
                    )
                    keypoints_list.append(keypoints)
        
        # Save visualization
        if keypoints_list:
            depth_vis = cv2.normalize(processed_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            if len(depth_vis.shape) == 2:
                depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
            output_image = draw_pose(depth_vis, keypoints_list[0])
            cv2.imwrite(os.path.join(self.output_dir, "pose_overlay.png"), output_image)
            print(f"✓ Detected {len(keypoints_list)} person(s)")
        else:
            print("⚠ No keypoints detected")
        
        return keypoints_list[0] if keypoints_list else None
    
    def _generate_segmentation_masks(self, processed_image):
        """Step 3: Generate segmentation masks using SAM3."""
        print("\n[3/5] Generating segmentation masks...")
        
        # Load SAM3 model
        if self.sam_model is None:
            self.sam_model = build_sam3_image_model()
            self.sam_processor = Sam3Processor(self.sam_model)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(processed_image)
        
        # Run SAM3
        inference_state = self.sam_processor.set_image(pil_image)
        output = self.sam_processor.set_text_prompt(
            state=inference_state,
            prompt="full body of a person"
        )
        
        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]
        
        # Save mask visualizations
        self._save_mask_visualizations(processed_image, masks, scores)
        
        print(f"✓ Generated {len(masks)} mask(s)")
        return masks, boxes, scores
    
    def _save_mask_visualizations(self, image, masks, scores):
        """Save mask overlay visualizations."""
        import matplotlib.pyplot as plt
        
        for idx, (mask, score) in enumerate(zip(masks, scores)):
            if score < 0.5:
                continue
            
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.imshow(mask.squeeze(0).cpu().numpy(), alpha=0.5, cmap='jet')
            plt.axis('off')
            plt.savefig(
                os.path.join(self.output_dir, f"sam3_mask_{idx}_score{score:.2f}.png"),
                bbox_inches='tight',
                pad_inches=0
            )
            plt.close()
    
    def _create_point_cloud(self, depth_map, rgb_image):
        """Step 4: Create 3D point cloud from depth map."""
        print("\n[4/5] Creating 3D point cloud...")
        
        points, colors = project_to_3d(
            depth_map=depth_map,
            img_rgb=rgb_image,
            camera_intrinsics=self.cam_intrinsics,
            center_around_origin=True
        )
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        point_cloud_array = np.asarray(pcd.points)
        
        # Save point cloud
        o3d.io.write_point_cloud(os.path.join(self.output_dir, "point_cloud.ply"), pcd)
        
        print(f"✓ Point cloud created: {len(point_cloud_array)} points")
        return point_cloud_array, pcd
    
    def _cleanup_gpu(self):
        """Free GPU memory from preprocessing models."""
        print("\nCleaning up GPU memory...")
        
        if self.depth_model is not None:
            self.depth_model.to(device='cpu')
        
        if self.fov_estimator is not None:
            self.fov_estimator.fov_estimator.to(device='cpu')
        
        if self.sam_model is not None:
            self.sam_model = None
            self.sam_processor = None
        
        torch.cuda.empty_cache()
        print("✓ GPU memory freed")
    
    def _fit_smplx_model(self, image, depth_map, keypoints_2d, cam_intrinsics, 
                         point_cloud, gender, masks):
        """Step 5: Fit SMPL-X model to detected keypoints."""
        print("\n[5/5] Fitting SMPL-X model...")
        
        if keypoints_2d is None:
            print("⚠ Cannot fit SMPL-X: No keypoints detected")
            return None
        
        try:
            # Initialize fitter
            fitter = MetricSMPLFitter(
                smplx_model_path="./data/smplx",
                gender=gender,
                device=str(self.device),
                image=image
            )
            
            # Run fitting
            final_params = fitter.fit(
                keypoints_2d=keypoints_2d,
                cam_intrinsics=cam_intrinsics.cpu().numpy(),
                depth_map=depth_map,
                point_cloud=point_cloud,
                mask=masks[0].cpu().numpy(),
                conf_threshold=0.5
            )
            
            # final_params = fitter.optimize_phase2(
            #     depth_map=depth_map,
            #     cam_intrinsics=cam_intrinsics.cpu().numpy(),
            #     mask=masks[0].cpu().numpy(),
            #     iterations=200
            # )

            # Export meshes
            fitter.export_mesh(os.path.join(self.output_dir, "smplx_fitted.obj"))
            fitter.export_mesh(os.path.join(self.output_dir, "smplx_fitted.ply"))
            
            print("✓ SMPL-X fitting complete")
            print(f"✓ Meshes exported to {self.output_dir}/")
            
            # Store fitter for visualization
            self.fitter = fitter
            
            return final_params
            
        except Exception as e:
            print(f"✗ SMPL-X fitting failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_visualizations(self, image, cam_intrinsics, point_cloud, params):
        """Step 6: Generate visualization outputs."""
        print("\nGenerating visualizations...")
        
        try:
            visualize_fitting_results(
                self.fitter,
                image,
                params=params,
                cam_intrinsics=cam_intrinsics,
                point_cloud=point_cloud,
                output_path=os.path.join(self.output_dir, "fitting_visualization.png")
            )
            
            compare_phase1_phase2(
                self.fitter,
                image,
                cam_intrinsics,
                output_path=os.path.join(self.output_dir, "phase_comparison.png")
            )
            
            export_colored_mesh(
                self.fitter,
                image,
                cam_intrinsics,
                output_path=os.path.join(self.output_dir, "colored_smplx.ply")
            )
            
            print(f"✓ Visualizations saved to {self.output_dir}/")
            
        except Exception as e:
            print(f"⚠ Visualization generation failed: {e}")


def main():
    """Main entry point for the pipeline."""
    # Configuration
    IMAGE_PATH = "/home/khater/pose-check/tom.jpg"
    GENDER = "male"
    OUTPUT_DIR = "output"
    
    # Initialize and run pipeline
    pipeline = BodyReconstructionPipeline(output_dir=OUTPUT_DIR)
    
    results = pipeline.run(
        image_path=IMAGE_PATH,
        gender=GENDER,
        enable_visualization=True
    )
    
    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)
    print(f"All outputs saved to: {OUTPUT_DIR}/")
    
    return results


if __name__ == "__main__":
    main()
