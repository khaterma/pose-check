"""
Metric Alignment Pipeline
=========================
This script takes an input image and produces:
1. A depth-estimated point cloud from the image (using MOGE camera intrinsics)
2. A rendered SMPL-X mesh depth map from NLF predictions
3. Scale alignment between the two using one of two methods:
   - 'depth_ratio': Histogram-based depth ratio matching (faster)
   - 'point_cloud_3d': ICP-style 3D point cloud matching (more robust, better coverage)
4. Final aligned point cloud and metric SMPL-X mesh

Camera intrinsics are obtained from MOGE depth estimation (single source of truth).

Usage:
    python metric_alignment_pipeline.py --input path/to/image.jpg --output output_dir
    python metric_alignment_pipeline.py -i image.jpg -o output -m depth_ratio
    python metric_alignment_pipeline.py -i image.jpg -o output -m point_cloud_3d
"""

import argparse
import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d

# Pipeline imports
import config
from body_reconstruct import BodyReconstructionPipeline
from nlf import NLFSMPLFitter
from utils import project_to_3d



def estimate_scale_3d(smpl_points, depth_points, num_iterations=3, coverage_threshold=0.5):
    """
    Estimate scalar scale s using iterative closest point (ICP) style matching
    with comprehensive coverage of the SMPL-X mesh.
    
    This function ensures full coverage by:
    1. Using all SMPL-X vertices (not just a subset)
    2. Iteratively refining scale with bidirectional matching
    3. Using robust statistics to handle outliers
    4. Ensuring spatial coverage across different body regions
    
    Args:
        smpl_points: (N, 3) numpy array of SMPL vertices/points
        depth_points: (M, 3) numpy array of depth-backprojected points
        num_iterations: Number of ICP-style iterations for refinement
        coverage_threshold: Minimum fraction of SMPL points that should have valid matches
    
    Returns:
        scale: Scalar scale factor
    """
    from scipy.spatial import cKDTree
    
    # Center both point clouds to remove translation ambiguity
    smpl_centroid = smpl_points.mean(axis=0, keepdims=True)
    depth_centroid = depth_points.mean(axis=0, keepdims=True)
    smpl_centered = smpl_points - smpl_centroid
    depth_centered = depth_points - depth_centroid
    
    # Use ALL SMPL-X points for comprehensive coverage
    # Subsample depth points if too large (depth cloud is typically denser)
    max_depth_points = 50000
    if len(depth_centered) > max_depth_points:
        idx = np.random.choice(len(depth_centered), max_depth_points, replace=False)
        depth_sub = depth_centered[idx]
    else:
        depth_sub = depth_centered
    
    # Build KD-tree for depth points
    depth_tree = cKDTree(depth_sub)
    
    # Initial scale estimate using median distance ratios
    smpl_dists = np.linalg.norm(smpl_centered, axis=1)
    depth_dists = np.linalg.norm(depth_sub, axis=1)
    
    # Use percentile-based estimation for robustness
    smpl_p50 = np.percentile(smpl_dists, 50)
    depth_p50 = np.percentile(depth_dists, 50)
    scale = depth_p50 / (smpl_p50 + 1e-13)


    # Adjust scale to account for clothing buffer
    
    body_size = np.percentile(depth_dists, 90)  # use depth cloud for body size estimate
    # clothing_buffer_cm = 0.03 # 3 cm in meters
    # reduction_factor = 1.0 - (clothing_buffer_cm / body_size)
    # scale *= max(0.90, reduction_factor)  # at least 10% reduction, typically ~3%
    # print(f"Clothing buffer adjustment factor: {max(0.90, reduction_factor):.4f}")
    
    
    scale *= 0.99  # reduce scale by 1% to account for clothing
    
    print(f"Initial scale estimate (median distance): {scale:.6f}")
    
    # Initial quality assessment - find well-matched regions
    smpl_scaled_initial = smpl_centered * scale
    distances_initial, _ = depth_tree.query(smpl_scaled_initial, k=1)
    # body_extent = np.max(np.abs(smpl_scaled_initial)) * 2
    body_extent = body_size  # use depth cloud for body extent estimate
    
    # Define "good match" threshold (e.g., within 10% of body extent)
    good_match_threshold = body_extent * 0.10
    good_match_mask = distances_initial < good_match_threshold
    
    print(f"Initial good matches: {good_match_mask.sum()}/{len(smpl_centered)} ({good_match_mask.sum()/len(smpl_centered)*100:.1f}%)")
    
    if good_match_mask.sum() < 50:
        print("Warning: Too few initial good matches, using relaxed threshold")
        good_match_threshold = np.percentile(distances_initial, 30)
        good_match_mask = distances_initial < good_match_threshold
    
    # Use only the well-matched subset for refinement
    smpl_good = smpl_centered[good_match_mask]
    
    print(f"Refining scale using {len(smpl_good)} well-matched points")
    
    # Iterative refinement focusing only on well-matched regions
    for iteration in range(num_iterations):
        # Scale SMPL points with current estimate
        smpl_scaled = smpl_good * scale
        
        # Find nearest depth point for each well-matched SMPL point
        distances_fwd, indices_fwd = depth_tree.query(smpl_scaled, k=1)
        
        # Use tighter threshold since we're only working with good matches
        distance_threshold = body_extent * 0.12
        
        # Filter matches by distance (inliers within the good region)
        inlier_mask = distances_fwd < distance_threshold
        
        if inlier_mask.sum() < 50:
            print(f"Warning: Iteration {iteration+1} - Too few inliers ({inlier_mask.sum()}), keeping previous scale")
            continue
        
        # Get matched pairs from the good region only
        smpl_matched = smpl_good[inlier_mask]  # Original (unscaled) SMPL points
        depth_matched = depth_sub[indices_fwd[inlier_mask]]
        
        # Robust scale estimation using different methods and taking median
        scales = []
        
        # Method 1: Least squares - minimize ||depth - s * smpl||^2
        # Closed form: s = (smpl · depth) / (smpl · smpl)
        numerator = np.sum(smpl_matched * depth_matched)
        denominator = np.sum(smpl_matched * smpl_matched)
        if abs(denominator) > 1e-9:
            s1 = numerator / denominator
            if 0.01 < s1 < 1000:
                scales.append(s1)
        
        # Method 2: Per-point scale ratios (robust median)
        smpl_norms = np.linalg.norm(smpl_matched, axis=1)
        depth_norms = np.linalg.norm(depth_matched, axis=1)
        valid_norm_mask = smpl_norms > 1e-6
        if valid_norm_mask.sum() > 10:
            point_scales = depth_norms[valid_norm_mask] / smpl_norms[valid_norm_mask]
            s2 = np.median(point_scales)
            if 0.01 < s2 < 1000:
                scales.append(s2)
        
        # Method 3: Axis-wise scale estimation (handles anisotropic cases)
        for axis in range(3):
            smpl_axis = np.abs(smpl_matched[:, axis])
            depth_axis = np.abs(depth_matched[:, axis])
            valid_axis_mask = smpl_axis > 1e-6
            if valid_axis_mask.sum() > 10:
                axis_scales = depth_axis[valid_axis_mask] / smpl_axis[valid_axis_mask]
                s_axis = np.median(axis_scales)
                if 0.01 < s_axis < 1000:
                    scales.append(s_axis)
        
        if len(scales) == 0:
            print(f"Warning: Iteration {iteration+1} - No valid scale estimates")
            continue
        
        # Take median of all scale estimates for robustness
        new_scale = np.median(scales)
        
        # Compute residual for diagnostics
        residual = np.mean(np.linalg.norm(depth_matched - new_scale * smpl_matched, axis=1))
        
        print(f"Iteration {iteration+1}: scale={new_scale:.6f}, residual={residual:.6f}, "
              f"inliers={inlier_mask.sum()}/{len(smpl_good)}")
        
        # Update scale with damping to prevent oscillation
        scale = 0.7 * new_scale + 0.3 * scale
    
    # Final validation
    smpl_scaled_final = smpl_centered * scale
    distances_final, _ = depth_tree.query(smpl_scaled_final, k=1)
    final_coverage = (distances_final < body_extent * 0.2).sum() / len(smpl_centered)
    final_residual = np.median(distances_final)
    
    print(f"\nFinal scale: {scale:.6f}")
    print(f"Final coverage: {final_coverage*100:.1f}%")
    print(f"Final median residual: {final_residual:.6f}")
    
    return scale


def render_mesh(vertices, fitter, cam_intrinsics):
    """
    Render depth map from SMPL-X mesh vertices using PyTorch3D rasterization.
    
    Args:
        vertices: SMPL-X mesh vertices (B, N, 3)
        fitter: NLFSMPLFitter instance with SMPL-X model
        cam_intrinsics: Camera intrinsic matrix (3, 3) or (1, 3, 3)
    
    Returns:
        rendered_depth: Rendered depth map (H, W)
    """
    from pytorch3d.renderer import (
        RasterizationSettings,
        MeshRasterizer,
        PerspectiveCameras,
    )
    from pytorch3d.structures import Meshes

    device = vertices.device

    # Create SMPLX mesh
    faces = torch.from_numpy(fitter.smplx_model.faces.astype(np.int64)).long().to(device)
    mesh = Meshes(verts=[vertices.squeeze(0)], faces=[faces])
    
    # Extract H, W from intrinsics
    if cam_intrinsics.dim() == 3:
        cam_intrinsics = cam_intrinsics.squeeze(0)
    
    fx = -cam_intrinsics[0, 0]  # flipped sign for PyTorch3D
    fy = -cam_intrinsics[1, 1]
    cx = cam_intrinsics[0, 2]
    cy = cam_intrinsics[1, 2]

    H = int(cy.item() * 2)
    W = int(cx.item() * 2)
    # H = int( cx.item() * 2)
    # W = int( cy.item() * 2) # swapped to match image convention
    print(f"Rendering depth with image size: {W}x{H}")
    # Set up PyTorch3D camera
    cameras = PerspectiveCameras(
        focal_length=((fx, fy),),
        principal_point=((cx, cy),),
        image_size=((H, W),),
        device=device,
        in_ndc=False
    )

    # Rasterization settings
    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0
    )

    # Rasterize mesh
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )

    fragments = rasterizer(mesh)
    rendered_depth = fragments.zbuf[0, ..., 0]  # [H, W]
    return rendered_depth


def compute_scale_from_depth_ratio(rendered_depth_np, estimated_depth, min_ratio=20, plot_histogram=False, output_dir=None):
    """
    Compute scale factor by analyzing the ratio of rendered depth to estimated depth.
    
    Args:
        rendered_depth_np: Rendered depth from SMPL-X mesh (H, W) numpy array (already transposed)
        estimated_depth: Estimated depth from depth model (H, W) numpy array
        min_ratio: Minimum ratio threshold to filter invalid values
        plot_histogram: Whether to plot histogram of ratios
        output_dir: Directory to save histogram plot
    
    Returns:
        scale: Scale factor to multiply SMPL-X model by
        highest_bin_value: The most frequent ratio value
    """
    # Prepare rendered depth (filter positive values only)
    r = rendered_depth_np * (rendered_depth_np > 0)
    
    # Ensure same dimensions
    H = min(r.shape[0], estimated_depth.shape[0])
    W = min(r.shape[1], estimated_depth.shape[1])
    r = r[:H, :W]
    estimated_depth = estimated_depth[:H, :W]
    
    # Compute ratio
    ratio_depth_map = r / (estimated_depth + 1e-13)
    
    # Filter valid ratios
    valid_ratios = ratio_depth_map[ratio_depth_map > min_ratio]
    
    if valid_ratios.size == 0:
        print("Warning: No valid ratios found. Using default scale of 1.0")
        return 1.0, 1.0
    
    # Remove outliers
    valid_ratios = valid_ratios[valid_ratios < valid_ratios.mean() * 1.5 * valid_ratios.std()]
    
    if valid_ratios.size == 0:
        print("Warning: All ratios filtered as outliers. Using default scale of 1.0")
        return 1.0, 1.0
    
    # Find highest bin (mode) in histogram
    hist_counts, hist_bins = np.histogram(valid_ratios, bins=100)
    highest_bin_idx = hist_counts.argmax()
    highest_bin_value = hist_bins[highest_bin_idx]
    
    scale = 1.0 / highest_bin_value
    
    if plot_histogram and output_dir:
        plt.figure(figsize=(10, 6))
        plt.hist(valid_ratios, bins=100)
        plt.axvline(x=highest_bin_value, color='r', linestyle='--', label=f'Mode: {highest_bin_value:.2f}')
        plt.title("Histogram of Valid Depth Ratios")
        plt.xlabel("Depth Ratio (Rendered / Estimated)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "depth_ratio_histogram.png"), dpi=150)
        plt.close()
        print(f"Histogram saved to {os.path.join(output_dir, 'depth_ratio_histogram.png')}")
    
    return scale, highest_bin_value


def center_point_cloud(pcd):
    """Center a point cloud around the origin."""
    centroid = np.mean(pcd, axis=0)
    return pcd - centroid, centroid


def render_mesh_on_image(vertices, faces, image, cam_intrinsics, alpha=0.6):
    """
    Render mesh with filled faces onto an image using painter's algorithm.
    
    Args:
        vertices: (N, 3) mesh vertices in camera coordinates
        faces: (F, 3) face indices
        image: (H, W, 3) RGB image (uint8)
        cam_intrinsics: (3, 3) camera intrinsic matrix
        alpha: transparency of mesh overlay
    
    Returns:
        overlay: (H, W, 3) image with mesh rendered (uint8)
    """
    import cv2
    
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.cpu().numpy()
    if vertices.ndim == 3:
        vertices = vertices.squeeze(0)
    if isinstance(cam_intrinsics, torch.Tensor):
        cam_intrinsics = cam_intrinsics.cpu().numpy()
    if cam_intrinsics.ndim == 3:
        cam_intrinsics = cam_intrinsics.squeeze(0)
    
    fx, fy = cam_intrinsics[0, 0], cam_intrinsics[1, 1]
    cx, cy = cam_intrinsics[0, 2], cam_intrinsics[1, 2]
    H, W = image.shape[:2]
    
    # Project vertices to 2D
    z = vertices[:, 2]
    valid = z > 1e-6
    verts_2d = np.zeros((len(vertices), 2), dtype=np.float32)
    verts_2d[valid, 0] = fx * vertices[valid, 0] / z[valid] + cx
    verts_2d[valid, 1] = fy * vertices[valid, 1] / z[valid] + cy
    
    # Sort faces back-to-front (painter's algorithm)
    face_depths = np.mean(z[faces], axis=1)
    sorted_idx = np.argsort(-face_depths)
    
    # Render
    overlay = image.astype(np.float32).copy()
    mesh_layer = np.zeros_like(overlay)
    mesh_mask = np.zeros((H, W), dtype=np.float32)
    
    for fi in sorted_idx:
        face = faces[fi]
        if not np.all(valid[face]):
            continue
        
        pts = verts_2d[face].astype(np.int32)
        if np.all(pts[:, 0] < 0) or np.all(pts[:, 0] >= W) or np.all(pts[:, 1] < 0) or np.all(pts[:, 1] >= H):
            continue
        
        # Backface culling
        v0, v1, v2 = vertices[face]
        normal = np.cross(v1 - v0, v2 - v0)
        if normal[2] > 0:
            continue
        
        # Draw filled face with color based on depth (lighter = closer)
        depth_norm = (face_depths[fi] - face_depths.min()) / (face_depths.max() - face_depths.min() + 1e-6)
        color = (120 + 100 * (1 - depth_norm), 180 + 60 * (1 - depth_norm), 255)  # Light blue gradient
        cv2.fillPoly(mesh_layer, [pts], color)
        cv2.fillPoly(mesh_mask, [pts], 1.0)
    
    # Alpha blend
    mask_3ch = np.stack([mesh_mask] * 3, axis=-1)
    overlay = overlay * (1 - alpha * mask_3ch) + mesh_layer * alpha * mask_3ch
    
    return np.clip(overlay, 0, 255).astype(np.uint8)


class MetricAlignmentPipeline:
    """
    Pipeline for metric alignment of SMPL-X predictions with monocular depth estimation.
    
    Supports two alignment methods:
    - 'depth_ratio': Uses histogram-based depth ratio matching (faster, works well for frontal poses)
    - 'point_cloud_3d': Uses 3D point cloud ICP-style matching (more robust, better coverage)
    """
    
    def __init__(self, device=None, output_dir="output", alignment_method="point_cloud_3d"):
        """
        Initialize the metric alignment pipeline.
        
        Args:
            device: Torch device (cuda or cpu)
            output_dir: Directory for saving outputs
            alignment_method: 'depth_ratio' or 'point_cloud_3d'
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        self.alignment_method = alignment_method
        os.makedirs(output_dir, exist_ok=True)
        
        # Validate alignment method
        valid_methods = ["depth_ratio", "point_cloud_3d"]
        if alignment_method not in valid_methods:
            raise ValueError(f"alignment_method must be one of {valid_methods}, got '{alignment_method}'")
        
        # Initialize the body reconstruction pipeline
        self.pipeline = BodyReconstructionPipeline(device=self.device, output_dir=output_dir)
        
        print(f"MetricAlignmentPipeline initialized on {self.device}")
        print(f"Output directory: {output_dir}")
        print(f"Alignment method: {alignment_method}")
    
    def run(self, image_path, save_intermediates=True, plot_histogram=True):
        """
        Run the complete metric alignment pipeline.
        
        Args:
            image_path: Path to input image
            save_intermediates: Whether to save intermediate outputs
            plot_histogram: Whether to plot depth ratio histogram
        
        Returns:
            dict: Results containing scale, aligned vertices, point clouds, etc.
        """
        print("=" * 60)
        print("Starting Metric Alignment Pipeline")
        print("=" * 60)
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        print(f"Loaded image: {image_path} ({image.width}x{image.height})")
        
        # Step 1: Depth estimation and segmentation
        print("\n[Step 1] Depth estimation and segmentation...")
        depth_map, processed_image, cam_intrinsics_moge = self.pipeline._estimate_depth_and_fov(image_path)
        self.pipeline.cam_intrinsics = cam_intrinsics_moge
        
        masks, boxes, scores = self.pipeline._generate_segmentation_masks(processed_image)
        self.pipeline._cleanup_gpu()
        
        # Step 2: Resize depth and masks to original image size
        print("\n[Step 2] Resizing depth map and masks...")
        resized_depth = Image.fromarray(depth_map).resize((image.width, image.height), Image.NEAREST)
        resized_depth_np = np.array(resized_depth)
        
        resized_masks = []
        for mask in masks:
            mask_np = mask.detach().cpu().squeeze(0).numpy()
            resized_mask = Image.fromarray((mask_np * 255).astype(np.uint8)).resize(
                (image.width, image.height), Image.NEAREST
            )
            resized_masks.append(torch.from_numpy(np.array(resized_mask)) > 128)
        
        resized_masks = torch.stack(resized_masks).to(self.device)
        depth_map_tensor = torch.from_numpy(resized_depth_np).float().to(self.device)
        
        print(f"Resized shapes - Image: {image_np.shape}, Depth: {resized_depth_np.shape}, Masks: {resized_masks.shape}")
        
        # Step 3: Create point cloud from depth map
        print("\n[Step 3] Creating point cloud from depth estimation...")
        pointcloud_array, pcd = self.pipeline._create_point_cloud(
            resized_depth_np, image_np, sam_mask=resized_masks
        )
        print(f"Point cloud shape: {pointcloud_array.shape}")
        
        if save_intermediates:
            o3d.io.write_point_cloud(os.path.join(self.output_dir, "depth_point_cloud.ply"), pcd)
        
        # Step 4: Run NLF SMPL-X inference
        print("\n[Step 4] Running NLF SMPL-X inference...")
        fitter = NLFSMPLFitter(image=image_np, device=str(self.device))
        pred = fitter._infer_nlf(fitter.image)
        vertices = pred["vertices3d"][0].to(self.device)
        print(f"SMPL-X vertices shape: {vertices.shape}")
        
        # Step 5: Use MOGE camera intrinsics (single source of truth)
        print("\n[Step 5] Using MOGE camera intrinsics...")
        K = cam_intrinsics_moge.float().to(self.device)
        print(f"Camera intrinsic matrix K shape: {K.shape}")
        if K.dim() == 3:
            print(f"  fx={K[0,0,0].item():.2f}, fy={K[0,1,1].item():.2f}, cx={K[0,0,2].item():.2f}, cy={K[0,1,2].item():.2f}")
        else:
            print(f"  fx={K[0,0].item():.2f}, fy={K[1,1].item():.2f}, cx={K[0,2].item():.2f}, cy={K[1,2].item():.2f}")
        
        # Step 6: Render depth from SMPL-X mesh
        print("\n[Step 6] Rendering depth from SMPL-X mesh...")
        rendered_depth_np = render_mesh(vertices, fitter, K).cpu().numpy()
        print(f"Rendered depth shape: {rendered_depth_np.shape}")
        
        if save_intermediates:
            plt.imsave(os.path.join(self.output_dir, "rendered_depth_map.png"), rendered_depth_np, cmap='gray')
        
        # Step 7: Create point cloud from rendered depth
        print("\n[Step 7] Creating point cloud from rendered depth...")
        # Crop image to match rendered depth dimensions (they may differ by 1 pixel due to rounding)
        rd_h, rd_w = rendered_depth_np.shape
        image_for_render = image_np[:rd_h, :rd_w, :]
        print(f"Image cropped to: {image_for_render.shape} to match rendered depth: {rendered_depth_np.shape}")
        
        # project_to_3d returns (points, colors) where colors is normalized [0,1]
        point_cloud_rendered, colors_rendered = project_to_3d(
            rendered_depth_np, image_for_render, K.cpu()
        )
        pcd_rendered_o3d = o3d.geometry.PointCloud()
        pcd_rendered_o3d.points = o3d.utility.Vector3dVector(point_cloud_rendered)
        pcd_rendered_o3d.colors = o3d.utility.Vector3dVector(colors_rendered)
        
        if save_intermediates:
            o3d.io.write_point_cloud(os.path.join(self.output_dir, "rendered_point_cloud.ply"), pcd_rendered_o3d)
        
        # Step 8: Compute scale factor using selected method
        print(f"\n[Step 8] Computing scale factor using '{self.alignment_method}' method...")
        
        if self.alignment_method == "depth_ratio":
            # Use histogram-based depth ratio matching
            scale, highest_bin_value = compute_scale_from_depth_ratio(
                rendered_depth_np, resized_depth_np,
                min_ratio=20,
                plot_histogram=plot_histogram,
                output_dir=self.output_dir
            )
            print(f"Scale factor (depth ratio): {scale:.6f}")
            print(f"Highest bin value: {highest_bin_value:.4f}")
        else:  # point_cloud_3d
            # Use 3D point cloud ICP-style matching
            scale = estimate_scale_3d(
                point_cloud_rendered, pointcloud_array,
                num_iterations=3,
                coverage_threshold=0.8
            )
            print(f"Scale factor (3D point cloud): {scale:.6f}")
        
        # No focal ratio adjustment needed - using single MOGE intrinsics throughout
        adjusted_scale = scale
        print(f"Final adjusted scale: {adjusted_scale:.6f}")
        
        # Step 9: Apply transformation (center -> scale -> translate)
        print("\n[Step 9] Applying transformation to SMPL-X model...")
        
        # Compute centroids
        _, rendered_centroid = center_point_cloud(point_cloud_rendered)
        _, depth_centroid = center_point_cloud(pointcloud_array)
        
        # Convert centroids to torch tensors
        rendered_centroid_t = torch.from_numpy(rendered_centroid).float().to(self.device)
        depth_centroid_t = torch.from_numpy(depth_centroid).float().to(self.device)
        
        # Apply consistent transformation: center -> scale -> translate back to depth centroid
        def transform_points(pts):
            return (pts - rendered_centroid_t) * adjusted_scale + depth_centroid_t
        
        # Transform vertices and joints
        scaled_vertices = transform_points(pred["vertices3d"][0].to(self.device))
        scaled_joints = transform_points(pred["joints3d"][0].to(self.device))
        
        # Transform point clouds (numpy)
        point_cloud_rendered_aligned = (point_cloud_rendered - rendered_centroid) * adjusted_scale + depth_centroid
        
        # Step 10: Save aligned outputs
        print("\n[Step 10] Saving aligned outputs...")
        
        # Save aligned SMPL-X mesh
        fitter.fitted_params = {
            "vertices": scaled_vertices,
            "joints": scaled_joints,
        }
        mesh_output_path = os.path.join(self.output_dir, "aligned_smplx_mesh.obj")
        fitter.export_mesh(mesh_output_path)
        
        # Save combined aligned point cloud
        combined_pcd = o3d.geometry.PointCloud()
        combined_points = np.concatenate([point_cloud_rendered_aligned, pointcloud_array], axis=0)
        combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
        
        # Color: Red for SMPL-X, Green for depth estimation
        colors_smpl = np.tile([[1, 0, 0]], (point_cloud_rendered_aligned.shape[0], 1))
        colors_depth = np.tile([[0, 1, 0]], (pointcloud_array.shape[0], 1))
        combined_pcd.colors = o3d.utility.Vector3dVector(np.concatenate([colors_smpl, colors_depth], axis=0))
        combined_pcd = combined_pcd.voxel_down_sample(voxel_size=0.005)
        
        combined_output_path = os.path.join(self.output_dir, "aligned_combined_pointcloud.ply")
        o3d.io.write_point_cloud(combined_output_path, combined_pcd)
        print(f"Combined point cloud saved to: {combined_output_path}")
        
        # Render filled mesh on image
        mesh_overlay = render_mesh_on_image(
            scaled_vertices, fitter.smplx_model.faces, image_np, K.cpu().numpy(), alpha=0.9
        )
        mesh_output_path = os.path.join(self.output_dir, "mesh_visualization.png")
        Image.fromarray(mesh_overlay).save(mesh_output_path)
        print(f"Mesh visualization saved to: {mesh_output_path}")
        
        print("\n" + "=" * 60)
        print("Metric Alignment Pipeline Complete!")
        print("=" * 60)
        
        return {
            "scale": adjusted_scale,
            "alignment_method": self.alignment_method,
            "scaled_vertices": scaled_vertices,
            "scaled_joints": scaled_joints,
            "pointcloud_array": pointcloud_array,
            "rendered_pointcloud_aligned": point_cloud_rendered_aligned,
            "fitter": fitter,
            "K": K,
            "mesh_overlay": mesh_overlay,
        }


def main():
    parser = argparse.ArgumentParser(description="Metric Alignment Pipeline for SMPL-X")
    parser.add_argument("--input", "-i", type=str, default=config.INPUT_IMAGE,
                        help="Path to input image")
    parser.add_argument("--output", "-o", type=str, default=config.OUTPUT_DIR,
                        help="Output directory")
    parser.add_argument("--alignment-method", "-m", type=str, default="point_cloud_3d",
                        choices=["depth_ratio", "point_cloud_3d"],
                        help="Scale alignment method: 'depth_ratio' (histogram-based) or 'point_cloud_3d' (ICP-style, default)")
    parser.add_argument("--no-histogram", action="store_true",
                        help="Disable histogram plotting (only for depth_ratio method)")
    parser.add_argument("--no-intermediates", action="store_true",
                        help="Disable saving intermediate outputs")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU execution")
    
    args = parser.parse_args()
    
    device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pipeline = MetricAlignmentPipeline(
        device=device,
        output_dir=args.output,
        alignment_method=args.alignment_method
    )
    
    results = pipeline.run(
        image_path=args.input,
        save_intermediates=not args.no_intermediates,
        plot_histogram=not args.no_histogram
    )
    
    print(f"\nFinal scale factor: {results['scale']:.6f}")
    print(f"Alignment method used: {results['alignment_method']}")
    print(f"Outputs saved to: {args.output}")


if __name__ == "__main__":
    main()
