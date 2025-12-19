"""
Minimal SMPL-X Fitter
====================
Simple single-stage optimization following SMPLify-X approach.
Inputs: YOLO 2D keypoints + camera intrinsics
Output: Optimized SMPL-X parameters
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
import smplx



class MetricSMPLFitter:
    """
    Simple SMPL-X fitter using only reprojection loss and priors.
    Follows SMPLify-X approach.
    """
    
    def __init__(
        self,
        smplx_model_path: str = "./data/smplx",
        gender: str = "neutral",
        device: str = "cuda",
        image: Optional[np.ndarray] = None,
    ):
        """
        Initialize SMPL-X model.
        
        Args:
            smplx_model_path: Path to SMPL-X model files
            gender: 'neutral', 'male', or 'female'
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device)
        self.gender = gender
        
        # Initialize SMPL-X model
        self.smplx_model = smplx.create(
            model_path=smplx_model_path,
            model_type='smplx',
            gender=gender,
            use_face_contour=False,
            num_betas=10,
            num_expression_coeffs=10,
            ext='npz'
        ).to(self.device)
        
        if image is not None:
            self.image = image
        
        # YOLO (17 keypoints) to SMPL-X joint mapping
        # Only map the most reliable body joints
        self.keypoint_mapping = {
            5: 16,   # left_shoulder
            6: 17,   # right_shoulder
            7: 18,   # left_elbow
            8: 19,   # right_elbow
            9: 20,   # left_wrist
            10: 21,  # right_wrist
            11: 1,   # left_hip
            12: 2,   # right_hip
            13: 4,   # left_knee
            14: 5,   # right_knee
            15: 7,   # left_ankle
            16: 8,   # right_ankle
        }
        
        self.fitted_params = None
    
    def _initialize_params(
        self,
        keypoints_2d: np.ndarray,
        valid_mask: np.ndarray,
        cam_intrinsics: np.ndarray,
        init_depth: float = 5.0
    ) -> Dict[str, torch.Tensor]:
        """
        Initialize SMPL-X parameters.
        
        Args:
            keypoints_2d: [N, 2] 2D keypoints
            valid_mask: [N] boolean mask
            cam_intrinsics: [3, 3] camera intrinsics
            init_depth: Initial depth estimate
            
        Returns:
            params: Initialized parameters
        """
        # Initialize translation: place at center of valid keypoints at init_depth
        valid_kpts = keypoints_2d[valid_mask]
        
        if len(valid_kpts) > 0:
            # Get 2D center
            center_2d = valid_kpts.mean(axis=0)
            
            # Unproject to 3D at init_depth
            fx = cam_intrinsics[0, 0]
            fy = cam_intrinsics[1, 1]
            cx = cam_intrinsics[0, 2]
            cy = cam_intrinsics[1, 2]
            
            x = (center_2d[0] - cx) * init_depth / fx
            y = -(center_2d[1] - cy) * init_depth / fy
            z = init_depth
            
            transl = torch.tensor([[x, y, z]], dtype=torch.float32, device=self.device)
        else:
            transl = torch.tensor([[0.0, 0.0, init_depth]], dtype=torch.float32, device=self.device)
        
        # Initialize pose and shape to zero (neutral pose)
        global_orient = torch.zeros(1, 3, dtype=torch.float32, device=self.device)
        body_pose = torch.zeros(1, 63, dtype=torch.float32, device=self.device)
        betas = torch.zeros(1, 10, dtype=torch.float32, device=self.device)
        
        # Fixed parameters (hands, face)
        left_hand_pose = torch.zeros(1, 6, dtype=torch.float32, device=self.device)
        right_hand_pose = torch.zeros(1, 6, dtype=torch.float32, device=self.device)
        expression = torch.zeros(1, 10, dtype=torch.float32, device=self.device)
        
        return {
            'transl': transl,
            'global_orient': global_orient,
            'body_pose': body_pose,
            'betas': betas,
            'left_hand_pose': left_hand_pose,
            'right_hand_pose': right_hand_pose,
            'expression': expression,
        }
    
    def _project_points(
        self,
        points_3d: torch.Tensor,
        cam_intrinsics: torch.Tensor
    ) -> torch.Tensor:
        """
        Project 3D points to 2D using camera intrinsics.
        
        Args:
            points_3d: [N, 3] 3D points
            cam_intrinsics: [3, 3] camera intrinsics
            
        Returns:
            points_2d: [N, 2] projected 2D points
        """
        fx = cam_intrinsics[0, 0]
        fy = cam_intrinsics[1, 1]
        cx = cam_intrinsics[0, 2]
        cy = cam_intrinsics[1, 2]
        
        # Perspective projection
        x_2d = fx * points_3d[:, 0] / (points_3d[:, 2] + 1e-6) + cx
        y_2d = -fy * points_3d[:, 1] / (points_3d[:, 2] + 1e-6) + cy
        
        return torch.stack([x_2d, y_2d], dim=1)
    
    def fit(
        self,
        keypoints_2d: np.ndarray,
        cam_intrinsics: np.ndarray,
        conf_threshold: float = 0.3,
        num_iterations: int = 500,
        lr: float = 0.01,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Fit SMPL-X to 2D keypoints.
        
        Args:
            keypoints_2d: [N, 3] YOLO keypoints (x, y, confidence)
            cam_intrinsics: [3, 3] camera intrinsics matrix
            conf_threshold: Confidence threshold for valid keypoints
            num_iterations: Number of optimization iterations
            lr: Learning rate
            weights: Loss weights dict
            
        Returns:
            fitted_params: Optimized SMPL-X parameters
        """
        if weights is None:
            weights = {
                'reproj': 100.0,      # Reprojection error
                'pose_prior': 0.01,   # Pose regularization
                'shape_prior': 0.1,   # Shape regularization
            }
        
        print("\n" + "=" * 60)
        print("SMPL-X FITTING")
        print("=" * 60)
        
        # Prepare data
        valid_mask = keypoints_2d[:, 2] > conf_threshold
        
        if valid_mask.sum() < 6:
            raise ValueError(f"Not enough valid keypoints: {valid_mask.sum()}")
        
        print(f"Valid keypoints: {valid_mask.sum()}/{len(keypoints_2d)}")
        
        # Convert to torch
        keypoints_2d_torch = torch.from_numpy(keypoints_2d[:, :2]).float().to(self.device)
        valid_mask_torch = torch.from_numpy(valid_mask).bool().to(self.device)
        
        if cam_intrinsics.ndim == 3:
            cam_intrinsics = cam_intrinsics[0]
        cam_intrinsics_torch = torch.from_numpy(cam_intrinsics).float().to(self.device)
        
        # Initialize parameters
        params = self._initialize_params(
            keypoints_2d[:, :2],
            valid_mask,
            cam_intrinsics
        )
        
        # Set up optimization variables
        transl = params['transl'].clone().requires_grad_(True)
        global_orient = params['global_orient'].clone().requires_grad_(True)
        body_pose = params['body_pose'].clone().requires_grad_(True)
        betas = params['betas'].clone().requires_grad_(True)
        
        # Fixed parameters
        left_hand_pose = params['left_hand_pose']
        right_hand_pose = params['right_hand_pose']
        expression = params['expression']
        
        # Optimizer
        optimizer = torch.optim.Adam(
            [transl, global_orient, body_pose, betas],
            lr=lr
        )
        
        # Optimization loop
        print("\nOptimizing...")
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            # Forward pass through SMPL-X
            output = self.smplx_model(
                betas=betas,
                global_orient=global_orient,
                body_pose=body_pose,
                transl=transl,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                expression=expression,
                return_verts=False
            )
            
            # Get SMPL-X joints
            smpl_joints = output.joints[0]  # [127, 3]
            
            # Map to YOLO keypoints and compute reprojection loss
            pred_2d_list = []
            target_2d_list = []
            
            for yolo_idx, smpl_idx in self.keypoint_mapping.items():
                if not valid_mask_torch[yolo_idx]:
                    continue
                
                # Get 3D joint
                joint_3d = smpl_joints[smpl_idx]
                
                # Project to 2D
                joint_2d = self._project_points(
                    joint_3d.unsqueeze(0),
                    cam_intrinsics_torch
                )[0]
                
                pred_2d_list.append(joint_2d)
                target_2d_list.append(keypoints_2d_torch[yolo_idx])

                # ## overlay both prediction and target on the image for visualization
                # import cv2
                # import matplotlib.pyplot as plt
                # img = self.image.copy()
                # pred_point = joint_2d.clone().detach().cpu().numpy()
                # pred_point = (int(pred_point[0]), int(pred_point[1]))
                # # pred_point = (int(joint_2d[0].cpu().numpy()), int(joint_2d[1].cpu().numpy()))
                # target_point = (int(keypoints_2d_torch[yolo_idx][0].cpu().numpy()), int(keypoints_2d_torch[yolo_idx][1].cpu().numpy()))
                # cv2.circle(img, pred_point, 5, (0, 255, 0), -1)  # Green for prediction
                # cv2.circle(img, target_point, 5, (0, 0, 255), -1)  # Red for target
                # plt.imshow(img)
                # plt.axis('off')
                # plt.show()
            
            if len(pred_2d_list) < 3:
                print(f"Warning: Only {len(pred_2d_list)} valid joints")
                continue
            
            pred_2d = torch.stack(pred_2d_list)
            target_2d = torch.stack(target_2d_list)
            
            # Compute losses
            # L_reproj: Reprojection error
            loss_reproj = torch.mean((pred_2d - target_2d) ** 2)
            
            # L_pose: Pose prior (penalize large rotations)
            loss_pose = torch.mean(body_pose ** 2)
            
            # L_shape: Shape prior (keep betas small)
            loss_shape = torch.mean(betas ** 2)
            
            # Total loss
            loss = (
                weights['reproj'] * loss_reproj +
                weights['pose_prior'] * loss_pose +
                weights['shape_prior'] * loss_shape
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Logging
            if iteration % 50 == 0:
                print(
                    f"Iter {iteration:04d} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Reproj: {loss_reproj.item():.4f} | "
                    f"Pose: {loss_pose.item():.6f} | "
                    f"Shape: {loss_shape.item():.6f}"
                )
        
        print("\n" + "=" * 60)
        print("FITTING COMPLETE")
        print("=" * 60)
        
        # Store results
        self.fitted_params = {
            'transl': transl.detach(),
            'global_orient': global_orient.detach(),
            'body_pose': body_pose.detach(),
            'betas': betas.detach(),
            'left_hand_pose': left_hand_pose,
            'right_hand_pose': right_hand_pose,
            'expression': expression,
        }
        
        return self.fitted_params
    

        
    
    def get_mesh(
        self,
        params: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get SMPL-X mesh vertices and faces.
        
        Args:
            params: SMPL-X parameters (uses fitted_params if None)
            
        Returns:
            vertices: [V, 3] mesh vertices
            faces: [F, 3] mesh faces
        """
        if params is None:
            params = self.fitted_params
            if params is None:
                raise ValueError("No fitted parameters. Run fit() first.")
        
        with torch.no_grad():
            output = self.smplx_model(**params, return_verts=True)
            vertices = output.vertices[0].cpu().numpy()
            faces = self.smplx_model.faces
        
        return vertices, faces
    
    def export_mesh(
        self,
        output_path: str,
        params: Optional[Dict[str, torch.Tensor]] = None
    ):
        """Export SMPL-X mesh to file."""
        import trimesh
        
        vertices, faces = self.get_mesh(params)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(output_path)
        print(f"Mesh exported to {output_path}")


if __name__ == "__main__":
    # Example usage
    print("MetricSMPLFitter - Simple SMPL-X optimization")
    print("Usage: fitter.fit(keypoints_2d, cam_intrinsics)")
