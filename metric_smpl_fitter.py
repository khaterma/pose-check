"""
Metric SMPL-X Fitter
====================
Two-phase fitting approach:
  Phase 1: Skeleton fitting using lifted 3D joints
  Phase 2: Surface refinement using point cloud and silhouette
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List
import smplx
from scipy.ndimage import median_filter
from scipy.optimize import minimize
from pytorch3d.structures import Pointclouds
from pytorch3d.loss import chamfer_distance
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
)
import warnings


class MetricSMPLFitter:
    """
    Two-phase metric SMPL-X fitting pipeline.
    
    Phase 1: Skeleton initialization and fitting
    Phase 2: Surface refinement with point cloud and silhouette
    """
    
    def __init__(
        self,
        smplx_model_path: str = "./data/smplx",
        gender: str = "neutral",
        device: str = "cuda",
        use_vposer: bool = False,
        vposer_ckpt: Optional[str] = None,
    ):
        """
        Initialize the SMPL-X fitter.
        
        Args:
            smplx_model_path: Path to SMPL-X model files
            gender: 'neutral', 'male', or 'female'
            device: 'cuda' or 'cpu'
            use_vposer: Whether to use VPoser prior
            vposer_ckpt: Path to VPoser checkpoint
        """
        self.device = torch.device(device)
        self.gender = gender
        self.use_vposer = use_vposer
        
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
        
        # VPoser initialization (if available)
        self.vposer = None
        if use_vposer and vposer_ckpt is not None:
            try:
                from human_body_prior.tools.model_loader import load_model
                from human_body_prior.models.vposer_model import VPoser
                self.vposer = load_model(
                    vposer_ckpt,
                    model_code=VPoser,
                    remove_words_in_model_weights='vp_model.',
                    disable_grad=True
                ).to(self.device)
                self.vposer.eval()
                print("VPoser loaded successfully")
            except Exception as e:
                warnings.warn(f"Could not load VPoser: {e}. Using L2 pose prior instead.")
                self.vposer = None
        
        # YOLO to SMPL-X joint mapping (YOLO has 17 keypoints)
        # YOLO keypoints: 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear,
        # 5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow, 
        # 9=left_wrist, 10=right_wrist, 11=left_hip, 12=right_hip,
        # 13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle
        #
        # SMPL-X joint indices (first 22 joints): 0=pelvis, 1=left_hip, 2=right_hip, 3=spine1, 4=left_knee, 
        # 5=right_knee, 6=spine2, 7=left_ankle, 8=right_ankle, 9=spine3, 10=left_foot, 11=right_foot, 
        # 12=neck, 13=left_collar, 14=right_collar, 15=head, 16=left_shoulder, 17=right_shoulder,
        # 18=left_elbow, 19=right_elbow, 20=left_wrist, 21=right_wrist
        self.yolo_to_smplx_mapping = {
            0: 15,   # nose -> head
            1: 15,   # left_eye -> head
            2: 15,   # right_eye -> head
            3: 15,   # left_ear -> head
            4: 15,   # right_ear -> head
            5: 16,   # left_shoulder -> left_shoulder
            6: 17,   # right_shoulder -> right_shoulder
            7: 18,   # left_elbow -> left_elbow
            8: 19,   # right_elbow -> right_elbow
            9: 20,   # left_wrist -> left_wrist
            10: 21,  # right_wrist -> right_wrist
            11: 1,   # left_hip -> left_hip
            12: 2,   # right_hip -> right_hip
            13: 4,   # left_knee -> left_knee
            14: 5,   # right_knee -> right_knee
            15: 7,   # left_ankle -> left_ankle
            16: 8,   # right_ankle -> right_ankle
        }
        
        # Optimization parameters
        self.phase1_params = None
        self.phase2_params = None
    
        
    def lift_2d_to_3d(
        self,
        keypoints_2d: np.ndarray,
        depth_map: np.ndarray,
        cam_intrinsics: np.ndarray,
        conf_threshold: float = 0.3,
        median_filter_size: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Lift 2D keypoints to 3D using metric depth map.
        
        Args:
            keypoints_2d: [N, 3] array with (x, y, confidence)
            depth_map: [H, W] metric depth map
            cam_intrinsics: [3, 3] camera intrinsics matrix
            conf_threshold: Minimum confidence threshold
            median_filter_size: Size of median filter (e.g., 5x5)
            
        Returns:
            keypoints_3d: [N, 3] lifted 3D keypoints
            valid_mask: [N] boolean mask for valid keypoints
            filtered_depth: [H, W] median-filtered depth map
        """
        # Apply median filter to depth map to avoid background noise
        # filtered_depth = median_filter(depth_map, size=median_filter_size)
        filtered_depth = depth_map # Commenting out median filter for better accuracy
        
        # Extract camera parameters
        fx = cam_intrinsics[0, 0]
        fy = cam_intrinsics[1, 1]
        cx = cam_intrinsics[0, 2]
        cy = cam_intrinsics[1, 2]
        
        keypoints_3d = []
        valid_mask = []
        
        for i, kpt in enumerate(keypoints_2d):
            x_2d, y_2d, conf = kpt
            
            # Check confidence and bounds
            if conf < conf_threshold:
                keypoints_3d.append([0, 0, 0])
                valid_mask.append(False)
                continue
                
            x_2d_int = int(np.clip(x_2d, 0, depth_map.shape[1] - 1))
            y_2d_int = int(np.clip(y_2d, 0, depth_map.shape[0] - 1))
            
            # Get median-filtered depth
            z = filtered_depth[y_2d_int, x_2d_int]
            
            if z <= 0 or not np.isfinite(z):
                keypoints_3d.append([0, 0, 0])
                valid_mask.append(False)
                continue
            
            # Unproject to 3D (OpenCV convention)
            x = (x_2d - cx) * z / fx
            y = -(y_2d - cy) * z / fy
            R = np.array([
            [-1,  0,  0],
            [ 0,  1,  0],
            [ 0,  0, -1]
                ])
            points = np.array([x, y, z])
            # points = points @ R.T
            
            keypoints_3d.append(points)
            valid_mask.append(True)
        
        return (
            np.array(keypoints_3d),
            np.array(valid_mask),
            filtered_depth
        )
    

    def generate_sam_mask(
        self,
        image: np.ndarray,
        keypoints_2d: np.ndarray,
        sam_checkpoint: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate person segmentation mask using SAM.
        
        Args:
            image: [H, W, 3] RGB image
            keypoints_2d: [N, 3] 2D keypoints for prompt
            sam_checkpoint: Path to SAM checkpoint
            
        Returns:
            mask: [H, W] binary mask
        """
        try:
            from segment_anything import sam_model_registry, SamPredictor
            
            if sam_checkpoint is None:
                # Try to find SAM checkpoint
                import os
                possible_paths = [
                    "sam_vit_h_4b8939.pth",
                    "./weights/sam_vit_h_4b8939.pth",
                    "/home/khater/.cache/sam_vit_h_4b8939.pth"
                ]
                sam_checkpoint = None
                for path in possible_paths:
                    if os.path.exists(path):
                        sam_checkpoint = path
                        break
                        
                if sam_checkpoint is None:
                    warnings.warn("SAM checkpoint not found. Using simple threshold mask.")
                    return self._generate_simple_mask(image, keypoints_2d)
            
            # Load SAM
            sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
            sam.to(device=self.device)
            predictor = SamPredictor(sam)
            
            # Prepare image
            predictor.set_image(image)
            
            # Use valid keypoints as point prompts
            valid_kpts = keypoints_2d[keypoints_2d[:, 2] > 0.3][:, :2]
            
            if len(valid_kpts) == 0:
                return self._generate_simple_mask(image, keypoints_2d)
            
            # Sample a few keypoints as prompts
            num_points = min(5, len(valid_kpts))
            indices = np.linspace(0, len(valid_kpts) - 1, num_points, dtype=int)
            input_points = valid_kpts[indices]
            input_labels = np.ones(num_points)
            
            masks, scores, _ = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )
            
            # Select best mask
            best_mask = masks[np.argmax(scores)]
            return best_mask.astype(np.uint8)
            
        except Exception as e:
            warnings.warn(f"SAM failed: {e}. Using simple mask.")
            return self._generate_simple_mask(image, keypoints_2d)
    
    def _generate_simple_mask(
        self,
        image: np.ndarray,
        keypoints_2d: np.ndarray
    ) -> np.ndarray:
        """Generate a simple mask using keypoint bounding box + dilation."""
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        valid_kpts = keypoints_2d[keypoints_2d[:, 2] > 0.3][:, :2]
        if len(valid_kpts) == 0:
            return mask
        
        # Create bounding box with padding
        x_min = int(valid_kpts[:, 0].min())
        x_max = int(valid_kpts[:, 0].max())
        y_min = int(valid_kpts[:, 1].min())
        y_max = int(valid_kpts[:, 1].max())
        
        # Add padding
        pad = 50
        x_min = max(0, x_min - pad)
        x_max = min(image.shape[1], x_max + pad)
        y_min = max(0, y_min - pad)
        y_max = min(image.shape[0], y_max + pad)
        
        # Create ellipse mask
        center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
        axes = ((x_max - x_min) // 2, (y_max - y_min) // 2)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
        
        # Dilate
        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        return mask
    
    def initialize_smplx_params(
        self,
        keypoints_3d: np.ndarray,
        valid_mask: np.ndarray
    ) -> Dict[str, torch.Tensor]:
        """
        Initialize SMPL-X parameters using 3D keypoints.
        
        Args:
            keypoints_3d: [N, 3] lifted 3D keypoints
            valid_mask: [N] boolean mask
            
        Returns:
            params: Dictionary of initialized parameters
        """
        # Get valid keypoints
        valid_kpts = keypoints_3d[valid_mask]
        
        if len(valid_kpts) < 3:
            raise ValueError("Not enough valid keypoints for initialization")
        
        # Initialize translation as midpoint of hips (SMPL-X pelvis location)
        # YOLO indices: 11=left_hip, 12=right_hip
        left_hip_idx = 11
        right_hip_idx = 12
        
        if valid_mask[left_hip_idx] and valid_mask[right_hip_idx]:
            # Use midpoint of hips as pelvis location
            hip_midpoint = (keypoints_3d[left_hip_idx] + keypoints_3d[right_hip_idx]) / 2.0
            transl = torch.tensor(
                hip_midpoint,
                dtype=torch.float32,
                device=self.device
            )
        else:
            # Fallback to centroid if hips not detected
            transl = torch.tensor(
                valid_kpts.mean(axis=0),
                dtype=torch.float32,
                device=self.device
            )
        
        # Initialize global orientation (yaw) using hip direction
        # Try to find left and right hips (YOLO indices 11, 12 -> SMPL-X 1, 2)
        left_hip_idx = 11
        right_hip_idx = 12
        
        global_orient = torch.zeros(1, 3, dtype=torch.float32, device=self.device)
        
        if valid_mask[left_hip_idx] and valid_mask[right_hip_idx]:
            left_hip = keypoints_3d[left_hip_idx]
            right_hip = keypoints_3d[right_hip_idx]
            
            # Compute direction vector (left to right)
            hip_dir = right_hip - left_hip
            hip_dir_xz = np.array([hip_dir[0], 0, hip_dir[2]])
            
            # Compute yaw angle
            if np.linalg.norm(hip_dir_xz) > 1e-6:
                hip_dir_xz = hip_dir_xz / np.linalg.norm(hip_dir_xz)
                # Reference direction is [1, 0, 0] (right in SMPL)
                ref_dir = np.array([1, 0, 0])
                cos_angle = np.dot(hip_dir_xz, ref_dir)
                sin_angle = np.cross(hip_dir_xz, ref_dir)[1]
                yaw = np.arctan2(sin_angle, cos_angle)
                
                # Set global orientation (rotation around Y axis)
                global_orient[0, 1] = yaw
        
        # Initialize other parameters to defaults
        body_pose = torch.zeros(1, 63, dtype=torch.float32, device=self.device)  # 21 joints * 3
        
        # Initialize betas to slightly positive values for a more natural starting body shape
        betas = torch.zeros(1, 10, dtype=torch.float32, device=self.device)
        betas[0, 0] = 0.5  # Slightly positive first shape component for realistic proportions
        
        # Expression and hand pose (set to zero)
        expression = torch.zeros(1, 10, dtype=torch.float32, device=self.device)
        left_hand_pose = torch.zeros(1, 6, dtype=torch.float32, device=self.device)  # 6 PCA components
        right_hand_pose = torch.zeros(1, 6, dtype=torch.float32, device=self.device)  # 6 PCA components
        
        return {
            'transl': transl.unsqueeze(0),
            'global_orient': global_orient,
            'body_pose': body_pose,
            'betas': betas,
            'expression': expression,
            'left_hand_pose': left_hand_pose,
            'right_hand_pose': right_hand_pose,
        }
    
    def phase1_skeleton_fitting(
        self,
        keypoints_3d: torch.Tensor,
        keypoints_2d: torch.Tensor,
        valid_mask: torch.Tensor,
        cam_intrinsics: torch.Tensor,
        num_iterations: int = 300,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Phase 1: Fit SMPL skeleton to 3D joints using L-BFGS.
        
        Args:
            keypoints_3d: [N, 3] lifted 3D keypoints
            keypoints_2d: [N, 2] 2D keypoints
            valid_mask: [N] boolean mask
            cam_intrinsics: [3, 3] camera intrinsics
            num_iterations: Number of optimization iterations
            weights: Loss weights
            
        Returns:
            optimized_params: Optimized SMPL-X parameters
        """
        if weights is None:
            weights = {
                'L_3D': 10.0,      # Reduced from 100 to prevent over-fitting
                'L_2D': 1.0,       # Reduced from 10
                'L_Prior': 0.1,    # Reduced from 1.0 to allow more natural poses
                'L_Shape': 1.0,    # Increased from 0.1 to preserve body shape
            }
        
        print("=" * 50)
        print("Phase 1: Skeleton Fitting")
        print("=" * 50)
        
        # Initialize parameters
        init_params = self.initialize_smplx_params(
            keypoints_3d.cpu().numpy(),
            valid_mask.cpu().numpy()
        )
        
        # Set up optimization variables
        transl = init_params['transl'].clone().requires_grad_(True)
        global_orient = init_params['global_orient'].clone().requires_grad_(True)
        body_pose = init_params['body_pose'].clone().requires_grad_(True)
        betas = init_params['betas'].clone().requires_grad_(True)
        
        # Fixed parameters
        expression = init_params['expression'].clone()
        left_hand_pose = init_params['left_hand_pose'].clone()
        right_hand_pose = init_params['right_hand_pose'].clone()
        
        # Optimizer - use more conservative learning rate
        params_to_optimize = [transl, global_orient, body_pose, betas]
        optimizer = torch.optim.LBFGS(
            params_to_optimize,
            lr=0.5,  # Reduced from 1.0 for more stable optimization
            max_iter=20,
            line_search_fn='strong_wolfe'
        )
        
        # Camera parameters
        fx = cam_intrinsics[0, 0]
        fy = cam_intrinsics[1, 1]
        cx = cam_intrinsics[0, 2]
        cy = cam_intrinsics[1, 2]
        
        iter_count = [0]
        
        def closure():
            optimizer.zero_grad()
            
            # Forward pass
            output = self.smplx_model(
                betas=betas,
                global_orient=global_orient,
                body_pose=body_pose,
                transl=transl,
                expression=expression,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                return_verts=True
            )
            
            smpl_joints = output.joints[:, :22]  # Get first 22 body joints (0-21)
            
            # Map SMPL joints to YOLO keypoints
            mapped_joints_3d = []
            mapped_joints_2d = []
            target_3d = []
            target_2d = []
            
            for yolo_idx, smpl_idx in self.yolo_to_smplx_mapping.items():
                if valid_mask[yolo_idx]:
                    mapped_joints_3d.append(smpl_joints[0, smpl_idx])
                    target_3d.append(keypoints_3d[yolo_idx])
                    
                    # Project SMPL joint to 2D
                    joint_3d = smpl_joints[0, smpl_idx]
                    x_2d = fx * joint_3d[0] / joint_3d[2] + cx
                    y_2d = fy * joint_3d[1] / joint_3d[2] + cy
                    mapped_joints_2d.append(torch.stack([x_2d, y_2d]))
                    target_2d.append(keypoints_2d[yolo_idx])
            
            if len(mapped_joints_3d) == 0:
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            
            mapped_joints_3d = torch.stack(mapped_joints_3d)
            target_3d = torch.stack(target_3d)
            
            # L_3D: 3D joint loss
            loss_3d = F.mse_loss(mapped_joints_3d, target_3d)
            
            # L_2D: Reprojection loss
            loss_2d = torch.tensor(0.0, device=self.device)
            if len(mapped_joints_2d) > 0:
                mapped_joints_2d = torch.stack(mapped_joints_2d)
                target_2d = torch.stack(target_2d)
                loss_2d = F.mse_loss(mapped_joints_2d, target_2d)
            
            # L_Prior: Pose prior (regularize but don't force T-pose)
            loss_prior = torch.tensor(0.0, device=self.device)
            if self.vposer is not None:
                # VPoser prior
                body_pose_aa = body_pose.view(1, -1, 3)
                loss_prior = torch.mean(body_pose_aa ** 2)
            else:
                # Gentle L2 prior - only penalize extreme poses
                # Use a threshold to avoid forcing T-pose
                pose_magnitudes = torch.abs(body_pose)
                loss_prior = torch.mean(torch.clamp(pose_magnitudes - 0.5, min=0.0) ** 2)
            
            # L_Shape: Shape regularization
            loss_shape = torch.mean(betas ** 2)
            
            # Total loss
            loss = (
                weights['L_3D'] * loss_3d +
                weights['L_2D'] * loss_2d +
                weights['L_Prior'] * loss_prior +
                weights['L_Shape'] * loss_shape
            )
            
            if iter_count[0] % 10 == 0:
                print(f"Iter {iter_count[0]}: Loss={loss.item():.4f}, "
                      f"L_3D={loss_3d.item():.4f}, L_2D={loss_2d.item():.4f}, "
                      f"L_Prior={loss_prior.item():.4f}, L_Shape={loss_shape.item():.4f}")
            
            iter_count[0] += 1
            
            loss.backward()
            return loss
        
        # Run optimization
        for i in range(num_iterations // 20):
            optimizer.step(closure)
        
        print(f"\nPhase 1 completed after {iter_count[0]} iterations")
        
        # Return optimized parameters
        return {
            'transl': transl.detach(),
            'global_orient': global_orient.detach(),
            'body_pose': body_pose.detach(),
            'betas': betas.detach(),
            'expression': expression,
            'left_hand_pose': left_hand_pose,
            'right_hand_pose': right_hand_pose,
        }
    
    def phase2_surface_refinement(
        self,
        phase1_params: Dict[str, torch.Tensor],
        point_cloud: torch.Tensor,
        silhouette_mask: torch.Tensor,
        cam_intrinsics: torch.Tensor,
        image_size: Tuple[int, int],
        num_iterations: int = 500,
        weights: Optional[Dict[str, float]] = None,
        clothing_buffer: float = 0.03,  # 3cm
    ) -> Dict[str, torch.Tensor]:
        """
        Phase 2: Refine SMPL surface to point cloud and silhouette.
        
        Args:
            phase1_params: Parameters from Phase 1
            point_cloud: [M, 3] metric point cloud
            silhouette_mask: [H, W] binary silhouette mask
            cam_intrinsics: [3, 3] camera intrinsics
            image_size: (H, W) image dimensions
            num_iterations: Number of optimization iterations
            weights: Loss weights
            clothing_buffer: Clothing tolerance in meters (0.02-0.05)
            
        Returns:
            optimized_params: Refined SMPL-X parameters
        """
        if weights is None:
            weights = {
                'L_Chamfer': 10.0,      # Reduced from 1000 to prevent over-fitting
                'L_Silhouette': 1.0,    # Reduced from 10
                'L_Reg': 100.0,         # Increased from 10 to stay close to Phase 1
            }
        
        print("\n" + "=" * 50)
        print("Phase 2: Surface Refinement")
        print("=" * 50)
        
        # Clone parameters from Phase 1
        transl = phase1_params['transl'].clone().requires_grad_(True)
        global_orient = phase1_params['global_orient'].clone().requires_grad_(True)
        body_pose = phase1_params['body_pose'].clone().requires_grad_(True)
        betas = phase1_params['betas'].clone().requires_grad_(True)
        
        # Store Phase 1 params for regularization
        phase1_transl = phase1_params['transl'].clone()
        phase1_global_orient = phase1_params['global_orient'].clone()
        phase1_body_pose = phase1_params['body_pose'].clone()
        phase1_betas = phase1_params['betas'].clone()
        
        # Fixed parameters
        expression = phase1_params['expression'].clone()
        left_hand_pose = phase1_params['left_hand_pose'].clone()
        right_hand_pose = phase1_params['right_hand_pose'].clone()
        
        # Optimizer: Adam for surface refinement
        params_to_optimize = [transl, global_orient, body_pose, betas]
        optimizer = torch.optim.Adam(params_to_optimize, lr=0.01)
        
        # Prepare point cloud
        pcd = Pointclouds(points=[point_cloud])
        
        # Prepare silhouette mask
        H, W = image_size
        silhouette_tensor = torch.from_numpy(silhouette_mask).float().to(self.device)
        if silhouette_tensor.max() > 1.0:
            silhouette_tensor = silhouette_tensor / 255.0
        
        # Camera for rendering
        fx = cam_intrinsics[0, 0]
        fy = cam_intrinsics[1, 1]
        cx = cam_intrinsics[0, 2]
        cy = cam_intrinsics[1, 2]
        
        # PyTorch3D camera (note: different convention)
        # We need to convert from OpenCV to PyTorch3D
        focal_length = torch.tensor([[fx, fy]], device=self.device)
        principal_point = torch.tensor([[cx, cy]], device=self.device)
        
        cameras = PerspectiveCameras(
            focal_length=focal_length,
            principal_point=principal_point,
            image_size=((H, W),),
            device=self.device,
            in_ndc=False
        )
        
        # Rasterization settings
        raster_settings = RasterizationSettings(
            image_size=(H, W),
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        
        # Silhouette renderer
        silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftSilhouetteShader()
        )
        
        # Optimization loop
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            # Forward pass
            output = self.smplx_model(
                betas=betas,
                global_orient=global_orient,
                body_pose=body_pose,
                transl=transl,
                expression=expression,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                return_verts=True
            )
            
            vertices = output.vertices  # [1, V, 3]
            
            # L_Chamfer: One-way nearest neighbor with clothing buffer
            smpl_pcd = Pointclouds(points=[vertices[0]])
            
            # Compute distances from SMPL to point cloud
            loss_chamfer_raw, _ = chamfer_distance(smpl_pcd, pcd, single_directional=True)
            
            # Apply clothing buffer: don't penalize if within threshold
            # This is a soft version - we use a smooth penalty
            loss_chamfer = torch.clamp(loss_chamfer_raw - clothing_buffer, min=0.0)
            
            # L_Silhouette: Projected SMPL should be inside mask
            # This requires rendering - simplified version using projection
            loss_silhouette = torch.tensor(0.0, device=self.device)
            
            # Project vertices to 2D
            verts_2d_x = fx * vertices[0, :, 0] / (vertices[0, :, 2] + 1e-6) + cx
            verts_2d_y = fy * vertices[0, :, 1] / (vertices[0, :, 2] + 1e-6) + cy
            
            # Clip to image bounds
            verts_2d_x = torch.clamp(verts_2d_x, 0, W - 1)
            verts_2d_y = torch.clamp(verts_2d_y, 0, H - 1)
            
            # Sample mask at projected vertices
            # Use bilinear sampling
            verts_2d_x_norm = (verts_2d_x / (W - 1)) * 2 - 1
            verts_2d_y_norm = (verts_2d_y / (H - 1)) * 2 - 1
            grid = torch.stack([verts_2d_x_norm, verts_2d_y_norm], dim=-1).unsqueeze(0).unsqueeze(0)
            
            mask_values = F.grid_sample(
                silhouette_tensor.unsqueeze(0).unsqueeze(0),
                grid,
                align_corners=True,
                mode='bilinear'
            )
            
            # Penalize vertices outside the mask
            loss_silhouette = torch.mean((1.0 - mask_values.squeeze()) ** 2)
            
            # L_Reg: Regularization to Phase 1 results
            loss_reg = (
                F.mse_loss(transl, phase1_transl) +
                F.mse_loss(global_orient, phase1_global_orient) +
                F.mse_loss(body_pose, phase1_body_pose) +
                F.mse_loss(betas, phase1_betas)
            )
            
            # Total loss
            loss = (
                weights['L_Chamfer'] * loss_chamfer +
                weights['L_Silhouette'] * loss_silhouette +
                weights['L_Reg'] * loss_reg
            )
            
            if iteration % 50 == 0:
                print(f"Iter {iteration}: Loss={loss.item():.4f}, "
                      f"L_Chamfer={loss_chamfer.item():.6f}, "
                      f"L_Silhouette={loss_silhouette.item():.4f}, "
                      f"L_Reg={loss_reg.item():.4f}")
            
            loss.backward()
            optimizer.step()
        
        print(f"\nPhase 2 completed after {num_iterations} iterations")
        
        # Return optimized parameters
        return {
            'transl': transl.detach(),
            'global_orient': global_orient.detach(),
            'body_pose': body_pose.detach(),
            'betas': betas.detach(),
            'expression': expression,
            'left_hand_pose': left_hand_pose,
            'right_hand_pose': right_hand_pose,
        }
    
    def fit(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        keypoints_2d: np.ndarray,
        cam_intrinsics: np.ndarray,
        point_cloud: Optional[np.ndarray] = None,
        sam_checkpoint: Optional[str] = None,
        phase1_iterations: int = 300,
        phase2_iterations: int = 500,
        phase1_weights: Optional[Dict[str, float]] = None,
        phase2_weights: Optional[Dict[str, float]] = None,
        clothing_buffer: float = 0.03,
    ) -> Dict[str, torch.Tensor]:
        """
        Full two-phase fitting pipeline.
        
        Args:
            image: [H, W, 3] RGB image
            depth_map: [H, W] metric depth map
            keypoints_2d: [N, 3] YOLO keypoints (x, y, conf)
            cam_intrinsics: [3, 3] or [1, 3, 3] camera intrinsics
            point_cloud: [M, 3] optional pre-computed point cloud
            sam_checkpoint: Path to SAM checkpoint
            phase1_iterations: Phase 1 optimization iterations
            phase2_iterations: Phase 2 optimization iterations
            phase1_weights: Phase 1 loss weights
            phase2_weights: Phase 2 loss weights
            clothing_buffer: Clothing tolerance in meters
            
        Returns:
            final_params: Final optimized SMPL-X parameters
        """
        print("\n" + "=" * 60)
        print("METRIC SMPL-X FITTING PIPELINE")
        print("=" * 60)
        
        # Reshape intrinsics if needed
        if cam_intrinsics.ndim == 3:
            cam_intrinsics = cam_intrinsics[0]
        
        # Phase 0: Data Preparation
        print("\nPhase 0: Data Preparation")
        print("-" * 60)
        
        # Lift 2D keypoints to 3D
        keypoints_3d, valid_mask, filtered_depth = self.lift_2d_to_3d(
            keypoints_2d,
            depth_map,
            cam_intrinsics
        )
        
        print(f"Lifted {valid_mask.sum()}/{len(valid_mask)} keypoints to 3D")
        
        # Generate SAM mask
        print("Generating person segmentation mask...")
        silhouette_mask = self.generate_sam_mask(image, keypoints_2d, sam_checkpoint)
        print(f"Mask coverage: {silhouette_mask.sum() / silhouette_mask.size * 100:.1f}%")
        
        # Convert to torch tensors
        keypoints_3d_torch = torch.from_numpy(keypoints_3d).float().to(self.device)
        keypoints_2d_torch = torch.from_numpy(keypoints_2d[:, :2]).float().to(self.device)
        valid_mask_torch = torch.from_numpy(valid_mask).bool().to(self.device)
        print(f" camera intrinsics:\n{cam_intrinsics}")
        cam_intrinsics = np.array(cam_intrinsics)
        cam_intrinsics_torch = torch.from_numpy(cam_intrinsics).float().to(self.device)
        
        # Phase 1: Skeleton Fitting
        phase1_params = self.phase1_skeleton_fitting(
            keypoints_3d_torch,
            keypoints_2d_torch,
            valid_mask_torch,
            cam_intrinsics_torch,
            num_iterations=phase1_iterations,
            weights=phase1_weights
        )
        
        # Store Phase 1 results
        self.phase1_params = phase1_params
        
        # Phase 2: Surface Refinement
        if point_cloud is None:
            # Generate point cloud from depth map
            print("\nGenerating point cloud from depth map...")
            point_cloud = self._depth_to_point_cloud(filtered_depth, cam_intrinsics)
        
        # Subsample point cloud for efficiency
        if len(point_cloud) > 50000:
            indices = np.random.choice(len(point_cloud), 50000, replace=False)
            point_cloud = point_cloud[indices]
        
        point_cloud_torch = torch.from_numpy(point_cloud).float().to(self.device)
        
        phase2_params = self.phase2_surface_refinement(
            phase1_params,
            point_cloud_torch,
            silhouette_mask,
            cam_intrinsics_torch,
            image_size=(image.shape[0], image.shape[1]),
            num_iterations=phase2_iterations,
            weights=phase2_weights,
            clothing_buffer=clothing_buffer
        )
        
        # Store Phase 2 results
        self.phase2_params = phase2_params
        
        print("\n" + "=" * 60)
        print("FITTING COMPLETE")
        print("=" * 60)
        
        return phase2_params
    
    def _depth_to_point_cloud(
        self,
        depth_map: np.ndarray,
        cam_intrinsics: np.ndarray
    ) -> np.ndarray:
        """Convert depth map to point cloud."""
        H, W = depth_map.shape
        fx = cam_intrinsics[0, 0]
        fy = cam_intrinsics[1, 1]
        cx = cam_intrinsics[0, 2]
        cy = cam_intrinsics[1, 2]
        
        # Create mesh grid
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        
        # Unproject
        z = depth_map
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        # Stack and reshape
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        
        # Filter invalid points
        valid = (points[:, 2] > 0) & np.isfinite(points).all(axis=1)
        points = points[valid]
        
        return points
    
    def get_mesh(
        self,
        params: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get SMPL-X mesh vertices and faces.
        
        Args:
            params: SMPL-X parameters (uses phase2_params if None)
            
        Returns:
            vertices: [V, 3] mesh vertices
            faces: [F, 3] mesh faces
        """
        if params is None:
            params = self.phase2_params
            if params is None:
                raise ValueError("No fitted parameters available. Run fit() first.")
        
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
    print("MetricSMPLFitter initialized")
    print("Use fit() method to run the two-phase fitting pipeline")
