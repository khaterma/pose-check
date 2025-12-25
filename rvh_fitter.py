"""
RVH SMPL-X Fitter
=================
SMPL-X fitter using RVH-style multi-stage optimization approach.
Combines 2D reprojection, 3D keypoint lifting, chamfer distance, and point-mesh losses.

Inputs: YOLO 2D keypoints + camera intrinsics + depth map + point cloud
Output: Optimized SMPL-X parameters

Follows the optimization strategy from fit_SMPLH_pcloud.py:
- Stage 1: Global orientation optimization using 3D keypoints
- Stage 2: Full pose optimization using 3D keypoints  
- Stage 3: Fine-tuning with chamfer distance and point-to-mesh losses
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
import smplx
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import point_mesh_face_distance, chamfer_distance
from pytorch3d.ops import knn_points
import cv2
import sys
from tqdm import tqdm

sys.path.append('/home/khater/human_body_prior')

# VPoser import
try:
    from human_body_prior.tools.model_loader import load_model
    from human_body_prior.models.vposer_model import VPoser
    VPOSER_AVAILABLE = True
except ImportError:
    VPOSER_AVAILABLE = False
    print("Warning: VPoser not available. Install with: pip install git+https://github.com/nghorbani/human_body_prior")


class RVHSMPLXFitter:
    """
    SMPL-X fitter using RVH-style multi-stage optimization.
    
    Combines:
    - 2D reprojection error from joint poses
    - 3D keypoint lifting and optimization
    - Chamfer distance to point cloud
    - Point-to-mesh face distance
    - Shape and pose priors (VPoser optional)
    """
    
    def __init__(
        self,
        smplx_model_path: str = "./data/smplx",
        gender: str = "neutral",
        device: str = "cuda",
        image: Optional[np.ndarray] = None,
        vposer_model_path: Optional[str] = None,
        use_vposer: bool = True,
        debug: bool = False,
    ):
        """
        Initialize SMPL-X model and fitter.
        
        Args:
            smplx_model_path: Path to SMPL-X model files
            gender: 'neutral', 'male', or 'female'
            device: 'cuda' or 'cpu'
            vposer_model_path: Path to VPoser model
            use_vposer: Whether to use VPoser prior
            debug: Enable debug visualizations
        """
        self.device = torch.device(device)
        self.gender = gender
        self.debug = debug
        
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
        
        # Initialize VPoser
        self.use_vposer = use_vposer and VPOSER_AVAILABLE
        print(f"Use VPoser: {self.use_vposer}, VPOSER_AVAILABLE: {VPOSER_AVAILABLE}")
        self.vposer = None
        if self.use_vposer:
            if vposer_model_path is None:
                print("Warning: use_vposer=True but no vposer_model_path provided. Disabling VPoser.")
                self.use_vposer = False
            else:
                try:
                    self.vposer, _ = load_model(
                        vposer_model_path,
                        model_code=VPoser,
                        remove_words_in_model_weights='vp_model.',
                    )
                    self.vposer = self.vposer.to(self.device)
                    self.vposer.eval()
                    print(f"VPoser loaded from {vposer_model_path}")
                except Exception as e:
                    print(f"Failed to load VPoser: {e}. Disabling VPoser.")
                    self.use_vposer = False
        
        if image is not None:
            self.image = image
        
        # YOLO (17 keypoints) to SMPL-X joint mapping
        self.keypoint_mapping = {
            0: 15,    # nose
            5: 16,    # left_shoulder
            6: 17,    # right_shoulder
            7: 18,    # left_elbow
            8: 19,    # right_elbow
            9: 20,    # left_wrist
            10: 21,   # right_wrist
            11: 1,    # left_hip
            12: 2,    # right_hip
            13: 4,    # left_knee
            14: 5,    # right_knee
            15: 7,    # left_ankle
            16: 8,    # right_ankle
        }
        
        # Keypoint weights (following RVH style)
        self.keypoint_weights = torch.tensor([
            1.0,  # nose
            0.0, 0.0, 0.0, 0.0,  # eyes, ears (skip)
            1.0, 1.0,  # shoulders
            1.0, 1.0,  # elbows
            1.0, 1.0,  # wrists
            1.0, 1.0,  # hips
            1.0, 1.0,  # knees
            1.0, 1.0,  # ankles
        ], dtype=torch.float32, device=self.device).reshape(-1, 1)
        
        self.fitted_params = None
    
    def get_loss_weights(self, phase: str = 'global'):
        """
        Get loss weights for different optimization phases.
        Following fit_SMPLH_pcloud.py style.
        """
        if phase == 'global':
            # Initial global orientation optimization
            return {
                'joints3d': lambda cst, it: 40. ** 2 * cst / (1 + it),
                'reproj': lambda cst, it: 10. ** 2 * cst / (1 + it),
                'beta': lambda cst, it: 10. ** -2 * cst / (1 + it),  # Reduced to allow shape changes
                'pose_prior': lambda cst, it: 10. ** -5 * cst / (1 + it),
            }
        elif phase == 'pose':
            # Full pose optimization with keypoints - optimize shape for limb lengths
            return {
                'joints3d': lambda cst, it: 40. ** 2 * cst / (1 + it),
                'reproj': lambda cst, it: 10. ** 2 * cst / (1 + it),
                'beta': lambda cst, it: 10. ** -2 * cst / (1 + it),  # Reduced to allow shape changes
                'pose_prior': lambda cst, it: 10. ** -5 * cst / (1 + it),
            }
        elif phase == 'all':
            # Add chamfer distance - shape helps fit point cloud
            return {
                'joints3d': lambda cst, it: 40. ** 2 * cst / (1 + it),
                'reproj': lambda cst, it: 5. ** 2 * cst / (1 + it),
                'chamf': lambda cst, it: 30. ** 2 * cst / (1 + it),
                'beta': lambda cst, it: 10. ** -2 * cst / (1 + it),  # Reduced to allow shape changes
                'pose_prior': lambda cst, it: 10. ** -5 * cst / (1 + it),
            }
        elif phase == 'tune':
            # Fine-tuning with point-to-mesh - continue shape optimization
            return {
                'joints3d': lambda cst, it: 20. ** 2 * cst / (1 + it),
                'reproj': lambda cst, it: 5. ** 2 * cst / (1 + it),
                'chamf': lambda cst, it: 30. ** 2 * cst / (1 + it),
                'p2mf': lambda cst, it: 10. ** 2 * cst * (1 + it),
                'beta': lambda cst, it: 10. ** -2 * cst / (1 + it),  # Keep optimizing shape
                'pose_prior': lambda cst, it: 10. ** -5 * cst / (1 + it),
            }
        else:
            raise ValueError(f"Unknown phase: {phase}")
    
    @staticmethod
    def backward_step(loss_dict: Dict[str, torch.Tensor], 
                      weight_dict: Dict[str, callable], 
                      it: int) -> torch.Tensor:
        """Compute weighted total loss (following base_fitter.py style)."""
        w_loss = {}
        for k in loss_dict:
            if k in weight_dict:
                w_loss[k] = weight_dict[k](loss_dict[k], it)
        
        tot_loss = list(w_loss.values())
        tot_loss = torch.stack(tot_loss).sum()
        return tot_loss
    
    def _project_points(
        self,
        points_3d: torch.Tensor,
        cam_intrinsics: torch.Tensor
    ) -> torch.Tensor:
        """Project 3D points to 2D using camera intrinsics."""
        fx = cam_intrinsics[0, 0]
        fy = cam_intrinsics[1, 1]
        cx = cam_intrinsics[0, 2]
        cy = cam_intrinsics[1, 2]
        
        x_2d = fx * points_3d[:, 0] / (points_3d[:, 2] + 1e-6) + cx
        y_2d = fy * points_3d[:, 1] / (points_3d[:, 2] + 1e-6) + cy
        
        return torch.stack([x_2d, y_2d], dim=1)
    
    def _lift_keypoints_to_3d(
        self,
        keypoints_2d: torch.Tensor,
        depth_map: torch.Tensor,
        cam_intrinsics: torch.Tensor,
        confidence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Lift 2D keypoints to 3D using depth map.
        
        Returns:
            keypoints_3d: [N, 3] 3D keypoints
            valid_mask: [N] boolean mask for valid keypoints
        """
        fx = cam_intrinsics[0, 0]
        fy = cam_intrinsics[1, 1]
        cx = cam_intrinsics[0, 2]
        cy = cam_intrinsics[1, 2]
        
        H, W = depth_map.shape
        keypoints_3d = []
        valid_mask = []
        
        for i, (kp, conf) in enumerate(zip(keypoints_2d, confidence)):
            u, v = int(kp[0].item()), int(kp[1].item())
            
            # Check bounds and confidence
            if 0 <= u < W and 0 <= v < H and conf > 0.3:
                z = depth_map[v, u]
                if z > 0.1:  # Valid depth
                    x = (u - cx.item()) * z.item() / fx.item()
                    y = (v - cy.item()) * z.item() / fy.item()
                    keypoints_3d.append(torch.tensor([x, y, z], device=self.device))
                    valid_mask.append(True)
                else:
                    keypoints_3d.append(torch.zeros(3, device=self.device))
                    valid_mask.append(False)
            else:
                keypoints_3d.append(torch.zeros(3, device=self.device))
                valid_mask.append(False)
        
        return torch.stack(keypoints_3d), torch.tensor(valid_mask, device=self.device)
    
    def _compute_joints3d_loss(
        self,
        smpl_joints: torch.Tensor,
        keypoints_3d: torch.Tensor,
        valid_mask: torch.Tensor,
        confidence: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute 3D joint loss between SMPL joints and lifted keypoints.
        Following batch_3djoints_loss style from body_objectives.py.
        """
        loss = torch.tensor(0.0, device=self.device)
        count = 0
        
        for yolo_idx, smpl_idx in self.keypoint_mapping.items():
            if yolo_idx < len(valid_mask) and valid_mask[yolo_idx]:
                smpl_joint = smpl_joints[smpl_idx]
                target_joint = keypoints_3d[yolo_idx]
                conf = confidence[yolo_idx] if yolo_idx < len(confidence) else 1.0
                weight = self.keypoint_weights[yolo_idx].item() if yolo_idx < len(self.keypoint_weights) else 1.0
                
                joint_loss = torch.sum((smpl_joint - target_joint) ** 2) * conf * weight
                loss = loss + joint_loss
                count += 1
        
        if count > 0:
            loss = loss / count
        
        return loss
    
    def _compute_reproj_loss(
        self,
        smpl_joints: torch.Tensor,
        keypoints_2d: torch.Tensor,
        cam_intrinsics: torch.Tensor,
        valid_mask: torch.Tensor,
        confidence: torch.Tensor
    ) -> torch.Tensor:
        """Compute 2D reprojection loss."""
        pred_2d_list = []
        target_2d_list = []
        weights_list = []
        
        for yolo_idx, smpl_idx in self.keypoint_mapping.items():
            if yolo_idx < len(valid_mask) and valid_mask[yolo_idx]:
                joint_3d = smpl_joints[smpl_idx]
                joint_2d = self._project_points(joint_3d.unsqueeze(0), cam_intrinsics)[0]
                
                pred_2d_list.append(joint_2d)
                target_2d_list.append(keypoints_2d[yolo_idx])
                conf = confidence[yolo_idx] if yolo_idx < len(confidence) else 1.0
                weight = self.keypoint_weights[yolo_idx].item() if yolo_idx < len(self.keypoint_weights) else 1.0
                weights_list.append(conf * weight)
        
        if len(pred_2d_list) < 3:
            return torch.tensor(0.0, device=self.device)
        
        pred_2d = torch.stack(pred_2d_list)
        target_2d = torch.stack(target_2d_list)
        weights = torch.tensor(weights_list, device=self.device).unsqueeze(1)
        
        loss = torch.mean(((pred_2d - target_2d) ** 2) * weights)
        return loss
    
    def _compute_chamfer_loss(
        self,
        vertices: torch.Tensor,
        point_cloud: torch.Tensor,
        bidirectional: bool = False
    ) -> torch.Tensor:
        """
        Compute chamfer distance between SMPL vertices and point cloud.
        Following batch_chamfer style from torch_functions.py.
        """
        # Use KNN-based chamfer distance
        w1, w2 = (1.0, 1.0) if bidirectional else (1.0, 0.0)
        
        verts = vertices.unsqueeze(0)  # [1, V, 3]
        pc = point_cloud.unsqueeze(0)  # [1, N, 3]
        
        # Distance from vertices to point cloud
        dist_v2p = knn_points(verts, pc, K=1)
        loss_v2p = (dist_v2p.dists ** 0.5).mean()
        
        if bidirectional:
            # Distance from point cloud to vertices
            dist_p2v = knn_points(pc, verts, K=1)
            loss_p2v = (dist_p2v.dists ** 0.5).mean()
            return w1 * loss_v2p + w2 * loss_p2v
        
        return loss_v2p
    
    def _compute_p2mesh_loss(
        self,
        vertices: torch.Tensor,
        point_cloud: torch.Tensor
    ) -> torch.Tensor:
        """Compute point-to-mesh face distance."""
        faces = torch.from_numpy(self.smplx_model.faces).long().to(self.device)
        mesh = Meshes(verts=[vertices], faces=[faces])
        pclouds = Pointclouds([point_cloud])
        
        loss = point_mesh_face_distance(mesh, pclouds)
        return loss
    
    def _initialize_params(
        self,
        keypoints_2d: np.ndarray,
        valid_mask: np.ndarray,
        cam_intrinsics: np.ndarray,
        init_depth: float = 3.0
    ) -> Dict[str, torch.Tensor]:
        """Initialize SMPL-X parameters."""
        valid_kpts = keypoints_2d[valid_mask]
        
        if len(valid_kpts) > 0:
            center_2d = valid_kpts.mean(axis=0)
            fx = cam_intrinsics[0, 0]
            fy = cam_intrinsics[1, 1]
            cx = cam_intrinsics[0, 2]
            cy = cam_intrinsics[1, 2]
            
            x = (center_2d[0] - cx) * init_depth / fx
            y = (center_2d[1] - cy) * init_depth / fy
            z = init_depth
            
            transl = torch.tensor([[x, y, z]], dtype=torch.float32, device=self.device)
        else:
            transl = torch.tensor([[0.0, 0.0, init_depth]], dtype=torch.float32, device=self.device)
        
        # Global orientation facing camera
        global_orient = torch.tensor([[np.pi, 0, 0]], dtype=torch.float32, device=self.device)
        
        # Body pose
        if self.use_vposer:
            poser_latent = torch.zeros(1, 32, dtype=torch.float32, device=self.device)
        else:
            body_pose = torch.zeros(1, 63, dtype=torch.float32, device=self.device)
        
        betas = torch.zeros(1, 10, dtype=torch.float32, device=self.device)
        left_hand_pose = torch.zeros(1, 6, dtype=torch.float32, device=self.device)
        right_hand_pose = torch.zeros(1, 6, dtype=torch.float32, device=self.device)
        expression = torch.zeros(1, 10, dtype=torch.float32, device=self.device)
        
        params = {
            'transl': transl,
            'global_orient': global_orient,
            'betas': betas,
            'left_hand_pose': left_hand_pose,
            'right_hand_pose': right_hand_pose,
            'expression': expression,
        }
        
        if self.use_vposer:
            params['poser_latent'] = poser_latent
        else:
            params['body_pose'] = body_pose
        
        return params
    
    def fit(
        self,
        keypoints_2d: np.ndarray,
        cam_intrinsics: np.ndarray,
        depth_map: np.ndarray,
        mask: np.ndarray,
        feet_mask: np.ndarray,
        point_cloud: np.ndarray,
        conf_threshold: float = 0.3,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Fit SMPL-X using multi-stage RVH-style optimization.
        
        Args:
            keypoints_2d: [N, 3] YOLO keypoints (x, y, confidence)
            cam_intrinsics: [3, 3] camera intrinsics matrix
            depth_map: [H, W] depth map
            mask: [H, W] segmentation mask
            feet_mask: feet segmentation mask (for future use)
            point_cloud: [N, 3] 3D point cloud
            conf_threshold: Confidence threshold for valid keypoints
            
        Returns:
            fitted_params: Optimized SMPL-X parameters
        """
        print("\n" + "=" * 60)
        print("RVH SMPL-X FITTING (Multi-Stage Optimization)")
        if self.use_vposer:
            print("Using VPoser prior for body pose")
        print("=" * 60)
        
        # Prepare data
        valid_mask = keypoints_2d[:, 2] > conf_threshold
        
        if valid_mask.sum() < 6:
            raise ValueError(f"Not enough valid keypoints: {valid_mask.sum()}")
        
        print(f"Valid keypoints: {valid_mask.sum()}/{len(keypoints_2d)}")
        
        # Convert to torch
        keypoints_2d_torch = torch.from_numpy(keypoints_2d[:, :2]).float().to(self.device)
        confidence_torch = torch.from_numpy(keypoints_2d[:, 2]).float().to(self.device)
        valid_mask_torch = torch.from_numpy(valid_mask).bool().to(self.device)
        depth_map_torch = torch.from_numpy(depth_map).float().to(self.device)
        point_cloud_torch = torch.from_numpy(point_cloud).float().to(self.device)
        
        if cam_intrinsics.ndim == 3:
            cam_intrinsics = cam_intrinsics[0]
        cam_intrinsics_torch = torch.from_numpy(cam_intrinsics).float().to(self.device)
        
        # Lift keypoints to 3D
        keypoints_3d, valid_3d_mask = self._lift_keypoints_to_3d(
            keypoints_2d_torch, depth_map_torch, cam_intrinsics_torch, confidence_torch
        )
        print(f"3D keypoints lifted: {valid_3d_mask.sum()}/{len(valid_3d_mask)}")
        
        # Initialize parameters
        params = self._initialize_params(
            keypoints_2d[:, :2], valid_mask, cam_intrinsics
        )
        
        # Set up optimization variables
        transl = params['transl'].clone().requires_grad_(True)
        global_orient = params['global_orient'].clone().requires_grad_(True)
        betas = params['betas'].clone().requires_grad_(True)
        left_hand_pose = params['left_hand_pose']
        right_hand_pose = params['right_hand_pose']
        expression = params['expression']
        
        if self.use_vposer:
            poser_latent = params['poser_latent'].clone().requires_grad_(True)
        else:
            body_pose = params['body_pose'].clone().requires_grad_(True)
        
        # ===================================================================
        # OPTIMIZATION STAGES (following fit_SMPLH_pcloud.py structure)
        # ===================================================================
        
        iter_for_global = 5
        iter_for_pose = 15
        iter_for_all = 15
        iter_for_tune = 3
        steps_per_iter = 10
        
        total_iterations = iter_for_global + iter_for_pose + iter_for_all + iter_for_tune
        
        print(f"\nOptimization plan:")
        print(f"  Global orientation: {iter_for_global} iters")
        print(f"  Full pose (keypoints): {iter_for_pose} iters")
        print(f"  Full pose + chamfer: {iter_for_all} iters")
        print(f"  Fine-tuning + p2mesh: {iter_for_tune} iters")
        print(f"  Steps per iteration: {steps_per_iter}")
        
        phase = 'global'
        weight_dict = self.get_loss_weights(phase)
        
        # Initial optimizer (global orientation only)
        if self.use_vposer:
            optimizer = torch.optim.Adam([transl, global_orient, betas], lr=0.01)
        else:
            optimizer = torch.optim.Adam([transl, global_orient, betas], lr=0.01)
        
        for it in tqdm(range(total_iterations), desc='Total Progress'):
            # Update phase and optimizer
            if it == iter_for_global:
                phase = 'pose'
                weight_dict = self.get_loss_weights(phase)
                if self.use_vposer:
                    optimizer = torch.optim.Adam([transl, global_orient, betas, poser_latent], lr=0.004)
                else:
                    optimizer = torch.optim.Adam([transl, global_orient, betas, body_pose], lr=0.004)
                print("\n→ Phase: Full pose optimization (keypoints)")
                
            elif it == iter_for_global + iter_for_pose:
                phase = 'all'
                weight_dict = self.get_loss_weights(phase)
                if self.use_vposer:
                    optimizer = torch.optim.Adam([transl, global_orient, betas, poser_latent], lr=0.002)
                else:
                    optimizer = torch.optim.Adam([transl, global_orient, betas, body_pose], lr=0.002)
                print("\n→ Phase: Chamfer distance optimization")
                
            elif it == iter_for_global + iter_for_pose + iter_for_all:
                phase = 'tune'
                weight_dict = self.get_loss_weights(phase)
                if self.use_vposer:
                    optimizer = torch.optim.Adam([transl, global_orient, betas, poser_latent], lr=0.001)
                else:
                    optimizer = torch.optim.Adam([transl, global_orient, betas, body_pose], lr=0.001)
                print("\n→ Phase: Fine-tuning with point-to-mesh (including shape)")
            
            # Inner optimization loop
            for step in range(steps_per_iter):
                optimizer.zero_grad()
                
                # Decode body pose from VPoser
                if self.use_vposer:
                    body_pose_decoded = self.vposer.decode(poser_latent, output_type='aa')['pose_body']
                else:
                    body_pose_decoded = body_pose
                
                # Forward pass
                output = self.smplx_model(
                    betas=betas,
                    global_orient=global_orient,
                    body_pose=body_pose_decoded,
                    transl=transl,
                    left_hand_pose=left_hand_pose,
                    right_hand_pose=right_hand_pose,
                    expression=expression,
                    return_verts=True
                )
                
                smpl_joints = output.joints[0]  # [127, 3]
                vertices = output.vertices[0]   # [V, 3]
                
                # Compute losses
                loss_dict = {}
                
                # 3D joint loss
                loss_dict['joints3d'] = self._compute_joints3d_loss(
                    smpl_joints, keypoints_3d, valid_3d_mask, confidence_torch
                )
                
                # 2D reprojection loss
                loss_dict['reproj'] = self._compute_reproj_loss(
                    smpl_joints, keypoints_2d_torch, cam_intrinsics_torch,
                    valid_mask_torch, confidence_torch
                )
                
                # Shape prior
                loss_dict['beta'] = torch.mean(betas ** 2)
                
                # Pose prior
                if self.use_vposer:
                    loss_dict['pose_prior'] = torch.mean(poser_latent ** 2)
                else:
                    loss_dict['pose_prior'] = torch.mean(body_pose_decoded ** 2)
                
                # Chamfer distance (for 'all' and 'tune' phases)
                if phase in ['all', 'tune']:
                    loss_dict['chamf'] = self._compute_chamfer_loss(
                        vertices, point_cloud_torch, bidirectional=False
                    )
                
                # Point-to-mesh distance (for 'tune' phase)
                if phase == 'tune':
                    loss_dict['p2mf'] = self._compute_p2mesh_loss(vertices, point_cloud_torch)
                
                # Compute weighted total loss
                total_loss = self.backward_step(loss_dict, weight_dict, it)
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
            
            # Logging
            if it % 5 == 0 or it == total_iterations - 1:
                loss_str = f"Iter {it:03d} ({phase})"
                for k, v in loss_dict.items():
                    weighted = weight_dict.get(k, lambda x, i: x)(v, it)
                    loss_str += f" | {k}: {weighted.item():.4f}"
                tqdm.write(loss_str)
                
                # Debug visualization
                if self.debug and hasattr(self, 'image'):
                    self._visualize_fitting(it, params={
                        'transl': transl.detach(),
                        'global_orient': global_orient.detach(),
                        'body_pose': body_pose_decoded.detach(),
                        'betas': betas.detach(),
                        'left_hand_pose': left_hand_pose,
                        'right_hand_pose': right_hand_pose,
                        'expression': expression,
                    })
        
        print("\n" + "=" * 60)
        print("FITTING COMPLETE")
        print("=" * 60)
        
        # Final body pose
        if self.use_vposer:
            with torch.no_grad():
                body_pose_final = self.vposer.decode(poser_latent, output_type='aa')['pose_body']
        else:
            body_pose_final = body_pose
        
        # Store results
        self.fitted_params = {
            'transl': transl.detach(),
            'global_orient': global_orient.detach(),
            'body_pose': body_pose_final.detach(),
            'betas': betas.detach(),
            'left_hand_pose': left_hand_pose,
            'right_hand_pose': right_hand_pose,
            'expression': expression,
        }
        
        if self.use_vposer:
            self.fitted_params['poser_latent'] = poser_latent.detach()
        
        return self.fitted_params
    
    def _visualize_fitting(self, iteration: int, params: Dict[str, torch.Tensor]):
        """Save debug visualization of current fitting state."""
        if not hasattr(self, 'image'):
            return
        
        overlay = self.project_mesh_on_image(params, self.image, None)
        cv2.imwrite(f"output/rvh_overlay_{iteration:04d}.png", overlay)
    
    def project_mesh_on_image(
        self,
        params: Dict[str, torch.Tensor],
        image: np.ndarray,
        cam_intrinsics: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Project SMPL mesh onto image for visualization."""
        vertices, faces = self.get_mesh(params)
        
        # Get joints for visualization
        smpl_output = self.smplx_model(
            betas=params['betas'],
            global_orient=params['global_orient'],
            body_pose=params['body_pose'],
            transl=params['transl'],
            left_hand_pose=params['left_hand_pose'],
            right_hand_pose=params['right_hand_pose'],
            expression=params['expression'],
            return_verts=False,
        )
        joints = smpl_output.joints[0].detach().cpu().numpy()
        
        # Use stored intrinsics if not provided
        if cam_intrinsics is None:
            if hasattr(self, 'cam_intrinsics_np'):
                cam_intrinsics = self.cam_intrinsics_np
            else:
                # Estimate from image size
                H, W = image.shape[:2]
                cam_intrinsics = np.array([
                    [W, 0, W/2],
                    [0, W, H/2],
                    [0, 0, 1]
                ])
        
        if cam_intrinsics.ndim == 3:
            cam_intrinsics = cam_intrinsics[0]
        
        fx, fy = cam_intrinsics[0, 0], cam_intrinsics[1, 1]
        cx, cy = cam_intrinsics[0, 2], cam_intrinsics[1, 2]
        
        overlay = image.copy()
        
        # Draw joints
        joint_indices = [15, 16, 17, 18, 19, 7, 8]  # Key joints to visualize
        joint_names = ['nose', 'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_ankle', 'r_ankle']
        
        for idx, name in zip(joint_indices, joint_names):
            joint_3d = joints[idx]
            if joint_3d[2] > 0:
                u = int(fx * joint_3d[0] / joint_3d[2] + cx)
                v = int(fy * joint_3d[1] / joint_3d[2] + cy)
                cv2.circle(overlay, (u, v), 5, (255, 0, 0), -1)
                cv2.putText(overlay, name, (u+5, v-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # Draw mesh edges
        vertices_2d = np.zeros((len(vertices), 2))
        for i, v in enumerate(vertices):
            if v[2] > 0:
                vertices_2d[i, 0] = fx * v[0] / v[2] + cx
                vertices_2d[i, 1] = fy * v[1] / v[2] + cy
        
        for face in faces[::10]:
            pts = vertices_2d[face].astype(np.int32)
            if np.all((pts[:, 0] >= 0) & (pts[:, 0] < image.shape[1]) &
                      (pts[:, 1] >= 0) & (pts[:, 1] < image.shape[0])):
                cv2.polylines(overlay, [pts], True, (0, 255, 0), 1)
        
        return overlay
    
    def get_mesh(
        self,
        params: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get SMPL-X mesh vertices and faces."""
        if params is None:
            params = self.fitted_params
            if params is None:
                raise ValueError("No fitted parameters. Run fit() first.")
        
        with torch.no_grad():
            output = self.smplx_model(
                betas=params['betas'],
                global_orient=params['global_orient'],
                body_pose=params['body_pose'],
                transl=params['transl'],
                left_hand_pose=params['left_hand_pose'],
                right_hand_pose=params['right_hand_pose'],
                expression=params['expression'],
                return_verts=True
            )
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
    print("RVHSMPLXFitter - Multi-stage SMPL-X optimization")
    print("Usage: fitter.fit(keypoints_2d, cam_intrinsics, depth_map, mask, feet_mask, point_cloud)")
