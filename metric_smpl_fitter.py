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
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import point_mesh_face_distance, chamfer_distance
import cv2
import sys
sys.path.append('/home/khater/human_body_prior')

# Add VPoser import
try:
    from human_body_prior.tools.model_loader import load_model
    from human_body_prior.models.vposer_model import VPoser
    VPOSER_AVAILABLE = True
except ImportError:
    VPOSER_AVAILABLE = False
    print("Warning: VPoser not available. Install with: pip install git+https://github.com/nghorbani/human_body_prior")

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
        vposer_model_path: Optional[str] = None,  # Add VPoser path
        use_vposer: bool = True,
    ):
        """
        Initialize SMPL-X model.
        
        Args:
            smplx_model_path: Path to SMPL-X model files
            gender: 'neutral', 'male', or 'female'
            device: 'cuda' or 'cpu'
            vposer_model_path: Path to VPoser model (e.g., './data/vposer_v2_05')
            use_vposer: Whether to use VPoser prior
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
        
        # Initialize VPoser
        self.use_vposer = use_vposer and VPOSER_AVAILABLE
        print(f"Use VPoser: {self.use_vposer}, VPOSER_AVAILABLE: {VPOSER_AVAILABLE}, vposer_model_path: {vposer_model_path}")
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
        
        # # YOLO (17 keypoints) to SMPL-X joint mapping
        # # Only map the most reliable body joints
        self.keypoint_mapping = {
            0: 15,    # nose
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
            17: 10, # left_foot
            18: 11, # right_foot
        }

        # YOLO (17 keypoints) to SMPL-X joint mapping
        # Only map the most reliable body joints  # switch left and right for some reason
        # FLIPPED
        # self.keypoint_mapping = {
        #     0: 15,    # nose
        #     5: 17,   # left_shoulder
        #     6: 16,   # right_shoulder
        #     7: 19,   # left_elbow
        #     8: 18,   # right_elbow
        #     9: 21,   # left_wrist
        #     10: 20,  # right_wrist
        #     11: 2,   # left_hip
        #     12: 1,   # right_hip
        #     13: 5,   # left_knee
        #     14: 4,   # right_knee
        #     15: 8,   # left_ankle
        #     16: 7,   # right_ankle
        #     17: 11, # left_foot
        #     18: 10, # right_foot
        #     }
        
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
            y = (center_2d[1] - cy) * init_depth / fy
            z = init_depth
            
            transl = torch.tensor([[x, y, z]], dtype=torch.float32, device=self.device)
        else:
            transl = torch.tensor([[0.0, 0.0, init_depth]], dtype=torch.float32, device=self.device)
        
        # Initialize pose and shape to zero (neutral pose)
        global_orient = torch.tensor([[np.pi, 0, 0]], dtype=torch.float32, device=self.device)
        # global_orient = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
        
        # Initialize body pose or VPoser latent code
        if self.use_vposer:
            # VPoser latent code (32-dimensional)
            poser_latent = torch.zeros(1, 32, dtype=torch.float32, device=self.device)
        else:
            body_pose = torch.zeros(1, 63, dtype=torch.float32, device=self.device)
        
        betas = torch.zeros(1, 10, dtype=torch.float32, device=self.device)
        
        # Fixed parameters (hands, face)
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
        y_2d = fy * points_3d[:, 1] / (points_3d[:, 2] + 1e-6) + cy
        
        return torch.stack([x_2d, y_2d], dim=1)
    
    def fit(
        self,
        keypoints_2d: np.ndarray,
        cam_intrinsics: np.ndarray,
        depth_map: np.ndarray,
        mask: np.ndarray,
        feet_mask: np.ndarray,
        point_cloud: np.ndarray,
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
                'reproj': 10.0,      # Reprojection error
                'pose_prior': 0.01,   # Pose regularization (or VPoser KL)
                'shape_prior': 0.1,   # Shape regularization
                'depth': 20.0,         # Depth supervision
                'p2mf': 1.0           # Point-to-mesh face distance
            }
        
        print("\n" + "=" * 60)
        print("SMPL-X FITTING")
        if self.use_vposer:
            print("Using VPoser prior for body pose")
        print("=" * 60)
        
        # Prepare data
        valid_mask = keypoints_2d[:, 2] > conf_threshold
        print(f" key points 2d shape: {keypoints_2d.shape},  confidence :{keypoints_2d[:, 2]} ")
        
        if valid_mask.sum() < 6:
            raise ValueError(f"Not enough valid keypoints: {valid_mask.sum()}")
        
        print(f"Valid keypoints: {valid_mask.sum()}/{len(keypoints_2d)}")
        
        # Convert to torch
        keypoints_2d_torch = torch.from_numpy(keypoints_2d[:, :2]).float().to(self.device)
        valid_mask_torch = torch.from_numpy(valid_mask).bool().to(self.device)
        valid_mask_torch = torch.cat([valid_mask_torch, torch.tensor([0, 0], dtype=torch.bool).to(self.device)], dim=0)
        depth_map_torch = torch.from_numpy(depth_map).float().to(self.device)
        mask_torch = torch.from_numpy(mask.astype(np.bool_)).to(self.device)
        
        if cam_intrinsics.ndim == 3:
            cam_intrinsics = cam_intrinsics[0]
        cam_intrinsics_torch = torch.from_numpy(cam_intrinsics).float().to(self.device)
        
        print(f" mask shape: {mask.shape}, depth_map shape: {depth_map.shape} ")

        depth_initalization = depth_map[mask.squeeze(0)].mean() 
        print(f" Depth initialization value: {depth_initalization} ")
        
        # Initialize parameters
        params = self._initialize_params(
            keypoints_2d[:, :2],
            valid_mask,
            cam_intrinsics,
            init_depth= depth_initalization
        )

        # visualize initial overlay
        init_overlay = self.project_mesh_on_image(params, self.image, cam_intrinsics)

        cv2.imwrite(f"output/overlay_init.png", init_overlay)
        
        # Set up optimization variables
        transl = params['transl'].clone().requires_grad_(True)
        global_orient = params['global_orient'].clone().requires_grad_(True)
        betas = params['betas'].clone().requires_grad_(True)
        left_hand_pose = params['left_hand_pose'].clone().requires_grad_(True)
        right_hand_pose = params['right_hand_pose'].clone().requires_grad_(True)
        
        # Body pose: VPoser latent or direct
        if self.use_vposer:
            poser_latent = params['poser_latent'].clone().requires_grad_(True)
            opt_params = [transl, global_orient, betas, poser_latent]
        else:
            body_pose = params['body_pose'].clone().requires_grad_(True)
            opt_params = [transl, global_orient, body_pose, betas]
        
        expression = params['expression']
        
        # Optimizer
        optimizer = torch.optim.Adam(opt_params, lr=lr)
        
        # Optimization loop
        print("\nOptimizing...")
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            # Decode body pose from VPoser if using it
            if self.use_vposer:
                body_pose = self.vposer.decode(poser_latent, output_type='aa')['pose_body']
                print(f" body_pose shape: {body_pose.shape} ")
            
            # Forward pass through SMPL-X
            output = self.smplx_model(
                betas=betas,
                global_orient=global_orient,
                body_pose=body_pose,
                transl=transl,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                expression=expression,
                return_verts=True
            )
            
            # Get SMPL-X joints
            smpl_joints = output.joints[0]  # [127, 3]
            vertices = output.vertices[0]  # [V, 3]
            
            # Map to YOLO keypoints and compute reprojection loss
            pred_2d_list = []
            target_2d_list = []
            
            for yolo_idx, smpl_idx in self.keypoint_mapping.items():
                if not valid_mask_torch[yolo_idx]:
                    continue
                
                joint_3d = smpl_joints[smpl_idx]
                joint_2d = self._project_points(
                    joint_3d.unsqueeze(0),
                    cam_intrinsics_torch
                )[0]
                
                pred_2d_list.append(joint_2d)
                target_2d_list.append(keypoints_2d_torch[yolo_idx])
            
            if len(pred_2d_list) < 3:
                print(f"Warning: Only {len(pred_2d_list)} valid joints")
                continue
            
            pred_2d = torch.stack(pred_2d_list)
            target_2d = torch.stack(target_2d_list)
            
            # Compute losses
            # L_reproj: Reprojection error
            loss_reproj = torch.mean((pred_2d - target_2d) ** 2)
            
            keypoints_3d, valid_mask_3d = self._lift_keypoints_to_3d(keypoints_2d_torch, depth_map_torch, cam_intrinsics_torch, keypoints_2d[:, 2])
            loss_3d_joints = self._compute_joints3d_loss(smpl_joints,keypoints_3d,valid_mask_3d, keypoints_2d[:, 2])
            # print(f"3D joints loss: {loss_3d_joints.item():.4f}")

            # L_pose: VPoser KL divergence or direct pose prior
            if self.use_vposer:
                # VPoser encourages latent codes near N(0, I)
                loss_pose = torch.mean(poser_latent ** 2)
                print(f" poser_latent norm: {torch.norm(poser_latent).item():.4f} ")
            else:
                # Direct pose prior
                loss_pose = torch.mean(body_pose ** 2)
            
            # L_shape: Shape prior (keep betas small)
            loss_shape = torch.mean(betas ** 2)

            # loss_depth = self.compute_depth_loss_with_rasterization(
            #     vertices=vertices, cam_intrinsics=cam_intrinsics_torch,point_cloud=torch.from_numpy(point_cloud).float().to(self.device), it=iteration)

            loss_depth = self.compute_depth_loss_with_point_cloud(
                vertices=vertices,
                cam_intrinsics=cam_intrinsics_torch,
                point_cloud=torch.from_numpy(point_cloud).float().to(self.device),
                it=iteration
            )
            # loss_depth = self.compute_depth_loss(
            #     vertices=vertices,
            #     cam_intrinsics=cam_intrinsics_torch,
            #     depth_map=depth_map_torch,
            #     mask=mask_torch,
            #     it=iteration
            # )
            
            loss_depth = torch.tensor(0.0).to(self.device)
            loss_s2m_face = torch.tensor(0.0).to(self.device)
            # if iteration > 300:
            #     faces = self.smplx_model.faces
            #     faces = torch.from_numpy(faces).long().to(self.device)
            #     mesh = Meshes(verts=[vertices], faces=[faces])
            #     pclouds = Pointclouds([torch.from_numpy(point_cloud).float().to(self.device)])

            #     loss_s2m_face = point_mesh_face_distance(mesh, pclouds)
            #     print(f"Loss S2M Face: {loss_s2m_face.item():.4f}")
            
            # Total loss
            loss = (
                weights['reproj'] * loss_reproj +
                weights['pose_prior'] * loss_pose +
                weights['shape_prior'] * loss_shape +
                weights['depth'] * loss_depth + 
                weights.get('p2mf', 0.0) * loss_s2m_face + 
                weights.get('3d_joints', 1.0) * loss_3d_joints
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Logging
            if iteration % 50 == 0:
                pose_type = "VPoser" if self.use_vposer else "Pose"
                print(
                    f"Iter {iteration:04d} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Reproj: {loss_reproj.item():.4f} | "
                    f"{pose_type}: {loss_pose.item():.6f} | "
                    f"Shape: {loss_shape.item():.6f} | "
                    f"Depth: {loss_depth.item():.4f}"
                    f" | P2MF: {loss_s2m_face.item():.4f}"
                )
                
                # Decode body_pose for visualization if using VPoser
                if self.use_vposer:
                    with torch.no_grad():
                        body_pose_vis = self.vposer.decode(poser_latent, output_type='aa')['pose_body']
                else:
                    body_pose_vis = body_pose
                
                params_vis = {
                    'transl': transl.detach(),
                    'global_orient': global_orient.detach(),
                    'body_pose': body_pose_vis.detach(),
                    'betas': betas.detach(),
                    'left_hand_pose': left_hand_pose,
                    'right_hand_pose': right_hand_pose,
                    'expression': expression,
                }
                init_overlay = self.project_mesh_on_image(params_vis, self.image, cam_intrinsics)
                cv2.imwrite(f"output/overlay_{iteration}.png", init_overlay)
        
        print("\n" + "=" * 60)
        print("FITTING COMPLETE")
        print("=" * 60)
        
        # Decode final body pose if using VPoser
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
                # weight = self.keypoint_weights[yolo_idx].item() if yolo_idx < len(self.keypoint_weights) else 1.0
                
                joint_loss = torch.sum((smpl_joint - target_joint) ** 2) * conf   #* weight
                loss = loss + joint_loss
                count += 1
        
        if count > 0:
            loss = loss / count
        
        return loss
    
    def compute_depth_loss_with_point_cloud(
        self,
        vertices: torch.Tensor,        # [V, 3]
        cam_intrinsics: torch.Tensor,  # [3, 3]
        point_cloud: torch.Tensor,     # [N, 3]
        it: int = 0
    ) -> torch.Tensor:
        """ 
        Compute depth loss between SMPLX vertices and point cloud. 
        Strategy:
        1. Project all vertices and create full z-buffer (no mask filtering)
        2. Sample the z_buffer closest to the camera. 
        3. apply chamfer distance between the point cloud and the z_buffer points.
        """
        device = vertices.device
        
        # ----------------------------------
        # 1. Project vertices to create z-buffer
        # ----------------------------------
        points_2d = self._project_points(vertices, cam_intrinsics)
        u = points_2d[:, 0].long()
        v = points_2d[:, 1].long()
        z = vertices[:, 2]
        
        # Assume some reasonable image dimensions (can be passed as parameter if needed)
        # Or extract from cam_intrinsics: H ≈ 2*cy, W ≈ 2*cx
        H = int(cam_intrinsics[1, 2].item() * 2)
        W = int(cam_intrinsics[0, 2].item() * 2)
        
        # Image bounds check
        inside = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)
        u, v, z = u[inside], v[inside], z[inside]
        
        if z.numel() == 0:
            return torch.zeros((), device=device)
        
        # ----------------------------------
        # 2. Build z-buffer (keep closest depth per pixel)
        # ----------------------------------
        pixel_ids = v * W + u
        unique_pixels, inverse_indices = torch.unique(pixel_ids, return_inverse=True)
        
        # Get minimum depth per pixel
        z_min = torch.full((unique_pixels.shape[0],), float('inf'), device=device)
        z_min.scatter_reduce_(0, inverse_indices, z, reduce='amin')
        
        # Reconstruct 3D points from z-buffer
        v_zbuf = unique_pixels // W
        u_zbuf = unique_pixels % W
        
        fx = cam_intrinsics[0, 0]
        fy = cam_intrinsics[1, 1]
        cx = cam_intrinsics[0, 2]
        cy = cam_intrinsics[1, 2]
        
        x_3d = (u_zbuf.float() - cx) * z_min / fx
        y_3d = (v_zbuf.float() - cy) * z_min / fy
        z_3d = z_min
        
        zbuffer_points = torch.stack([x_3d, y_3d, z_3d], dim=1)  # [M, 3]
        
        # ----------------------------------
        # 3. Compute chamfer distance
        # ----------------------------------
        # Ensure point_cloud is on the correct device
        if not isinstance(point_cloud, torch.Tensor):
            point_cloud = torch.from_numpy(point_cloud).float().to(device)
        else:
            point_cloud = point_cloud.to(device)
        
        # Create Pointclouds objects for chamfer distance
        pclouds_zbuf = Pointclouds([zbuffer_points])
        pclouds_gt = Pointclouds([point_cloud])
        
        loss, _ = chamfer_distance(pclouds_zbuf, pclouds_gt)
        
        if it % 50 == 0:
            print(f"Point cloud chamfer loss: {loss.item():.4f} "
                  f"(zbuf points: {zbuffer_points.shape[0]}, "
                  f"gt points: {point_cloud.shape[0]})")
            import matplotlib.pyplot as plt
            # Visualize z-buffer points
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(zbuffer_points[:, 0].detach().cpu().numpy(), zbuffer_points[:, 1].detach().cpu().numpy(), zbuffer_points[:, 2].detach().cpu().numpy(), c='r', s=1, label='Z-buffer Points')
            ax.scatter(point_cloud[:, 0].detach().cpu().numpy(), point_cloud[:, 1].detach().cpu().numpy(), point_cloud[:, 2].detach().cpu().numpy(), c='b', s=1, label='GT Point Cloud')
            ax.set_title('Z-buffer Points vs GT Point Cloud')
            ax.legend()
            plt.savefig(f"output/pointcloud_comparison_{it}.png")
            plt.close()
        
        return loss
    
    def compute_depth_loss(
        self,
        vertices: torch.Tensor,        # [V, 3]
        cam_intrinsics: torch.Tensor,  # [3, 3]
        depth_map: torch.Tensor,       # [H, W]
        mask: torch.Tensor,            # [H, W]
        it: int = 0
    ) -> torch.Tensor:
        """
        Compute depth loss between SMPLX z-buffer and masked depth map.
        
        Strategy:
        1. Project all vertices and create full z-buffer (no mask filtering)
        2. Sample both z-buffer and depth_map at mask locations
        3. Compare only at valid overlapping pixels
        """
        device = vertices.device
        H, W = depth_map.shape

        # ----------------------------------
        # 1. Project vertices (no filtering yet)
        # ----------------------------------
        points = self._project_points(vertices, cam_intrinsics)
        u = points[:, 0].long()
        v = points[:, 1].long()
        z = vertices[:, 2]

        # Image bounds check
        inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v, z = u[inside], v[inside], z[inside]

        if z.numel() == 0:
            return torch.zeros((), device=device)

        # ----------------------------------
        # 2. Build full z-buffer (all valid vertices)
        # ----------------------------------
        pixel_ids = v * W + u
        unique_pixels, inverse_indices = torch.unique(pixel_ids, return_inverse=True)

        # Get minimum depth per pixel (z-buffering)
        z_min = torch.full((unique_pixels.shape[0],), float('inf'), device=device)
        z_min.scatter_reduce_(0, inverse_indices, z, reduce='amin')

        # ----------------------------------
        # 3. Sample at mask locations only
        # ----------------------------------
        mask_flat = mask.reshape(-1)           # [H*W]
        depth_flat = depth_map.reshape(-1)     # [H*W]

        # Get mask and depth values at z-buffered pixel locations
        mask_at_pixels = mask_flat[unique_pixels]      # [num_unique_pixels]
        depth_at_pixels = depth_flat[unique_pixels]    # [num_unique_pixels]

        # Keep only pixels where mask is active
        valid_mask = mask_at_pixels > 0.5

        if not valid_mask.any():
            return torch.zeros((), device=device)

        # Final comparison: z-buffer vs ground truth at masked pixels
        z_buffer_masked = z_min[valid_mask]
        depth_gt_masked = depth_at_pixels[valid_mask]

        # ----------------------------------
        # Debug visualization
        # ----------------------------------
# ----------------------------------
# Debug visualization
# ----------------------------------
        # if it % 50 == 0:
        #     import matplotlib.pyplot as plt
            
        #     # Create full z-buffer image
        #     zbuffer_image = torch.zeros((H, W), device=device)
        #     zbuffer_image.view(-1)[unique_pixels] = z_min
            
        #     # Masked depth map (ensure it's float for proper masking)
        #     mask_float = mask.float() if mask.dtype == torch.bool else mask
        #     depth_masked = depth_map * mask_float
            
        #     # Save visualizations
        #     plt.imsave("depth_zbuffer.png", zbuffer_image.detach().cpu().numpy(), cmap='plasma')
        #     plt.imsave("depth_gt_masked.png", depth_masked.detach().cpu().numpy(), cmap='plasma')
            
        #     print(f"Depth loss computed on {valid_mask.sum()} masked pixels")
        #     print(f"Z-buffer range: [{z_buffer_masked.min():.3f}, {z_buffer_masked.max():.3f}]")
        #     print(f"GT depth range: [{depth_gt_masked.min():.3f}, {depth_gt_masked.max():.3f}]")
        # ----------------------------------
        # Compute loss
        # ----------------------------------
        loss = torch.mean((z_buffer_masked - depth_gt_masked) ** 2)
        return loss
    
    def compute_depth_loss_with_rasterization(
        self,
        vertices: torch.Tensor,
        cam_intrinsics: torch.Tensor,
        point_cloud: torch.Tensor,
        it: int = 0
    ) -> torch.Tensor:
        """
        Compute depth loss using proper triangle rasterization and chamfer distance.
        
        Strategy:
        1. Rasterize SMPLX mesh to get z-buffer
        2. Extract visible 3D points from z-buffer
        3. Compute chamfer distance with point cloud
        """
        from pytorch3d.renderer import (
            RasterizationSettings,
            MeshRasterizer,
            PerspectiveCameras,
        )
        from pytorch3d.structures import Meshes, Pointclouds
        from pytorch3d.loss import chamfer_distance
        
        device = vertices.device
        
        # Ensure point_cloud is on correct device
        if not isinstance(point_cloud, torch.Tensor):
            point_cloud = torch.from_numpy(point_cloud).float().to(device)
        else:
            point_cloud = point_cloud.to(device)
        
        # Create SMPLX mesh
        faces = torch.from_numpy(self.smplx_model.faces).long().to(device)
        mesh = Meshes(verts=[vertices], faces=[faces])
        
        # Extract H, W from intrinsics (assuming standard camera setup)
        fx = -cam_intrinsics[0, 0] # flipped sign for PyTorch3D
        fy = -cam_intrinsics[1, 1]
        cx = cam_intrinsics[0, 2]
        cy = cam_intrinsics[1, 2]
        
        H = int(cy.item() * 2)
        W = int(cx.item() * 2)
        
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
        
        # Extract valid z-buffer pixels (visible points)
        valid_mask = rendered_depth > 0
        
        if not valid_mask.any():
            return torch.zeros((), device=device)
        
        # Get pixel coordinates of valid depths
        v_coords, u_coords = torch.where(valid_mask)
        z_vals = rendered_depth[valid_mask]
        
        # Backproject to 3D
        x_3d = (u_coords.float() - cx) * z_vals / fx
        y_3d = (v_coords.float() - cy) * z_vals / fy
        z_3d = z_vals
        
        zbuffer_points = torch.stack([x_3d, y_3d, z_3d], dim=1)  # [M, 3]
        
        # Create Pointclouds for chamfer distance
        pclouds_zbuf = Pointclouds([zbuffer_points])
        pclouds_gt = Pointclouds([point_cloud])
        
        # Compute chamfer distance
        loss, _ = chamfer_distance(pclouds_zbuf, pclouds_gt)
        
        # Debug visualization
        if it % 50 == 0:
            print(f"Rasterized chamfer loss: {loss.item():.4f} "
                    f"(zbuf points: {zbuffer_points.shape[0]}, "
                    f"gt points: {point_cloud.shape[0]})")
            
            import matplotlib.pyplot as plt
            
            # Visualize 3D point clouds
            fig = plt.figure(figsize=(12, 5))
            
            # Rendered depth map
            ax1 = fig.add_subplot(121)
            depth_vis = rendered_depth.detach().cpu().numpy()
            depth_vis[depth_vis <= 0] = np.nan
            im = ax1.imshow(depth_vis, cmap='plasma')
            ax1.set_title(f'Rasterized Z-buffer (iter {it})')
            plt.colorbar(im, ax=ax1, fraction=0.046)
            
            # 3D point cloud comparison
            ax2 = fig.add_subplot(122, projection='3d')
            zbuf_np = zbuffer_points.detach().cpu().numpy()
            pc_np = point_cloud.detach().cpu().numpy()
            
            ax2.scatter(zbuf_np[::10, 0], zbuf_np[::10, 1], zbuf_np[::10, 2], 
                        c='r', s=1, label='Z-buffer', alpha=0.5)
            ax2.scatter(pc_np[::10, 0], pc_np[::10, 1], pc_np[::10, 2], 
                        c='b', s=1, label='GT Point Cloud', alpha=0.5)
            ax2.set_title('Point Cloud Comparison')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(f'output/rasterized_chamfer_iter_{it}.png', dpi=150)
            plt.close()
        
        return loss


    def project_mesh_on_image(self, params, image, cam_intrinsics):
        """
        Project SMPL mesh onto the image.
        """
        # Get mesh
        vertices, faces = self.get_mesh(params)
        
        ## i want to highligh joints as well on the image
        # Get joints
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
        joints = smpl_output.joints[0].detach().cpu().numpy()  # [127, 3]
        joints_list = [16, 17, 18, 19, 15, 7, 8]  # left_shoulder, right_shoulder, left_elbow, right_elbow, nose, left_ankle, right_ankle
        joint_names = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'nose', 'left_ankle', 'right_ankle']
        overlay = image.copy()
        for idx, name in zip(joints_list, joint_names):
            joint_3d = joints[idx]
            # Project to 2D
            if isinstance(cam_intrinsics, torch.Tensor):
                cam_intrinsics = cam_intrinsics.cpu().numpy()
            if cam_intrinsics.ndim == 3:
                cam_intrinsics = cam_intrinsics[0]
            

            fx = cam_intrinsics[0, 0]
            fy = cam_intrinsics[1, 1]
            cx = cam_intrinsics[0, 2]
            cy = cam_intrinsics[1, 2]
            if joint_3d[2] > 0:
                u = fx * joint_3d[0] / joint_3d[2] + cx
                v = fy * joint_3d[1] / joint_3d[2] + cy
                import cv2
                
                cv2.circle(overlay, (int(u), int(v)), 5, (255, 0, 0), -1)  # Blue for joints
                cv2.putText(overlay, name, (int(u)+5, int(v)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        if isinstance(cam_intrinsics, torch.Tensor):
            cam_intrinsics = cam_intrinsics.cpu().numpy()
        if cam_intrinsics.ndim == 3:
            cam_intrinsics = cam_intrinsics[0]
        
        fx = cam_intrinsics[0, 0]
        fy = cam_intrinsics[1, 1]
        cx = cam_intrinsics[0, 2]
        cy = cam_intrinsics[1, 2]
        
        # Project vertices
        vertices_2d = np.zeros((len(vertices), 2))
        for i, v in enumerate(vertices):
            if v[2] > 0:
                vertices_2d[i, 0] = fx * v[0] / v[2] + cx
                vertices_2d[i, 1] = fy * v[1] / v[2] + cy
        
        # Create overlay
        
        # Draw mesh edges
        for face in faces[::10]:  # Draw every 10th face for clarity
            pts = vertices_2d[face].astype(np.int32)
            
            # Check if all points are in image bounds
            import cv2
            if np.all((pts[:, 0] >= 0) & (pts[:, 0] < image.shape[1]) &
                    (pts[:, 1] >= 0) & (pts[:, 1] < image.shape[0])):
                cv2.polylines(overlay, [pts], True, (0, 255, 0), 1)
        
        return overlay

        
    
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
