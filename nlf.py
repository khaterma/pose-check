
"""
NLF-based SMPL-X fitter
=======================
Wraps the TorchScript NLF model so it can be used with the same interface as
`MetricSMPLFitter` inside `main_pipeline`. The NLF model already predicts
SMPL-X vertices, so this class focuses on loading the SMPL-X topology for
visualization, projecting to images, and providing the z-buffer / chamfer
loss against an observed point cloud.
"""

import torch
import torchvision  # Required for TorchScript model deps
import numpy as np
import cv2
import smplx
from typing import Dict, Optional, Tuple
from pytorch3d.structures import Pointclouds
from pytorch3d.loss import chamfer_distance
from smplfitter.pt import BodyModel, BodyFitter
from typing import List, Tuple
import random


class NLFSMPLFitter:
    """Drop-in replacement for `MetricSMPLFitter` using the NLF TorchScript model."""

    def __init__(
        self,
        model_path: str = "checkpoints/nlf_l_multi.torchscript",
        smplx_model_path: str = "./data/smplx",
        gender: str = "neutral",
        device: str = "cuda",
        image: Optional[np.ndarray] = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = torch.jit.load(model_path).to(self.device).eval()
        self.image = image

        # Load SMPL-X just to obtain faces/topology and enable exports/visuals.
        self.smplx_model = smplx.create(
            model_path=smplx_model_path,
            model_type="smplx",
            gender=gender,
            use_face_contour=False,
            num_betas=10,
            num_expression_coeffs=10,
            ext="npz",
        ).to(self.device)

        # Optional refinement fitter if we want to re-fit parametric pose/shape.
        self.body_model = BodyModel(
            "smplx",
            gender,
            num_betas=10,
            model_root=f"{smplx_model_path}/smplx",
        ).to(self.device)
        self.body_fitter = BodyFitter(self.body_model).to(self.device)
        self.body_fitter = torch.jit.script(self.body_fitter)

        self.prediction = None
        self.fitted_params: Optional[Dict[str, torch.Tensor]] = None

    def _prepare_image_tensor(self, image: np.ndarray) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            return image.to(self.device)
        tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        return tensor.to(self.device)

    def _infer_nlf(self, image: np.ndarray) -> Dict[str, torch.Tensor]:
        image_tensor = self._prepare_image_tensor(image)
        frame_batch = image_tensor.unsqueeze(0)
        with torch.inference_mode():
            pred = self.model.detect_smpl_batched(frame_batch, model_name="smplx")
        return pred

    def _project_points(self, points_3d: torch.Tensor, cam_intrinsics: torch.Tensor) -> torch.Tensor:
        fx = cam_intrinsics[0, 0]
        fy = cam_intrinsics[1, 1]
        cx = cam_intrinsics[0, 2]
        cy = cam_intrinsics[1, 2]
        x_2d = fx * points_3d[:, 0] / (points_3d[:, 2] + 1e-6) + cx
        y_2d = fy * points_3d[:, 1] / (points_3d[:, 2] + 1e-6) + cy
        return torch.stack([x_2d, y_2d], dim=1)

    def fit(
        self,
        keypoints_2d: Optional[np.ndarray] = None,
        cam_intrinsics: Optional[np.ndarray] = None,
        depth_map: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        feet_mask: Optional[np.ndarray] = None,
        point_cloud: Optional[np.ndarray] = None,
        **_: Dict,
    ) -> Dict[str, torch.Tensor]:
        """
        Run the NLF model and package outputs to mimic MetricSMPLFitter.
        Keypoint arguments are accepted for API compatibility but unused.
        """
        if self.image is None:
            raise ValueError("Image not provided to NLFSMPLFitter.")

        pred = self._infer_nlf(self.image)
        self.prediction = pred

        # Extract primary outputs
        pose = pred["pose"][0].to(self.device)
        betas = pred["betas"][0].to(self.device)
        transl = pred["trans"][0].to(self.device)
        vertices = pred["vertices3d"][0].to(self.device)
        joints = pred["joints3d"][0].to(self.device)

        # Optional parametric refinement to align with SMPL-X topology
        try:
            with torch.inference_mode():
                print(f" shapes of vertices: {vertices.shape}, joints: {joints.shape} ")
                fit_res = self.body_fitter.fit(vertices, joints, num_iter=3, beta_regularizer=1)
            pose_rotvecs = fit_res.get("pose_rotvecs", pose)
            shape_betas = fit_res.get("shape_betas", betas)
            trans_fitted = fit_res.get("trans", transl)
            print("NLF BodyFitter refinement succeeded.")
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(f"NLF BodyFitter refinement failed, using raw predictions: {exc}")
            pose_rotvecs, shape_betas, trans_fitted = pose.unsqueeze(0), betas.unsqueeze(0), transl.unsqueeze(0)

        # Store params compatible with smplx forward
        self.fitted_params = {
            "betas": shape_betas,
            "global_orient": pose_rotvecs[:, :3],
            "body_pose": pose_rotvecs[:, 3:],
            "transl": trans_fitted,
            "left_hand_pose": torch.zeros((1, 6), device=self.device),
            "right_hand_pose": torch.zeros((1, 6), device=self.device),
            "expression": torch.zeros((1, 10), device=self.device),
            "vertices": vertices,
            "joints": joints,
        }

        # Compute optional z-buffer chamfer loss for logging
        if cam_intrinsics is not None and point_cloud is not None:
            cam_intrinsics_torch = torch.from_numpy(cam_intrinsics[0] if cam_intrinsics.ndim == 3 else cam_intrinsics).float().to(self.device)
            _ = self.compute_depth_loss_with_point_cloud(vertices, cam_intrinsics_torch, point_cloud)

        return self.fitted_params

    def get_zbuffer(
        self,
        vertices: torch.Tensor,
        cam_intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        
        device = vertices.device
        points_2d = self._project_points(vertices, cam_intrinsics)
        u = points_2d[:, 0].long()
        v = points_2d[:, 1].long()
        z = vertices[:, 2]

        H = int(cam_intrinsics[1, 2].item() * 2)
        W = int(cam_intrinsics[0, 2].item() * 2)

        inside = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)
        u, v, z = u[inside], v[inside], z[inside]
        if z.numel() == 0:
            return torch.zeros((), device=device)

        pixel_ids = v * W + u
        unique_pixels, inverse_indices = torch.unique(pixel_ids, return_inverse=True)
        z_min = torch.full((unique_pixels.shape[0],), float("inf"), device=device)
        z_min.scatter_reduce_(0, inverse_indices, z, reduce="amin")

        v_zbuf = unique_pixels // W
        u_zbuf = unique_pixels % W

        fx = cam_intrinsics[0, 0]
        fy = cam_intrinsics[1, 1]
        cx = cam_intrinsics[0, 2]
        cy = cam_intrinsics[1, 2]

        x_3d = (u_zbuf.float() - cx) * z_min / fx
        y_3d = (v_zbuf.float() - cy) * z_min / fy
        z_3d = z_min
        zbuffer_points = torch.stack([x_3d, y_3d, z_3d], dim=1)
        return zbuffer_points
    
    def compute_depth_loss_with_point_cloud(
        self,
        vertices: torch.Tensor,
        cam_intrinsics: torch.Tensor,
        point_cloud: torch.Tensor,
        it: int = 0,
    ) -> torch.Tensor:
        
        zbuffer_points = self.get_zbuffer(vertices, cam_intrinsics)
        device = zbuffer_points.device

        if not isinstance(point_cloud, torch.Tensor):
            point_cloud = torch.from_numpy(point_cloud).float().to(device)
        else:
            point_cloud = point_cloud.to(device)

        pclouds_zbuf = Pointclouds([zbuffer_points])
        pclouds_gt = Pointclouds([point_cloud])
        loss, _ = chamfer_distance(pclouds_zbuf, pclouds_gt)

        if it % 50 == 0:
            print(
                f"Point cloud chamfer loss: {loss.item():.4f} "
                f"(zbuf points: {zbuffer_points.shape[0]}, gt points: {point_cloud.shape[0]})"
            )
        return loss
    

    def center_pcd(self, pcd: np.ndarray) -> np.ndarray:
        # this function centers the point cloud around the origin
        centroid = np.mean(pcd, axis=0)
        pcd -= centroid
        return pcd
    
    def ransac_scale(
        self,
        pairs: List[Tuple[float, float]], iters: int = 10000, tol: float = 0.05
        ) -> Tuple[float, int]:
        """RANSAC for scale s in s * d_sfm = d_metric.

        tol: relative tolerance |s*d_sfm - d_metric| <= tol * d_metric
        Returns best_scale, inlier_count.
        """
        if not pairs:
            raise ValueError("No depth pairs provided.")
        pairs_arr = np.array(pairs, dtype=float)  # (N,2)
        d_sfm = pairs_arr[:, 0]
        d_met = pairs_arr[:, 1]
        ratios = d_met / np.clip(d_sfm, 1e-9, None)
        best_s = np.median(ratios)
        best_inliers = 0
        n = len(pairs)
        for _ in range(iters):
            i = random.randint(0, n - 1)
            s_candidate = d_met[i] / max(d_sfm[i], 1e-9)
            pred = s_candidate * d_sfm
            err = np.abs(pred - d_met)
            inliers = np.sum(err <= tol * np.maximum(d_met, 1e-6))
            if inliers > best_inliers:
                best_inliers = inliers
                best_s = s_candidate

        return best_s, best_inliers


    def run_metric_optimization(
        self,
        cam_intrinsics: torch.Tensor,
        point_cloud: torch.Tensor,
        num_iters: int = 100,
    ) -> Dict[str, torch.Tensor]:
        # Placeholder for potential metric optimization steps
        # Currently returns the original vertices and joints without modification

        # get zbuffer points from vertices
        # device = vertices.device
        try:
            vertices = self.fitted_params["vertices"]
            joints = self.fitted_params["joints"]
        except Exception as exc:
            raise ValueError("Fitted parameters not available for metric optimization.") from exc

        zbuffer_points = self.get_zbuffer(vertices, cam_intrinsics)
        ## ensure zbuffer are numpy arrays
        zbuffer_points_np = zbuffer_points.detach().cpu().numpy()
        point_cloud_np = point_cloud.detach().cpu().numpy()
        # center both point clouds
        zbuffer_points_np = self.center_pcd(zbuffer_points_np)
        point_cloud_np = self.center_pcd(point_cloud_np)

        # pass only the z values of both point clouds
        z_depth = zbuffer_points_np[:, 2]
        pc_depth = point_cloud_np[:, 2]

        ## pass both point clouds to ransac scale
        scale, inliers = self.ransac_scale(
            list(zip(z_depth, pc_depth)), iters=1000, tol=0.1
        )
        print(f" RANSAC scale: {scale:.4f} with {inliers} inliers ")
        # scale the vertices and joints
        # vertices = vertices * scale
        # joints = joints * scale

        # ## fitted params update
        # self.fitted_params["vertices"] = vertices * scale
        # self.fitted_params["joints"] = joints * scale

        # save the point clouds for visualization
        np.save("zbuffer_points.npy", zbuffer_points_np * scale)
        np.save("point_cloud.npy", point_cloud_np)
        pcd = np.concatenate([zbuffer_points_np * scale, point_cloud_np], axis=0)
        np.save("combined_pcd.npy", pcd)

        return {
            "vertices": vertices,
            "joints": joints,
        }

    def project_mesh_on_image(
        self,
        params: Optional[Dict[str, torch.Tensor]],
        image: np.ndarray,
        cam_intrinsics: np.ndarray,
    ) -> np.ndarray:
        if params is None:
            params = self.fitted_params
        if params is None:
            raise ValueError("No fitted parameters available for projection.")

        if "vertices" in params:
            vertices = params["vertices"].detach().cpu().numpy()
        else:
            with torch.no_grad():
                output = self.smplx_model(**params, return_verts=True)
            vertices = output.vertices[0].cpu().numpy()

        faces = self.smplx_model.faces

        cam = cam_intrinsics[0] if cam_intrinsics.ndim == 3 else cam_intrinsics
        fx, fy, cx, cy = cam[0, 0], cam[1, 1], cam[0, 2], cam[1, 2]

        vertices_2d = np.zeros((len(vertices), 2))
        for i, v in enumerate(vertices):
            if v[2] > 0:
                vertices_2d[i, 0] = fx * v[0] / v[2] + cx
                vertices_2d[i, 1] = fy * v[1] / v[2] + cy

        overlay = image.copy()
        for face in faces[::10]:
            pts = vertices_2d[face].astype(np.int32)
            if np.all((pts[:, 0] >= 0) & (pts[:, 0] < image.shape[1]) & (pts[:, 1] >= 0) & (pts[:, 1] < image.shape[0])):
                cv2.polylines(overlay, [pts], True, (0, 255, 0), 1)
        return overlay

    def get_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.fitted_params is None:
            raise ValueError("No fitted parameters. Run fit() first.")
        if "vertices" in self.fitted_params:
            vertices = self.fitted_params["vertices"].detach().cpu().numpy()
        else:
            with torch.no_grad():
                output = self.smplx_model(**self.fitted_params, return_verts=True)
            vertices = output.vertices[0].cpu().numpy()
        faces = self.smplx_model.faces
        return vertices, faces

    def export_mesh(self, output_path: str, params: Optional[Dict[str, torch.Tensor]] = None) -> None:
        import trimesh
        if params is None:
            params = self.fitted_params
        if params is None:
            raise ValueError("No parameters provided for export.")

        # Temporarily swap fitted params if caller supplied a different one
        original = self.fitted_params
        self.fitted_params = params
        verts, faces = self.get_mesh()
        self.fitted_params = original
        trimesh.Trimesh(vertices=verts, faces=faces).export(output_path)
        print(f"Mesh exported to {output_path}")


__all__ = ["NLFSMPLFitter"]
