from pose import draw_pose, yolo_pose_inference
import torch
from depth_anything_3.api import DepthAnything3
import cv2
import numpy as np
from fov_estimator import FOVEstimator
import open3d as o3d
import os
import smplx

class SMPLXHandler:
    def __init__(self, model_folder='data', model_type='smplx', gender='neutral', device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # Create the SMPL-X layer
        self.smpl_layer = smplx.create(
            model_path=model_folder,
            model_type=model_type,
            gender=gender,
            num_betas=10,
            num_expression_coeffs=10,
            ext='pkl',
        ).to(self.device)

        print(f"Loaded {model_type} model on {self.device}")
        self.num_betas = self.smpl_layer.num_betas
        
        # COCO to SMPL-X joint mapping (approximate)
        self.coco_to_smplx = {
            0: 15,   # nose -> head
            5: 17,   # left_shoulder -> left_shoulder
            6: 16,   # right_shoulder -> right_shoulder
            7: 19,   # left_elbow -> left_elbow
            8: 18,   # right_elbow -> right_elbow
            9: 21,   # left_wrist -> left_wrist
            10: 20,  # right_wrist -> right_wrist
            11: 2,   # left_hip -> left_hip
            12: 1,   # right_hip -> right_hip
            13: 5,   # left_knee -> left_knee
            14: 4,   # right_knee -> right_knee
            15: 8,   # left_ankle -> left_ankle
            16: 7,   # right_ankle -> right_ankle
        }
    
    def initialize_betas(self):
        betas = torch.zeros(1, self.num_betas, device=self.device)
        return betas

    def coco_to_smplx_pose(self, coco_keypoints_3d, num_iterations=100, lr=0.01):
        """
        Convert COCO 3D keypoints to SMPL-X pose parameters using inverse kinematics.
        
        Args:
            coco_keypoints_3d: numpy array of shape [17, 3] or [N, 17, 3] (COCO keypoints)
            num_iterations: number of optimization iterations
            lr: learning rate for optimization
            
        Returns:
            body_pose: torch tensor of optimized body pose parameters
            global_orient: torch tensor of global orientation
        """
        # Convert to torch tensor
        if isinstance(coco_keypoints_3d, np.ndarray):
            coco_keypoints_3d = torch.from_numpy(coco_keypoints_3d).float().to(self.device)
        
        # Handle batch dimension
        if coco_keypoints_3d.dim() == 2:
            coco_keypoints_3d = coco_keypoints_3d.unsqueeze(0)
        
        batch_size = coco_keypoints_3d.shape[0]
        
        # Initialize pose parameters (axis-angle representation)
        global_orient = torch.zeros(batch_size, 3, device=self.device, requires_grad=True)
        body_pose = torch.zeros(batch_size, 21, 3, device=self.device, requires_grad=True)  # 21 body joints
        betas = self.initialize_betas()
        
        # Optimizer
        optimizer = torch.optim.Adam([global_orient, body_pose], lr=lr)
        
        # Optimization loop
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            # Forward pass through SMPL-X
            output = self.smpl_layer(
                betas=betas,
                global_orient=global_orient,
                body_pose=body_pose.reshape(batch_size, -1)
            )
            
            # Get predicted joint positions
            pred_joints = output.joints  # [batch_size, num_joints, 3]
            
            # Calculate loss for mapped keypoints
            loss = 0
            count = 0
            for coco_idx, smplx_idx in self.coco_to_smplx.items():
                if coco_idx < coco_keypoints_3d.shape[1]:
                    target = coco_keypoints_3d[:, coco_idx, :]
                    pred = pred_joints[:, smplx_idx, :]
                    loss += torch.nn.functional.mse_loss(pred, target)
                    count += 1
            
            loss = loss / count if count > 0 else loss
            
            # Regularization to prevent extreme poses
            loss += 0.001 * torch.mean(body_pose ** 2)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if iteration % 20 == 0:
                print(f"Iteration {iteration}, Loss: {loss.item():.6f}")
        
        return body_pose.detach(), global_orient.detach()

    def get_mesh(self, body_pose, global_orient, betas=None):
        """Generate SMPL-X mesh from pose parameters."""
        if betas is None:
            betas = self.initialize_betas()
        
        output = self.smpl_layer(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose.reshape(body_pose.shape[0], -1)
        )
        
        return output


# create_point_cloud(depth_map, image, cam_intrinsics):
def create_point_cloud(depth_map, image, cam_intrinsics):
    width, height = depth_map.shape[1], depth_map.shape[0]
    fx = cam_intrinsics[0, 0, 0]
    fy = cam_intrinsics[0, 1, 1]
    # Generate mesh grid and calculate point cloud coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x = (x - width / 2) / fx
    y = (y - height / 2) / fy
    z = np.array(depth_map)
    points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
    colors = np.array(image).reshape(-1, 3) / 255.0

    # Create the point cloud and save it to the output directory
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # o3d.io.write_point_cloud(os.path.join("output", "point_cloud.ply"), pcd)
    return pcd

def lift2d_keypoints_to_3d(keypoints_2d, depth_map, cam_intrinsics):
    fx = cam_intrinsics[0, 0, 0]
    fy = cam_intrinsics[0, 1, 1]
    cx = cam_intrinsics[0, 0, 2]
    cy = cam_intrinsics[0, 1, 2]

    keypoints_3d = []
    for keypoint in keypoints_2d:
        x_2d, y_2d = int(keypoint[0]), int(keypoint[1])
        z = depth_map[y_2d, x_2d]
        x = (x_2d - cx) * z / fx
        y = (y_2d - cy) * z / fy
        keypoints_3d.append([x, y, z])
    return np.array(keypoints_3d)



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
    model = model.to(device=device)

    images = ["/home/khater/pose-check/tom.jpg"]

    prediction = model.inference(
        images,
        export_dir="output",
    )
    depth_prediction = prediction.depth[0]
    print(prediction.processed_images.shape)

    fov_estimator = FOVEstimator(name="moge2", device=device)
    cam_intrinsics = fov_estimator.get_cam_intrinsics(img=prediction.processed_images[0])
    results = yolo_pose_inference(images)
    ratio_h = prediction.processed_images.shape[1] / cv2.imread(images[0]).shape[0]
    ratio_w = prediction.processed_images.shape[2] / cv2.imread(images[0]).shape[1]
    # handle_yolo_results(prediction.processed_images[0], results, ratio_w=ratio_w, ratio_h=ratio_h)
    # pcd = create_point_cloud(depth_prediction, prediction.processed_images[0], cam_intrinsics)
    # key_points_3d = lift2d_keypoints_to_3d(
    #     n depth_prediction, cam_intrinsics
    # )
    # depth_normalized = cv2.normalize(depth_prediction, None, 0, 255, cv2.NORM_MINMAX)