import numpy as np
import cv2
# from PIL import Image
def project_to_3d(depth_map, img_rgb, camera_intrinsics, sam_mask=None):
    # Mask valid pixels
    mask = (depth_map > 0.1) & (depth_map < 4.5)
    height, width = depth_map.shape
    u,v = np.meshgrid(np.arange(width), np.arange(height))
    # fx = 520.9761
    # fy = 520.9761
    # cx = 252.0
    # cy = 168.0
    if camera_intrinsics is None:
        camera_intrinsics = np.array([[520.9761, 0, 252.0],
                                      [0, 520.9761, 168.0],
                                      [0, 0, 1]])
        
    if camera_intrinsics.ndim == 3:
        camera_intrinsics = camera_intrinsics[0]

    camera_intrinsics = camera_intrinsics.numpy()
    
    fx = camera_intrinsics[0, 0]
    fy = camera_intrinsics[1, 1]
    cx = camera_intrinsics[0, 2]
    cy = camera_intrinsics[1, 2]    

    if sam_mask is not None:
        sam_mask = sam_mask.detach().cpu().numpy().astype(bool) # shape (H, W)
        # print(f" SAM mask shape: {sam_mask.shape} ")
        mask = sam_mask[0]
        # mask = mask & (sam_mask > 0)

    # Extract valid data
    z = depth_map[mask]
    u_valid = u[mask]
    v_valid = v[mask]
    colors = img_rgb[v_valid, u_valid]

    # Pinhole projection
    x = (u_valid - cx) * z / fx
    y = (v_valid - cy) * z / fy  # Flip Y

    points = np.stack([x, y, z], axis=1)

    # # project back the points to form the image (for verification)

    # points_2d = np.zeros((height, width, 3), dtype=np.uint8)
    # for i in range(points.shape[0]):
    #     u_coord = int((points[i, 0] * fx) / points[i, 2] + cx)
    #     v_coord = int((points[i, 1] * fy) / points[i, 2] + cy)
    #     if 0 <= u_coord < width and 0 <= v_coord < height:
    #         points_2d[v_coord, u_coord] = colors[i]
    
    # cv2.imwrite("projected_points.png", points_2d)


    colors = colors / 255.0
    return points, colors


## apply the same exact transformation when lifting 2D keypoints to 3D points.

def lift2d_keypoints_to_3d(keypoints_2d, depth_map, cam_intrinsics, center_around_origin=False):
    if cam_intrinsics.ndim == 3:
        cam_intrinsics = cam_intrinsics[0]
    fx = cam_intrinsics[0, 0]
    fy = cam_intrinsics[1, 1]
    cx = cam_intrinsics[0, 2]
    cy = cam_intrinsics[1, 2]
    
    keypoints_3d = []
    for keypoint in keypoints_2d:
        u = int(keypoint[0])
        v = int(keypoint[1])
        z = depth_map[v, u]
        
        # Pinhole projection (same as project_to_3d)
        x = (u - cx) * z / fx
        y = -(v - cy) * z / fy  # Flip Y
        
        points = np.array([x, y, z])
        keypoints_3d.append(points)
    
    keypoints_3d = np.array(keypoints_3d)
    
    # Apply same centering logic as project_to_3d if needed
    # if center_around_origin:
    #     center = keypoints_3d.mean(axis=0, keepdims=True)
    #     keypoints_3d = keypoints_3d - center
    #     keypoints_3d[:, 2] = -keypoints_3d[:, 2]  # Flip Z
    #     keypoints_3d[:, 2] += 1.0  # Shift forward
    
    return keypoints_3d