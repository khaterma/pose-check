from pose import draw_pose, yolo_pose_inference
import torch
from depth_anything_3.api import DepthAnything3
import cv2
import numpy as np
from fov_estimator import FOVEstimator
import open3d as o3d
import os
from metric_smpl_fitter import MetricSMPLFitter
from smpl_visualization import (
    visualize_fitting_results, 
    compare_phase1_phase2,
    export_colored_mesh
)

def main():
    # Load model from Hugging Face Hub


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
    model = model.to(device=device)


    ## run infernce on tom cruise image



    images = ["/home/khater/pose-check/tom.jpg"]

    prediction = model.inference(
        images,
        export_dir="output",
        # export_format="colmap"  # Options: glb, npz, ply, mini_npz, gs_ply, gs_video
    )
    depth_prediction = prediction.depth[0]
    print(prediction.processed_images.shape)  # Processed images: [N, H, W, 3] uint8

    fov_estimator = FOVEstimator(name="moge2", device=device)
    cam_intrinsics = fov_estimator.get_cam_intrinsics(img=prediction.processed_images[0])
    # create point cloud
    point_cloud_array, pcd = create_point_cloud(depth_prediction, prediction.processed_images[0], cam_intrinsics)
    # save depth map as image

    # show point cloud in open3d viewer
 

    depth_normalized = cv2.normalize(depth_prediction, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_normalized.astype(np.uint8)
    cv2.imwrite("output/tom_depth.png", depth_uint8)

    image_shape = cv2.imread(images[0]).shape
    ratio_h = prediction.processed_images.shape[1] / image_shape[0]
    ratio_w = prediction.processed_images.shape[2] / image_shape[1]

    results = yolo_pose_inference(images)
    keypoints = handle_yolo_results(depth_uint8, results, ratio_w=ratio_w, ratio_h=ratio_h)
    # # open the image and get its shape
    keypoints_3d = lift2d_keypoints_to_3d(keypoints[0], depth_prediction, cam_intrinsics)
    # visualize 3d keypoints using open3d
    keypoint_spheres = []
    for keypoint in keypoints_3d:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        sphere.translate(keypoint)
        sphere.paint_uniform_color([1, 0, 0])  # Red color
        keypoint_spheres.append(sphere)

    o3d.visualization.draw_geometries(keypoint_spheres + [pcd])

    print("Estimated Camera Intrinsics:", cam_intrinsics)

    # # Run SMPL-X Fitting
    if keypoints is not None and len(keypoints) > 0:
        print("\n" + "="*60)
        print("Running Metric SMPL-X Fitting...")
        print("="*60)
        
        # Initialize fitter
        fitter = MetricSMPLFitter(
            smplx_model_path="./data/smplx",
            gender="neutral",
            device=str(device),
            use_vposer=False  # Set to True if you have VPoser checkpoint
        )
        
        # Run two-phase fitting
        try:
            final_params = fitter.fit(
                image=prediction.processed_images[0],
                depth_map=depth_prediction,
                keypoints_2d=keypoints[0],  # First person
                cam_intrinsics=cam_intrinsics,
                point_cloud=point_cloud_array,
                sam_checkpoint=None,  # Will use simple mask if SAM not available
                phase1_iterations=300,
                phase2_iterations=500,
                clothing_buffer=0.03  # 3cm tolerance
            )
            
            # Export the fitted mesh
            fitter.export_mesh("output/smplx_fitted.obj")
            fitter.export_mesh("output/smplx_fitted.ply")
            
            print("\nSMPL-X mesh exported to output/smplx_fitted.obj and .ply")
            
            # Generate visualizations
            print("\nGenerating visualizations...")
            try:
                visualize_fitting_results(
                    fitter,
                    prediction.processed_images[0],
                    params=final_params,
                    cam_intrinsics=cam_intrinsics,
                    point_cloud=point_cloud_array,
                    output_path="output/fitting_visualization.png"
                )
                
                compare_phase1_phase2(
                    fitter,
                    prediction.processed_images[0],
                    cam_intrinsics,
                    output_path="output/phase_comparison.png"
                )
                
                export_colored_mesh(
                    fitter,
                    prediction.processed_images[0],
                    cam_intrinsics,
                    output_path="output/colored_smplx.ply"
                )
                
                print("Visualizations saved to output/")
            except Exception as e:
                print(f"Visualization failed (non-critical): {e}")
            
        except Exception as e:
            print(f"SMPL-X fitting failed: {e}")
            import traceback
            traceback.print_exc()

    # print(f" ratio h: {ratio_h} , ratio w: {ratio_w} ")
    # # prediction_opencv = cv2.cvtColor(prediction.processed_images[0], cv2.COLOR_RGB2BGR)
    





    # import matplotlib.pyplot as plt
    # plt.imshow(prediction.depth[0])
    # plt.show()



def handle_yolo_results(image, results, ratio_w=1.0, ratio_h=1.0):
 # Access the results
        key_points = []     
   
        print(f" image shape : {image.shape}")
        for result in results:
            if result.keypoints is not None:
                for person_kpts in result.keypoints.xy.cpu().numpy():
                    # Add confidence if needed:
                    conf = result.keypoints.conf.cpu().numpy()[0]
                    keypoints = np.concatenate(
                        [person_kpts * [ratio_w, ratio_h], conf[:, None]], axis=1
                    )
                    key_points.append(keypoints)
                    output_image = draw_pose(image, keypoints)
        # key_points_all.append(key_points)
        # cv2.imshow("Pose", img)
        # save image with poses
        # output_path = image_path.replace(".jpg", "_pose.jpg")
        cv2.imwrite("overlayed_image.png", output_image)
        
        return key_points  # Return keypoints for SMPL fitting


### apply FOV Estimator and then try to build the point cloud. ## done

### initalize HMR model and try to get the 3D mesh from the 2D image. priors are depth map, 2d pose, and sam3 masks.


# create_point_cloud(depth_map, image, cam_intrinsics):
def create_point_cloud(depth_map, image, cam_intrinsics):
    print(f" cam intrinsics: {cam_intrinsics} ")
    cam_intrinsics = cam_intrinsics.cpu().numpy()
    width, height = depth_map.shape[1], depth_map.shape[0]
    fx = cam_intrinsics[0, 0, 0]
    fy = cam_intrinsics[0, 1, 1]
    cx = cam_intrinsics[0, 0, 2]
    cy = cam_intrinsics[0, 1, 2]

    x, y = np.meshgrid(np.arange(width), np.arange(height))


    x = (x - cx) / fx
    y = -(y - cy) / fy   # <-- flip Y axis here
    z = np.array(depth_map)

    points = np.stack((x * z, y * z, z), axis=-1).reshape(-1, 3)


    print(f" z max: {np.max(z)} , z min: {np.min(z)} ")
    points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
    colors = np.array(image).reshape(-1, 3) / 255.0
    R = np.array([
    [-1,  0,  0],
    [ 0,  1,  0],
    [ 0,  0, -1]
        ])

    points = points @ R.T


    # Create the point cloud and save it to the output directory
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join("output", "point_cloud.ply"), pcd)
    ##show point cloud in open3d viewer
    # o3d.visualization.draw_geometries([pcd])
    
    # Return point cloud array for SMPL fitting
    return points, pcd

## apply the same exact transformation when lifting 2D keypoints to 3D points.

def lift2d_keypoints_to_3d(keypoints_2d, depth_map, cam_intrinsics):
    cam_intrinsics = cam_intrinsics.cpu().numpy()
    fx = cam_intrinsics[0, 0, 0]
    fy = cam_intrinsics[0, 1, 1]
    cx = cam_intrinsics[0, 0, 2]
    cy = cam_intrinsics[0, 1, 2]

    keypoints_3d = []
    for keypoint in keypoints_2d:
        x_2d, y_2d = int(keypoint[0]), int(keypoint[1])
        z = depth_map[y_2d, x_2d]
        x = (x_2d - cx) * z / fx
        y = -(y_2d - cy) * z / fy
        R = np.array([
        [-1,  0,  0],
        [ 0,  1,  0],
        [ 0,  0, -1]
            ])
        points = np.array([x, y, z])
        points = points @ R.T
        keypoints_3d.append(points)
    return np.array(keypoints_3d)

if __name__ == "__main__":
    main()