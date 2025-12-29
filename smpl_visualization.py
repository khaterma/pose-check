"""
Visualization utilities for Metric SMPL-X Fitting results
"""

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh


def visualize_fitting_results(
    fitter,
    image,
    params=None,
    cam_intrinsics=None,
    point_cloud=None,
    output_path="output/fitting_visualization.png"
):
    """
    Create a comprehensive visualization of fitting results.
    
    Args:
        fitter: MetricSMPLFitter instance
        image: Original RGB image
        params: SMPL-X parameters (uses phase2_params if None)
        cam_intrinsics: Camera intrinsics
        point_cloud: Optional point cloud for comparison
        output_path: Where to save the visualization
    """
    if params is None:
        params = fitter.phase2_params
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 10))
    
    # 1. Original image
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis('off')
    
    # 2. SMPL mesh projected on image
    ax2 = fig.add_subplot(2, 3, 2)
    projected_img = project_mesh_on_image(
        fitter, params, image, cam_intrinsics
    )
    ax2.imshow(projected_img)
    ax2.set_title("SMPL-X Projection")
    ax2.axis('off')
    
    # 3. SMPL mesh 3D view (front)
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    vertices, faces = fitter.get_mesh(params)
    plot_mesh_3d(ax3, vertices, faces, "Front View")
    
    # 4. SMPL mesh 3D view (side)
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    plot_mesh_3d(ax4, vertices, faces, "Side View", azim=-90)
    
    # 5. Comparison with point cloud (if provided)
    if point_cloud is not None:
        ax5 = fig.add_subplot(2, 3, 5, projection='3d')
        plot_mesh_vs_pointcloud(ax5, vertices, point_cloud)
    else:
        ax5 = fig.add_subplot(2, 3, 5, projection='3d')
        plot_mesh_3d(ax5, vertices, faces, "Top View", elev=90)
    
    # 6. Joint locations
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    plot_joints_3d(ax6, fitter, params)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    plt.close()


def project_mesh_on_image(fitter, params, image, cam_intrinsics):
    """
    Project SMPL mesh onto the image.
    """
    # Get mesh
    vertices, faces = fitter.get_mesh(params)
    
    # Convert to numpy
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
    overlay = image.copy()
    
    # Draw mesh edges
    for face in faces[::10]:  # Draw every 10th face for clarity
        pts = vertices_2d[face].astype(np.int32)
        
        # Check if all points are in image bounds
        if np.all((pts[:, 0] >= 0) & (pts[:, 0] < image.shape[1]) &
                  (pts[:, 1] >= 0) & (pts[:, 1] < image.shape[0])):
            cv2.polylines(overlay, [pts], True, (0, 255, 0), 1)
    
    return overlay


def plot_mesh_3d(ax, vertices, faces, title, azim=45, elev=20):
    """
    Plot 3D mesh.
    """
    # Create mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Plot
    ax.plot_trisurf(
        vertices[:, 0], vertices[:, 1], vertices[:, 2],
        triangles=faces,
        color='lightblue',
        alpha=0.8,
        edgecolor='gray',
        linewidth=0.1
    )
    
    # Set equal aspect ratio
    max_range = np.array([
        vertices[:, 0].max() - vertices[:, 0].min(),
        vertices[:, 1].max() - vertices[:, 1].min(),
        vertices[:, 2].max() - vertices[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
    mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
    mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)


def plot_mesh_vs_pointcloud(ax, vertices, point_cloud, num_points=5000):
    """
    Plot mesh overlaid with point cloud.
    """
    # Subsample point cloud for visualization
    if len(point_cloud) > num_points:
        indices = np.random.choice(len(point_cloud), num_points, replace=False)
        point_cloud_vis = point_cloud[indices]
    else:
        point_cloud_vis = point_cloud
    
    # Plot point cloud
    ax.scatter(
        point_cloud_vis[:, 0],
        point_cloud_vis[:, 1],
        point_cloud_vis[:, 2],
        c='blue',
        marker='.',
        s=1,
        alpha=0.3,
        label='Point Cloud'
    )
    
    # Plot mesh vertices
    ax.scatter(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        c='red',
        marker='.',
        s=5,
        alpha=0.5,
        label='SMPL-X'
    )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('SMPL-X vs Point Cloud')
    ax.legend()


def plot_joints_3d(ax, fitter, params):
    """
    Plot SMPL joint locations in 3D.
    """
    with torch.no_grad():
        output = fitter.smplx_model(**params, return_verts=True)
        joints = output.joints[0].cpu().numpy()
    
    # Plot joints
    ax.scatter(
        joints[:, 0],
        joints[:, 1],
        joints[:, 2],
        c='red',
        marker='o',
        s=50,
        label='Joints'
    )
    
    # Draw skeleton connections
    skeleton = [
        (0, 1), (0, 2), (0, 3),  # Root to hips and spine
        (1, 4), (2, 5),  # Hips to knees
        (4, 7), (5, 8),  # Knees to ankles
        (3, 6), (6, 9),  # Spine to neck and head
        (9, 12), (9, 13), (9, 14),  # Head to jaw and eyes
        (9, 16), (9, 17),  # Neck to shoulders
        (16, 18), (17, 19),  # Shoulders to elbows
        (18, 20), (19, 21),  # Elbows to wrists
    ]
    
    for joint1, joint2 in skeleton:
        if joint1 < len(joints) and joint2 < len(joints):
            ax.plot(
                [joints[joint1, 0], joints[joint2, 0]],
                [joints[joint1, 1], joints[joint2, 1]],
                [joints[joint1, 2], joints[joint2, 2]],
                'b-',
                linewidth=2
            )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('SMPL-X Skeleton')
    ax.legend()


def compare_phase1_phase2(
    fitter,
    image,
    cam_intrinsics,
    output_path="output/phase_comparison.png"
):
    """
    Compare Phase 1 (skeleton) and Phase 2 (surface) results side-by-side.
    """
    if fitter.phase1_params is None or fitter.phase2_params is None:
        print("Both phases must be completed for comparison")
        return
    
    fig = plt.figure(figsize=(20, 8))
    
    # Phase 1 results
    ax1 = fig.add_subplot(2, 3, 1)
    phase1_img = project_mesh_on_image(
        fitter, fitter.phase1_params, image, cam_intrinsics
    )
    ax1.imshow(phase1_img)
    ax1.set_title("Phase 1: Skeleton Fitting")
    ax1.axis('off')
    
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    vertices1, faces1 = fitter.get_mesh(fitter.phase1_params)
    plot_mesh_3d(ax2, vertices1, faces1, "Phase 1: Front View")
    
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    plot_mesh_3d(ax3, vertices1, faces1, "Phase 1: Side View", azim=-90)
    
    # Phase 2 results
    ax4 = fig.add_subplot(2, 3, 4)
    phase2_img = project_mesh_on_image(
        fitter, fitter.phase2_params, image, cam_intrinsics
    )
    ax4.imshow(phase2_img)
    ax4.set_title("Phase 2: Surface Refinement")
    ax4.axis('off')
    
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    vertices2, faces2 = fitter.get_mesh(fitter.phase2_params)
    plot_mesh_3d(ax5, vertices2, faces2, "Phase 2: Front View")
    
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    plot_mesh_3d(ax6, vertices2, faces2, "Phase 2: Side View", azim=-90)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Phase comparison saved to {output_path}")
    plt.close()


def export_colored_mesh(fitter, image, cam_intrinsics, output_path="output/colored_mesh.ply"):
    """
    Export SMPL mesh with colors from the image.
    """
    vertices, faces = fitter.get_mesh()
    
    # Project vertices to image to get colors
    if isinstance(cam_intrinsics, torch.Tensor):
        cam_intrinsics = cam_intrinsics.cpu().numpy()
    if cam_intrinsics.ndim == 3:
        cam_intrinsics = cam_intrinsics[0]
    
    fx = cam_intrinsics[0, 0]
    fy = cam_intrinsics[1, 1]
    cx = cam_intrinsics[0, 2]
    cy = cam_intrinsics[1, 2]
    
    # Get colors from image
    vertex_colors = np.zeros((len(vertices), 3), dtype=np.uint8)
    
    for i, v in enumerate(vertices):
        if v[2] > 0:
            u = int(fx * v[0] / v[2] + cx)
            v_coord = int(fy * v[1] / v[2] + cy)
            
            if 0 <= u < image.shape[1] and 0 <= v_coord < image.shape[0]:
                vertex_colors[i] = image[v_coord, u]
            else:
                vertex_colors[i] = [128, 128, 128]  # Gray for out-of-bounds
        else:
            vertex_colors[i] = [128, 128, 128]
    
    # Create colored mesh
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=vertex_colors
    )
    
    # Export
    mesh.export(output_path)
    print(f"Colored mesh exported to {output_path}")


if __name__ == "__main__":
    print("Visualization utilities for Metric SMPL-X Fitting")
    print("Use these functions after running the fitting pipeline")
    print("\nExample usage:")
    print("  from smpl_visualization import visualize_fitting_results")
    print("  visualize_fitting_results(fitter, image, cam_intrinsics=intrinsics)")
