"""
Coordinate System Diagnostic Script
====================================
This script tests projection/unprojection consistency and identifies
coordinate system mismatches that cause the SMPL-X model to face the wrong direction.
"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow
import smplx
from typing import Dict, Tuple
import os


class CoordinateSystemDiagnostic:
    """Diagnose projection inconsistencies in SMPL-X fitting pipeline."""
    
    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Initialize SMPL-X for testing
        self.smplx_model = smplx.create(
            model_path="./data/smplx",
            model_type='smplx',
            gender='neutral',
            use_face_contour=False,
            num_betas=10,
            num_expression_coeffs=10,
            ext='npz'
        ).to(self.device)
        
        # Test camera intrinsics (typical values)
        self.cam_intrinsics = np.array([
            [520.0, 0.0, 320.0],
            [0.0, 520.0, 240.0],
            [0.0, 0.0, 1.0]
        ])
        
        self.image_size = (480, 640)  # H, W
        
    def test_projection_consistency(self):
        """Test if project -> unproject -> project gives same result."""
        print("\n" + "="*70)
        print("TEST 1: Projection/Unprojection Consistency")
        print("="*70)
        
        # Create test 3D points
        test_points_3d = np.array([
            [0.0, 0.0, 5.0],      # Center
            [0.5, 0.5, 5.0],      # Top-right
            [-0.5, -0.5, 5.0],    # Bottom-left
            [0.0, 1.0, 5.0],      # Up
            [0.0, -1.0, 5.0],     # Down
        ])
        
        print("\n📍 Original 3D Points (X, Y, Z):")
        for i, pt in enumerate(test_points_3d):
            print(f"  Point {i}: {pt}")
        
        # Project to 2D (using current implementation)
        points_2d_v1 = self._project_v1(test_points_3d)
        points_2d_v2 = self._project_v2(test_points_3d)
        
        print("\n📷 Projected 2D Points (u, v):")
        print("  Version 1 (with -Y flip):")
        for i, pt in enumerate(points_2d_v1):
            print(f"    Point {i}: ({pt[0]:.1f}, {pt[1]:.1f})")
        
        print("\n  Version 2 (without -Y flip):")
        for i, pt in enumerate(points_2d_v2):
            print(f"    Point {i}: ({pt[0]:.1f}, {pt[1]:.1f})")
        
        # Unproject back to 3D
        points_3d_recovered_v1 = self._unproject(points_2d_v1, test_points_3d[:, 2])
        points_3d_recovered_v2 = self._unproject(points_2d_v2, test_points_3d[:, 2])
        
        print("\n🔄 Unprojected 3D Points:")
        print("  From Version 1:")
        for i, pt in enumerate(points_3d_recovered_v1):
            print(f"    Point {i}: {pt}")
            
        print("\n  From Version 2:")
        for i, pt in enumerate(points_3d_recovered_v2):
            print(f"    Point {i}: {pt}")
        
        # Check consistency
        error_v1 = np.abs(test_points_3d - points_3d_recovered_v1).max()
        error_v2 = np.abs(test_points_3d - points_3d_recovered_v2).max()
        
        print(f"\n✅ Round-trip Error V1 (with -Y): {error_v1:.6f}")
        print(f"✅ Round-trip Error V2 (without -Y): {error_v2:.6f}")
        
        if error_v1 < 1e-5:
            print("   → V1 is CONSISTENT")
        else:
            print("   → V1 has INCONSISTENCY ⚠️")
            
        if error_v2 < 1e-5:
            print("   → V2 is CONSISTENT")
        else:
            print("   → V2 has INCONSISTENCY ⚠️")
    
    def test_smplx_coordinate_system(self):
        """Test SMPL-X default coordinate system."""
        print("\n" + "="*70)
        print("TEST 2: SMPL-X Coordinate System")
        print("="*70)
        
        # Create neutral pose
        with torch.no_grad():
            output = self.smplx_model(
                return_verts=True,
                return_full_pose=True
            )
            
            joints = output.joints[0].cpu().numpy()
            vertices = output.vertices[0].cpu().numpy()
        
        # Analyze joint positions
        nose_idx = 55  # Approximate nose joint
        left_shoulder_idx = 16
        right_shoulder_idx = 17
        left_hip_idx = 1
        right_hip_idx = 2
        pelvis_idx = 0
        
        print("\n📊 Key Joint Positions (X, Y, Z):")
        print(f"  Pelvis:         {joints[pelvis_idx]}")
        print(f"  Left Hip:       {joints[left_hip_idx]}")
        print(f"  Right Hip:      {joints[right_hip_idx]}")
        print(f"  Left Shoulder:  {joints[left_shoulder_idx]}")
        print(f"  Right Shoulder: {joints[right_shoulder_idx]}")
        
        # Determine coordinate conventions
        pelvis = joints[pelvis_idx]
        left_shoulder = joints[left_shoulder_idx]
        right_shoulder = joints[right_shoulder_idx]
        
        # Check Y-axis (up/down)
        shoulder_center = (left_shoulder + right_shoulder) / 2
        vertical_vec = shoulder_center - pelvis
        
        print(f"\n📏 Pelvis to Shoulders Vector: {vertical_vec}")
        
        if vertical_vec[1] > 0:
            print("  → Y-axis points UP (OpenGL convention) ✓")
            y_convention = "UP"
        else:
            print("  → Y-axis points DOWN (OpenCV convention)")
            y_convention = "DOWN"
        
        # Check Z-axis (forward/backward)
        mean_z = vertices[:, 2].mean()
        print(f"\n📏 Mean Z coordinate: {mean_z:.4f}")
        
        if mean_z > 0:
            print("  → Z-axis points TOWARD camera (OpenGL convention) ✓")
            z_convention = "TOWARD"
        else:
            print("  → Z-axis points AWAY from camera (OpenCV convention)")
            z_convention = "AWAY"
        
        # Check X-axis (left/right)
        lr_vec = right_shoulder - left_shoulder
        print(f"\n📏 Left to Right Shoulder Vector: {lr_vec}")
        
        if lr_vec[0] > 0:
            print("  → X-axis points RIGHT (standard)")
            x_convention = "RIGHT"
        else:
            print("  → X-axis points LEFT")
            x_convention = "LEFT"
        
        print("\n" + "="*70)
        print("SMPL-X COORDINATE SYSTEM:")
        print(f"  X: {x_convention}")
        print(f"  Y: {y_convention}")
        print(f"  Z: {z_convention}")
        print("="*70)
        
        return x_convention, y_convention, z_convention
    
    def visualize_projection_mismatch(self, save_path="output/diagnostic_projection.png"):
        """Visualize what happens with different projection conventions."""
        print("\n" + "="*70)
        print("TEST 3: Visual Projection Comparison")
        print("="*70)
        
        # Generate SMPL-X at origin
        transl = torch.tensor([[0.0, 0.0, 5.0]], device=self.device)
        
        with torch.no_grad():
            output = self.smplx_model(transl=transl, return_verts=True)
            joints = output.joints[0].cpu().numpy()
        
        # Get key joints
        joint_indices = {
            'left_shoulder': 16,
            'right_shoulder': 17,
            'left_hip': 1,
            'right_hip': 2,
            'left_knee': 4,
            'right_knee': 5,
        }
        
        # Project with both versions
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Version 1: With Y-flip (current)
        joints_2d_v1 = self._project_v1(joints)
        self._plot_skeleton(axes[0], joints_2d_v1, joint_indices, "V1: y_2d = -fy*y/z + cy")
        
        # Version 2: Without Y-flip
        joints_2d_v2 = self._project_v2(joints)
        self._plot_skeleton(axes[1], joints_2d_v2, joint_indices, "V2: y_2d = fy*y/z + cy")
        
        # Version 3: Flip SMPL-X Y before projection
        joints_flipped = joints.copy()
        joints_flipped[:, 1] = -joints_flipped[:, 1]
        joints_2d_v3 = self._project_v2(joints_flipped)
        self._plot_skeleton(axes[2], joints_2d_v3, joint_indices, "V3: Flip SMPL-X Y, then project")
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n💾 Saved visualization to: {save_path}")
        plt.close()
    
    def test_with_real_keypoints(self, keypoints_2d: np.ndarray):
        """Test which projection gives best alignment with real detected keypoints."""
        print("\n" + "="*70)
        print("TEST 4: Real Keypoint Alignment")
        print("="*70)
        
        if keypoints_2d is None:
            print("⚠️  No real keypoints provided, skipping this test")
            return
        
        print("Testing projection methods against detected keypoints...")
        # This would compare projected SMPL-X joints with detected 2D keypoints
        # Implementation depends on having real data
    
    def _project_v1(self, points_3d: np.ndarray) -> np.ndarray:
        """Current projection (with -Y flip)."""
        fx, fy = self.cam_intrinsics[0, 0], self.cam_intrinsics[1, 1]
        cx, cy = self.cam_intrinsics[0, 2], self.cam_intrinsics[1, 2]
        
        x_2d = fx * points_3d[:, 0] / points_3d[:, 2] + cx
        y_2d = -fy * points_3d[:, 1] / points_3d[:, 2] + cy  # ← NEGATIVE
        
        return np.stack([x_2d, y_2d], axis=1)
    
    def _project_v2(self, points_3d: np.ndarray) -> np.ndarray:
        """Alternative projection (without -Y flip)."""
        fx, fy = self.cam_intrinsics[0, 0], self.cam_intrinsics[1, 1]
        cx, cy = self.cam_intrinsics[0, 2], self.cam_intrinsics[1, 2]
        
        x_2d = fx * points_3d[:, 0] / points_3d[:, 2] + cx
        y_2d = fy * points_3d[:, 1] / points_3d[:, 2] + cy  # ← NO NEGATIVE
        
        return np.stack([x_2d, y_2d], axis=1)
    
    def _unproject(self, points_2d: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Unproject 2D points to 3D (matching utils.py)."""
        fx, fy = self.cam_intrinsics[0, 0], self.cam_intrinsics[1, 1]
        cx, cy = self.cam_intrinsics[0, 2], self.cam_intrinsics[1, 2]
        
        x = (points_2d[:, 0] - cx) * depth / fx
        y = -(points_2d[:, 1] - cy) * depth / fy  # ← Matches utils.py
        z = depth
        
        return np.stack([x, y, z], axis=1)
    
    def _plot_skeleton(self, ax, joints_2d, joint_indices, title):
        """Plot skeleton on image."""
        H, W = self.image_size
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)  # Flip Y-axis for image coordinates
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('u (pixels)')
        ax.set_ylabel('v (pixels)')
        ax.grid(True, alpha=0.3)
        
        # Draw joints
        for name, idx in joint_indices.items():
            if idx < len(joints_2d):
                u, v = joints_2d[idx]
                color = 'red' if 'left' in name else 'blue'
                ax.plot(u, v, 'o', color=color, markersize=8, label=name if idx == list(joint_indices.values())[0] else "")
                ax.text(u+5, v-5, name.replace('_', ' ').title(), fontsize=8)
        
        # Draw skeleton connections
        connections = [
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            ('left_shoulder', 'right_shoulder'),
            ('left_hip', 'left_knee'),
            ('right_hip', 'right_knee'),
        ]
        
        for joint1, joint2 in connections:
            if joint1 in joint_indices and joint2 in joint_indices:
                idx1, idx2 = joint_indices[joint1], joint_indices[joint2]
                if idx1 < len(joints_2d) and idx2 < len(joints_2d):
                    u1, v1 = joints_2d[idx1]
                    u2, v2 = joints_2d[idx2]
                    ax.plot([u1, u2], [v1, v2], 'g-', linewidth=2, alpha=0.6)
    
    def run_all_tests(self):
        """Run all diagnostic tests."""
        print("\n" + "🔬"*35)
        print("COORDINATE SYSTEM DIAGNOSTIC SUITE")
        print("🔬"*35)
        
        self.test_projection_consistency()
        x_conv, y_conv, z_conv = self.test_smplx_coordinate_system()
        self.visualize_projection_mismatch()
        
        print("\n" + "="*70)
        print("DIAGNOSIS COMPLETE")
        print("="*70)
        
        print("\n📋 SUMMARY:")
        print(f"  • SMPL-X uses: X={x_conv}, Y={y_conv}, Z={z_conv}")
        print(f"  • Depth map likely uses: X=RIGHT, Y=DOWN, Z=AWAY (OpenCV)")
        
        print("\n💡 RECOMMENDED FIX:")
        if y_conv == "UP":
            print("  1. SMPL-X uses Y-UP, but depth/pose use Y-DOWN")
            print("  2. In _project_points(), flip Y-axis of SMPL-X before projection:")
            print("     ```python")
            print("     points_cv = points_3d.clone()")
            print("     points_cv[:, 1] = -points_cv[:, 1]  # Convert Y-UP → Y-DOWN")
            print("     x_2d = fx * points_cv[:, 0] / points_cv[:, 2] + cx")
            print("     y_2d = fy * points_cv[:, 1] / points_cv[:, 2] + cy")
            print("     ```")
        
        if z_conv == "TOWARD":
            print("  3. SMPL-X Z points toward camera, depth map Z points away")
            print("     Consider flipping Z as well if needed")
        
        print("\n" + "="*70)


def main():
    """Run the diagnostic."""
    diagnostic = CoordinateSystemDiagnostic(device="cuda")
    diagnostic.run_all_tests()
    
    print("\n✅ Check the generated visualization in: output/diagnostic_projection.png")
    print("   This shows how different projection methods affect the skeleton.")


if __name__ == "__main__":
    main()