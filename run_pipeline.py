"""
Simple entry point for the 3D body reconstruction pipeline.
Adjust settings in config.py before running.
"""

import torch
from main_pipeline import BodyReconstructionPipeline
import config


def main():
    """Run the reconstruction pipeline with settings from config.py"""
    
    # Determine device
    if config.FORCE_CPU:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Initialize pipeline
    pipeline = BodyReconstructionPipeline(
        device=device,
        output_dir=config.OUTPUT_DIR
    )
    
    # Run pipeline
    results = pipeline.run(
        image_path=config.INPUT_IMAGE,
        gender=config.GENDER,
        enable_visualization=config.ENABLE_VISUALIZATION
    )
    
    print("\n" + "="*60)
    print("✓ Pipeline Complete!")
    print("="*60)
    print(f"Results saved to: {config.OUTPUT_DIR}/")
    print("\nOutput files:")
    print("  - pose_overlay.png       : 2D pose visualization")
    print("  - sam3_mask_*.png        : Segmentation masks")
    print("  - point_cloud.ply        : 3D point cloud")
    print("  - smplx_fitted.obj/.ply  : Fitted body mesh")
    if config.ENABLE_VISUALIZATION:
        print("  - fitting_visualization.png")
        print("  - phase_comparison.png")
        print("  - colored_smplx.ply")
    
    return results


if __name__ == "__main__":
    main()
