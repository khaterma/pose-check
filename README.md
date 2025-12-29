# Pose-check

Pose-check aligns metric SMPL-X reconstructions with depth-aware point clouds derived from single RGB inputs. The repository bundles a depth estimation frontend, pose fitting backend
## Key components
- `metric_alignment_pipeline.py`: main driver that estimates depth with Depth-Anything-3, fits a SMPL-X mesh through NLF, renders the mesh depth map, and aligns scales via histogram ratios or 3D ICP-style matching.
- `body_reconstruct.py` & `vitpose.py`: integrate the reconstruction stack and YOLO/VPoser pose priors needed to initialize the SMPL-X fitting process.
- `nlf.py`: SMPL-X optimization wrapper (NLF) responsible for producing expressive meshes that match 2D cues.
- `utils.py`, `fov_estimator.py`: shared utilities for projection, intrinsics, and field-of-view estimation.
- `Depth-Anything-3/` and `da3_streaming/`: copies of the DA3 metric depth estimator and streaming helper scripts that supply the depth point cloud used at alignment time.

## Repository layout
- `config.py`: central settings (input/output paths, chosen depth model, FOV estimator, fitting hyperparameters, visualization toggles).
- `data/`: SMPL-X assets (`SMPLX_FEMALE.npz`, etc.) and Depth-Anything checkpoints.
- `files/`: extra helpers such as `metric_smpl_fitter.py`, `rvh_fitter.py`, utilities for overlaying depth with pose.
- `checkpoints/`: pretrained tensors (NLF, VitPose, YOLO) referenced by the pipeline.
- `input/` & `output/`: sample inputs and destinations for rendered overlays, depth maps, and aligned meshes.

## Prerequisites
1. **Python**: 3.10+ recommended (GPU execution requires CUDA-compatible PyTorch).
2. **Dependencies**: install via `pip install -r requirements_smpl.txt` and add extras listed in the comments (PyTorch3D, smplx, segment-anything, etc.).
3. **CUDA / GPU**: strongly recommended for SMPL fitting and rendering but the pipeline can run on CPU by setting `FORCE_CPU = True` in `config.py`.
4. **Data**: ensure `data/smplx` contains the SMPL-X `.npz` files—already provided in this repo under `data/smplx/smplx/`.
5. **Depth Anything v3 + SAM3 + NLF**: install the dependencies described above (micromamba-based script, apt packages, optional extras) and point `config.DEPTH_MODEL` to your chosen Depth Anything checkpoint.

## Quick start
```bash
python -m pip install -r requirements_smpl.txt
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
# optional extras shown in requirements (segment-anything, human_body_prior)

python metric_alignment_pipeline.py \
  --input input/omar.jpg \
  --output output/ \
  --mode depth_ratio
```
Default behavior reads `config.INPUT_IMAGE` and `config.OUTPUT_DIR` if you omit CLI flags. Switch alignment strategy with `-m point_cloud_3d` for the ICP-style matcher.

## Usage notes
- `metric_alignment_pipeline.py` prints diagnostics such as final scale, coverage, and histogram peaks, and saves overlays/histograms when `config.SAVE_INTERMEDIATE_OUTPUTS` is enabled.
- Use `Depth-Anything-3/` to train or tune the depth model separately and drop new checkpoints into `checkpoints/` that match `config.DEPTH_MODEL`.
- `files/overlay_depth_and_pose.py` is handy for visual comparisons between the fitted mesh and the estimated depth map.

## Suggested workflow
1. Drop a high-resolution person image into `input/` (or change `config.INPUT_IMAGE`).
2. Run `metric_alignment_pipeline.py` to get aligned point clouds and mesh depth maps in `output/`.
3. Inspect plots and overlays generated in the output directory to validate scale and coverage.

## TODO: installation steps & environments
- Write a dedicated installation guide for Depth-Anything-3, SAM3, and NLF (micromamba bootstrap, apt packages, pip/conda list, optional fast JPEG/AV builds).
- Capture the environment setup command sequence (`micromamba env create ...`, `micromamba activate nlf`, `pip install ...`) inside a script or `docs/setup.md` so it can be rerun without copying from memory.
- Document how to source `~/.bashrc`, export `MAMBA_ROOT_PREFIX`, and activate the project environment before running the pipeline (`cd /home/khater/pose-check && micromamba activate nlf`).
- Decide on and document a stable requirements lockfile for pip/conda so contributors can reproduce the GPU stack (PyTorch3D, DA3 checkpoints, SAM3/NLF extras).

## Next steps
1. Add a `requirements.txt` and installation steps.
2. Document expected GPU/LUT memory usage for large images when running PyTorch3D rasterization.
3. Automate evaluation by comparing aligned meshes against ground-truth scans (if available) using `files/metric_smpl_fitter.py`.
