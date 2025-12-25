"""
Pipeline Configuration
Adjust these settings to customize the reconstruction pipeline.
"""

# Input/Output Settings
# INPUT_IMAGE = "/home/khater/pose-check/tom.jpg"
# INPUT_IMAGE = "omar3.jpg"
INPUT_IMAGE = "mohamed.jpg"
OUTPUT_DIR = "output"

# Model Settings
SMPLX_MODEL_PATH = "./data/smplx"
GENDER = "male"  # Options: "male", "female", "neutral"

# Depth Model Settings
DEPTH_MODEL = "depth-anything/DA3METRIC-LARGE"

# FOV Estimator Settings
FOV_ESTIMATOR_NAME = "moge2"

# SAM3 Settings
SAM3_PROMPT = "full body of a person"
SAM3_CONFIDENCE_THRESHOLD = 0.5

# YOLO Pose Settings
POSE_CONFIDENCE_THRESHOLD = 0.5

# SMPL-X Fitting Settings
FITTING_CONFIG = {
    "phase1_iterations": 300,
    "phase2_iterations": 500,
    "clothing_buffer": 0.03,  # 3cm tolerance for clothing
    "conf_threshold": 0.5
}

# Visualization Settings
ENABLE_VISUALIZATION = True
SAVE_INTERMEDIATE_OUTPUTS = True

# GPU Settings
FORCE_CPU = False  # Set to True to force CPU-only execution
