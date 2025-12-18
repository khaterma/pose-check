import torch
import smplx
import os
import numpy as np



# --- Configuration ---
# 1. Path to your models folder (the directory containing 'smpl', 'smplx', etc.)
MODEL_FOLDER = 'data'
# 2. Choose the specific model
MODEL_TYPE = 'smplx' # Options: 'smpl', 'smplh', 'smplx'
GENDER = 'neutral' # Options: 'neutral', 'male', 'female'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # Use 'cuda' if you have a GPU

# 3. Create the SMPL layer (a PyTorch module)
smpl_layer = smplx.create(
    model_path=MODEL_FOLDER,
    model_type=MODEL_TYPE,
    gender=GENDER,
    # This is important for batch processing
    num_betas=10,        # Number of shape coefficients
    num_expression_coeffs=10, # Only for SMPL-X
    ext='pkl',           # File extension of the model
).to(DEVICE)

print(f"Loaded {MODEL_TYPE} model on {DEVICE}")

print("SMPL Layer:", smpl_layer)

num_betas = smpl_layer.num_betas

# Initialize a batch of 1 person with an average shape (zeros)
betas = torch.zeros(1, num_betas, device=DEVICE)

# Example: Make the person taller (often beta[0] controls height)
# A value of +3.0 is a 3-standard-deviation change from the mean.
betas[0, 0] = 3.0

# 1. Global Orientation (No rotation - facing forward)
global_orient = torch.zeros(1, 1, 3, device=DEVICE)

# 2. Body Pose (T-pose/A-pose - all joints straight)
body_pose = torch.zeros(1, smpl_layer.NUM_BODY_JOINTS, 3, device=DEVICE)

# Example: Lift the right elbow (Index 18 is the right elbow)
# Axis-Angle: [Angle_X, Angle_Y, Angle_Z]
# A rotation around the X-axis (forward/backward) by ~60 degrees (in radians)
elbow_bend = np.deg2rad(40.0) 
# body_pose[0, 18, 0] = elbow_bend
# body_pose = torch.zeros((1, 63))

LEFT_SHOULDER = 15
RIGHT_SHOULDER = 16

# Rotate shoulders up (X axis)
body_pose[0, LEFT_SHOULDER + 0] = -0.4
body_pose[0, RIGHT_SHOULDER + 0] = 4


# Place the root joint at 3D coordinate (0, 0, 1.5)
transl = torch.tensor([[0., 0., 1.5]], device=DEVICE)

# --- 4. Forward Pass ---
output = smpl_layer(
    betas=betas,
    body_pose=body_pose,
    global_orient=global_orient,
    transl=transl,
    return_verts=True  # Tells the model to calculate the vertex positions
)

# --- 5. Extract Results ---

# 3D Coordinates of the mesh vertices (N x 6890 x 3)
vertices = output.vertices.detach().cpu().numpy().squeeze() 

# 3D Coordinates of the joint locations (N x 24 x 3)
joints = output.joints.detach().cpu().numpy().squeeze() 

# The Triangulation Faces (always the same for a given SMPL model)
faces = smpl_layer.faces

print(f"Generated {vertices.shape[0]} vertices and {joints.shape} joints.")

# The generated mesh can now be visualized using tools like Open3D or trimesh.
# You can save the mesh to an OBJ file:
import trimesh
trimesh.Trimesh(vertices=vertices, faces=faces).export('posed_mesh.obj')


print(f" joints {joints} ")