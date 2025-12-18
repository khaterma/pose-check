# import glob, os, torch
# from depth_anything_3.api import DepthAnything3
# device = torch.device("cuda")
# model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
# model = model.to(device=device)
# example_path = "assets/examples/SOH"
# # images = sorted(glob.glob(os.path.join(example_path, "*.png")))
# images = ["/home/khater/pose-check/000000000785.jpg", "/home/khater/pose-check/tom.jpg"]
# prediction = model.inference(
#     images,
# )

# # show depth image shape
# import matplotlib.pyplot as plt
# plt.imshow(prediction.depth[1])
# plt.show()
# # prediction.processed_images : [N, H, W, 3] uint8   array
# print(prediction.processed_images.shape)
# # prediction.depth            : [N, H, W]    float32 array
# print(prediction.depth.shape)  
# # prediction.conf             : [N, H, W]    float32 array
# # print(prediction.conf.shape)  
# # prediction.extrinsics       : [N, 3, 4]    float32 array # opencv w2c or colmap format
# # print(prediction.extrinsics.shape)
# # prediction.intrinsics       : [N, 3, 3]    float32 array
# print(prediction.intrinsics.shape)
import torch
from depth_anything_3.api import DepthAnything3

# Load model from Hugging Face Hub
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
model = model.to(device=device)

# Run inference on images
# images = ["image1.jpg", "image2.jpg"]  # List of image paths, PIL Images, or numpy arrays
images = ["/home/khater/pose-check/000000000785.jpg", "/home/khater/pose-check/tom.jpg", "/home/khater/pose-check/bus.jpg"]

prediction = model.inference(
    images,
    export_dir="output",
    # export_format="colmap"  # Options: glb, npz, ply, mini_npz, gs_ply, gs_video
)
import matplotlib.pyplot as plt
plt.imshow(prediction.depth[2])
plt.show()
# Access results
print(prediction.depth.shape)        # Depth maps: [N, H, W] float32
print(prediction.conf.shape)         # Confidence maps: [N, H, W] float32
print(prediction.extrinsics.shape)   # Camera poses (w2c): [N, 3, 4] float32
print(prediction.intrinsics.shape)   # Camera intrinsics: [N, 3, 3] float32
