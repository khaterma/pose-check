from ultralytics import YOLO
import cv2
import numpy as np

SKELETON = [
    (5, 6),   # shoulders
    (5, 7), (7, 9),    # left arm
    (6, 8), (8, 10),   # right arm
    (11, 12),          # hips
    (5, 11), (6, 12),  # torso
    (11, 13), (13, 15),# left leg
    (12, 14), (14, 16)# right leg
]


### inference function using ultralytics YOLOv11-pose model

def yolo_pose_inference(image_paths):
    # Load a model
    model = YOLO("yolo11n-pose.pt")  # load an official model

    key_points_all = []

    results_dict = {}

    for image_path in image_paths:
        # Predict with the model
        results = model(image_path)  # predict on an image
    return results



def main():
    # Load a model
    model = YOLO("checkpoints/yolo11n-pose.pt")  # load an official model

    # Predict with the model
    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    results = model("/home/khater/pose-check/output/tom.jpg")  # predict on an image
    img = cv2.imread("/home/khater/pose-check/output/tom.jpg")

    # Access the results
    key_points = []
    for result in results:
        xy = result.keypoints.xy  # x and y coordinates
        xyn = result.keypoints.xyn  # normalized
        kpts = result.keypoints.data  # x, y, visibility (if available)
        # print kpts class 
        # print("Keypoints class:", type(kpts))
        # key_points.append(kpts)
        print("Keypoints (x, y, v):\n", kpts)
        print(f" keypoints shape: {kpts.shape} ")
        print("X and Y coordinates:\n", xy)
        print("Normalized coordinates:\n", xyn)
        # print(f" results {result}")
        if result.keypoints is not None:
            for person_kpts in result.keypoints.xy.cpu().numpy():
                # Add confidence if needed:
                conf = result.keypoints.conf.cpu().numpy()[0]
                keypoints = np.concatenate(
                    [person_kpts, conf[:, None]], axis=1
                )
                img = draw_pose(img, keypoints)
                

    cv2.imshow("Pose", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Visualize the results


def draw_pose(image, keypoints, conf_threshold=0.5):
    """
    image: BGR image (OpenCV)
    keypoints: shape (17, 3) -> (x, y, confidence)
    """

    # Draw keypoints
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > conf_threshold:
            cv2.circle(image, (int(x), int(y)), 4, (0, 255, 0), -1)
            ## print confidence value on the image 
            # if conf < 0.93:
            cv2.putText(image, f"{conf:.2f}", (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Draw skeleton
    for i, j in SKELETON:
        if keypoints[i][2] > conf_threshold and keypoints[j][2] > conf_threshold:
            pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
            pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
            cv2.line(image, pt1, pt2, (255, 0, 0), 2)

    return image



if __name__ == "__main__":
    main()



