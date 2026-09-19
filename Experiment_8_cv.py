# Name: Abhay Singh Tomar
# BTech Cse 3rd year
# sec:"B"
# Cu24250144
# ROll no. = 1

# Experiment No. 8

# Experiment Name
# Application of Optical Flow for Real-Time Object Tracking and Motion Analysis

# Aim
# To implement an optical flow-based motion analysis system for real-time object tracking and evaluate its effectiveness in detecting and tracking moving objects in video sequences using Python and OpenCV.

import cv2
import numpy as np
import matplotlib.pyplot as plt

from google.colab import files
from IPython.display import display, HTML
from base64 import b64encode

print("Libraries imported successfully!")

uploaded = files.upload()

video_path = list(uploaded.keys())[0]

print("Video uploaded successfully:", video_path)

cap = cv2.VideoCapture(video_path)

ret, old_frame = cap.read()

if not ret:
    print("Error: Unable to read video")
else:
    print("Video loaded successfully!")

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

feature_params = dict(
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

p0 = cv2.goodFeaturesToTrack(
    old_gray,
    mask=None,
    **feature_params
)

print("Feature points detected:", len(p0) if p0 is not None else 0)

lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(
        cv2.TERM_CRITERIA_EPS |
        cv2.TERM_CRITERIA_COUNT,
        10,
        0.03
    )
)

mask = np.zeros_like(old_frame)

trajectory_points = []
motion_data = []

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if p0 is None or len(p0) == 0:
        break

    p1, st, err = cv2.calcOpticalFlowPyrLK(
        old_gray,
        frame_gray,
        p0,
        None,
        **lk_params
    )

    if p1 is None:
        break

    good_new = p1[st == 1]
    good_old = p0[st == 1]

    for new, old in zip(good_new, good_old):

        x_new, y_new = new.ravel()
        x_old, y_old = old.ravel()

        dx = x_new - x_old
        dy = y_new - y_old

        magnitude = np.sqrt(dx**2 + dy**2)

        angle = np.degrees(np.arctan2(dy, dx))

        motion_data.append(
            [frame_count, dx, dy, magnitude, angle]
        )

        cv2.line(
            mask,
            (int(x_new), int(y_new)),
            (int(x_old), int(y_old)),
            (0, 255, 0),
            2
        )

        cv2.circle(
            frame,
            (int(x_new), int(y_new)),
            5,
            (0, 0, 255),
            -1
        )

    output = cv2.add(frame, mask)

    trajectory_points.append(output)

    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    frame_count += 1

cap.release()

print("Lucas-Kanade tracking completed!")
print("Frames processed:", frame_count)
print("Motion records:", len(motion_data))

if len(trajectory_points) > 0:

    selected_frames = np.linspace(
        0,
        len(trajectory_points) - 1,
        min(6, len(trajectory_points)),
        dtype=int
    )

    plt.figure(figsize=(16, 10))

    for i, index in enumerate(selected_frames):

        image = cv2.cvtColor(
            trajectory_points[index],
            cv2.COLOR_BGR2RGB
        )

        plt.subplot(2, 3, i + 1)
        plt.imshow(image)
        plt.title(f"Frame {index}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

else:
    print("No tracking frames available.")

if len(motion_data) > 0:

    motion_array = np.array(motion_data)

    magnitudes = motion_array[:, 3]

    average_magnitude = np.mean(magnitudes)
    maximum_magnitude = np.max(magnitudes)

    print("Average displacement:", average_magnitude)
    print("Maximum displacement:", maximum_magnitude)

else:
    print("No motion data available.")

if len(motion_data) > 0:

    plt.figure(figsize=(10, 5))

    plt.plot(
        motion_array[:, 3],
        label="Motion Magnitude"
    )

    plt.title("Optical Flow Motion Magnitude")
    plt.xlabel("Motion Sample")
    plt.ylabel("Displacement (Pixels)")
    plt.legend()
    plt.grid(True)

    plt.show()

else:
    print("No motion data available.")

cap = cv2.VideoCapture(video_path)

ret, previous_frame = cap.read()

if not ret:
    print("Unable to read video")
else:

    previous_gray = cv2.cvtColor(
        previous_frame,
        cv2.COLOR_BGR2GRAY
    )

    dense_results = []
    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        current_gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        flow = cv2.calcOpticalFlowFarneback(
            previous_gray,
            current_gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0
        )

        magnitude, angle = cv2.cartToPolar(
            flow[..., 0],
            flow[..., 1]
        )

        hsv = np.zeros_like(
            cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        )

        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(
            magnitude,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        flow_rgb = cv2.cvtColor(
            hsv,
            cv2.COLOR_HSV2RGB
        )

        dense_results.append(flow_rgb)

        previous_gray = current_gray.copy()
        frame_number += 1

    cap.release()

    print("Farneback Optical Flow completed!")
    print("Frames processed:", frame_number)

if len(dense_results) > 0:

    selected_frames = np.linspace(
        0,
        len(dense_results) - 1,
        min(6, len(dense_results)),
        dtype=int
    )

    plt.figure(figsize=(16, 10))

    for i, index in enumerate(selected_frames):

        plt.subplot(2, 3, i + 1)
        plt.imshow(dense_results[index])
        plt.title(f"Frame {index}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

else:
    print("No dense flow results available.")


# QUESTION AND ANSWER

# Q1. How does object tracking differ from object detection?
# Answer: Object detection identifies objects in an image or video frame, whereas object tracking follows the movement of identified objects across multiple frames. Detection determines the location of an object, while tracking maintains its identity and movement over time.

# Q2. Explain how Optical Flow can be used for real-time object tracking.
# Answer: Optical Flow estimates the movement of pixels or feature points between consecutive video frames. By tracking these changes, the system can calculate object displacement, direction, and motion trajectories. This allows continuous object tracking without performing object detection independently in every frame.

# Q3. What is the role of Shi-Tomasi Corner Detection in the Lucas-Kanade Optical Flow algorithm?
# Answer: Shi-Tomasi Corner Detection identifies strong and distinctive feature points in an image. These points are used as input for the Lucas-Kanade algorithm. Lucas-Kanade then tracks their movement between consecutive frames. This improves tracking by focusing on points that can be reliably identified.

# Q4. Why is Optical Flow suitable for motion analysis in videos?
# Answer: Optical Flow estimates the movement of pixels between consecutive frames. It provides information about displacement, direction, and motion magnitude. This makes it useful for analyzing object movement, detecting motion patterns, and estimating trajectories in video sequences.

# Q5. What challenges arise while tracking fast-moving or partially occluded objects?
# Answer: Fast-moving objects can move significantly between consecutive frames, making feature matching difficult. Partial occlusion can hide important feature points. Other challenges include motion blur, illumination changes, low-texture surfaces, and camera movement. These conditions can reduce tracking accuracy.

# Q6. Compare Optical Flow-based tracking with deep learning-based object tracking methods.
# Answer:
# Optical Flow estimates motion between consecutive frames using pixel or feature-point changes. It can be computationally efficient and does not necessarily require a trained object detection model.
# Deep learning-based tracking methods use trained neural networks to detect, identify, and track objects. They can provide semantic object information but may require more computational resources and training data.

# Q7. How can Optical Flow be used in traffic monitoring and autonomous driving systems?
# Answer: Optical Flow can estimate the movement of vehicles, pedestrians, and other objects in video streams. In traffic monitoring, it can help analyze vehicle direction and motion patterns. In autonomous driving, optical flow can contribute to motion estimation, obstacle movement analysis, and navigation-related perception.

# Q8. Explain the effect of camera motion on Optical Flow estimation.
# Answer: When the camera moves, many or all background pixels may appear to move. Optical Flow then measures both camera-induced motion and object motion. This can make it difficult to distinguish independently moving objects. Camera motion compensation or additional scene analysis can help reduce this problem.

# Q9. Mention five real-world applications where motion analysis using Optical Flow is commonly employed.
# Answer:
# Traffic monitoring
# Autonomous navigation
# Video surveillance
# Human activity recognition
# Sports analytics

# Q10. How can Optical Flow improve the performance of surveillance, robotics, and human activity recognition systems?
# Answer: Optical Flow provides motion information that can be used to identify movement patterns and track objects. In surveillance, it can help detect unusual motion. In robotics, it can support motion estimation and navigation. In human activity recognition, it can help analyze body movement and activity patterns.
