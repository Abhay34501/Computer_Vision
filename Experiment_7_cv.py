# Name: Abhay Singh Tomar
# BTech Cse 3rd year
# sec:"B"
# Cu24250144
# ROll no. = 1

# Experiment No. 7

# Experiment Name
# Motion Estimation using Optical Flow Algorithms in Video Sequences

# Aim
# To implement optical flow algorithms for estimating the motion of objects between consecutive video frames and analyze motion patterns for dynamic scene understanding using Python and OpenCV.

import cv2
import numpy as np
import matplotlib.pyplot as plt

print("Libraries imported successfully!")

from google.colab import files

uploaded = files.upload()

video_name = list(uploaded.keys())[0]

cap = cv2.VideoCapture(video_name)

if not cap.isOpened():
    print("Error: Could not open video")
else:
    print("Video opened successfully!")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

print("Total Frames:", total_frames)
print("FPS:", fps)

ret, frame1 = cap.read()
ret2, frame2 = cap.read()

if ret and ret2:
    print("Two frames read successfully!")
else:
    print("Error reading frames")

frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(frame1_rgb)
plt.title("Frame 1")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(frame2_rgb)
plt.title("Frame 2")
plt.axis("off")

plt.show()

gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(gray1, cmap="gray")
plt.title("Grayscale Frame 1")
plt.axis("off")

plt.subplot(1, 2, 2)
hsv = np.zeros_like(frame1)

hsv[..., 1] = 255

hsv[..., 0] = angle * 180 / np.pi / 2

hsv[..., 2] = cv2.normalize(
    magnitude,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

dense_flow_rgb = cv2.cvtColor(
    hsv,
    cv2.COLOR_HSV2RGB
)

plt.figure(figsize=(10, 7))
plt.imshow(dense_flow_rgb)
plt.title("Farneback Dense Optical Flow")
plt.axis("off")
plt.show()

plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.imshow(lk_result)
plt.title("Lucas-Kanade Sparse Optical Flow")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(dense_flow_rgb)
plt.title("Farneback Dense Optical Flow")
plt.axis("off")

plt.tight_layout()
plt.show()

average_magnitude = np.mean(magnitude)
maximum_magnitude = np.max(magnitude)

print("Motion Analysis")
print("=" * 40)

print("Average Motion Magnitude:", average_magnitude)
print("Maximum Motion Magnitude:", maximum_magnitude)
print("Tracked Points:", len(good_new))

cap.release()

print("Video processing completed!")


# # EXPERIMENT 7 — QUESTION ANSWERS

# ### Q1. What is Optical Flow? How is it used in computer vision?

# **Answer:**
# Optical Flow is a technique used to estimate the apparent motion of objects or surfaces between consecutive video frames. It is used in object tracking, motion detection, video analysis, autonomous driving, and surveillance systems.

# ### Q2. Explain the working principle of the Lucas-Kanade Optical Flow algorithm.

# **Answer:**
# Lucas-Kanade estimates the motion of selected feature points between two consecutive frames. It assumes that the motion is approximately constant within a small image region. It uses image intensity changes and gradients to calculate the movement of these points.

# ### Q3. What is Dense Optical Flow? How does it differ from Sparse Optical Flow?

# **Answer:**
# Dense Optical Flow estimates motion for nearly every pixel in an image. Sparse Optical Flow estimates motion only for selected feature points. Dense flow provides more complete motion information, while sparse flow generally requires fewer computations.

# ### Q4. Compare the Lucas-Kanade and Farneback optical flow algorithms.

# **Answer:**
# Lucas-Kanade is a sparse optical flow method that tracks selected feature points. Farneback is a dense method that estimates motion across the image. Lucas-Kanade is useful for point tracking, while Farneback is suitable for detailed motion analysis.

# ### Q5. What assumptions are made while computing optical flow?

# **Answer:**
# Optical flow commonly assumes that pixel brightness remains approximately constant during motion. It also assumes that motion is relatively small between frames and that neighboring pixels have similar motion. These assumptions may not hold in all real-world situations.

# ### Q6. What factors can affect the accuracy of optical flow estimation?

# **Answer:**
# The accuracy of optical flow can be affected by object speed, lighting changes, camera movement, image noise, occlusion, and large motion between frames. Blurred or low-texture regions can also make motion estimation difficult.

# ### Q7. Mention five real-world applications of optical flow in computer vision.

# **Answer:**
# Five applications of optical flow are:

# 1. Autonomous driving
# 2. Video surveillance
# 3. Object tracking
# 4. Gesture recognition
# 5. Sports analytics

# ### Q8. Why are grayscale images generally used for optical flow computation?

# **Answer:**
# Grayscale images simplify optical flow computation by using a single intensity channel instead of multiple color channels. This reduces computational complexity and allows algorithms to focus on intensity changes and image gradients.

# ### Q9. What are the limitations of optical flow algorithms in real-world environments?

# **Answer:**
# Optical flow algorithms can be affected by lighting changes, fast-moving objects, occlusion, noise, and camera motion. They may also produce inaccurate results when objects move significantly between consecutive frames or when the image lacks distinctive features.

# ### Q10. How does optical flow contribute to applications such as autonomous driving, video surveillance, and action recognition?

# **Answer:**
# Optical flow helps estimate the movement of objects and the camera in video sequences. In autonomous driving, it supports motion analysis and obstacle detection. In surveillance, it helps track moving objects, while in action recognition, it provides information about body and object movements.


# print("Video uploaded:", video_name)
