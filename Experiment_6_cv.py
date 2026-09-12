# Name: Abhay Singh Tomar
# BTech Cse 3rd year
# sec:"B"
# Cu24250144
# ROll no. = 1

# Experiment No. 6

# Experiment Name
# Implementation and Comparative Analysis of Image Segmentation Techniques using Python and OpenCV

# Aim
# To implement and evaluate various image segmentation techniques for partitioning digital images into meaningful regions, thereby facilitating object localization, scene understanding, and subsequent computer vision tasks.

import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

print("Libraries imported successfully!")

from google.colab import files

uploaded = files.upload()

image_name = list(uploaded.keys())[0]
image = cv2.imread(image_name)

print("Image loaded successfully!")
print("Image Shape:", image.shape)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8, 6))
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(8, 6))
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
plt.show()

blurred = cv2.GaussianBlur(gray, (5, 5), 0)

plt.figure(figsize=(8, 6))
plt.imshow(blurred, cmap="gray")
plt.title("Gaussian Blurred Image")
plt.axis("off")
plt.show()

_, global_thresh = cv2.threshold(
    blurred,
    127,
    255,
    cv2.THRESH_BINARY
)

plt.figure(figsize=(8, 6))
plt.imshow(global_thresh, cmap="gray")
plt.title("Global Thresholding")
plt.axis("off")
plt.show()

otsu_value, otsu_thresh = cv2.threshold(
    blurred,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

print("Optimal Otsu Threshold:", otsu_value)

plt.figure(figsize=(8, 6))
plt.imshow(otsu_thresh, cmap="gray")
plt.title("Otsu's Thresholding")
plt.axis("off")
plt.show()

adaptive_thresh = cv2.adaptiveThreshold(
    blurred,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)

plt.figure(figsize=(8, 6))
plt.imshow(adaptive_thresh, cmap="gray")
plt.title("Adaptive Thresholding")
plt.axis("off")
plt.show()

# Convert threshold image to binary
binary = otsu_thresh

# Remove noise
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    kernel,
    iterations=2
)

# Sure background
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Distance transform
dist_transform = cv2.distanceTransform(
    opening,
    cv2.DIST_L2,
    5
)

# Sure foreground
_, sure_fg = cv2.threshold(
    dist_transform,
    0.7 * dist_transform.max(),
    255,
    0
)

sure_fg = np.uint8(sure_fg)

# Unknown region
unknown = cv2.subtract(
    sure_bg,
    sure_fg
)

# Marker labelling
num_labels, markers = cv2.connectedComponents(sure_fg)

markers = markers + 1
markers[unknown == 255] = 0

# Apply Watershed
watershed_image = image.copy()

markers = cv2.watershed(
    watershed_image,
    markers
)

watershed_image[markers == -1] = [255, 0, 0]

watershed_rgb = cv2.cvtColor(
    watershed_image,
    cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(10, 7))
plt.imshow(watershed_rgb)
plt.title("Watershed Segmentation")
plt.axis("off")
plt.show()

# Convert image from BGR to RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshape image pixels
pixels = rgb_image.reshape((-1, 3))

# Convert to float
pixels = np.float32(pixels)

# Number of clusters
k = 3

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

labels = kmeans.fit_predict(pixels)

centers = np.uint8(kmeans.cluster_centers_)

segmented_pixels = centers[labels]

segmented_image = segmented_pixels.reshape(rgb_image.shape)

plt.figure(figsize=(8, 6))
plt.imshow(segmented_image)
plt.title("K-Means Image Segmentation")
plt.axis("off")
plt.show()

plt.figure(figsize=(16, 10))

plt.subplot(2, 3, 1)
plt.imshow(image_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(global_thresh, cmap="gray")
plt.title("Global Threshold")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(otsu_thresh, cmap="gray")
plt.title("Otsu Threshold")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(adaptive_thresh, cmap="gray")
plt.title("Adaptive Threshold")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(watershed_rgb)
plt.title("Watershed")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(segmented_image)
plt.title("K-Means")
plt.axis("off")

plt.tight_layout()
plt.show()


# EXPERIMENT 6 — QUESTION ANSWERS
# Q1. What is image segmentation? Why is it considered a fundamental step in computer vision?
# Answer: Image segmentation is the process of dividing an image into meaningful regions or objects. It is fundamental because it helps isolate important areas of an image for tasks such as object detection, recognition, medical analysis, and scene understanding.

# Q2. Differentiate between Image Segmentation and Image Classification.
# Answer: Image segmentation divides an image into different regions or objects, while image classification assigns a label to the entire image. Segmentation provides information about the location and boundaries of objects, whereas classification mainly identifies what the image represents.

# Q3. Explain the working principle of Global Thresholding, Otsu's Thresholding, and Adaptive Thresholding.
# Answer: Global Thresholding uses one fixed threshold for the entire image. Otsu's Thresholding automatically selects an optimal threshold from the image histogram. Adaptive Thresholding calculates different threshold values for different local regions, making it useful for uneven lighting conditions.

# Q4. What is the Watershed Algorithm? Why is it useful for separating overlapping objects?
# Answer: Watershed is a segmentation algorithm that treats an image like a topographic surface and identifies boundaries between regions. It is useful for separating touching or overlapping objects because it can identify boundaries between individual objects.

# Q5. How is K-Means Clustering applied to image segmentation?
# Answer: In K-Means segmentation, image pixels are grouped into a fixed number of clusters based on their properties such as color. Each pixel is assigned to the nearest cluster, and the resulting clusters form different segmented regions.

# Q6. Compare threshold-based segmentation and clustering-based segmentation techniques.
# Answer: Threshold-based segmentation separates pixels using selected intensity values. Clustering-based segmentation groups similar pixels into clusters. Thresholding is generally simpler, while clustering can handle more complex color-based segmentation.

# Q7. What challenges are encountered while segmenting images with complex backgrounds or varying illumination?
# Answer: Complex backgrounds can make it difficult to distinguish objects from the background. Varying illumination can also change pixel intensity and reduce segmentation accuracy. Noise, shadows, and similar colors can create additional difficulties.

# Q8. Mention five real-world applications where image segmentation plays a critical role.
# Answer: Five applications of image segmentation are:

# Medical image analysis
# Object detection
# Autonomous vehicles
# Satellite image analysis
# Industrial inspection
# Q9. How does image segmentation improve the performance of object detection and image recognition systems?
# Answer: Segmentation separates important objects from the background and provides their boundaries. This reduces unnecessary information and allows recognition and detection systems to focus on relevant regions, improving their performance.

# Q10. Compare traditional image segmentation techniques with deep learning-based segmentation methods such as U-Net and Mask R-CNN.
# Answer: Traditional techniques such as thresholding and K-Means are simpler and require less computational power. Deep learning methods such as U-Net and Mask R-CNN can learn complex features automatically and generally provide better results on difficult images, but they require more data and computational resources.
