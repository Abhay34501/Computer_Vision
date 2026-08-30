# Name: Abhay Singh Tomar
# BTech Cse, 3rd year
# sec: "B"
# Cu24250144
# ROll no. = 1

# Experiment No. 1

# Experiment Name

# Implementation of Fundamental Image Processing Operations using Python and OpenCV

# To develop a comprehensive understanding of fundamental image processing techniques by
# implementing image acquisition, storage, color space conversion, geometric transformations,
# and image enhancement operations using Python and *OpenCV*


# Import required libraries

import cv2
import numpy as np
import matplotlib.pyplot as plt

print("Libraries imported successfully!")

from google.colab import files

# Upload image
uploaded = files.upload()

# Get uploaded image name
filename = list(uploaded.keys())[0]

# Read image using OpenCV
img = cv2.imread(filename)

# Check image
if img is None:
    raise ValueError("Image could not be loaded.")

print("Image loaded successfully!")
print("File name:", filename)

# Convert BGR image to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display using Matplotlib
plt.figure(figsize=(8, 6))
plt.imshow(img_rgb)
plt.title("Original Color Image")
plt.axis("off")
plt.show()

# Image properties

height, width, channels = img.shape

print("Image Properties")
print("-" * 40)
print("Height       :", height)
print("Width        :", width)
print("Channels     :", channels)
print("Image Shape  :", img.shape)
print("Data Type    :", img.dtype)
print("Resolution   :", width, "x", height)

# Save image in JPEG and PNG formats

cv2.imwrite("output_image.jpg", img)
cv2.imwrite("output_image.png", img)

print("Image saved successfully!")
print("Saved files:")
print("1. output_image.jpg")
print("2. output_image.png")

# Convert image to different color spaces

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

print("Color space conversion completed!")

plt.figure(figsize=(15, 5))

# Grayscale
plt.subplot(1, 3, 1)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

# HSV
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
plt.title("HSV Image")
plt.axis("off")

# LAB
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
plt.title("LAB Image")
plt.axis("off")

plt.tight_layout()
plt.show()

# Resize image

resized = cv2.resize(img, (600, 400))

plt.figure(figsize=(8, 5))
plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
plt.title("Resized Image")
plt.axis("off")
plt.show()

print("Original Size :", img.shape[1], "x", img.shape[0])
print("New Size      :", resized.shape[1], "x", resized.shape[0])

# Rotate image by 45 degrees

h, w = img.shape[:2]
center = (w // 2, h // 2)

# Rotation matrix
M = cv2.getRotationMatrix2D(center, 45, 1.0)

rotated = cv2.warpAffine(img, M, (w, h))

plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
plt.title("Rotated Image (45 Degrees)")
plt.axis("off")
plt.show()

# Horizontal flip

horizontal_flip = cv2.flip(img, 1)

plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(horizontal_flip, cv2.COLOR_BGR2RGB))
plt.title("Horizontal Flip")
plt.axis("off")
plt.show()

# Vertical flip

vertical_flip = cv2.flip(img, 0)

plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(vertical_flip, cv2.COLOR_BGR2RGB))
plt.title("Vertical Flip")
plt.axis("off")
plt.show()

# Generate negative image

negative = cv2.bitwise_not(img)

plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(negative, cv2.COLOR_BGR2RGB))
plt.title("Negative / Complement Image")
plt.axis("off")
plt.show()

# Crop a Region of Interest (ROI)

h, w = img.shape[:2]

# Select central region
x1 = int(w * 0.25)
y1 = int(h * 0.25)
x2 = int(w * 0.75)
y2 = int(h * 0.75)

roi = img[y1:y2, x1:x2]

plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
plt.title("Cropped Region of Interest (ROI)")
plt.axis("off")
plt.show()

print("ROI Shape:", roi.shape)

# Complete comparison of original and processed images

plt.figure(figsize=(18, 10))

images = [
    (img_rgb, "Original"),
    (gray, "Grayscale"),
    (cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), "HSV"),
    (cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), "LAB"),
    (cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), "Resized"),
    (cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB), "Rotated 45°"),
    (cv2.cvtColor(horizontal_flip, cv2.COLOR_BGR2RGB), "Horizontal Flip"),
    (cv2.cvtColor(vertical_flip, cv2.COLOR_BGR2RGB), "Vertical Flip"),
    (cv2.cvtColor(negative, cv2.COLOR_BGR2RGB), "Negative"),
    (cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), "ROI")
]

for i, (image, title) in enumerate(images):
    plt.subplot(2, 5, i + 1)

    if len(image.shape) == 2:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image)

    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()

print("=" * 65)
print("OBSERVATIONS")
print("=" * 65)

print("""
1. The color image was successfully loaded using OpenCV and
   displayed using Matplotlib.

2. The image properties such as height, width, channels,
   resolution, and data type were successfully examined.

3. The image was successfully saved in JPEG and PNG formats.
   JPEG provides compression, while PNG preserves image
   information using lossless compression.

4. Grayscale conversion reduced the three color channels into
   a single intensity channel.

5. HSV and LAB color spaces represented the image using
   different color and intensity characteristics.

6. Resizing changed the dimensions of the image, while rotation
   changed its orientation.

7. Horizontal and vertical flipping changed the spatial
   orientation of the image.

8. The negative operation inverted the pixel intensity values
   and produced a complementary image.

9. ROI cropping allowed a specific region of the image to be
   selected for further analysis.

10. These preprocessing operations help prepare images for
    advanced computer vision tasks.
""")

print("=" * 65)

print("=" * 65)
print("RESULT")
print("=" * 65)

print("""
Fundamental image processing operations were successfully
implemented using Python and OpenCV. Image acquisition,
storage, color space conversion, resizing, rotation, flipping,
image complement generation, and ROI extraction were performed
successfully.

These operations are important preprocessing steps for
computer vision applications such as feature extraction,
object detection, image classification, and image analysis.
""")

# Questions and Answers

# Q1. What is a digital image? Differentiate between grayscale and color images.

# **Answer:**
# A digital image is a representation of a visual scene in the form of a two-dimensional array of pixels. Each pixel contains numerical information representing the intensity or color at that particular position. Digital images are stored and processed by computers using these pixel values.

# A grayscale image contains only one channel in which each pixel represents an intensity value, generally ranging from black to white. A color image contains multiple channels, commonly three channels representing colors such as Red, Green, and Blue. Therefore, grayscale images require less storage and processing compared to color images, while color images contain more visual information.

# Q2. Explain the difference between RGB, BGR, HSV, and LAB color spaces.

# **Answer:**
# RGB represents an image using Red, Green, and Blue color components. It is commonly used for displaying color images. BGR also uses three color components, but the order is Blue, Green, and Red. OpenCV normally stores color images in BGR format.

# HSV represents colors using Hue, Saturation, and Value. Hue represents the basic color, Saturation represents the amount of color, and Value represents brightness. This color space is useful for many color-based image processing tasks.

# LAB represents an image using Lightness and two color components. It is designed to represent colors more independently from brightness and is useful in applications involving color analysis and image enhancement.

# Q3. What is the purpose of converting an image to grayscale before further processing?

# **Answer:**
# Converting an image to grayscale reduces the number of color channels from three to one. As a result, the amount of data that needs to be processed is reduced, making many image-processing operations faster and simpler. Grayscale images contain intensity information that is sufficient for several tasks such as edge detection, thresholding, segmentation, and feature extraction. Therefore, grayscale conversion is commonly used as an initial preprocessing step when color information is not required.

# Q4. Explain the concept of image complement (negative image) and mention its practical applications.

# **Answer:**
# An image complement, also called a negative image, is produced by inverting the intensity value of every pixel in an image. For an 8-bit image, the complement of a pixel can be obtained by subtracting its intensity from 255. Thus, dark pixels become bright and bright pixels become dark.

# Image negatives can be useful for enhancing certain details that are difficult to observe in the original image. They can be applied in areas such as medical image analysis, photographic processing, and visual inspection where inverted intensity information may make specific features more visible.

# Q5. Differentiate between image resizing, cropping, and scaling.

# **Answer:**
# Image resizing changes the overall dimensions of an image by specifying a new width and height. It may increase or decrease the number of pixels in the image.

# Cropping removes unwanted portions of an image and keeps only a selected region. It is useful when only a particular part of an image is required for analysis.

# Scaling changes the size of an image according to a specific scale factor. For example, an image can be scaled to 50% of its original size or enlarged to twice its original size. Therefore, resizing mainly specifies new dimensions, cropping selects a region, and scaling changes size according to a factor.

# Q6. What is a Region of Interest (ROI), and why is it important in computer vision?

# **Answer:**
# A Region of Interest, or ROI, is a specific portion of an image selected for further processing or analysis. Instead of applying an operation to the complete image, processing can be restricted to the selected region.

# ROI is important in computer vision because it reduces unnecessary computation and allows the system to focus on the area containing useful information. For example, in object detection or medical image analysis, processing only the relevant region can make the task faster and more efficient while improving the analysis of important features.

# Q7. Why is OpenCV preferred over conventional image processing libraries for computer vision applications?

# **Answer:**
# OpenCV is widely used in computer vision because it provides a large collection of optimized functions for image processing, computer vision, video processing, and related tasks. It supports operations such as image reading and writing, color conversion, geometric transformations, filtering, feature detection, and object detection.

# OpenCV also works efficiently with NumPy arrays and supports programming languages such as Python and C++. Its extensive functionality, good performance, open-source nature, and strong community support make it a suitable library for developing computer vision applications.

# Q8. Explain how image resolution and pixel intensity influence image quality.

# **Answer:**
# Image resolution refers to the number of pixels used to represent an image. Higher resolution generally provides more pixels and allows more visual details to be represented, resulting in a clearer image when viewed at an appropriate size. Lower resolution contains fewer pixels and may lose fine details.

# Pixel intensity represents the brightness level of a pixel. In a typical 8-bit grayscale image, intensity values range from 0 for black to 255 for white. The distribution of pixel intensities affects brightness, contrast, and the visibility of image features. Therefore, both resolution and intensity information have an important influence on the overall quality of an image.

# Q9. Mention five real-world applications where basic image preprocessing is an essential step.

# **Answer:**
# Basic image preprocessing is an essential part of many real-world computer vision systems. In **medical image analysis**, preprocessing improves image quality before detecting or analyzing abnormalities. In **face recognition**, images may be resized, normalized, or converted into suitable color spaces before recognition.

# Preprocessing is also important in **Optical Character Recognition (OCR)**, where noise removal and grayscale conversion can improve text recognition. In **autonomous vehicles**, camera images are processed before detecting roads, objects, and obstacles. It is also used in **industrial automation**, where images are enhanced and analyzed for quality inspection and defect detection.

# Q10. How do image preprocessing techniques improve the performance of feature extraction and deep learning models?

# **Answer:**
# Image preprocessing improves the performance of feature extraction and deep learning models by preparing the input images in a consistent and useful form. Operations such as resizing ensure that images have a common input dimension, while grayscale conversion can reduce unnecessary color information when it is not required. Other preprocessing operations can improve contrast, remove unwanted information, and focus processing on important regions.

# As a result, feature extraction algorithms can identify important patterns more effectively. Deep learning models can also benefit from consistent and properly prepared input data, which can improve training efficiency and help the model learn useful features more effectively. Thus, preprocessing acts as an important step before feature extraction, classification, and other computer vision tasks.

"""
