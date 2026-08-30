# Name: Abhay Singh Tomar
# BTech Cse, 3rd year
# sec: "B"
# Cu24250144
# ROll no. = 1

# Experiment No. 2

# Experiment Name

# Contrast Enhancement and Histogram-Based Image Processing using Python and OpenCV

# Image contrast plays a significant role in the performance of computer vision systems. Poor illumination, sensor limitations, or environmental conditions often result in low-contrast images, making feature extraction and object recognition difficult. Histogram-based enhancement techniques improve the distribution of pixel intensities, thereby enhancing image visibility and facilitating subsequent image analysis tasks.

# Import required libraries

import cv2
import numpy as np
import matplotlib.pyplot as plt

print("Libraries imported successfully!")

from google.colab import files

print("Upload an image:")
uploaded = files.upload()

filename = list(uploaded.keys())[0]

# Read image
img = cv2.imread(filename)

if img is None:
    raise ValueError("Image could not be loaded.")

print("Image uploaded successfully!")
print("File:", filename)

# Convert color image to grayscale

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("Image converted to grayscale!")
print("Image Shape:", gray.shape)

plt.figure(figsize=(8, 6))

plt.imshow(gray, cmap="gray")
plt.title("Original Grayscale Image")
plt.axis("off")

plt.show()

# Calculate histogram

hist_original = cv2.calcHist([gray], [0], None, [256], [0, 256])

plt.figure(figsize=(10, 5))

plt.plot(hist_original)
plt.title("Histogram of Original Image")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.xlim([0, 256])

plt.show()

# Contrast Stretching

min_val = np.min(gray)
max_val = np.max(gray)

contrast_stretched = ((gray - min_val) / (max_val - min_val)) * 255
contrast_stretched = np.uint8(contrast_stretched)

print("Contrast stretching completed!")
print("Original minimum intensity:", min_val)
print("Original maximum intensity:", max_val)
print("New minimum intensity:", np.min(contrast_stretched))
print("New maximum intensity:", np.max(contrast_stretched))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(gray, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(contrast_stretched, cmap="gray")
plt.title("Contrast Stretched Image")
plt.axis("off")

plt.tight_layout()
plt.show()

# Histogram Equalization

hist_equalized = cv2.equalizeHist(gray)

print("Histogram Equalization completed!")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(gray, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(hist_equalized, cmap="gray")
plt.title("Histogram Equalized Image")
plt.axis("off")

plt.tight_layout()
plt.show()

# Create CLAHE object

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

# Apply CLAHE
clahe_image = clahe.apply(gray)

print("CLAHE applied successfully!")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(gray, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(hist_equalized, cmap="gray")
plt.title("Histogram Equalization")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(clahe_image, cmap="gray")
plt.title("CLAHE")
plt.axis("off")

plt.tight_layout()
plt.show()

# Histogram of Contrast Stretched Image

hist_stretched = cv2.calcHist(
    [contrast_stretched],
    [0],
    None,
    [256],
    [0, 256]
)

plt.figure(figsize=(10, 5))

plt.plot(hist_stretched)
plt.title("Histogram of Contrast Stretched Image")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.xlim([0, 256])

plt.show()

# Histogram of Equalized Image

hist_equalized_plot = cv2.calcHist(
    [hist_equalized],
    [0],
    None,
    [256],
    [0, 256]
)

plt.figure(figsize=(10, 5))

plt.plot(hist_equalized_plot)
plt.title("Histogram of Histogram Equalized Image")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.xlim([0, 256])

plt.show()

# Histogram of CLAHE Image

hist_clahe = cv2.calcHist(
    [clahe_image],
    [0],
    None,
    [256],
    [0, 256]
)

plt.figure(figsize=(10, 5))

plt.plot(hist_clahe)
plt.title("Histogram of CLAHE Enhanced Image")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.xlim([0, 256])

plt.show()

plt.figure(figsize=(12, 6))

plt.plot(hist_original, label="Original")
plt.plot(hist_equalized_plot, label="Histogram Equalized")
plt.plot(hist_clahe, label="CLAHE")

plt.title("Comparison of Image Histograms")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.xlim([0, 256])
plt.legend()

plt.show()

plt.figure(figsize=(16, 10))

images = [
    (gray, "Original Image"),
    (contrast_stretched, "Contrast Stretching"),
    (hist_equalized, "Histogram Equalization"),
    (clahe_image, "CLAHE")
]

for i, (image, title) in enumerate(images):
    plt.subplot(2, 2, i + 1)
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()

print("=" * 65)
print("OBSERVATIONS")
print("=" * 65)

print("""
1. Original Image:
   The original image may contain a limited range of pixel
   intensities, resulting in low or uneven contrast.

2. Contrast Stretching:
   Contrast stretching expands the intensity range of the image.
   It improves the difference between dark and bright regions.

3. Histogram Equalization:
   Histogram Equalization redistributes pixel intensities and
   improves the overall contrast of the image.

4. CLAHE:
   CLAHE performs adaptive histogram equalization on local
   regions and limits excessive contrast enhancement.

5. Histogram Comparison:
   The enhanced images show a more distributed intensity
   range compared with the original image.

6. Visual Quality:
   Contrast enhancement makes important structures and
   features more visible, which can help later image analysis.
""")

print("=" * 65)

print("=" * 65)
print("RESULT")
print("=" * 65)

print("""
Contrast enhancement and histogram-based image processing
techniques were successfully implemented using Python and
OpenCV.

Contrast Stretching, Histogram Equalization, and CLAHE were
applied successfully. The enhanced images showed improved
contrast and better visibility of image details compared with
the original image.
""")

# =================================================================
# RESULT
# =================================================================

# Contrast enhancement and histogram-based image processing
# techniques were successfully implemented using Python and
# OpenCV.

# Contrast Stretching, Histogram Equalization, and CLAHE were
# applied successfully. The enhanced images showed improved
# contrast and better visibility of image details compared with
# the original image.

# Questions and Answers
# Q1. What is image contrast, and why is it important in image processing?
# Answer: Image contrast refers to the difference in intensity between different regions or objects in an image. An image with good contrast has clearly distinguishable dark and bright regions, while a low-contrast image may appear dull and its important details may not be clearly visible.

# Contrast is important because it directly affects the visibility of image features. Improving contrast can make edges, objects, and structures easier to identify. It is therefore an important preprocessing step in computer vision applications such as feature extraction, segmentation, object detection, and image recognition.

# Q2. Explain the concept of an image histogram. What information does it provide?
# Answer: An image histogram is a graphical representation of the distribution of pixel intensity values in an image. For a grayscale image, the horizontal axis generally represents intensity values from 0 to 255, while the vertical axis represents the number of pixels having each intensity value.

# The histogram provides information about the brightness and contrast characteristics of an image. If the intensity values are concentrated in a small range, the image may have low contrast. A histogram spread over a wider intensity range generally indicates better utilization of the available intensity levels.

# Q3. Differentiate between Histogram Stretching and Histogram Equalization.
# Answer: Histogram Stretching, also called contrast stretching, improves contrast by expanding the existing range of pixel intensities to a wider range. It generally maintains the relative ordering of the intensity values while increasing the dynamic range.

# Histogram Equalization works by redistributing pixel intensity values based on the cumulative distribution of the image histogram. Its purpose is to produce a more evenly distributed intensity range and improve overall contrast.

# Therefore, contrast stretching mainly expands the intensity range, whereas histogram equalization redistributes intensities to enhance contrast.

# Q4. What is Contrast Limited Adaptive Histogram Equalization (CLAHE)? How does it differ from standard Histogram Equalization?
# Answer: Contrast Limited Adaptive Histogram Equalization, or CLAHE, is an enhancement technique that performs histogram equalization on small local regions of an image rather than applying one histogram operation to the entire image. It also limits the amplification of contrast to prevent excessive enhancement of noise.

# Standard Histogram Equalization is a global technique because it uses the histogram of the complete image. CLAHE is adaptive and considers local regions, making it more suitable for images with varying illumination or areas having different contrast levels.

# Q5. Why is histogram equalization commonly applied before feature extraction and image segmentation?
# Answer: Histogram equalization is commonly applied before feature extraction and image segmentation because it can improve the contrast and visibility of important image features. When an image has poor contrast, edges, boundaries, and object regions may be difficult to distinguish.

# By redistributing pixel intensities, histogram equalization can make these structures more visible. This provides feature extraction and segmentation algorithms with more distinguishable image information and can therefore improve the quality of subsequent image analysis.

# Q6. Mention three real-world applications where histogram equalization is widely used.
# Answer: Histogram equalization can be used in several real-world image-processing applications. One important application is medical imaging, where contrast enhancement can help make structures in medical images more visible. It can also be used in satellite and remote-sensing images to improve the visibility of geographical features.

# Another application is low-light photography, where contrast enhancement can make objects and details more visible. These applications demonstrate the importance of histogram-based enhancement in improving image quality for further analysis. The experiment material specifically mentions medical images, satellite images, and low-light photographs as application areas.

# Q7. What are the limitations of global histogram equalization?
# Answer: Global histogram equalization considers the histogram of the entire image, so it may not perform equally well in all regions. If an image contains areas with different illumination levels, improving the global contrast may cause some regions to become excessively bright or dark.

# It can also enhance unwanted noise in certain images, particularly when the original image contains noisy regions. Another limitation is that global equalization does not provide direct local control over contrast. Techniques such as CLAHE can address some of these limitations by performing adaptive enhancement on local regions.

# Q8. How does contrast enhancement improve the performance of object detection and recognition systems?
# Answer: Contrast enhancement improves object detection and recognition systems by making important visual features more distinguishable. Better contrast can improve the visibility of object boundaries, edges, textures, and other patterns that are useful for detection and recognition.

# When preprocessing produces a clearer image, feature extraction algorithms can obtain more useful information from the input. This can help subsequent computer vision processes identify objects more effectively, particularly when the original image suffers from poor illumination or low contrast.

# Q9. Compare histogram-based enhancement techniques with brightness adjustment methods.
# Answer: Brightness adjustment mainly changes the overall intensity of an image by making it brighter or darker. It does not necessarily improve the difference between different intensity regions. Therefore, simply increasing brightness may not make details more visible in a low-contrast image.

# Histogram-based enhancement techniques analyze the distribution of pixel intensities and modify that distribution to improve contrast. Contrast stretching expands the intensity range, while histogram equalization redistributes intensity values. CLAHE further performs local contrast enhancement. Thus, histogram-based techniques provide more control over contrast than simple brightness adjustment.

# Q10. Why is CLAHE preferred for medical imaging and low-light image enhancement?
# Answer: CLAHE is useful for medical imaging and low-light images because these images often contain regions with different illumination and contrast levels. A global enhancement method may improve one region while producing excessive enhancement in another region.

# CLAHE works on local regions and improves contrast according to the characteristics of each region. Its contrast-limiting mechanism also helps prevent excessive amplification of noise. Therefore, CLAHE can provide better local feature visibility and more controlled enhancement for images with uneven illumination, which is important in medical and low-light image analysis.

