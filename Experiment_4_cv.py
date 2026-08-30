# Name: Abhay Singh Tomar
# BTech Cse 3rd year
# sec:"B"
# Cu24250144
# ROll no. = 1

# Experiment No. 3

# Experiment Name

# Frequency Domain Image Filtering using Fourier Transform

# Aim
# To implement image filtering in the frequency domain using Fourier Transform and analyze the effectiveness of frequencybased filtering techniques for noise removal, image enhancement, and feature preservation using Python and OpenCV.

import cv2
import numpy as np
import matplotlib.pyplot as plt

print("Libraries imported successfully!")

from google.colab import files

print("Upload the image:")
uploaded = files.upload()

filename = list(uploaded.keys())[0]

print("Image uploaded:", filename)

image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise ValueError("Image could not be loaded.")

print("Image loaded successfully!")
print("Image Shape:", image.shape)

plt.figure(figsize=(8, 6))

plt.imshow(image, cmap='gray')
plt.title("Original Grayscale Image")
plt.axis("off")

plt.show()

# Convert image to floating point
image_float = np.float32(image)

# Apply Discrete Fourier Transform
dft = cv2.dft(image_float, flags=cv2.DFT_COMPLEX_OUTPUT)

print("DFT calculated successfully!")
print("DFT Shape:", dft.shape)

dft_shift = np.fft.fftshift(dft)

print("Zero-frequency component shifted to the center.")

magnitude_spectrum = cv2.magnitude(
    dft_shift[:, :, 0],
    dft_shift[:, :, 1]
)

# Log transformation for better visualization
magnitude_spectrum = np.log(magnitude_spectrum + 1)

# Normalize to 0-255
magnitude_spectrum = cv2.normalize(
    magnitude_spectrum,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

magnitude_spectrum = np.uint8(magnitude_spectrum)

plt.figure(figsize=(8, 6))

plt.imshow(magnitude_spectrum, cmap='gray')
plt.title("Magnitude Spectrum")
plt.axis("off")

plt.show()

rows, cols = image.shape

crow, ccol = rows // 2, cols // 2

# Create a mask
low_pass_mask = np.zeros((rows, cols, 2), np.float32)

# Radius of low-pass filter
radius = 50

# Create circular low-pass region
y, x = np.ogrid[:rows, :cols]

distance = np.sqrt((x - ccol)**2 + (y - crow)**2)

low_pass_mask[distance <= radius] = 1

print("Low-Pass Filter created successfully!")

# Apply Low-Pass Filter
low_pass_dft = dft_shift * low_pass_mask

print("Low-Pass Filter applied successfully!")

# Shift back
low_pass_ishift = np.fft.ifftshift(low_pass_dft)

# Inverse DFT
low_pass_image = cv2.idft(low_pass_ishift)

# Calculate magnitude
low_pass_image = cv2.magnitude(
    low_pass_image[:, :, 0],
    low_pass_image[:, :, 1]
)

# Normalize
low_pass_image = cv2.normalize(
    low_pass_image,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

low_pass_image = np.uint8(low_pass_image)

plt.figure(figsize=(8, 6))

plt.imshow(low_pass_image, cmap='gray')
plt.title("Low-Pass Filtered Image")
plt.axis("off")

plt.show()

# Create High-Pass Filter
high_pass_mask = np.ones((rows, cols, 2), np.float32)

# Remove low-frequency center
high_pass_mask[distance <= radius] = 0

print("High-Pass Filter created successfully!")

# Apply High-Pass Filter
high_pass_dft = dft_shift * high_pass_mask

print("High-Pass Filter applied successfully!")

# Shift back
high_pass_ishift = np.fft.ifftshift(high_pass_dft)

# Inverse DFT
high_pass_image = cv2.idft(high_pass_ishift)

# Calculate magnitude
high_pass_image = cv2.magnitude(
    high_pass_image[:, :, 0],
    high_pass_image[:, :, 1]
)

# Normalize
high_pass_image = cv2.normalize(
    high_pass_image,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

high_pass_image = np.uint8(high_pass_image)

plt.figure(figsize=(8, 6))

plt.imshow(high_pass_image, cmap='gray')
plt.title("High-Pass Filtered Image")
plt.axis("off")

plt.show()

plt.figure(figsize=(16, 10))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title("Magnitude Spectrum")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(low_pass_image, cmap='gray')
plt.title("Low-Pass Filtered Image")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(high_pass_image, cmap='gray')
plt.title("High-Pass Filtered Image")
plt.axis("off")

plt.tight_layout()
plt.show()

print("=" * 60)
print("OBSERVATIONS")
print("=" * 60)

print("""
1. Original Image:
   The original grayscale image contains both low-frequency
   and high-frequency information.

2. Magnitude Spectrum:
   The Fourier Transform represents the image in terms of
   frequency components. Low-frequency components are located
   near the center after shifting.

3. Low-Pass Filter:
   The Low-Pass Filter removes high-frequency components.
   The resulting image becomes smoother and fine details
   are reduced.

4. High-Pass Filter:
   The High-Pass Filter removes low-frequency components.
   It emphasizes edges, boundaries, and fine details.

5. Comparison:
   Low-Pass filtering is useful for smoothing and noise
   reduction, while High-Pass filtering is useful for
   edge enhancement and feature extraction.
""")

print("=" * 60)

print("=" * 60)
print("RESULT")
print("=" * 60)

print("""
Frequency domain image filtering was successfully implemented
using the Discrete Fourier Transform (DFT) and Inverse Fourier
Transform (IDFT).

The Low-Pass Filter reduced high-frequency components and
produced a smoother image, while the High-Pass Filter enhanced
edges and fine details.
""")

# ============================================================
# RESULT
# ============================================================

# Frequency domain image filtering was successfully implemented
# using the Discrete Fourier Transform (DFT) and Inverse Fourier
# Transform (IDFT).

# The Low-Pass Filter reduced high-frequency components and
# produced a smoother image, while the High-Pass Filter enhanced
# edges and fine details.

# EXPERIMENT 3 - ANSWERS TO QUESTIONS

# Q1. What is the Fourier Transform? Why is it important in digital image processing?
# Answer: The Fourier Transform is a mathematical technique used to convert an image from the spatial domain into the frequency domain. In the frequency domain, an image is represented in terms of different frequency components. Low-frequency components generally represent slowly varying regions such as smooth backgrounds, while high-frequency components represent edges, boundaries, and fine details. Fourier Transform is important in digital image processing because it allows an image to be analyzed and modified based on its frequency information. It is widely used for image filtering, noise removal, image enhancement, restoration, and feature analysis.

# Q2. Differentiate between the spatial domain and the frequency domain.
# Answer: The spatial domain represents an image directly using its pixel intensity values. Image processing operations in the spatial domain are performed directly on individual pixels or their neighboring pixels. For example, averaging and median filtering are spatial-domain operations.

# The frequency domain represents an image using its frequency components obtained through a transformation such as the Fourier Transform. Instead of directly modifying pixels, filtering is performed by modifying selected frequency components. Spatial-domain processing is generally simple for local operations, while frequency-domain processing is useful when selective control over image frequencies is required.

# Q3. What is the significance of the Discrete Fourier Transform (DFT) in image processing?
# Answer: The Discrete Fourier Transform (DFT) is used to convert a digital image from the spatial domain into the frequency domain. It decomposes the image into different frequency components, making it possible to study the frequency characteristics of the image. The DFT is significant because it provides a basis for frequency-domain filtering and image analysis. By modifying specific frequency components, unwanted noise can be reduced and important features can be enhanced. DFT is also useful in applications such as image restoration, image compression, enhancement, and feature extraction.

# Q4. Explain the purpose of shifting the zero-frequency component to the center of the frequency spectrum.
# Answer: After applying the Fourier Transform, the zero-frequency component, which represents the lowest frequency, is normally located at the corner of the frequency spectrum. For easier visualization and analysis, this component is shifted to the center of the spectrum using a frequency-shifting operation. After shifting, the low-frequency components are concentrated around the center, while the high-frequency components appear farther away from the center. This arrangement makes the frequency spectrum more intuitive and makes it easier to design and apply Low-Pass and High-Pass filters.

# Q5. Compare Low-Pass Frequency Filters and High-Pass Frequency Filters with suitable applications.
# Answer: A Low-Pass Frequency Filter allows low-frequency components to pass while reducing or removing high-frequency components. Since high-frequency components are generally associated with rapid intensity changes and fine details, Low-Pass filtering produces a smoother image. It is commonly used for image smoothing and noise reduction.

# A High-Pass Frequency Filter works in the opposite manner. It suppresses low-frequency components and allows high-frequency components to pass. As a result, edges, boundaries, and fine details become more prominent. High-Pass filtering is therefore useful for edge enhancement, sharpening, and feature extraction in images.

# Q6. What is the role of the Inverse Fourier Transform (IDFT) in image reconstruction?
# Answer: The Inverse Fourier Transform, or IDFT, is used to convert an image from the frequency domain back into the spatial domain. After applying a frequency-domain filter, the image exists as modified frequency components rather than directly viewable pixel values. The IDFT reconstructs the filtered image from these modified components. This allows the final result to be displayed and analyzed as a normal image. Therefore, IDFT is an essential step in frequency-domain filtering because it provides the reconstructed image after the required frequency components have been removed, preserved, or enhanced.

# Q7. Why is frequency domain filtering preferred for certain image enhancement tasks?
# Answer: Frequency-domain filtering is preferred for certain image enhancement tasks because it provides direct control over different frequency components of an image. Different types of information can be selectively suppressed or enhanced according to their frequency characteristics. For example, Low-Pass filtering can reduce high-frequency noise and smooth an image, while High-Pass filtering can emphasize edges and fine details. Frequency-domain filtering is also useful for removing periodic noise and performing image restoration. Therefore, it is particularly effective when the required enhancement depends on specific frequency components.

# Q8. Mention four real-world applications where Fourier Transform is used in computer vision and image analysis.
# Answer: Fourier Transform has several important applications in computer vision and image analysis. It is used in medical image enhancement to improve the quality and clarity of medical images. It is also used in satellite image analysis for processing and analyzing remote-sensing images. Another important application is image restoration, where frequency-based techniques can help reduce unwanted noise and degradation. Fourier Transform is also useful in biometric systems, where image processing techniques can be applied to information such as fingerprints and other biometric patterns. These applications demonstrate the usefulness of frequency-domain analysis in practical image-processing systems.

# Q9. Compare frequency domain filtering with spatial domain filtering based on computational efficiency and practical applications.
# Answer: Spatial-domain filtering directly modifies pixel values using a filtering kernel or the neighboring pixels. It is relatively simple and intuitive and is commonly used for operations such as smoothing, sharpening, and edge detection. However, for some large filtering operations, directly processing a large kernel over all pixels can become computationally expensive.

# Frequency-domain filtering first transforms the image into the frequency domain and then modifies the required frequency components. It can be more efficient for certain large-scale filtering operations and is particularly useful for periodic noise removal, image restoration, and frequency-based enhancement. The choice between the two methods depends on the size of the operation and the requirements of the application.

# Q10. How does frequency domain filtering improve the performance of image restoration and feature extraction techniques?
# Answer: Frequency-domain filtering improves image restoration and feature extraction by allowing unwanted and useful frequency components to be handled separately. Noise and unwanted information can often be reduced by suppressing specific frequency components, while important high-frequency information such as edges and fine details can be enhanced when required. This produces a cleaner and more informative image. As a result, subsequent image-processing tasks such as feature extraction, object analysis, and image restoration can work with improved image information. Therefore, frequency-domain filtering can help improve both the quality of the reconstructed image and the visibility of important features.
