# Name: Abhay Singh Tomar
# BTech Cse 3rd year
# sec:"B"
# Cu24250144
# ROll no. = 1

# Experiment No. 9 

# Experiment name
# Object Detection and Recognition using Deep Learning Models on Standard Image Datasets

# Aim
# To implement a deep learning-based object detection and recognition system using pre-trained models and evaluate its performance on standard image datasets for identifying and localizing multiple objects.

# code
!pip install -q ultralytics opencv-python matplotlib

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from google.colab import files
from ultralytics import YOLO

print("Libraries imported successfully!")

uploaded = files.upload()

image_path = list(uploaded.keys())[0]

print("Uploaded Image:", image_path)

image = cv2.imread(image_path)

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 7))
plt.imshow(image_rgb)
plt.axis("off")
plt.title("Original Image")
plt.show()

model = YOLO("yolov8n.pt")

print("YOLOv8 model loaded successfully!")

results = model(image_path)

print("Object detection completed!")

result_image = results[0].plot()

result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 8))
plt.imshow(result_rgb)
plt.axis("off")
plt.title("Object Detection Result")
plt.show()

result = results[0]

if result.boxes is not None and len(result.boxes) > 0:

    print("Detected Objects:\n")

    for i, box in enumerate(result.boxes):

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = model.names[class_id]

        print(
            f"{i+1}. Object: {class_name} | "
            f"Confidence: {confidence:.2f}"
        )

else:
    print("No objects detected.")

if result.boxes is not None and len(result.boxes) > 0:

    print("Bounding Box Coordinates:\n")

    for i, box in enumerate(result.boxes):

        coordinates = box.xyxy[0].cpu().numpy()

        x1, y1, x2, y2 = coordinates

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        print(f"Object {i+1}: {model.names[class_id]}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Bounding Box: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
        print("-" * 50)

  confidence_threshold = 0.40

filtered_results = model(
    image_path,
    conf=confidence_threshold
)

filtered_image = filtered_results[0].plot()

filtered_rgb = cv2.cvtColor(
    filtered_image,
    cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(12, 8))
plt.imshow(filtered_rgb)
plt.axis("off")
plt.title("Detection with Confidence Threshold = 0.40")
plt.show()

output_path = "object_detection_result.jpg"

cv2.imwrite(
    output_path,
    filtered_results[0].plot()
)

print("Result saved as:", output_path)

# QUESTIONS — ANSWERS
# Q1. Differentiate between image classification, object detection, and image segmentation.
# Answer: Image classification assigns a class or category to the complete image. Object detection identifies one or more objects in an image and locates them using bounding boxes along with class labels. Image segmentation goes one step further by classifying individual pixels of an image to identify the exact region or shape of objects. Thus, classification tells what is present, detection tells what and where, while segmentation tells what and which pixels belong to it.

# Q2. Explain the working principle of the YOLO (You Only Look Once) object detection algorithm.
# Answer: YOLO stands for You Only Look Once. It is a deep learning-based object detection algorithm that processes the complete image in a single pass through a neural network. The model predicts the locations of objects, their class labels, and confidence scores. Since the image is processed in one pass, YOLO provides fast object detection and is suitable for real-time applications such as surveillance and autonomous systems. The experiment specifically includes YOLOv8 as one of the pre-trained models that can be used for object detection.

# Q3. Compare YOLO, SSD, and Faster R-CNN based on speed, accuracy, and practical applications.
# Answer: YOLO, SSD, and Faster R-CNN are deep learning-based object detection models, but they differ in speed and accuracy.

# YOLO: It is designed for fast object detection and is suitable for real-time applications.
# SSD (Single Shot Detector): It is also a single-stage detector that provides a good balance between detection speed and accuracy.
# Faster R-CNN: It generally focuses more on detection accuracy but can be slower compared with single-stage detectors.
# YOLO and SSD are commonly suitable for real-time applications, while Faster R-CNN can be used when detailed and accurate detection is more important. The experiment asks for comparison based on detection accuracy and inference speed when multiple models are used.

# Q4. What is a bounding box? Why is it important in object detection?
# Answer: A bounding box is a rectangular box drawn around a detected object in an image. It represents the location of the object using coordinates such as the top-left and bottom-right points. Bounding boxes are important because they allow the system to identify not only what object is present, but also where the object is located in the image. Object detection systems use bounding boxes together with class labels to localize objects.

# Q5. Explain the significance of confidence score and Intersection over Union (IoU) in object detection.
# Answer: A confidence score represents the model's confidence in its prediction that a detected region contains a particular object. A higher confidence score generally indicates a stronger prediction.

# Intersection over Union (IoU) measures the amount of overlap between the predicted bounding box and the actual ground-truth bounding box. It is used to evaluate how accurately an object has been localized. Therefore, confidence score helps evaluate the certainty of a prediction, while IoU helps evaluate the accuracy of the predicted object's location.

# Q6. What are the advantages of using pre-trained deep learning models over traditional image processing techniques?
# Answer: Pre-trained deep learning models have already learned useful visual features from large standard datasets. They can therefore be used for object detection without training a complete model from the beginning. They can reduce development and training time and can detect complex objects more effectively than many traditional hand-crafted image processing techniques. Models such as YOLOv8, SSD, and Faster R-CNN can be used as pre-trained object detection models.

# Q7. Why are datasets such as MS COCO and Pascal VOC widely used for object detection research?
# Answer: MS COCO and Pascal VOC are standard datasets containing images and object annotations used for computer vision research. They provide suitable data for training and evaluating object detection systems. These datasets contain multiple object categories and provide information required for object localization and recognition. Therefore, they are widely used as standard datasets and benchmarks for object detection research.

# Q8. Mention five real-world applications of object detection in computer vision.
# Answer: Five real-world applications of object detection are:

# Autonomous vehicles – detecting vehicles, pedestrians, traffic signs, and other objects.
# Surveillance systems – detecting and monitoring objects or people in security environments.
# Healthcare – assisting in the detection and localization of relevant objects in medical images.
# Smart cities – monitoring traffic and objects in urban environments.
# Retail analytics – detecting products and analyzing objects in retail environments.
# The experiment specifically mentions autonomous vehicles, surveillance systems, healthcare, smart cities, retail analytics, and industrial inspection as application areas.

# Q9. What challenges are commonly encountered while detecting objects in complex real-world environments?
# Answer: Object detection can face several challenges in complex real-world environments. These include different object sizes, poor lighting conditions, occlusion, overlapping objects, complex backgrounds, different object orientations, and low-quality images. These factors can make it difficult for a model to correctly identify and localize objects and may result in false detections or missed objects. The experiment also asks students to analyze correctly detected objects, false detections, and missed objects.

# Q10. How can object detection systems be further improved using recent advancements in deep learning and computer vision?
# Answer: Object detection systems can be improved by using newer and more efficient deep learning architectures, better training datasets, improved data augmentation techniques, and advanced feature extraction methods. Models can also be optimized to improve detection accuracy and inference speed. Using better hardware and optimized models can help in real-time applications. Comparing different models based on accuracy and inference speed can also help in selecting an appropriate model for a particular application.
