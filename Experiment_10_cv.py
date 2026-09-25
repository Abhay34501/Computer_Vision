# Name: Abhay Singh Tomar
# BTech Cse 3rd year
# sec:"B"
# Cu24250144
# ROll no. = 1

# Experiment No. 10

# Experiment name
# Character, Digit, or Face Classification using Convolutional Neural Networks (CNNs)

# Aim
# To design, train, and evaluate a Convolutional Neural Network (CNN) for character, digit, or face classification using a standard image dataset and analyze its performance using appropriate evaluation metrics.

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

print("TensorFlow Version:", tf.__version__)
print("Libraries imported successfully!")

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Training images:", x_train.shape)
print("Training labels:", y_train.shape)
print("Testing images:", x_test.shape)
print("Testing labels:", y_test.shape)

plt.figure(figsize=(12, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title("Label: " + str(y_train[i]))
    plt.axis("off")

plt.tight_layout()
plt.show()

# Normalize pixel values from 0-255 to 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

print("Training shape:", x_train.shape)
print("Testing shape:", x_test.shape)

# Use last 10,000 training samples for validation
x_val = x_train[-10000:]
y_val = y_train[-10000:]

x_train_new = x_train[:-10000]
y_train_new = y_train[:-10000]

print("Training data:", x_train_new.shape)
print("Validation data:", x_val.shape)
print("Testing data:", x_test.shape)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks(range(10))
plt.yticks(range(10))

plt.show()

print(
    classification_report(
        y_test,
        y_pred
    )
)

index = 100

sample_image = x_test[index]

prediction = model.predict(
    np.expand_dims(sample_image, axis=0),
    verbose=0
)

predicted_digit = np.argmax(prediction)

actual_digit = y_test[index]

plt.figure(figsize=(4, 4))

plt.imshow(
    sample_image.squeeze(),
    cmap="gray"
)

plt.title(
    f"Predicted: {predicted_digit} | Actual: {actual_digit}"
)

plt.axis("off")
plt.show()

print("Predicted Digit:", predicted_digit)
print("Actual Digit:", actual_digit)

plt.figure(figsize=(12, 6))

for i in range(15):

    prediction = model.predict(
        np.expand_dims(x_test[i], axis=0),
        verbose=0
    )

    predicted_digit = np.argmax(prediction)

    plt.subplot(3, 5, i + 1)
    plt.imshow(
        x_test[i].squeeze(),
        cmap="gray"
    )

    plt.title(
        f"P: {predicted_digit} | A: {y_test[i]}"
    )

    plt.axis("off")

plt.tight_layout()
plt.show()

# QUESTIONS — ANSWERS
# Q1. What is image classification? How does it differ from object detection and image segmentation?

# Answer: Image classification assigns a class to an image. Object detection identifies and locates objects using bounding boxes, while image segmentation identifies the exact pixels belonging to objects.

# Q2. Explain the architecture and working principle of a Convolutional Neural Network (CNN).

# Answer: A CNN uses convolution, pooling, and fully connected layers to extract image features and classify them. It automatically learns useful features from images.

# Q3. What is the role of Convolutional Layers, Pooling Layers, and Fully Connected Layers in a CNN?

# Answer: Convolutional layers extract features, pooling layers reduce image dimensions, and fully connected layers perform the final classification.

# Q4. Why is image normalization performed before training a deep learning model?

# Answer: Normalization scales pixel values to a smaller range, usually 0 to 1. It makes training faster and more stable.

# Q5. Explain the purpose of activation functions such as ReLU and Softmax in CNNs.

# Answer: ReLU adds non-linearity and helps the network learn complex features. Softmax converts the final outputs into class probabilities.

# Q6. What is a Confusion Matrix? How is it used to evaluate classification performance?

# Answer: A confusion matrix compares actual and predicted classes. It shows correctly and incorrectly classified samples for each class.

# Q7. Differentiate between Accuracy, Precision, Recall, and F1-Score with suitable examples.

# Answer: Accuracy measures overall correct predictions. Precision measures correct positive predictions. Recall measures correctly identified actual positives. F1-Score combines Precision and Recall.

# Q8. Compare traditional feature-based image classification methods (e.g., SIFT/HOG + SVM) with CNN-based classification.

# Answer: Traditional methods require manually extracted features such as SIFT or HOG. CNNs automatically learn important features directly from images.

# Q9. Mention five real-world applications of CNN-based image classification.

# Answer:

# Handwritten digit recognition
# Facial recognition
# Biometric authentication
# Medical diagnosis
# Intelligent document processing
# Q10. How can techniques such as Data Augmentation, Transfer Learning, and Hyperparameter Tuning improve the performance of deep learning-based image classification models?

# Answer: Data augmentation creates more varied training images, transfer learning reuses knowledge from existing models, and hyperparameter tuning finds better training settings. These techniques can improve model performance.
