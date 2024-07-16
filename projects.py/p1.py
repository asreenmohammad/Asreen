import numpy as np
import os
import cv2

# Define paths to image folders
train_folder = '/home/asreen-mohammad/Desktop/project/train'
validation_folder = '/home/asreen-mohammad/Desktop/project/validation'
test_folder = '/home/asreen-mohammad/Desktop/project/test'

# Function to load images from a folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
            if img is not None:
                img = cv2.resize(img, (28, 28))  # Resize image to 28x28 (adjust as needed)
                images.append(img.flatten())  # Flatten the image and add to list
    return images

# Load training images and labels
train_images = load_images_from_folder(train_folder)
train_outputs = np.zeros(len(train_images))  # Assuming binary classification (0 or 1)

# Load validation images and labels
validation_images = load_images_from_folder(validation_folder)
validation_outputs = np.zeros(len(validation_images))  # Assuming binary classification (0 or 1)

# Load test images and labels
test_images = load_images_from_folder(test_folder)
test_outputs = np.zeros(len(test_images))  # Assuming binary classification (0 or 1)

# Initialize weights (+1 for bias)
weights = np.zeros(train_images[0].shape[0] + 1)

# Training the perceptron
learning_rate = 0.1
epochs = 10

for epoch in range(epochs):
    # Training
    train_results = []
    for inputs, output in zip(train_images, train_outputs):
        inputs_with_bias = np.insert(inputs, 0, 1)  # Add bias term
        prediction = np.dot(inputs_with_bias, weights)
        prediction = 1 if prediction > 0 else 0
        weights += learning_rate * (output - prediction) * inputs_with_bias
        train_results.append(prediction == output)
    print(f"Epoch {epoch + 1}: Training accuracy:", np.mean(train_results) if train_results else 0)

    # Validation
    validation_results = []
    for inputs, output in zip(validation_images, validation_outputs):
        inputs_with_bias = np.insert(inputs, 0, 1)  # Add bias term
        prediction = np.dot(inputs_with_bias, weights)
        prediction = 1 if prediction > 0 else 0
        validation_results.append(prediction == output)
    print(f"Epoch {epoch + 1}: Validation accuracy:", np.mean(validation_results) if validation_results else 0)

# Testing
test_results = []
for inputs, output in zip(test_images, test_outputs):
    inputs_with_bias = np.insert(inputs, 0, 1)  # Add bias term
    prediction = np.dot(inputs_with_bias, weights)
    prediction = 1 if prediction > 0 else 0
    test_results.append(prediction == output)
print("Test accuracy:", np.mean(test_results) if test_results else 0)




