import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import os

# Set dataset path
dataset_path = r"C:\dermetology ai\archive (1)\Split_smol\train"

# Validate dataset path
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

# Allowed image file formats
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")

# Load dataset function
def load_images_and_labels():
    images = []
    labels = []
    class_names = os.listdir(dataset_path)

    for label, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_path, class_name)
        
        if not os.path.isdir(class_path):  # Skip non-folder items
            continue

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            
            # Skip non-image files
            if not img_name.lower().endswith(valid_extensions):
                print(f"Skipping non-image file: {img_name}")
                continue

            # Read and verify image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not load image {img_path}")
                continue  # Skip this image

            img = cv2.resize(img, (224, 224))
            img = img / 255.0  # Normalize
            images.append(img)
            labels.append(label)

    if not images:
        raise ValueError("No valid images found. Check dataset path and image files.")

    return np.array(images), np.array(labels), class_names

# Load images and labels
X, y, class_names = load_images_and_labels()

# Create CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(len(class_names), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=10, batch_size=16)

# Save model
model.save("dermatology_model.h5")
print("Model training complete and saved as 'dermatology_model.h5'")
