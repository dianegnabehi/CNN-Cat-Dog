#!/usr/bin/env python3
import os
import sys

# Set env vars early to suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Temporarily silence stderr for noisy C++ backend logs (cuDNN, cuBLAS, etc.)
original_stderr = sys.stderr
sys.stderr = open(os.devnull, "w")

import tensorflow as tf

# Restore stderr after TF initializes
sys.stderr = original_stderr


"""
Train a CNN to classify cats vs dogs using Keras.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import to_categorical

# ──────────────────────────────────────────────────────────────
# Parameters and Paths
# ──────────────────────────────────────────────────────────────
TRAIN_DIR = '/.../training_data'
TEST_DIR = '/.../testing_data'

IMG_SIZE = 150
BATCH_SIZE = 32
EPOCHS = 10

# ──────────────────────────────────────────────────────────────
# Data Preparation
# ──────────────────────────────────────────────────────────────
print("🔄 Preparing training and validation data...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# ──────────────────────────────────────────────────────────────
# Model Definition
# ──────────────────────────────────────────────────────────────
print("\n🔧 Building model...\n")

model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # Two classes: cat, dog
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────
print("\nStarting training...\n")

model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    verbose=2
)

# ──────────────────────────────────────────────────────────────
# Saving
# ──────────────────────────────────────────────────────────────
model.save("cat_dog_model.keras")
print("\nModel saved: cat_dog_model.keras")

# ──────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────
print("\nEvaluating model on test data...")

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

loss, accuracy = model.evaluate(test_generator, verbose=0)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# ──────────────────────────────────────────────────────────────
# Prediction on One Image
# ──────────────────────────────────────────────────────────────
def predict_image(image_path):
    print(f"\n🔍 Predicting image: {image_path}")
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0]
    class_indices = list(train_generator.class_indices.keys())
    label = class_indices[np.argmax(prediction)]
    emoji = "🐱" if label == "cats" else "🐶"
    print(f"🖼️ Prediction for `{os.path.basename(image_path)}`: {emoji} {label.capitalize()}")

# Predict a sample image from test set
first_class = os.listdir(TEST_DIR)[0]
example_image = os.path.join(TEST_DIR, first_class, os.listdir(os.path.join(TEST_DIR, first_class))[0])
predict_image(example_image)
