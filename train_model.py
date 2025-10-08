#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# -----------------------------
# CONFIG
# -----------------------------
dataset_dir = "dataset"  # your dataset folder
img_height = 224
img_width = 224
batch_size = 32
epochs = 10  # you can increase later

# -----------------------------
# LOAD DATA
# -----------------------------
print("🚀 Loading dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Capture class names
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"✅ Found {num_classes} classes: {class_names}")

# -----------------------------
# DATA PREFETCHING AND NORMALIZATION
# -----------------------------
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# -----------------------------
# MODEL CREATION
# -----------------------------
print("🚀 Creating model...")
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# -----------------------------
# TRAIN MODEL
# -----------------------------
print("🚀 Training model...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# -----------------------------
# SAVE MODEL
# -----------------------------
model_dir = "poster_model"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "poster_model.h5")
model.save(model_path)
print(f"✅ Model saved to {model_path}")

# -----------------------------
# OPTIONAL: Plot training curves
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()
plt.show()