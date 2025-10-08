import os
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# === CONFIG ===
MODEL_PATH = "poster_model/poster_model.h5"
DATASET_DIR = "dataset"
EMBEDDINGS_PATH = "poster_embeddings.pkl"

# === LOAD MODEL ===
model = load_model(MODEL_PATH)
print("✅ Model loaded")

# --- CALL THE MODEL ONCE TO INITIALIZE INPUTS ---
dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
_ = model.predict(dummy_input, verbose=0)
print("✅ Model initialized")

# Now we can safely access model.layers[-2]
feature_extractor = model.layers[-2]
print("✅ Feature extractor ready")

# === EXTRACT EMBEDDINGS ===
embeddings = {}

for label in sorted(os.listdir(DATASET_DIR)):
    label_path = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(label_path):
        continue

    embeddings[label] = []
    for img_file in os.listdir(label_path):
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(label_path, img_file)
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Use model.predict instead of feature_extractor.predict
        features = model.predict(x, verbose=0)
        embeddings[label].append(features.flatten())

# === SAVE EMBEDDINGS ===
with open(EMBEDDINGS_PATH, "wb") as f:
    pickle.dump(embeddings, f)

print(f"✅ Embeddings saved to {EMBEDDINGS_PATH}")