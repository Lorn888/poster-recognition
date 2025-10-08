import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# === CONFIG ===
MODEL_PATH = "poster_model/poster_model.h5"
EMBEDDINGS_PATH = "poster_embeddings.pkl"
DATASET_DIR = "dataset"
QUERY_IMAGE = "query.jpg"  # replace with your query image path

# === LOAD MODEL ===
model = load_model(MODEL_PATH)
print("✅ Model loaded")

# Initialize the model by calling it once
dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
_ = model.predict(dummy_input, verbose=0)
print("✅ Model initialized")

# === LOAD EMBEDDINGS ===
with open(EMBEDDINGS_PATH, "rb") as f:
    embeddings = pickle.load(f)
print("✅ Embeddings loaded")

# === LOAD QUERY IMAGE AND EXTRACT FEATURES ===
img = image.load_img(QUERY_IMAGE, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

query_feat = model.predict(x, verbose=0).flatten()
print("✅ Query features extracted")

# === FIND MOST SIMILAR IMAGE ===
best_score = -1
best_match = None

for label, features_list in embeddings.items():
    for f in features_list:
        score = cosine_similarity([query_feat], [f])[0][0]
        if score > best_score:
            best_score = score
            best_match = (label, score)

# === SHOW RESULTS ===
if best_match:
    label, score = best_match
    print(f"🎯 Best match: {label} (similarity: {score:.4f})")

    # Visualize the query and a random image from the best match folder
    matched_folder = os.path.join(DATASET_DIR, label)
    matched_img = os.listdir(matched_folder)[0]
    matched_img_path = os.path.join(matched_folder, matched_img)

    query_img = image.load_img(QUERY_IMAGE)
    matched_img = image.load_img(matched_img_path)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(query_img)
    plt.title("Query Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(matched_img)
    plt.title(f"Matched: {label}")
    plt.axis("off")

    plt.show()
else:
    print("❌ No match found.")