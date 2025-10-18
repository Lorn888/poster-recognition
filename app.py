from flask_cors import CORS
from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import io
from PIL import Image

# -----------------------
# App Setup
# -----------------------
app = Flask(__name__)
CORS(app, origins=["*"])  # Allow all origins or specify yours

# -----------------------
# Load Model and Embeddings
# -----------------------
MODEL_PATH = "poster_model_fixed.h5"
EMBEDDINGS_PATH = "poster_embeddings.pkl"

print("✅ Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded")

with open(EMBEDDINGS_PATH, "rb") as f:
    embeddings = pickle.load(f)
print("✅ Embeddings loaded")

# -----------------------
# Routes
# -----------------------
@app.route("/")
def home():
    return jsonify({"message": "Poster Recognition API is running 🚀"})


@app.route("/search", methods=["POST"])
def search_poster():
    try:
        # Check file uploaded
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img = img.resize((224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Get feature vector from model
        query_feat = model.predict(x)[0]

        # Compare with embeddings
        best_match, best_score = None, -1
        for label, feats in embeddings.items():
            for f in feats:
                score = cosine_similarity([query_feat], [f])[0][0]
                if score > best_score:
                    best_score, best_match = score, label

        return jsonify({
            "best_match": best_match,
            "similarity": round(float(best_score), 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------
# Run App
# -----------------------
if __name__ == "__main__":
    # Debug off for production
    app.run(host="0.0.0.0", port=10000)