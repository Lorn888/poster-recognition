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

app = Flask(__name__)
CORS(app, origins=["https://lorn888.github.io"])

# Load model and embeddings once
MODEL_PATH = "poster_model/poster_model.h5"
EMBEDDINGS_PATH = "poster_embeddings.pkl"

print("✅ Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded")

with open(EMBEDDINGS_PATH, "rb") as f:
    embeddings = pickle.load(f)
print("✅ Embeddings loaded")

@app.route("/")
def home():
    return jsonify({"message": "Poster Recognition API is running 🚀"})

@app.route("/search", methods=["POST"])
def search_poster():
    try:
        # Expecting an uploaded image file
        file = request.files["file"]
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img = img.resize((224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        query_feat = model.predict(x)[0]

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)