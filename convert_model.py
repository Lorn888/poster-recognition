from tensorflow.keras.models import load_model
import numpy as np

MODEL_PATH = "poster_model_fixed.h5"
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# Test with dummy input
dummy_input = np.random.rand(1, 224, 224, 3).astype("float32")
output = model.predict(dummy_input)
print("✅ Forward pass successful, output shape:", output.shape)