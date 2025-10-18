from tensorflow.keras.models import load_model, save_model

MODEL_PATH = "poster_model\poster_model.h5"
NEW_MODEL_PATH = "fixed_model.h5"

# Load with `compile=False` to avoid some errors
model = load_model(MODEL_PATH, compile=False)

# Save it again
save_model(model, NEW_MODEL_PATH)
print("Model re-saved successfully.")