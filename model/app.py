import os
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from model.modelFile.model import build_model
from pathlib import Path

# --- Config ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / 'modelFile' / 'brain_tumor_model.h5'))
IMG_SIZE = 150
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary', 'unlabeled']

# --- Load Pretrained Model ---
try:
    print(f"[INFO] Loading model from: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train and save the model first.")
    # model = load_model(str(MODEL_PATH), compile=False) # Old way
    model = build_model(img_size=IMG_SIZE, num_classes=len(CLASS_NAMES)) # Build model structure
    model.load_weights(str(MODEL_PATH)) # Load only weights
    print("[INFO] Model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    raise SystemExit("Exiting: Pretrained model could not be loaded.")

# --- Flask App ---
app = Flask(__name__)
origins = os.getenv("ALLOWED_ORIGINS", '["https://codestrom-hackathon.vercel.app", "http://localhost:5173"]')
CORS(app, resources={r"/*": {"origins": json.loads(origins)}})
print("[INFO] CORS configured.")

@app.route('/')
def home():
    return 'Flask app is running!', 200

@app.route('/health')
def health():
    return jsonify({'status': 'ok'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        print("[ERROR] No file part in request.")
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        print("[ERROR] No selected file.")
        return jsonify({'error': 'No selected file'}), 400

    try:
        filename = secure_filename(file.filename)
        print(f"[INFO] Received file: {filename}")

        # Accept both file and stream for image loading
        img = image.load_img(file.stream, target_size=(IMG_SIZE, IMG_SIZE))
        x = image.img_to_array(img)
        x = np.expand_dims(x / 255.0, axis=0)

        preds = model.predict(x)
        pred_class = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))

        print(f"[INFO] Prediction: {CLASS_NAMES[pred_class]} ({confidence:.2f})")

        return jsonify({
            'predicted_class': CLASS_NAMES[pred_class],
            'confidence': confidence,
            'all_confidences': {CLASS_NAMES[i]: float(preds[0][i]) for i in range(len(CLASS_NAMES))}
        })

    except Exception as e:
        print(f"[ERROR] Exception during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

# --- Start the Server ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
