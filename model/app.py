import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from pathlib import Path

# --- Config ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'brain_tumor_model.keras'
IMG_SIZE = 150
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary', 'unlabeled']

# --- Load Pretrained Model ---
try:
    print(f"[INFO] Loading model from: {MODEL_PATH}")
    model = load_model(str(MODEL_PATH), compile=False)
    print("[INFO] Model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    raise SystemExit("Exiting: Pretrained model could not be loaded.")

# --- Flask App ---
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": [
    "https://codestrom-hackathon.vercel.app",
    "http://localhost:5173"
]}})
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

        img = image.load_img(file, target_size=(IMG_SIZE, IMG_SIZE))
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
