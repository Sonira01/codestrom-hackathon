import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from pathlib import Path

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'brain_tumor_model.keras'
IMG_SIZE = 150
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary', 'unlabeled']

# --- Load pretrained model ---
try:
    print(f"[DEBUG] Attempting to load model from: {MODEL_PATH}")
    model = load_model(str(MODEL_PATH))
    print("[INFO] Model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load model from {MODEL_PATH}: {e}")
    raise SystemExit("Deployment failed: pretrained model could not be loaded.")

# --- Flask App ---
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://codestrom-hackathon.vercel.app", "http://localhost:5173"]}})
print("[INFO] CORS configured successfully for Vercel and localhost.")

@app.route('/')
def home():
    print("[DEBUG] Root endpoint '/' hit.")
    return 'Flask app is running!', 200

@app.route('/health')
def health():
    print("[DEBUG] Health check requested.")
    return jsonify({'status': 'ok'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    print("[DEBUG] /predict endpoint hit.")
    if 'file' not in request.files:
        print("[WARN] No file part in request.")
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        print("[WARN] Empty filename submitted.")
        return jsonify({'error': 'No selected file'}), 400

    try:
        filename = secure_filename(file.filename)
        print(f"[INFO] Received file: {filename}")

        img = image.load_img(file, target_size=(IMG_SIZE, IMG_SIZE))
        print("[DEBUG] Image loaded and resized.")

        x = image.img_to_array(img)
        x = np.expand_dims(x / 255.0, axis=0)
        print("[DEBUG] Image converted to array and normalized.")

        preds = model.predict(x)
        print(f"[DEBUG] Model prediction complete: {preds}")

        pred_class = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))
        print(f"[INFO] Predicted class: {CLASS_NAMES[pred_class]}, Confidence: {confidence:.4f}")

        return jsonify({
            'predicted_class': CLASS_NAMES[pred_class],
            'confidence': confidence,
            'all_confidences': {CLASS_NAMES[i]: float(preds[0][i]) for i in range(len(CLASS_NAMES))}
        })

    except Exception as e:
        print(f"[ERROR] Exception during prediction: {e}")
        return jsonify({'error': str(e)}), 500
