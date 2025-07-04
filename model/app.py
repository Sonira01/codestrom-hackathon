import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from flask_cors import CORS
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models
from pathlib import Path # Ensure Path is imported for use outside main guard
import sys
import stat

# --- Utility: Directory and Path Safety ---
def ensure_dir_and_writable(path):
    """Ensure directory exists and is writable. Raise error if not."""
    d = Path(path)
    if d.is_file():
        d = d.parent
    d.mkdir(parents=True, exist_ok=True)
    if not os.access(str(d), os.W_OK):
        raise PermissionError(f"Directory {d} is not writable!")
    # Extra: check not root
    if str(d) == '/' or str(d) == '':
        raise PermissionError(f"Refusing to write to root directory: {d}")
    return d

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
<<<<<<< HEAD
DEFAULT_MODEL_PATH = BASE_DIR / 'modelFile' / 'brain_tumor_model.keras'
=======
DEFAULT_MODEL_PATH = Path('/tmp/brain_tumor_model.keras')

>>>>>>> 98dbd10362e802089202f1e86d457374da9ac0d2
DEFAULT_TRAIN_DATA_DIR = BASE_DIR.parent / 'data' / 'brain-data' / 'Training'

MODEL_PATH = os.environ.get('MODEL_PATH', str(DEFAULT_MODEL_PATH))
TRAIN_DATA_DIR = os.environ.get('TRAIN_DATA_DIR', str(DEFAULT_TRAIN_DATA_DIR))
IMG_SIZE = 150
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary', 'unlabeled']  # Replace with real class names

# --- Model Loading and Initial Training ---
try:
    print(f"Loading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    if model.output_shape[-1] != len(CLASS_NAMES):
        print(f"Warning: Model output shape {model.output_shape[-1]} "
              f"does not match number of classes {len(CLASS_NAMES)}. Rebuilding model.")
        raise ValueError('Model output shape does not match number of classes.')
except Exception as e:
    print(f"Failed to load model from {MODEL_PATH} or model mismatch: {e}")
    print("Attempting to build and potentially run initial training...")

    def build_model_for_app():
        m = models.Sequential([
            layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(len(CLASS_NAMES), activation='softmax')
        ])
        m.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
        return m
    model = build_model_for_app()

    def initial_train():
        print(f"Attempting initial training. Data directory: {TRAIN_DATA_DIR}")
        try:
            ensure_dir_and_writable(TRAIN_DATA_DIR)
        except Exception as err:
            print(f"[ERROR] Training data directory not writable: {err}")
            # Save untrained model if not present
            try:
                ensure_dir_and_writable(MODEL_PATH)
                if not Path(MODEL_PATH).exists():
                    model.save(MODEL_PATH)
                    print(f"Saved untrained model structure to {MODEL_PATH}")
            except Exception as save_err:
                print(f"[ERROR] Could not save untrained model: {save_err}")
            return
        if not Path(TRAIN_DATA_DIR).exists() or not any(Path(TRAIN_DATA_DIR).iterdir()):
            print(f"Training data directory {TRAIN_DATA_DIR} is empty or does not exist. Skipping initial training. Model will be untrained.")
            try:
                ensure_dir_and_writable(MODEL_PATH)
                if not Path(MODEL_PATH).exists():
                    model.save(MODEL_PATH)
                    print(f"Saved untrained model structure to {MODEL_PATH}")
            except Exception as save_err:
                print(f"[ERROR] Could not save untrained model: {save_err}")
            return
        print(f"Using training data from: {TRAIN_DATA_DIR}")
        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)
        try:
            train_gen = datagen.flow_from_directory(
                TRAIN_DATA_DIR,
                target_size=(IMG_SIZE, IMG_SIZE),
                batch_size=16,
                class_mode='categorical',
                subset='training'
            )
        except Exception as flow_exc:
            print(f"Error during flow_from_directory for training data: {flow_exc}")
            print("Skipping initial training.")
            try:
                ensure_dir_and_writable(MODEL_PATH)
                if not Path(MODEL_PATH).exists():
                    model.save(MODEL_PATH)
                    print(f"Saved untrained model structure to {MODEL_PATH}")
            except Exception as save_err:
                print(f"[ERROR] Could not save untrained model: {save_err}")
            return
        val_gen = datagen.flow_from_directory(
            TRAIN_DATA_DIR,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=16,
            class_mode='categorical',
            subset='validation'
        )
        if train_gen.samples == 0:
            print("No training samples found by ImageDataGenerator. Skipping model.fit.")
            try:
                ensure_dir_and_writable(MODEL_PATH)
                if not Path(MODEL_PATH).exists():
                    model.save(MODEL_PATH)
                    print(f"Saved untrained model structure to {MODEL_PATH}")
            except Exception as save_err:
                print(f"[ERROR] Could not save untrained model: {save_err}")
            return
        print(f"Starting model.fit with {train_gen.samples} training samples, {val_gen.samples} validation samples.")
        model.fit(train_gen, validation_data=val_gen, epochs=3)
        try:
            ensure_dir_and_writable(MODEL_PATH)
            model.save(MODEL_PATH)
            print(f"Initial training complete. Model saved to {MODEL_PATH}")
        except Exception as save_err:
            print(f"[ERROR] Could not save trained model: {save_err}")
    initial_train()

app = Flask(__name__)

# --- CORS Configuration ---
# Allow configuring origins via environment variable, defaulting to localhost for development
ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', 'http://localhost:5173,https://codestrom-hackathon.vercel.app')
if isinstance(ALLOWED_ORIGINS, str):
    ALLOWED_ORIGINS = [origin.strip() for origin in ALLOWED_ORIGINS.split(',')]

CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})
print(f"CORS configured for origins: {ALLOWED_ORIGINS}")


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        label = request.form.get('label', 'unlabeled')
        save_dir_base = Path(TRAIN_DATA_DIR)
        save_dir = save_dir_base / label
        try:
            ensure_dir_and_writable(save_dir)
        except Exception as err:
            return jsonify({'error': f'Cannot save uploaded file: {err}'}), 500
        filename = secure_filename(file.filename)
        save_path = save_dir / filename
        file.seek(0)
        file.save(save_path)
        img = image.load_img(save_path, target_size=(IMG_SIZE, IMG_SIZE))
        x = image.img_to_array(img)
        x = x / 255.0
        x = np.expand_dims(x, axis=0)
        preds = model.predict(x)
        pred_class = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))
        return jsonify({
            'predicted_class': CLASS_NAMES[pred_class],
            'confidence': confidence,
            'all_confidences': {CLASS_NAMES[i]: float(preds[0][i]) for i in range(len(CLASS_NAMES))},
            'retrained': False
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def retrain_model():
    print(f"Starting model retraining using data from {TRAIN_DATA_DIR}...")
    train_data_path = Path(TRAIN_DATA_DIR)
    try:
        ensure_dir_and_writable(train_data_path)
    except Exception as err:
        print(f"[ERROR] Training data directory not writable: {err}")
        return False
    if not train_data_path.exists() or not any(train_data_path.iterdir()):
        print(f"Training directory {TRAIN_DATA_DIR} is empty or does not exist. Skipping retraining.")
        return False
    sub_dirs = [d for d in train_data_path.iterdir() if d.is_dir()]
    if not sub_dirs:
        print(f"No subdirectories found in {TRAIN_DATA_DIR}. Skipping retraining as ImageDataGenerator needs class folders.")
        return False
    found_images = False
    for sub_dir in sub_dirs:
        if any(sub_dir.glob('*.jpg')) or any(sub_dir.glob('*.jpeg')) or any(sub_dir.glob('*.png')) or any(sub_dir.glob('*.bmp')):
            found_images = True
            break
    if not found_images:
        print(f"No images found in subdirectories of {TRAIN_DATA_DIR}. Skipping retraining.")
        return False
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)
    try:
        train_gen = datagen.flow_from_directory(
            TRAIN_DATA_DIR,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=16,
            class_mode='categorical',
            subset='training'
        )
    except Exception as flow_exc:
        print(f"Error during flow_from_directory for retraining data: {flow_exc}")
        print("Skipping retraining.")
        return False
    val_gen = datagen.flow_from_directory(
        TRAIN_DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=16,
        class_mode='categorical',
        subset='validation'
    )
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    print(f"Retraining: Found {train_gen.samples} training samples and {val_gen.samples} validation samples.")
    if train_gen.samples == 0:
        print("Retraining: No training samples found by ImageDataGenerator. Skipping model.fit.")
        return False
    model.fit(train_gen, validation_data=val_gen, epochs=1)
    try:
        ensure_dir_and_writable(MODEL_PATH)
        model.save(MODEL_PATH)
        print(f"Model retrained and saved to {MODEL_PATH}")
    except Exception as save_err:
        print(f"[ERROR] Could not save retrained model: {save_err}")
        return False
    return True

@app.route('/trigger_retrain', methods=['POST'])
def trigger_retrain_endpoint():
    try:
        print("Retrain endpoint called.")
        if retrain_model():
            return jsonify({'message': 'Model retraining initiated and completed successfully.'}), 200
        else:
            return jsonify({'message': 'Model retraining skipped (e.g., no data or data issue).'}), 200
    except Exception as e:
        print(f"Error during retraining: {str(e)}")
        return jsonify({'error': f'Error during retraining: {str(e)}'}), 500

@app.route('/')
def health_check():
    return 'Flask app is running!', 200

@app.route('/health')
def health():
    return jsonify({'status': 'ok'}), 200
