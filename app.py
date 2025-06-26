from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os
import shutil

app = Flask(__name__, template_folder='.', static_folder='static')  # Explicit static folder
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'model/butterfly_classifier.h5'
DATA_TRAIN_DIR = 'data/train'

# Creating upload folder and static folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(os.getcwd(), 'static'), exist_ok=True)  # Ensure static folder exists

# Loading the model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

# Image size for the model
img_size = (224, 224)

# Loading class labels from the training directory
if not os.path.exists(DATA_TRAIN_DIR):
    raise FileNotFoundError(f"Training directory not found at {DATA_TRAIN_DIR}")
class_labels = sorted([d for d in os.listdir(DATA_TRAIN_DIR) if os.path.isdir(os.path.join(DATA_TRAIN_DIR, d))])
if not class_labels:
    raise ValueError("No class labels found in the training directory")

def predict_species(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize(img_size)
        img_array = image.img_to_array(img) / 255.0
        img_array = tf.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)
        class_index = tf.argmax(predictions[0]).numpy()
        confidence = float(predictions[0][class_index])
        if class_index >= len(class_labels):
            raise ValueError(f"Prediction index {class_index} out of range for {len(class_labels)} classes")
        return class_labels[class_index], confidence
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Validating file extension
    allowed_extensions = {'.jpg', '.jpeg', '.png'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        return jsonify({'error': f'Invalid file type. Allowed: {", ".join(allowed_extensions)}'}), 400

    # Saving the uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        # Predicting species
        species, confidence = predict_species(file_path)
        return jsonify({
            'predicted_species': species,
            'confidence': confidence
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Cleaning up the uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)