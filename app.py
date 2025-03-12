import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for

# Initialize Flask app
app = Flask(__name__)

# Load the trained deep learning model
MODEL_PATH = "dermatology_model.h5"  # Make sure this model exists in the directory
model = tf.keras.models.load_model(MODEL_PATH)

# Define class names based on your dataset
CLASS_NAMES = ["Acne", "Eczema", "Melanoma", "Psoriasis", "Rosacea", "Other"]  # Update with actual classes

# Ensure the 'uploads' directory exists
UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Home Page Route
@app.route('/')
def home():
    return render_template('subfolder/index.html')

# Image Upload and Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Preprocess the image
        img = cv2.imread(file_path)
        img = cv2.resize(img, (224, 224))  # Resize to match model input size
        img = img / 255.0  # Normalize pixel values
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Make prediction
        predictions = model.predict(img)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = round(100 * np.max(predictions), 2)

        return render_template('result.html', filename=file.filename, predicted_class=predicted_class, confidence=confidence)

# Route to handle favicon.ico error
@app.route('/favicon.ico')
def favicon():
    return "", 204  # Returns a blank response

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

