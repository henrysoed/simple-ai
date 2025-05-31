from flask import Flask, render_template, request, jsonify
import numpy as np
import base64
import io
from PIL import Image
import tensorflow as tf
import json
import datetime
import os
from neural_network import SimpleNeuralNetwork

app = Flask(__name__)

# Load the trained model
model_path = 'mnist_digit_classifier.h5'
try:
    model = tf.keras.models.load_model(model_path)
except OSError:
    print(f"Error: Model file '{model_path}' not found. Please train the model first by running train_digit_classifier.py")

# Fungsi untuk menyimpan feedback
def save_feedback(image_data, prediction, correction, confidence):
    feedback_file = 'feedback_data.json'
    
    # Buat data feedback
    feedback = {
        'timestamp': str(datetime.datetime.now()),
        'image': image_data,
        'prediction': int(prediction),
        'correction': int(correction) if correction is not None else None,
        'confidence': float(confidence)
    }
    
    # Load data yang sudah ada atau buat baru
    if os.path.exists(feedback_file):
        with open(feedback_file, 'r') as f:
            try:
                all_feedback = json.load(f)
            except json.JSONDecodeError:
                all_feedback = []
    else:
        all_feedback = []
    
    # Tambahkan feedback baru
    all_feedback.append(feedback)
    
    # Simpan kembali ke file
    with open(feedback_file, 'w') as f:
        json.dump(all_feedback, f)

# Route untuk halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# API endpoint untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    # Ambil data gambar dari request
    img_data = request.json['image'].split(',')[1]
    img_bytes = base64.b64decode(img_data)
    
    # Konversi ke gambar dan resize
    img = Image.open(io.BytesIO(img_bytes)).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)
    
    # Normalisasi dan reshape untuk model
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    # Prediksi dengan model
    predictions = model.predict(img_array)
    digit = np.argmax(predictions[0])
    confidence = float(predictions[0][digit])
    
    return jsonify({
        'digit': int(digit),
        'confidence': confidence,
        'probabilities': predictions[0].tolist()
    })

# API endpoint untuk menyimpan feedback
@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    image_data = data['image']
    prediction = data['prediction']
    correction = data['correction']
    confidence = data['confidence']
    
    save_feedback(image_data, prediction, correction, confidence)
    
    return jsonify({'status': 'success'})

# API endpoint untuk Neural Network demo
@app.route('/neural_network_demo', methods=['POST'])
def neural_network_demo():
    data = request.json
    input_data = np.array(data['input'])
    
    # Buat instance neural network
    nn = SimpleNeuralNetwork(
        input_size=data.get('input_size', 2), 
        hidden_size=data.get('hidden_size', 4), 
        output_size=data.get('output_size', 1),
        activation=data.get('activation', 'sigmoid')
    )
    
    # Forward pass
    _, output = nn.forward(input_data)
    
    return jsonify({
        'output': output.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
