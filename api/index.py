from flask import Flask, render_template, request, jsonify
import numpy as np
import base64
import io
from PIL import Image
import json
import datetime
import os
from neural_network import SimpleNeuralNetwork

app = Flask(__name__)

# Untuk Vercel, kita tidak bisa load model TensorFlow yang besar
# Jadi kita akan fokus pada Neural Network demo saja
# atau buat model yang lebih ringan

# Fungsi untuk menyimpan feedback (disederhanakan untuk serverless)
def save_feedback(image_data, prediction, correction, confidence):
    # Untuk production, Anda bisa menggunakan database cloud
    # Sementara ini hanya return success
    return {"status": "saved"}

# Route untuk halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# API endpoint untuk Neural Network demo
@app.route('/api/neural_network_demo', methods=['POST'])
def neural_network_demo():
    try:
        data = request.json
        input_data = np.array(data['input']).reshape(1, -1)
        
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
            'output': output.tolist(),
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

# API endpoint untuk training demo
@app.route('/api/train_demo', methods=['POST'])
def train_demo():
    try:
        data = request.json
        
        # Data XOR sebagai contoh
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])
        
        # Buat dan latih neural network
        nn = SimpleNeuralNetwork(
            input_size=2, 
            hidden_size=data.get('hidden_size', 4), 
            output_size=1,
            learning_rate=data.get('learning_rate', 0.1),
            activation=data.get('activation', 'sigmoid')
        )
        
        # Training dengan epochs yang disesuaikan
        epochs = min(data.get('epochs', 1000), 2000)  # Batasi epochs untuk serverless
        nn.train(X, y, epochs=epochs, verbose=False)
        
        # Test prediksi
        predictions = nn.predict(X)
        
        return jsonify({
            'predictions': predictions.tolist(),
            'loss_history': nn.loss_history[-10:],  # Hanya 10 terakhir
            'final_loss': nn.loss_history[-1] if nn.loss_history else 0,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

# API endpoint untuk testing berbagai activation functions
@app.route('/api/test_activations', methods=['POST'])
def test_activations():
    try:
        data = request.json
        activations = ['sigmoid', 'relu', 'tanh']
        results = {}
        
        # Data XOR
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])
        
        for activation in activations:
            nn = SimpleNeuralNetwork(
                input_size=2, 
                hidden_size=data.get('hidden_size', 4), 
                output_size=1,
                learning_rate=data.get('learning_rate', 0.1),
                activation=activation
            )
            
            # Training singkat untuk demo
            nn.train(X, y, epochs=500, verbose=False)
            predictions = nn.predict(X)
            
            results[activation] = {
                'predictions': predictions.tolist(),
                'final_loss': nn.loss_history[-1] if nn.loss_history else 0
            }
        
        return jsonify({
            'results': results,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

# API untuk mendapatkan info tentang neural network
@app.route('/api/info', methods=['GET'])
def info():
    return jsonify({
        'title': 'Neural Network Demo',
        'description': 'Simple neural network implementation with backpropagation',
        'features': [
            'XOR problem solver',
            'Multiple activation functions (sigmoid, ReLU, tanh)',
            'Real-time training visualization',
            'Interactive parameter tuning'
        ],
        'status': 'online'
    })

# Handler untuk Vercel
def handler(request):
    return app(request.environ, lambda status, headers: None)

if __name__ == '__main__':
    app.run(debug=True)
