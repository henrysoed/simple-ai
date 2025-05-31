import base64
import io
import os
import requests
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageDraw
import json
from datetime import datetime

# Define model path and URL
model_path = 'mnist_digit_classifier.h5'
MODEL_DOWNLOAD_URL = 'https://drive.google.com/uc?export=download&id=1IHCZuvxXlAP0zb6RdL0g0PEEmuJplWVn'
feedback_data_file = 'feedback_data.json'

# Function to download the model
def download_model_if_not_exists(url, path):
    if not os.path.exists(path):
        print(f"Model not found at {path}, downloading from {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Model downloaded successfully to {path}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading model: {e}")
            exit()
        except Exception as e:
            print(f"An unexpected error occurred during model download: {e}")
            exit()
    else:
        print(f"Model found at {path}, skipping download.")

# Function to save feedback data
def save_feedback_data(image_data, predicted_digit, correct_digit, is_correct):
    feedback_entry = {
        'timestamp': datetime.now().isoformat(),
        'image_data': image_data,
        'predicted_digit': int(predicted_digit),
        'correct_digit': int(correct_digit),
        'is_correct': bool(is_correct)
    }
    
    # Load existing feedback data
    feedback_list = []
    if os.path.exists(feedback_data_file):
        try:
            with open(feedback_data_file, 'r') as f:
                feedback_list = json.load(f)
        except:
            feedback_list = []
    
    # Add new feedback
    feedback_list.append(feedback_entry)
    
    # Save updated feedback data
    with open(feedback_data_file, 'w') as f:
        json.dump(feedback_list, f)
    
    print(f"Feedback saved: predicted={predicted_digit}, correct={correct_digit}, is_correct={is_correct}")

# Function to retrain model with feedback data
def retrain_model_with_feedback():
    global model
    
    if not os.path.exists(feedback_data_file):
        print("No feedback data available for retraining")
        return
    
    try:
        with open(feedback_data_file, 'r') as f:
            feedback_list = json.load(f)
        
        # Filter incorrect predictions for retraining
        incorrect_feedback = [fb for fb in feedback_list if not fb['is_correct']]
        
        if len(incorrect_feedback) < 5:  # Need at least 5 samples to retrain
            print(f"Not enough incorrect samples for retraining: {len(incorrect_feedback)}")
            return
        
        print(f"Retraining with {len(incorrect_feedback)} incorrect samples...")
        
        # Prepare training data
        X_retrain = []
        y_retrain = []
        
        for feedback in incorrect_feedback:
            # Process image data
            processed_image = preprocess_image_data(feedback['image_data'])
            X_retrain.append(processed_image[0])  # Remove batch dimension
            
            # Create one-hot encoded label
            label = np.zeros(10)
            label[feedback['correct_digit']] = 1
            y_retrain.append(label)
        
        X_retrain = np.array(X_retrain)
        y_retrain = np.array(y_retrain)
        
        # Retrain model with very small learning rate
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        # Train for a few epochs
        model.fit(X_retrain, y_retrain, epochs=3, verbose=1, batch_size=min(32, len(X_retrain)))
        
        # Save updated model
        model.save(model_path)
        print("Model retrained and saved successfully!")
        
    except Exception as e:
        print(f"Error during retraining: {e}")

# Download the model before loading
download_model_if_not_exists(MODEL_DOWNLOAD_URL, model_path)

# Load the trained model
try:
    model = tf.keras.models.load_model(model_path)
except OSError:
    print(f"Error: Model file '{model_path}' not found. Make sure it's in the same directory.")
    exit()

app = Flask(__name__)

def preprocess_image_data(image_data_url):
    # Decode the base64 image
    image_data = base64.b64decode(image_data_url.split(',')[1])
    pil_image = Image.open(io.BytesIO(image_data)).convert('L')

    # Find bounding box of the drawing
    bbox = pil_image.getbbox()

    if bbox is None:
        # No drawing, return a blank image
        img_final_for_model_array = np.zeros((28, 28), dtype=np.float32)
    else:
        # Crop to the bounding box
        img_cropped = pil_image.crop(bbox)

        target_fit_size = 22
        original_width, original_height = img_cropped.size

        if original_width > original_height:
            new_width_cropped = target_fit_size
            new_height_cropped = int(original_height * (new_width_cropped / original_width))
        else:
            new_height_cropped = target_fit_size
            new_width_cropped = int(original_width * (new_height_cropped / original_height))
        
        new_width_cropped = max(1, new_width_cropped)
        new_height_cropped = max(1, new_height_cropped)

        img_resized_cropped = img_cropped.resize((new_width_cropped, new_height_cropped), Image.Resampling.LANCZOS)

        img_display_28x28 = Image.new("L", (28, 28), "black")
        paste_x = (28 - new_width_cropped) // 2
        paste_y = (28 - new_height_cropped) // 2
        img_display_28x28.paste(img_resized_cropped, (paste_x, paste_y))

        img_array = np.array(img_display_28x28)
        img_final_for_model_array = img_array.astype('float32') / 255.0

    # Reshape for the model
    if model.input_shape == (None, 28, 28):
        img_processed = np.expand_dims(img_final_for_model_array, axis=0)
    elif model.input_shape == (None, 28, 28, 1):
        img_processed = np.expand_dims(img_final_for_model_array, axis=0)
        img_processed = np.expand_dims(img_processed, axis=-1)
    elif model.input_shape == (None, 784):
        img_processed = img_final_for_model_array.reshape(1, 28*28)
    else:
        raise ValueError(f"Unexpected model input shape: {model.input_shape}")
    
    return img_processed

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data_url = data['image']
    
    try:
        processed_image = preprocess_image_data(image_data_url)
        prediction_probabilities = model.predict(processed_image)[0]
        predicted_digit = int(np.argmax(prediction_probabilities))
        probabilities = [float(p) for p in prediction_probabilities]
        
        # Store image data for potential feedback
        session_id = datetime.now().timestamp()
        
        return jsonify({
            'prediction': predicted_digit, 
            'probabilities': probabilities,
            'session_id': session_id,
            'image_data': image_data_url  # Keep for feedback
        })
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    
    try:
        image_data = data['image_data']
        predicted_digit = data['predicted_digit']
        is_correct = data['is_correct']
        correct_digit = data.get('correct_digit', predicted_digit)
        
        # Save feedback data
        save_feedback_data(image_data, predicted_digit, correct_digit, is_correct)
        
        # If we have enough incorrect samples, retrain the model
        if not is_correct:
            retrain_model_with_feedback()
        
        return jsonify({'status': 'success', 'message': 'Feedback received'})
        
    except Exception as e:
        print(f"Error processing feedback: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    try:
        if not os.path.exists(feedback_data_file):
            return jsonify({'total_feedback': 0, 'correct_predictions': 0, 'accuracy': 0})
        
        with open(feedback_data_file, 'r') as f:
            feedback_list = json.load(f)
        
        total_feedback = len(feedback_list)
        correct_predictions = sum(1 for fb in feedback_list if fb['is_correct'])
        accuracy = (correct_predictions / total_feedback * 100) if total_feedback > 0 else 0
        
        return jsonify({
            'total_feedback': total_feedback,
            'correct_predictions': correct_predictions,
            'accuracy': round(accuracy, 2)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
