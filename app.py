\
import base64
import io
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageDraw

# Load the trained model
model_path = 'mnist_digit_classifier.h5'
try:
    model = tf.keras.models.load_model(model_path)
except OSError:
    print(f"Error: Model file '{model_path}' not found. Make sure it's in the same directory.")
    exit()

app = Flask(__name__)

def preprocess_image_data(image_data_url):
    # Decode the base64 image
    image_data = base64.b64decode(image_data_url.split(',')[1])
    pil_image = Image.open(io.BytesIO(image_data)).convert('L') # Convert to grayscale

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
        predicted_digit = int(np.argmax(prediction_probabilities)) # Ensure it's a standard int
        probabilities = [float(p) for p in prediction_probabilities] # Ensure they are standard floats
        return jsonify({'prediction': predicted_digit, 'probabilities': probabilities})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
