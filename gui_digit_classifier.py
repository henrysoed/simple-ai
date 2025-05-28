import tkinter as tk
from tkinter import filedialog, Canvas, Button, Label
from PIL import Image, ImageDraw, ImageTk
import numpy as np
import tensorflow as tf

# Load the trained model
model_path = 'mnist_digit_classifier.h5'
try:
    model = tf.keras.models.load_model(model_path)
except OSError:
    print(f"Error: Model file '{model_path}' not found. Please train the model first by running train_digit_classifier.py")
    exit()

class DigitClassifierApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Digit Classifier Neural Network")

        # --- UI Elements ---
        # Input Canvas
        self.input_label = Label(root_window, text="Input", font=("Arial", 16))
        self.input_label.grid(row=0, column=0, pady=5)
        self.canvas_input = Canvas(root_window, width=280, height=280, bg='black', highlightthickness=1, highlightbackground="grey")
        self.canvas_input.grid(row=1, column=0, padx=10, pady=5)
        self.canvas_input.bind("<B1-Motion>", self.paint)
        self.last_x, self.last_y = None, None
        self.image_input = Image.new("L", (280, 280), "black")
        self.draw_input = ImageDraw.Draw(self.image_input)

        # Processed Image Display
        self.processed_label = Label(root_window, text="Processed (28x28)", font=("Arial", 16))
        self.processed_label.grid(row=0, column=1, pady=5)
        self.canvas_processed_pil = Image.new("L", (280, 280), "lightgrey") # Placeholder
        self.canvas_processed_tk = ImageTk.PhotoImage(self.canvas_processed_pil)
        self.canvas_processed_display = Label(root_window, image=self.canvas_processed_tk, relief=tk.RAISED, borderwidth=1)
        self.canvas_processed_display.grid(row=1, column=1, padx=10, pady=5)

        # Control Buttons
        self.controls_frame = tk.Frame(root_window)
        self.controls_frame.grid(row=2, column=0, columnspan=1, pady=10)

        self.btn_check = Button(self.controls_frame, text="Check Digit", command=self.check_digit, width=15, font=("Arial", 12), bg="#FFC0CB", relief=tk.RAISED)
        self.btn_check.pack(side=tk.LEFT, padx=5)

        self.btn_clear = Button(self.controls_frame, text="Clear", command=self.clear_canvas, width=10, font=("Arial", 12), bg="#D3D3D3", relief=tk.RAISED)
        self.btn_clear.pack(side=tk.LEFT, padx=5)

        # Prediction Display
        self.prediction_label_text = Label(root_window, text="Prediction:", font=("Arial", 18, "bold"))
        self.prediction_label_text.grid(row=3, column=0, columnspan=2, pady=10)

        # Bar Chart for Probabilities
        self.prob_canvas_height = 150
        self.prob_canvas_width = 400
        self.prob_canvas = Canvas(root_window, width=self.prob_canvas_width, height=self.prob_canvas_height, bg='white', highlightthickness=1, highlightbackground="grey")
        self.prob_canvas.grid(row=4, column=0, columnspan=2, pady=10)
        self.draw_initial_bars()

    def paint(self, event):
        if self.last_x and self.last_y:
            self.canvas_input.create_line(self.last_x, self.last_y, event.x, event.y,
                                       width=20, fill='white', capstyle=tk.ROUND, smooth=tk.TRUE)
            self.draw_input.line([self.last_x, self.last_y, event.x, event.y],
                                 fill='white', width=20, joint="curve")
        self.last_x = event.x
        self.last_y = event.y

    def clear_canvas(self):
        self.canvas_input.delete("all")
        self.draw_input.rectangle([0, 0, 280, 280], fill="black") # Clear PIL image
        self.last_x, self.last_y = None, None
        # Clear processed image
        self.canvas_processed_pil = Image.new("L", (280, 280), "lightgrey")
        self.canvas_processed_tk = ImageTk.PhotoImage(self.canvas_processed_pil)
        self.canvas_processed_display.configure(image=self.canvas_processed_tk)
        self.canvas_processed_display.image = self.canvas_processed_tk # Keep a reference
        # Clear prediction and bars
        self.prediction_label_text.config(text="Prediction:")
        self.draw_initial_bars()

    def preprocess_image(self, pil_image):
        # Resize to 28x28
        img_resized = pil_image.resize((28, 28), Image.Resampling.LANCZOS)
        # Convert to numpy array
        img_array = np.array(img_resized)
        # Normalize (if your model expects this, MNIST usually does)
        img_array = img_array.astype('float32') / 255.0
        # Reshape for the model (1, 28, 28, 1) if using Conv2D or (1, 784) if Flattened
        # Assuming model input is (None, 28, 28) or (None, 28, 28, 1) for CNN
        # If model input is (None, 784) for a Dense layer after Flatten, reshape to (1, 28*28)
        if model.input_shape == (None, 28, 28):
            img_processed = np.expand_dims(img_array, axis=0)
        elif model.input_shape == (None, 28, 28, 1):
            img_processed = np.expand_dims(img_array, axis=0)
            img_processed = np.expand_dims(img_processed, axis=-1)
        elif model.input_shape == (None, 784):
            img_processed = img_array.reshape(1, 28*28)
        else:
            raise ValueError(f"Unexpected model input shape: {model.input_shape}")
        return img_processed, img_resized

    def check_digit(self):
        # Preprocess the drawn image
        processed_input, display_img_28x28 = self.preprocess_image(self.image_input)

        # Display the 28x28 processed image
        # Enlarge it for better visibility on the canvas_processed_display
        display_img_enlarged = display_img_28x28.resize((280, 280), Image.Resampling.NEAREST)
        self.canvas_processed_pil = display_img_enlarged
        self.canvas_processed_tk = ImageTk.PhotoImage(self.canvas_processed_pil)
        self.canvas_processed_display.configure(image=self.canvas_processed_tk)
        self.canvas_processed_display.image = self.canvas_processed_tk # Keep a reference

        # Make prediction
        prediction_probabilities = model.predict(processed_input)[0]
        predicted_digit = np.argmax(prediction_probabilities)

        self.prediction_label_text.config(text=f"Prediction: {predicted_digit}")
        self.update_probability_bars(prediction_probabilities)

    def draw_initial_bars(self):
        self.prob_canvas.delete("all")
        bar_width = (self.prob_canvas_width - 20) / 10 # 10 bars, 10px padding total
        max_bar_height = self.prob_canvas_height - 30 # Top/bottom padding

        for i in range(10):
            x0 = 10 + i * bar_width
            y0 = self.prob_canvas_height - 15 # Bottom line of bar
            x1 = x0 + bar_width * 0.8 # Bar width (80% of slot)
            y1 = y0 - 1 # Minimal height bar
            self.prob_canvas.create_rectangle(x0, y0, x1, y1, fill="#D3D3D3", outline="black", tags=f"bar_{i}")
            self.prob_canvas.create_text(x0 + (bar_width * 0.8) / 2, self.prob_canvas_height - 10,
                                       text=str(i), anchor=tk.N, font=("Arial", 10))
        # Y-axis labels
        self.prob_canvas.create_text(5, self.prob_canvas_height - 15, text="0.0", anchor=tk.W, font=("Arial", 8))
        self.prob_canvas.create_text(5, (self.prob_canvas_height - 15) / 2, text="0.5", anchor=tk.W, font=("Arial", 8))
        self.prob_canvas.create_text(5, 15, text="1.0", anchor=tk.W, font=("Arial", 8))

    def update_probability_bars(self, probabilities):
        self.prob_canvas.delete("all") # Clear previous bars and text
        bar_width = (self.prob_canvas_width - 40) / 10 # Adjusted for padding
        max_bar_height = self.prob_canvas_height - 30 # For text labels below
        left_padding = 30 # Space for Y-axis labels

        for i in range(10):
            prob = probabilities[i]
            bar_height = prob * max_bar_height
            x0 = left_padding + i * bar_width + (bar_width * 0.1) # Small gap between bars
            y0 = self.prob_canvas_height - 15 # Bottom of the bar (aligned with 0.0)
            x1 = left_padding + (i + 1) * bar_width - (bar_width * 0.1)
            y1 = y0 - bar_height # Top of the bar

            fill_color = "#4A86E8" if i == np.argmax(probabilities) else "#D3D3D3"
            self.prob_canvas.create_rectangle(x0, y0, x1, y1, fill=fill_color, outline="black", tags=f"bar_{i}")
            self.prob_canvas.create_text(x0 + (x1 - x0) / 2, self.prob_canvas_height - 10,
                                       text=str(i), anchor=tk.N, font=("Arial", 10))

        # Redraw Y-axis labels (as they are cleared by delete("all"))
        self.prob_canvas.create_text(left_padding - 5, self.prob_canvas_height - 15, text="0.0", anchor=tk.E, font=("Arial", 8))
        self.prob_canvas.create_text(left_padding - 5, (self.prob_canvas_height - 15 + 15 - max_bar_height)/2 + 15 , text="0.5", anchor=tk.E, font=("Arial", 8))
        self.prob_canvas.create_text(left_padding - 5, 15, text="1.0", anchor=tk.E, font=("Arial", 8))

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitClassifierApp(root)
    root.mainloop()
