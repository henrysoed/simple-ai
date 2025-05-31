import tkinter as tk
from tkinter import filedialog, Canvas, Button, Label, simpledialog, messagebox
from PIL import Image, ImageDraw, ImageTk
import numpy as np
import tensorflow as tf
import json
import datetime
import base64
import io
import os

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

        # Add retrain button
        self.btn_retrain = Button(self.controls_frame, text="Retrain AI", command=self.retrain_with_feedback, width=12, font=("Arial", 12), bg="#87CEEB", relief=tk.RAISED)
        self.btn_retrain.pack(side=tk.LEFT, padx=5)
        
        # Prediction Display
        self.prediction_label_text = Label(root_window, text="Prediction:", font=("Arial", 18, "bold"))
        self.prediction_label_text.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Feedback Buttons Frame
        self.feedback_frame = tk.Frame(root_window)
        self.feedback_frame.grid(row=4, column=0, columnspan=2, pady=5)
        
        self.btn_correct = Button(self.feedback_frame, text="✓ Correct", command=self.feedback_correct, 
                                 width=12, font=("Arial", 12), bg="#90EE90", relief=tk.RAISED, state=tk.DISABLED)
        self.btn_correct.pack(side=tk.LEFT, padx=10)
        
        self.btn_wrong = Button(self.feedback_frame, text="✗ Wrong", command=self.feedback_wrong, 
                               width=12, font=("Arial", 12), bg="#FFB6C1", relief=tk.RAISED, state=tk.DISABLED)
        self.btn_wrong.pack(side=tk.LEFT, padx=10)
        
        # Feedback status label
        self.feedback_status = Label(root_window, text="", font=("Arial", 12))
        self.feedback_status.grid(row=5, column=0, columnspan=2, pady=5)

        # Feedback statistics label
        self.stats_label = Label(root_window, text="", font=("Arial", 10), fg="gray")
        self.stats_label.grid(row=7, column=0, columnspan=2, pady=5)
        
        # Bar Chart for Probabilities
        self.prob_canvas_height = 150
        self.prob_canvas_width = 400
        self.prob_canvas = Canvas(root_window, width=self.prob_canvas_width, height=self.prob_canvas_height, bg='white', highlightthickness=1, highlightbackground="grey")
        self.prob_canvas.grid(row=6, column=0, columnspan=2, pady=10)
        
        # Variables to store current prediction data
        self.current_predicted_digit = None
        self.current_processed_input = None
        self.current_display_img = None
        
        # Initialize UI elements
        self.draw_initial_bars()
        self.update_feedback_stats()

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
        
        # Reset feedback buttons and status
        self.btn_correct.config(state=tk.DISABLED)
        self.btn_wrong.config(state=tk.DISABLED)
        self.feedback_status.config(text="")
        
        # Clear current prediction data
        self.current_predicted_digit = None
        self.current_processed_input = None
        self.current_display_img = None

    def preprocess_image(self, pil_image):
        # Find bounding box of the drawing
        bbox = pil_image.getbbox()

        if bbox is None:
            # No drawing, return a blank image (already normalized as zeros)
            img_final_for_model_array = np.zeros((28, 28), dtype=np.float32)
            # For display, a black PIL image
            img_display_28x28 = Image.new("L", (28, 28), "black")
        else:
            # Crop to the bounding box
            img_cropped = pil_image.crop(bbox)

            # Resize the cropped image to fit within a target box (e.g., 20x20 or 22x22)
            # while maintaining aspect ratio, then center it on a 28x28 canvas.
            target_fit_size = 22  # Fit the longest dimension to 22 pixels
            original_width, original_height = img_cropped.size

            if original_width > original_height:
                new_width_cropped = target_fit_size
                new_height_cropped = int(original_height * (new_width_cropped / original_width))
            else:
                new_height_cropped = target_fit_size
                new_width_cropped = int(original_width * (new_height_cropped / original_height))
            
            # Ensure dimensions are at least 1 pixel
            new_width_cropped = max(1, new_width_cropped)
            new_height_cropped = max(1, new_height_cropped)

            img_resized_cropped = img_cropped.resize((new_width_cropped, new_height_cropped), Image.Resampling.LANCZOS)

            # Create a new 28x28 black image to serve as canvas
            img_display_28x28 = Image.new("L", (28, 28), "black")

            # Calculate position to paste the resized crop to center it
            paste_x = (28 - new_width_cropped) // 2
            paste_y = (28 - new_height_cropped) // 2
            img_display_28x28.paste(img_resized_cropped, (paste_x, paste_y))

            # Convert the 28x28 PIL image (now with centered digit) to numpy array for the model
            img_array = np.array(img_display_28x28)
            # Normalize
            img_final_for_model_array = img_array.astype('float32') / 255.0

        # Reshape for the model (1, 28, 28, 1) if using Conv2D or (1, 784) if Flattened
        # This part uses img_final_for_model_array which is the normalized 28x28 numpy array
        if model.input_shape == (None, 28, 28):
            img_processed = np.expand_dims(img_final_for_model_array, axis=0)
        elif model.input_shape == (None, 28, 28, 1):
            img_processed = np.expand_dims(img_final_for_model_array, axis=0)
            img_processed = np.expand_dims(img_processed, axis=-1)
        elif model.input_shape == (None, 784):
            img_processed = img_final_for_model_array.reshape(1, 28*28)
        else:
            raise ValueError(f"Unexpected model input shape: {model.input_shape}")
        
        return img_processed, img_display_28x28 # Return the processed numpy array and the 28x28 PIL image for display

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

        # Store current prediction data for feedback
        self.current_predicted_digit = predicted_digit
        self.current_processed_input = processed_input
        self.current_display_img = display_img_28x28

        self.prediction_label_text.config(text=f"Prediction: {predicted_digit}")
        self.update_probability_bars(prediction_probabilities)
        
        # Enable feedback buttons
        self.btn_correct.config(state=tk.NORMAL)
        self.btn_wrong.config(state=tk.NORMAL)
        self.feedback_status.config(text="Is this prediction correct?", fg="black")

    def feedback_correct(self):
        """Handle correct prediction feedback"""
        self.save_feedback(self.current_predicted_digit, self.current_predicted_digit, True)
        self.feedback_status.config(text="✓ Feedback saved: Prediction was correct!", fg="green")
        self.btn_correct.config(state=tk.DISABLED)
        self.btn_wrong.config(state=tk.DISABLED)

    def feedback_wrong(self):
        """Handle wrong prediction feedback"""
        # Ask user for the correct digit
        correct_digit = simpledialog.askinteger(
            "Correct Digit", 
            f"The AI predicted {self.current_predicted_digit}.\nWhat is the correct digit (0-9)?",
            minvalue=0, maxvalue=9
        )
        
        if correct_digit is not None:
            self.save_feedback(self.current_predicted_digit, correct_digit, False)
            self.feedback_status.config(text=f"✓ Feedback saved: Correct digit was {correct_digit}", fg="orange")
        else:
            self.feedback_status.config(text="Feedback cancelled", fg="gray")
        
        self.btn_correct.config(state=tk.DISABLED)
        self.btn_wrong.config(state=tk.DISABLED)

    def save_feedback(self, predicted_digit, correct_digit, is_correct):
        """Save feedback data to JSON file"""
        try:
            # Convert current display image to base64 for storage
            img_bytes = io.BytesIO()
            self.current_display_img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            img_data = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
            img_data_url = f"data:image/png;base64,{img_data}"
            
            # Create feedback entry
            feedback_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "image_data": img_data_url,
                "predicted_digit": int(predicted_digit),
                "correct_digit": int(correct_digit),
                "is_correct": bool(is_correct)
            }
            
            # Load existing feedback data
            try:
                with open('feedback_data.json', 'r') as f:
                    feedback_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                feedback_data = []
            
            # Add new feedback
            feedback_data.append(feedback_entry)
            
            # Save updated feedback data
            with open('feedback_data.json', 'w') as f:
                json.dump(feedback_data, f, indent=2)
                
            print(f"Feedback saved: Predicted={predicted_digit}, Correct={correct_digit}, Is_Correct={is_correct}")
            
            # Update feedback statistics display
            self.update_feedback_stats()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save feedback: {str(e)}")
            print(f"Error saving feedback: {e}")

    def retrain_with_feedback(self):
        """Retrain the model using feedback data"""
        try:
            # Load feedback data
            try:
                with open('feedback_data.json', 'r') as f:
                    feedback_data = json.load(f)
            except FileNotFoundError:
                messagebox.showerror("Error", "No feedback data found. Please provide feedback first.")
                return
            
            if len(feedback_data) == 0:
                messagebox.showinfo("Info", "No feedback data available for retraining.")
                return
            
            # Show progress dialog
            self.feedback_status.config(text="Loading feedback data for retraining...", fg="blue")
            self.root.update()
            
            # Prepare training data from feedback
            X_feedback = []
            y_feedback = []
            
            # Determine target shape based on model input
            if len(model.input_shape) < 2:
                messagebox.showerror("Error", f"Unexpected model input shape: {model.input_shape}")
                return
                
            if model.input_shape[1:] == (28, 28):
                target_shape = (28, 28)
                reshape_func = lambda x: x
            elif model.input_shape[1:] == (28, 28, 1):
                target_shape = (28, 28, 1)
                reshape_func = lambda x: np.expand_dims(x, axis=-1)
            elif model.input_shape[1:] == (784,):
                target_shape = (784,)
                reshape_func = lambda x: x.reshape(784)
            else:
                messagebox.showerror("Error", f"Unsupported model input shape: {model.input_shape}")
                return
            
            print(f"Target shape for training: {target_shape}")
            
            for entry in feedback_data:
                try:
                    # Decode base64 image
                    img_data = entry['image_data'].split(',')[1]  # Remove data:image/png;base64,
                    img_bytes = base64.b64decode(img_data)
                    img = Image.open(io.BytesIO(img_bytes))
                    
                    # Ensure image is grayscale and exactly 28x28
                    if img.mode != 'L':
                        img = img.convert('L')
                    
                    # Resize to exactly 28x28
                    img = img.resize((28, 28), Image.Resampling.LANCZOS)
                    
                    # Convert to numpy array and normalize
                    img_array = np.array(img).astype('float32') / 255.0
                    
                    # Apply the appropriate reshape function based on model requirements
                    img_processed = reshape_func(img_array)
                    
                    # Debug info
                    if len(X_feedback) == 0:
                        print(f"First processed image shape: {img_processed.shape}")
                    
                    X_feedback.append(img_processed)
                    y_feedback.append(entry['correct_digit'])
                    
                except Exception as e:
                    print(f"Error processing feedback entry: {str(e)}")
                    continue
            
            if len(X_feedback) == 0:
                messagebox.showerror("Error", "No valid feedback data found for retraining.")
                return
            
            # Convert to numpy arrays with consistent shape using stack
            try:
                X_feedback_array = np.stack(X_feedback)
                y_feedback_array = np.array(y_feedback)
                
                # Print final shape for debugging
                print(f"Final X_feedback shape: {X_feedback_array.shape}")
                
                # Convert labels to one-hot encoding
                y_feedback_onehot = tf.keras.utils.to_categorical(y_feedback_array, 10)
                
                self.feedback_status.config(text=f"Retraining with {len(X_feedback)} feedback samples...", fg="blue")
                self.root.update()
                
                # Fine-tune the model with feedback data
                # Use a lower learning rate for fine-tuning
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
                
                # Train for a few epochs with the feedback data
                batch_size = min(32, len(X_feedback))
                if batch_size < 1:
                    batch_size = 1  # Ensure at least batch size of 1
                    
                print(f"Training model with input shape {X_feedback_array.shape} and labels shape {y_feedback_onehot.shape}")
                print(f"Using batch size: {batch_size}")
                
                history = model.fit(X_feedback_array, y_feedback_onehot, 
                                epochs=5, 
                                batch_size=batch_size,
                                verbose=1)  # Set to 1 to show training progress
                                
                # Save the updated model
                model.save('mnist_digit_classifier.h5')
                
                # Show completion message
                final_accuracy = history.history['accuracy'][-1]
                self.feedback_status.config(
                    text=f"✓ AI retrained successfully! Final accuracy: {final_accuracy:.3f}", 
                    fg="green"
                )
                
                messagebox.showinfo("Success", 
                                f"AI has been retrained with {len(X_feedback)} feedback samples!\n"
                                f"Final training accuracy: {final_accuracy:.3f}\n"
                                f"Model saved to mnist_digit_classifier.h5")
                
            except Exception as e:
                print(f"Error during array conversion or training: {e}")
                messagebox.showerror("Training Error", f"Error in training process: {str(e)}")
                self.feedback_status.config(text="Training failed", fg="red")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to retrain model: {str(e)}")
            self.feedback_status.config(text="Retraining failed", fg="red")
            print(f"Error retraining model: {e}")

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

    def load_feedback_stats(self):
        """Load and display feedback statistics"""
        try:
            with open('feedback_data.json', 'r') as f:
                feedback_data = json.load(f)
            
            total_feedback = len(feedback_data)
            correct_predictions = sum(1 for entry in feedback_data if entry['is_correct'])
            accuracy = (correct_predictions / total_feedback * 100) if total_feedback > 0 else 0
            
            return total_feedback, correct_predictions, accuracy
        except:
            return 0, 0, 0

    def update_feedback_stats(self):
        """Update feedback statistics display"""
        total, correct, accuracy = self.load_feedback_stats()
        if total > 0:
            self.stats_label.config(
                text=f"Feedback Stats: {total} samples, {correct} correct ({accuracy:.1f}% accuracy)"
            )
        else:
            self.stats_label.config(text="No feedback data yet")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitClassifierApp(root)
    root.mainloop()
