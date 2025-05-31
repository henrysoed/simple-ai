import numpy as np
import tensorflow as tf
from tkinter import messagebox
import json
import io
import base64
from PIL import Image

def fixed_retrain_with_feedback(model, feedback_status_widget, root):
    """Retrain the model using feedback data - fixed version that handles shapes correctly"""
    try:
        # Load feedback data
        with open('feedback_data.json', 'r') as f:
            feedback_data = json.load(f)
        
        if len(feedback_data) == 0:
            messagebox.showinfo("Info", "No feedback data available for retraining.")
            return
        
        # Show progress dialog
        feedback_status_widget.config(text="Loading feedback data for retraining...", fg="blue")
        root.update()
        
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
                print(f"Processed image shape: {img_processed.shape}")
                
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
            
            feedback_status_widget.config(text=f"Retraining with {len(X_feedback)} feedback samples...", fg="blue")
            root.update()
            
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
            feedback_status_widget.config(
                text=f"✓ AI retrained successfully! Final accuracy: {final_accuracy:.3f}", 
                fg="green"
            )
            
            messagebox.showinfo("Success", 
                            f"AI has been retrained with {len(X_feedback)} feedback samples!\n"
                            f"Final training accuracy: {final_accuracy:.3f}\n"
                            f"Model saved to mnist_digit_classifier.h5")
            
            return True
            
        except Exception as e:
            print(f"Error during array conversion or training: {e}")
            messagebox.showerror("Training Error", f"Error in training process: {str(e)}")
            feedback_status_widget.config(text="Training failed", fg="red")
            return False
            
    except FileNotFoundError:
        messagebox.showerror("Error", "No feedback data file found. Please provide some feedback first.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to retrain model: {str(e)}")
        feedback_status_widget.config(text="Retraining failed", fg="red")
        print(f"Error retraining model: {e}")
        return False