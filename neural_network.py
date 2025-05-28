import numpy as np
import matplotlib.pyplot as plt

class SimpleNeuralNetwork:
    """
    Neural Network sederhana dengan satu hidden layer
    Menggunakan sigmoid activation function
    """
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        """
        Inisialisasi neural network
        
        Args:
            input_size: Jumlah input neurons
            hidden_size: Jumlah hidden neurons  
            output_size: Jumlah output neurons
            learning_rate: Learning rate untuk gradient descent
        """
        self.learning_rate = learning_rate
        
        # Inisialisasi weights secara random (Xavier initialization)
        # W1: weights dari input ke hidden layer
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        # b1: bias untuk hidden layer
        self.b1 = np.zeros((1, hidden_size))
        
        # W2: weights dari hidden ke output layer
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        # b2: bias untuk output layer
        self.b2 = np.zeros((1, output_size))
        
        # List untuk menyimpan loss history
        self.loss_history = []
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        # Clip x untuk mencegah overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Turunan dari sigmoid function"""
        return x * (1 - x)
    
    def forward(self, X):
        """
        Forward propagation
        
        Args:
            X: Input data (batch_size, input_size)
            
        Returns:
            Tuple berisi (hidden_output, final_output)
        """
        # Hidden layer calculation
        # z1 = X * W1 + b1
        self.z1 = np.dot(X, self.W1) + self.b1
        # a1 = sigmoid(z1)
        self.a1 = self.sigmoid(self.z1)
        
        # Output layer calculation
        # z2 = a1 * W2 + b2
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        # a2 = sigmoid(z2) - final output
        self.a2 = self.sigmoid(self.z2)
        
        return self.a1, self.a2
    
    def compute_loss(self, y_true, y_pred):
        """
        Menghitung Mean Squared Error loss
        
        Args:
            y_true: Target values
            y_pred: Predicted values
            
        Returns:
            MSE loss
        """
        m = y_true.shape[0]  # jumlah samples
        loss = np.sum((y_true - y_pred) ** 2) / (2 * m)
        return loss
    
    def backward(self, X, y_true, y_pred):
        """
        Backward propagation (Backpropagation)
        Menghitung gradients menggunakan chain rule
        
        Args:
            X: Input data
            y_true: Target values
            y_pred: Predicted values
        """
        m = X.shape[0]  # jumlah samples
        
        # Gradient untuk output layer
        # dL/da2 = -(y_true - y_pred) = (y_pred - y_true)
        dL_da2 = (y_pred - y_true)
        
        # dL/dz2 = dL/da2 * da2/dz2 = dL/da2 * sigmoid'(z2)
        dL_dz2 = dL_da2 * self.sigmoid_derivative(self.a2)
        
        # dL/dW2 = dL/dz2 * dz2/dW2 = a1.T * dL/dz2
        dL_dW2 = np.dot(self.a1.T, dL_dz2) / m
        
        # dL/db2 = dL/dz2
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True) / m
        
        # Gradient untuk hidden layer
        # dL/da1 = dL/dz2 * dz2/da1 = dL/dz2 * W2.T
        dL_da1 = np.dot(dL_dz2, self.W2.T)
        
        # dL/dz1 = dL/da1 * da1/dz1 = dL/da1 * sigmoid'(z1)
        dL_dz1 = dL_da1 * self.sigmoid_derivative(self.a1)
        
        # dL/dW1 = dL/dz1 * dz1/dW1 = X.T * dL/dz1
        dL_dW1 = np.dot(X.T, dL_dz1) / m
        
        # dL/db1 = dL/dz1
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True) / m
        
        return dL_dW1, dL_db1, dL_dW2, dL_db2
    
    def update_weights(self, dL_dW1, dL_db1, dL_dW2, dL_db2):
        """
        Update weights menggunakan gradient descent
        
        Args:
            dL_dW1, dL_db1, dL_dW2, dL_db2: Gradients yang dihitung dari backpropagation
        """
        # Gradient descent: W = W - learning_rate * gradient
        self.W1 -= self.learning_rate * dL_dW1
        self.b1 -= self.learning_rate * dL_db1
        self.W2 -= self.learning_rate * dL_dW2
        self.b2 -= self.learning_rate * dL_db2
    
    def train(self, X, y, epochs=1000, verbose=True):
        """
        Training neural network
        
        Args:
            X: Input data
            y: Target data
            epochs: Jumlah iterasi training
            verbose: Print progress atau tidak
        """
        for epoch in range(epochs):
            # Forward propagation
            _, y_pred = self.forward(X)
            
            # Compute loss
            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)
            
            # Backward propagation
            dL_dW1, dL_db1, dL_dW2, dL_db2 = self.backward(X, y, y_pred)
            
            # Update weights using gradient descent
            self.update_weights(dL_dW1, dL_db1, dL_dW2, dL_db2)
            
            # Print progress
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
    
    def predict(self, X):
        """
        Prediksi menggunakan trained model
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        _, predictions = self.forward(X)
        return predictions
    
    def plot_loss(self):
        """Plot loss curve selama training"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history)
        plt.title('Training Loss Over Time')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    print("Neural Network Implementation Ready!")
    print("Implementasi dasar neural network dengan backpropagation dan gradient descent selesai.")
