# Demo Comparison Different Activation Functions
import numpy as np
import matplotlib.pyplot as plt
from neural_network import SimpleNeuralNetwork

def compare_activations():
    """
    Compare performance neural network dengan activation functions berbeda
    pada XOR problem
    """
    print("=== Comparison Activation Functions ===")
    print("Testing Sigmoid vs ReLU vs Tanh pada XOR problem")
    print()
    
    # XOR dataset
    X = np.array([
        [0, 0],
        [0, 1], 
        [1, 0],
        [1, 1]
    ])
    
    y = np.array([
        [0],
        [1], 
        [1],
        [0]
    ])
    
    activations = ['sigmoid', 'relu', 'tanh']
    results = {}
    
    plt.figure(figsize=(15, 5))
    
    for i, activation in enumerate(activations):
        print(f"Training dengan {activation} activation...")
        
        # Create neural network
        nn = SimpleNeuralNetwork(
            input_size=2,
            hidden_size=6,
            output_size=1,
            learning_rate=1.0,
            activation=activation
        )
        
        # Train
        nn.train(X, y, epochs=2000, verbose=False)
        
        # Test
        predictions = nn.predict(X)
        pred_classes = (predictions > 0.5).astype(int)
        accuracy = np.mean(pred_classes == y) * 100
        
        results[activation] = {
            'accuracy': accuracy,
            'loss_history': nn.loss_history,
            'final_loss': nn.loss_history[-1]
        }
        
        print(f"  Final accuracy: {accuracy:.1f}%")
        print(f"  Final loss: {nn.loss_history[-1]:.6f}")
        
        # Plot loss curve
        plt.subplot(1, 3, i+1)
        plt.plot(nn.loss_history)
        plt.title(f'{activation.capitalize()} Activation')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # Print predictions
        print(f"  Predictions:")
        for j in range(len(X)):
            pred_val = predictions[j][0]
            pred_class = 1 if pred_val > 0.5 else 0
            actual = y[j][0]
            print(f"    {X[j]} -> {pred_val:.4f} -> {pred_class} (actual: {actual})")
        print()
    
    plt.suptitle('Loss Curves Comparison')
    plt.tight_layout()
    plt.show()
    
    # Summary
    print("=== Summary ===")
    for activation in activations:
        result = results[activation]
        print(f"{activation.capitalize()}: Accuracy={result['accuracy']:.1f}%, Final Loss={result['final_loss']:.6f}")

def visualize_activations():
    """
    Visualize different activation functions
    """
    x = np.linspace(-5, 5, 100)
    
    plt.figure(figsize=(15, 5))
    
    # Sigmoid
    plt.subplot(1, 3, 1)
    sigmoid_y = 1 / (1 + np.exp(-x))
    plt.plot(x, sigmoid_y, 'b-', linewidth=2)
    plt.title('Sigmoid Function')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.ylim(-0.1, 1.1)
    
    # ReLU
    plt.subplot(1, 3, 2)
    relu_y = np.maximum(0, x)
    plt.plot(x, relu_y, 'r-', linewidth=2)
    plt.title('ReLU Function')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    
    # Tanh
    plt.subplot(1, 3, 3)
    tanh_y = np.tanh(x)
    plt.plot(x, tanh_y, 'g-', linewidth=2)
    plt.title('Tanh Function')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.ylim(-1.1, 1.1)
    
    plt.suptitle('Activation Functions Comparison')
    plt.tight_layout()
    plt.show()
    
    print("\nKarakteristik Activation Functions:")
    print("1. Sigmoid: Output range [0,1], smooth gradient, prone to vanishing gradient")
    print("2. ReLU: Output range [0,∞), simple computation, dead neuron problem")
    print("3. Tanh: Output range [-1,1], zero-centered, better than sigmoid")

def main():
    print("Demo perbandingan activation functions pada neural network\n")
    
    # Visualize activation functions
    print("1. Visualisasi activation functions:")
    visualize_activations()
    
    print("\n" + "="*60 + "\n")
    
    # Compare performance
    print("2. Perbandingan performance:")
    compare_activations()

if __name__ == "__main__":
    main()
