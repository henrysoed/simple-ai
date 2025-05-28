# Demo Neural Network dengan XOR Problem
import numpy as np
from neural_network import SimpleNeuralNetwork

def main():
    print("=== XOR Problem Demo ===")
    print("Neural Network akan belajar XOR function:")
    print("Input (0,0) -> Output 0")
    print("Input (0,1) -> Output 1") 
    print("Input (1,0) -> Output 1")
    print("Input (1,1) -> Output 0")
    print()
    
    # Dataset XOR
    # Input: 2 features (x1, x2)
    X = np.array([
        [0, 0],
        [0, 1], 
        [1, 0],
        [1, 1]
    ])
    
    # Target output untuk XOR
    y = np.array([
        [0],  # 0 XOR 0 = 0
        [1],  # 0 XOR 1 = 1
        [1],  # 1 XOR 0 = 1
        [0]   # 1 XOR 1 = 0
    ])
    
    print("Training Data:")
    for i in range(len(X)):
        print(f"Input: {X[i]} -> Target: {y[i][0]}")
    print()
    
    # Buat neural network
    # 2 input neurons, 4 hidden neurons, 1 output neuron
    nn = SimpleNeuralNetwork(
        input_size=2, 
        hidden_size=4, 
        output_size=1, 
        learning_rate=1.0  # Learning rate tinggi untuk dataset kecil
    )
    
    print("Mulai training...")
    print("Architecture: 2 -> 4 -> 1 (input -> hidden -> output)")
    print()
    
    # Training
    nn.train(X, y, epochs=2000, verbose=True)
    
    print("\n=== Testing Hasil Training ===")
    predictions = nn.predict(X)
    
    print("Hasil prediksi:")
    for i in range(len(X)):
        pred_val = predictions[i][0]
        pred_class = 1 if pred_val > 0.5 else 0
        actual = y[i][0]
        
        print(f"Input: {X[i]} -> Prediction: {pred_val:.4f} -> Class: {pred_class} -> Actual: {actual}")
        
    # Hitung akurasi
    pred_classes = (predictions > 0.5).astype(int)
    accuracy = np.mean(pred_classes == y) * 100
    print(f"\nAkurasi: {accuracy:.1f}%")
    
    # Plot loss curve
    nn.plot_loss()

if __name__ == "__main__":
    main()
