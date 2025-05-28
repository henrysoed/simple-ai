# Demo Neural Network dengan Binary Classification
import numpy as np
import matplotlib.pyplot as plt
from neural_network import SimpleNeuralNetwork

def generate_spiral_data(n_samples=200, noise=0.1):
    """
    Generate spiral dataset untuk binary classification
    Dataset ini tidak linear separable, cocok untuk test neural network
    """
    np.random.seed(42)  # untuk reproducible results
    
    # Generate spiral data
    n_per_class = n_samples // 2
    
    # Class 0 (spiral pertama)
    t0 = np.linspace(0, 2*np.pi, n_per_class)
    r0 = t0 + noise * np.random.randn(n_per_class)
    x0 = r0 * np.cos(t0)
    y0 = r0 * np.sin(t0)
    
    # Class 1 (spiral kedua - shifted)
    t1 = np.linspace(0, 2*np.pi, n_per_class) + np.pi
    r1 = t1 + noise * np.random.randn(n_per_class) 
    x1 = r1 * np.cos(t1)
    y1 = r1 * np.sin(t1)
    
    # Combine data
    X = np.vstack([
        np.column_stack([x0, y0]),
        np.column_stack([x1, y1])
    ])
    
    y = np.vstack([
        np.zeros((n_per_class, 1)),
        np.ones((n_per_class, 1))
    ])
    
    # Shuffle data
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    return X, y

def plot_decision_boundary(X, y, nn, title="Decision Boundary"):
    """
    Plot decision boundary dari neural network
    """
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Data points
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='RdYlBu', alpha=0.7)
    plt.title('Training Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(scatter)
    
    # Plot 2: Decision boundary
    plt.subplot(1, 2, 2)
    
    # Create mesh untuk decision boundary
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    # Predict pada setiap point di mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = nn.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='RdYlBu', edgecolors='black')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(scatter)
    
    plt.tight_layout()
    plt.show()

def main():
    print("=== Binary Classification Demo ===")
    print("Dataset: Spiral data (non-linear)")
    print("Neural network akan belajar memisahkan 2 spiral")
    print()
    
    # Generate spiral dataset
    X, y = generate_spiral_data(n_samples=200, noise=0.2)
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {np.sum(y==0)} samples class 0, {np.sum(y==1)} samples class 1")
    print()
    
    # Normalize data untuk training yang lebih stabil
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_normalized = (X - X_mean) / X_std
    
    # Buat neural network
    nn = SimpleNeuralNetwork(
        input_size=2,
        hidden_size=10,  # Lebih banyak hidden neurons untuk data yang kompleks
        output_size=1,
        learning_rate=0.5
    )
    
    print("Neural Network Architecture: 2 -> 10 -> 1")
    print("Training dimulai...")
    print()
    
    # Training
    nn.train(X_normalized, y, epochs=3000, verbose=True)
    
    print("\n=== Evaluasi Model ===")
    predictions = nn.predict(X_normalized)
    pred_classes = (predictions > 0.5).astype(int)
    
    # Hitung akurasi
    accuracy = np.mean(pred_classes == y) * 100
    print(f"Training Accuracy: {accuracy:.1f}%")
    
    # Hitung confusion matrix sederhana
    tp = np.sum((pred_classes == 1) & (y == 1))
    tn = np.sum((pred_classes == 0) & (y == 0))
    fp = np.sum((pred_classes == 1) & (y == 0))
    fn = np.sum((pred_classes == 0) & (y == 1))
    
    print(f"True Positives: {tp}")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    
    # Plot results
    print("\nPlotting results...")
    plot_decision_boundary(X_normalized, y, nn, "Neural Network Decision Boundary")
    
    # Plot loss curve
    nn.plot_loss()

if __name__ == "__main__":
    main()
