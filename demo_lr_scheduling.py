# Demo Learning Rate Scheduling
import numpy as np
import matplotlib.pyplot as plt
from neural_network import SimpleNeuralNetwork

def compare_learning_rates():
    """
    Membandingkan training dengan dan tanpa learning rate scheduling
    """
    print("=== Learning Rate Scheduling Demo ===")
    print("Membandingkan fixed LR vs decaying LR pada spiral classification")
    print()
    
    # Generate complex dataset (spiral-like)
    np.random.seed(42)
    n_samples = 100
    
    # Generate two interleaved spirals
    t = np.linspace(0, 4*np.pi, n_samples//2)
    r = t
    
    # First spiral
    x1 = r * np.cos(t) + 0.1 * np.random.randn(n_samples//2)
    y1 = r * np.sin(t) + 0.1 * np.random.randn(n_samples//2)
    
    # Second spiral (shifted)
    x2 = r * np.cos(t + np.pi) + 0.1 * np.random.randn(n_samples//2)
    y2 = r * np.sin(t + np.pi) + 0.1 * np.random.randn(n_samples//2)
    
    # Combine data
    X = np.vstack([
        np.column_stack([x1, y1]),
        np.column_stack([x2, y2])
    ])
    
    y = np.vstack([
        np.zeros((n_samples//2, 1)),
        np.ones((n_samples//2, 1))
    ])
    
    # Normalize data
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    print(f"Dataset shape: {X.shape}")
    print("Training dengan 2 strategi berbeda:\n")
    
    # Strategy 1: Fixed learning rate
    print("1. Fixed Learning Rate = 0.3")
    nn_fixed = SimpleNeuralNetwork(
        input_size=2,
        hidden_size=8,
        output_size=1,
        learning_rate=0.3,
        activation='tanh'
    )
    
    nn_fixed.train(X, y, epochs=1000, verbose=False)
    
    pred_fixed = nn_fixed.predict(X)
    acc_fixed = np.mean((pred_fixed > 0.5) == y) * 100
    print(f"   Final accuracy: {acc_fixed:.1f}%")
    print(f"   Final loss: {nn_fixed.loss_history[-1]:.6f}")
    
    # Strategy 2: Learning rate scheduling
    print("\n2. Learning Rate Scheduling (start=0.5, decay=0.995)")
    nn_scheduled = SimpleNeuralNetwork(
        input_size=2,
        hidden_size=8,
        output_size=1,
        learning_rate=0.5,
        activation='tanh'
    )
    
    nn_scheduled.train_with_lr_schedule(X, y, epochs=1000, lr_decay=0.995, verbose=False)
    
    pred_scheduled = nn_scheduled.predict(X)
    acc_scheduled = np.mean((pred_scheduled > 0.5) == y) * 100
    print(f"   Final accuracy: {acc_scheduled:.1f}%")
    print(f"   Final loss: {nn_scheduled.loss_history[-1]:.6f}")
    
    # Plot comparison
    plt.figure(figsize=(15, 10))
    
    # Loss curves comparison
    plt.subplot(2, 3, 1)
    plt.plot(nn_fixed.loss_history, label='Fixed LR=0.3', alpha=0.8)
    plt.plot(nn_scheduled.loss_history, label='Scheduled LR', alpha=0.8)
    plt.title('Loss Curves Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Training data
    plt.subplot(2, 3, 2)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='RdYlBu', alpha=0.7)
    plt.title('Training Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(scatter)
    
    # Decision boundaries
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    # Fixed LR decision boundary
    plt.subplot(2, 3, 4)
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z1 = nn_fixed.predict(mesh_points)
    Z1 = Z1.reshape(xx.shape)
    plt.contourf(xx, yy, Z1, levels=50, alpha=0.8, cmap='RdYlBu')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='RdYlBu', edgecolors='black')
    plt.title(f'Fixed LR (Acc: {acc_fixed:.1f}%)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Scheduled LR decision boundary
    plt.subplot(2, 3, 5)
    Z2 = nn_scheduled.predict(mesh_points)
    Z2 = Z2.reshape(xx.shape)
    plt.contourf(xx, yy, Z2, levels=50, alpha=0.8, cmap='RdYlBu')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='RdYlBu', edgecolors='black')
    plt.title(f'Scheduled LR (Acc: {acc_scheduled:.1f}%)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Learning rate schedule visualization
    plt.subplot(2, 3, 3)
    epochs = range(1000)
    lr_schedule = [0.5 * (0.995 ** epoch) for epoch in epochs]
    plt.plot(epochs, lr_schedule)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== Kesimpulan ===")
    print(f"Fixed LR:     Accuracy {acc_fixed:.1f}%, Final Loss {nn_fixed.loss_history[-1]:.6f}")
    print(f"Scheduled LR: Accuracy {acc_scheduled:.1f}%, Final Loss {nn_scheduled.loss_history[-1]:.6f}")
    
    if acc_scheduled > acc_fixed:
        print("✓ Learning rate scheduling memberikan hasil yang lebih baik!")
    else:
        print("→ Dalam kasus ini, fixed learning rate sudah cukup baik")

def demonstrate_lr_effect():
    """
    Demo effect learning rate yang berbeda pada training
    """
    print("\n" + "="*60)
    print("=== Effect Learning Rate pada Training ===")
    print("Testing learning rates: 0.01, 0.1, 1.0, 10.0 pada XOR problem")
    print()
    
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    learning_rates = [0.01, 0.1, 1.0, 10.0]
    results = {}
    
    plt.figure(figsize=(15, 5))
    
    for i, lr in enumerate(learning_rates):
        print(f"Learning Rate: {lr}")
        
        nn = SimpleNeuralNetwork(
            input_size=2,
            hidden_size=6,
            output_size=1,
            learning_rate=lr,
            activation='sigmoid'
        )
        
        nn.train(X, y, epochs=1000, verbose=False)
        
        predictions = nn.predict(X)
        accuracy = np.mean((predictions > 0.5) == y) * 100
        
        results[lr] = {
            'accuracy': accuracy,
            'loss_history': nn.loss_history,
            'final_loss': nn.loss_history[-1]
        }
        
        print(f"  Final accuracy: {accuracy:.1f}%")
        print(f"  Final loss: {nn.loss_history[-1]:.6f}")
        
        # Plot loss curve
        plt.subplot(1, 4, i+1)
        plt.plot(nn.loss_history)
        plt.title(f'LR = {lr}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        
        if lr == 10.0 and nn.loss_history[-1] > 1:
            plt.ylim(0, 2)  # Limit y-axis for very high learning rates
        
        print()
    
    plt.suptitle('Effect of Different Learning Rates')
    plt.tight_layout()
    plt.show()
    
    print("Interpretasi:")
    print("• LR terlalu kecil (0.01): Training lambat, mungkin tidak converge")
    print("• LR optimal (0.1-1.0): Balance antara speed dan stability")
    print("• LR terlalu besar (10.0): Training tidak stabil, loss bisa explode")

def main():
    print("Demo Learning Rate Scheduling dan Optimization")
    print("="*60)
    
    # Demo 1: Learning rate scheduling
    compare_learning_rates()
    
    # Demo 2: Effect of different learning rates
    demonstrate_lr_effect()

if __name__ == "__main__":
    main()
