# Demo Lengkap Neural Network
"""
Demo komprehensif yang menunjukkan semua fitur neural network:
1. Basic XOR classification
2. Multiple activation functions
3. Learning rate scheduling
4. Complex spiral classification
"""

import numpy as np
import matplotlib.pyplot as plt
from neural_network import SimpleNeuralNetwork

def banner(title):
    """Print banner untuk setiap section"""
    print("\n" + "="*60)
    print(f"🧠 {title}")
    print("="*60)

def demo_xor_basic():
    """Demo dasar XOR problem"""
    banner("DEMO 1: Basic XOR Classification")
    
    print("XOR adalah problem klasik yang membuktikan perlunya hidden layer")
    print("Problem: 0⊕0=0, 0⊕1=1, 1⊕0=1, 1⊕1=0")
    print()
    
    # Data XOR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Neural Network
    nn = SimpleNeuralNetwork(
        input_size=2, 
        hidden_size=4, 
        output_size=1, 
        learning_rate=1.0
    )
    
    print("🔧 Training neural network...")
    nn.train(X, y, epochs=1000, verbose=False)
    
    # Test
    predictions = nn.predict(X)
    accuracy = np.mean((predictions > 0.5) == y) * 100
    
    print(f"✅ Training selesai! Akurasi: {accuracy:.1f}%")
    print(f"📊 Final loss: {nn.loss_history[-1]:.6f}")
    
    print("\n📋 Hasil Prediksi:")
    for i in range(len(X)):
        pred_val = predictions[i][0]
        pred_class = 1 if pred_val > 0.5 else 0
        actual = y[i][0]
        status = "✓" if pred_class == actual else "✗"
        print(f"   {status} Input {X[i]} → Pred: {pred_val:.4f} → Class: {pred_class} (Target: {actual})")

def demo_activation_comparison():
    """Demo perbandingan activation functions"""
    banner("DEMO 2: Activation Functions Comparison")
    
    print("Membandingkan 3 activation functions pada XOR problem:")
    print("• Sigmoid: σ(x) = 1/(1+e^(-x))")
    print("• ReLU: f(x) = max(0,x)")
    print("• Tanh: f(x) = tanh(x)")
    print()
    
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    activations = ['sigmoid', 'relu', 'tanh']
    results = {}
    
    for activation in activations:
        print(f"🔧 Training dengan {activation.upper()}...")
        
        nn = SimpleNeuralNetwork(
            input_size=2,
            hidden_size=6,
            output_size=1,
            learning_rate=1.0,
            activation=activation
        )
        
        nn.train(X, y, epochs=1500, verbose=False)
        
        predictions = nn.predict(X)
        accuracy = np.mean((predictions > 0.5) == y) * 100
        
        results[activation] = {
            'accuracy': accuracy,
            'final_loss': nn.loss_history[-1],
            'epochs_to_converge': len([l for l in nn.loss_history if l > 0.01])
        }
        
        print(f"   ✅ Accuracy: {accuracy:.1f}% | Loss: {nn.loss_history[-1]:.6f}")
    
    print("\n📊 Summary Comparison:")
    print(f"{'Activation':<10} {'Accuracy':<10} {'Final Loss':<12} {'Convergence':<12}")
    print("-" * 50)
    
    best_activation = max(results.keys(), key=lambda k: results[k]['accuracy'])
    
    for activation, result in results.items():
        marker = "🏆" if activation == best_activation else "  "
        print(f"{marker} {activation:<8} {result['accuracy']:<10.1f}% "
              f"{result['final_loss']:<12.6f} {result['epochs_to_converge']:<12} epochs")

def demo_learning_rate_scheduling():
    """Demo learning rate scheduling"""
    banner("DEMO 3: Learning Rate Scheduling")
    
    print("Membandingkan fixed learning rate vs adaptive learning rate")
    print("Dataset: Complex spiral classification")
    print()
    
    # Generate spiral data
    np.random.seed(42)
    n_samples = 100
    
    t = np.linspace(0, 4*np.pi, n_samples//2)
    r = t
    
    # Spiral 1
    x1 = r * np.cos(t) + 0.2 * np.random.randn(n_samples//2)
    y1 = r * np.sin(t) + 0.2 * np.random.randn(n_samples//2)
    
    # Spiral 2
    x2 = r * np.cos(t + np.pi) + 0.2 * np.random.randn(n_samples//2)
    y2 = r * np.sin(t + np.pi) + 0.2 * np.random.randn(n_samples//2)
    
    X = np.vstack([
        np.column_stack([x1, y1]),
        np.column_stack([x2, y2])
    ])
    
    y = np.vstack([
        np.zeros((n_samples//2, 1)),
        np.ones((n_samples//2, 1))
    ])
    
    # Normalize
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    print(f"📊 Dataset: {X.shape[0]} samples, 2 features")
    print("🔧 Training dengan 2 strategi:")
    
    # Strategy 1: Fixed LR
    print("\n1️⃣ Fixed Learning Rate (LR = 0.3)")
    nn_fixed = SimpleNeuralNetwork(
        input_size=2, hidden_size=8, output_size=1, 
        learning_rate=0.3, activation='tanh'
    )
    nn_fixed.train(X, y, epochs=800, verbose=False)
    
    acc_fixed = np.mean((nn_fixed.predict(X) > 0.5) == y) * 100
    print(f"   ✅ Accuracy: {acc_fixed:.1f}% | Final Loss: {nn_fixed.loss_history[-1]:.6f}")
    
    # Strategy 2: LR Scheduling
    print("\n2️⃣ Learning Rate Scheduling (Start=0.5, Decay=0.996)")
    nn_scheduled = SimpleNeuralNetwork(
        input_size=2, hidden_size=8, output_size=1, 
        learning_rate=0.5, activation='tanh'
    )
    nn_scheduled.train_with_lr_schedule(X, y, epochs=800, lr_decay=0.996, verbose=False)
    
    acc_scheduled = np.mean((nn_scheduled.predict(X) > 0.5) == y) * 100
    print(f"   ✅ Accuracy: {acc_scheduled:.1f}% | Final Loss: {nn_scheduled.loss_history[-1]:.6f}")
    
    print(f"\n🏆 Winner: {'Scheduled LR' if acc_scheduled > acc_fixed else 'Fixed LR'} "
          f"({max(acc_scheduled, acc_fixed):.1f}% vs {min(acc_scheduled, acc_fixed):.1f}%)")

def demo_architecture_comparison():
    """Demo perbandingan arsitektur neural network"""
    banner("DEMO 4: Architecture Comparison")
    
    print("Membandingkan berbagai ukuran hidden layer pada XOR problem")
    print("Architecture tested: 2-2-1, 2-4-1, 2-8-1, 2-16-1")
    print()
    
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    hidden_sizes = [2, 4, 8, 16]
    results = {}
    
    for hidden_size in hidden_sizes:
        print(f"🔧 Testing architecture 2-{hidden_size}-1...")
        
        nn = SimpleNeuralNetwork(
            input_size=2,
            hidden_size=hidden_size,
            output_size=1,
            learning_rate=1.0,
            activation='sigmoid'
        )
        
        nn.train(X, y, epochs=1000, verbose=False)
        
        predictions = nn.predict(X)
        accuracy = np.mean((predictions > 0.5) == y) * 100
        
        results[hidden_size] = {
            'accuracy': accuracy,
            'final_loss': nn.loss_history[-1],
            'parameters': 2*hidden_size + hidden_size + hidden_size*1 + 1  # W1 + b1 + W2 + b2
        }
        
        print(f"   ✅ Accuracy: {accuracy:.1f}% | Parameters: {results[hidden_size]['parameters']}")
    
    print("\n📊 Architecture Analysis:")
    print(f"{'Architecture':<12} {'Accuracy':<10} {'Parameters':<12} {'Efficiency'}")
    print("-" * 55)
    
    for hidden_size, result in results.items():
        efficiency = result['accuracy'] / result['parameters'] * 100
        print(f"2-{hidden_size:<2}-1       {result['accuracy']:<10.1f}% "
              f"{result['parameters']:<12} {efficiency:.2f}")

def demo_visualization():
    """Create visualization untuk neural network"""
    banner("DEMO 5: Neural Network Visualization")
    
    print("Visualisasi decision boundary untuk different scenarios")
    print()
    
    # XOR visualization
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([[0], [1], [1], [0]])
    
    nn_xor = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=1.0)
    nn_xor.train(X_xor, y_xor, epochs=1000, verbose=False)
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: XOR decision boundary
    plt.subplot(1, 3, 1)
    
    # Create mesh
    h = 0.02
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = nn_xor.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
    colors = ['red', 'blue']
    for i in range(len(X_xor)):
        plt.scatter(X_xor[i, 0], X_xor[i, 1], 
                   c=colors[int(y_xor[i])], s=100, edgecolors='black', linewidth=2)
    plt.title('XOR Decision Boundary')
    plt.xlabel('x1')
    plt.ylabel('x2')
    
    # Plot 2: Loss curves comparison
    plt.subplot(1, 3, 2)
    activations = ['sigmoid', 'relu', 'tanh']
    colors = ['blue', 'red', 'green']
    
    for i, activation in enumerate(activations):
        nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1, 
                               learning_rate=1.0, activation=activation)
        nn.train(X_xor, y_xor, epochs=500, verbose=False)
        plt.plot(nn.loss_history, label=activation.capitalize(), color=colors[i], alpha=0.8)
    
    plt.title('Loss Curves by Activation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Learning rate effect
    plt.subplot(1, 3, 3)
    epochs = range(500)
    lr_fixed = [0.5] * 500
    lr_decay = [0.5 * (0.995 ** epoch) for epoch in epochs]
    
    plt.plot(epochs, lr_fixed, label='Fixed LR', linewidth=2)
    plt.plot(epochs, lr_decay, label='Decaying LR', linewidth=2)
    plt.title('Learning Rate Scheduling')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("📊 Visualizations generated successfully!")

def main():
    """Main demo function"""
    print("🚀 NEURAL NETWORK COMPLETE DEMO")
    print("Implementasi lengkap neural network dengan backpropagation")
    print("Developed step-by-step dengan Git commits")
    
    # Run all demos
    demo_xor_basic()
    demo_activation_comparison()
    demo_learning_rate_scheduling()
    demo_architecture_comparison()
    demo_visualization()
    
    # Final summary
    banner("SUMMARY & CONCLUSIONS")
    print("🎯 Key Learning Points:")
    print("1. ✅ XOR problem membutuhkan hidden layer (non-linear classification)")
    print("2. ✅ ReLU often performs better than Sigmoid (faster convergence)")
    print("3. ✅ Learning rate scheduling dapat improve training stability")
    print("4. ✅ Architecture size vs performance tradeoff")
    print("5. ✅ Backpropagation successfully implements gradient descent")
    print()
    print("📚 Neural Network Components Implemented:")
    print("   • Forward Propagation dengan matrix operations")
    print("   • Backpropagation menggunakan chain rule")
    print("   • Multiple activation functions (Sigmoid, ReLU, Tanh)")
    print("   • Learning rate scheduling untuk optimization")
    print("   • Xavier weight initialization")
    print("   • MSE loss function")
    print()
    print("🔬 Tested On:")
    print("   • XOR Classification (classic test)")
    print("   • Spiral Classification (complex non-linear)")
    print("   • Various architectures and hyperparameters")
    print()
    print("🏆 Project completed successfully dengan commit history yang clear!")

if __name__ == "__main__":
    main()
