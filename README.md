# Simple Neural Network dengan Backpropagation

Implementasi neural network sederhana dari scratch menggunakan Python dan NumPy. Neural network ini menggunakan backpropagation dan gradient descent untuk learning.

## Fitur Utama

- **Forward Propagation**: Perhitungan output dari input melalui hidden layer
- **Backpropagation**: Menghitung gradients menggunakan chain rule
- **Gradient Descent**: Update weights untuk meminimalkan loss
- **Multiple Activation Functions**: Sigmoid, ReLU, Tanh
- **Learning Rate Scheduling**: Adaptive learning rate untuk better convergence
- **Xavier Initialization**: Inisialisasi weights yang baik untuk training

## Struktur Neural Network

```
Input Layer -> Hidden Layer -> Output Layer
     2              4-10           1
```

## Komponen Penting

### 1. Forward Propagation
```python
# Hidden layer: z1 = X * W1 + b1, a1 = activation(z1)  
# Output layer: z2 = a1 * W2 + b2, a2 = sigmoid(z2)
```

### 2. Backpropagation
```python
# Hitung gradients menggunakan chain rule
# dL/dW2 = a1.T * dL/dz2
# dL/dW1 = X.T * dL/dz1
```

### 3. Gradient Descent
```python
# Update weights: W = W - learning_rate * gradient
```

### 4. Activation Functions
- **Sigmoid**: σ(x) = 1/(1+e^(-x)) - Output [0,1]
- **ReLU**: f(x) = max(0,x) - Output [0,∞)
- **Tanh**: f(x) = tanh(x) - Output [-1,1]

## Demo yang Tersedia

1. **XOR Problem** (`demo_xor.py`): Problem klasik yang tidak bisa diselesaikan single perceptron
2. **Spiral Classification** (`demo_spiral.py`): Binary classification pada data non-linear
3. **Activation Functions Comparison** (`demo_activations.py`): Perbandingan performa sigmoid vs ReLU vs tanh
4. **Learning Rate Scheduling** (`demo_lr_scheduling.py`): Optimasi training dengan adaptive learning rate

## Cara Penggunaan

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Jalankan demo XOR:
```bash
python demo_xor.py
```

3. Jalankan demo spiral classification:
```bash
python demo_spiral.py
```

4. Jalankan demo perbandingan activation functions:
```bash
python demo_activations.py
```

5. Jalankan demo learning rate scheduling:
```bash
python demo_lr_scheduling.py
```

## Penjelasan Algoritma

### Backpropagation Algorithm
1. **Forward pass**: Hitung output dari input
2. **Compute loss**: Hitung error antara prediction dan target
3. **Backward pass**: Hitung gradients menggunakan chain rule
4. **Update weights**: Gunakan gradient descent untuk update parameters

### Gradient Descent
- Learning rate mengontrol seberapa besar step update
- Terlalu besar: bisa overshoot minimum
- Terlalu kecil: training lambat
- Learning rate scheduling: mulai besar, lalu decay untuk fine-tuning

### Kenapa Butuh Hidden Layer?
- Single perceptron hanya bisa linear classification
- Hidden layer memungkinkan non-linear decision boundary
- Semakin banyak hidden neurons, semakin kompleks pattern yang bisa dipelajari

### Activation Functions Comparison
- **Sigmoid**: Smooth gradients, prone to vanishing gradient
- **ReLU**: Fast computation, bisa dead neurons
- **Tanh**: Zero-centered, better gradient flow than sigmoid

## Development Steps (Git Commits)

Project ini dikembangkan secara step-by-step dengan commit history yang jelas:

1. **Basic Neural Network**: Implementasi dasar forward/backpropagation
2. **XOR Demo**: Test klasik untuk validasi network
3. **Spiral Classification**: Demo pada data non-linear kompleks
4. **Documentation**: README dan requirements
5. **Multiple Activations**: Support untuk ReLU, Tanh selain Sigmoid
6. **Learning Rate Scheduling**: Optimasi training dengan adaptive LR

## Performance Benchmarks

Pada XOR Problem (4 data points):
- **Sigmoid**: 100% accuracy, converge dalam ~1000 epochs
- **ReLU**: 100% accuracy, converge lebih cepat
- **Tanh**: 100% accuracy, gradients yang bagus

Pada Spiral Classification (200 data points):
- **Baseline**: ~74% accuracy dengan proper hyperparameters
- **Learning Rate Scheduling**: Dapat improve convergence speed
