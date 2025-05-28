# Simple Neural Network dengan Backpropagation

Implementasi neural network sederhana dari scratch menggunakan Python dan NumPy. Neural network ini menggunakan backpropagation dan gradient descent untuk learning.

## Fitur Utama

- **Forward Propagation**: Perhitungan output dari input melalui hidden layer
- **Backpropagation**: Menghitung gradients menggunakan chain rule
- **Gradient Descent**: Update weights untuk meminimalkan loss
- **Sigmoid Activation**: Activation function untuk non-linearity
- **Xavier Initialization**: Inisialisasi weights yang baik untuk training

## Struktur Neural Network

```
Input Layer -> Hidden Layer -> Output Layer
     2              4              1
```

## Komponen Penting

### 1. Forward Propagation
```python
# Hidden layer: z1 = X * W1 + b1, a1 = sigmoid(z1)  
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

## Demo yang Tersedia

1. **XOR Problem** (`demo_xor.py`): Problem klasik yang tidak bisa diselesaikan single perceptron
2. **Spiral Classification** (`demo_spiral.py`): Binary classification pada data non-linear

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

### Kenapa Butuh Hidden Layer?
- Single perceptron hanya bisa linear classification
- Hidden layer memungkinkan non-linear decision boundary
- Semakin banyak hidden neurons, semakin kompleks pattern yang bisa dipelajari
