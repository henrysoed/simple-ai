# Neural Network Demo - Vercel Deployment

🧠 **Interactive Neural Network Demo** deployed on Vercel

## 🚀 Live Demo
Visit: [Your Vercel URL will be here after deployment]

## 📋 Features
- Interactive XOR problem training
- Real-time neural network visualization  
- Multiple activation function comparison (Sigmoid, ReLU, Tanh)
- Responsive web interface
- Serverless deployment on Vercel

## 🛠️ Local Development

### Prerequisites
- Python 3.8+
- pip

### Setup
1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run locally:
   ```bash
   python api/index.py
   ```
4. Open http://localhost:5000

## 📦 Deployment to Vercel

### One-Click Deploy
[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/yourusername/neural-network-demo)

### Manual Deploy
1. Install Vercel CLI:
   ```bash
   npm i -g vercel
   ```

2. Login to Vercel:
   ```bash
   vercel login
   ```

3. Deploy:
   ```bash
   vercel
   ```

4. Follow the prompts and your app will be live!

## 🏗️ Project Structure
```
├── api/
│   ├── index.py          # Main Flask app (serverless function)
│   └── neural_network.py # Neural network implementation
├── templates/
│   └── index.html        # Web interface
├── vercel.json           # Vercel configuration
├── requirements.txt      # Python dependencies
└── .vercelignore        # Files to ignore during deployment
```

## 🧪 API Endpoints
- `GET /` - Main web interface
- `POST /api/train_demo` - Train neural network on XOR problem
- `POST /api/test_activations` - Compare activation functions
- `GET /api/info` - Get API information

## 🎯 What You Can Do
1. **Train XOR Network**: Watch a neural network learn the XOR function
2. **Adjust Parameters**: Change hidden layer size, learning rate, epochs
3. **Compare Functions**: See how Sigmoid, ReLU, and Tanh perform
4. **Real-time Results**: Get instant feedback on network performance

## 🔧 Tech Stack
- **Backend**: Python + Flask (Serverless)
- **Frontend**: HTML + JavaScript (Vanilla)
- **ML Library**: NumPy (Custom neural network implementation)
- **Deployment**: Vercel
- **Styling**: CSS3 with modern gradients

## 📊 Neural Network Details
- **Architecture**: Input → Hidden → Output (fully connected)
- **Training**: Backpropagation with gradient descent
- **Problem**: XOR function (non-linearly separable)
- **Activations**: Sigmoid, ReLU, Tanh support

## 🤝 Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch  
5. Open a Pull Request

## 📄 License
MIT License - feel free to use this for educational purposes!

---
Built with ❤️ for learning neural networks
