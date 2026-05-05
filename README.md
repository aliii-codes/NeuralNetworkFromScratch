# 🧠 NeuralNetworkFromScratch
**Build and train a neural network from the ground up!**

[![GitHub stars](https://img.shields.io/github/stars/aliii-codes/NeuralNetworkFromScratch?style=for-the-badge)](https://github.com/aliii-codes/NeuralNetworkFromScratch/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/aliii-codes/NeuralNetworkFromScratch?style=for-the-badge)](https://github.com/aliii-codes/NeuralNetworkFromScratch/network)
[![GitHub issues](https://img.shields.io/github/issues/aliii-codes/NeuralNetworkFromScratch?style=for-the-badge)](https://github.com/aliii-codes/NeuralNetworkFromScratch/issues)
[![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)](LICENSE)

![Spiral Classification](spiral_classification.png)

## ✨ Highlights
- **Educational Focus**: Learn core neural network concepts through a simple, intuitive implementation.
- **From-Scratch Implementation**: No deep learning frameworks—just pure NumPy.
- **Non-Linear Classification**: Handles complex datasets like spirals with ease.

## 📚 Features

| Feature                     | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| **2-Layer Architecture**    | One hidden layer with ReLU and an output layer with softmax activation.     |
| **Kaiming Initialization**  | Weights initialized using Kaiming method for improved convergence.          |
| **L2 Regularization**       | Prevents overfitting by penalizing large weights.                           |
| **Decision Boundary Plot**  | Visualize how the network classifies data.                                  |
| **Training Metrics**        | Tracks accuracy and loss over epochs for performance monitoring.            |

## 🛠️ Tech Stack

| Category       | Technologies                                                                 |
|----------------|------------------------------------------------------------------------------|
| **Language**   | ![Python](https://img.shields.io/badge/python-3.8+-blue?style=flat-square)  |
| **Numerics**   | ![NumPy](https://img.shields.io/badge/numpy-1.20+-blue?style=flat-square)   |
| **Visualization** | ![Matplotlib](https://img.shields.io/badge/matplotlib-3.3+-green?style=flat-square) |

## 🚀 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/aliii-codes/NeuralNetworkFromScratch.git
   cd NeuralNetworkFromScratch
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃 Usage
Train the neural network on a synthetic spiral dataset:
```bash
python neural_net.py
```
This will generate a `spiral_classification.png` plot showing the decision boundary.

## 📁 Project Structure
```
NeuralNetworkFromScratch/
├── neural_net.py          # Core neural network implementation
├── utils.py               # Utility functions for dataset and plotting
└── test.ipynb             # Jupyter notebook for experimentation
```

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -m "Add new feature"`
4. Push to branch: `git push origin feature/new-feature`
5. Open a pull request

## 🐞 Bug Reports & Feature Requests
Found an issue? Have a great idea? [Open an issue](https://github.com/aliii-codes/NeuralNetworkFromScratch/issues/new)

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Acknowledgements**: Inspired by classic machine learning implementations and educational resources.
