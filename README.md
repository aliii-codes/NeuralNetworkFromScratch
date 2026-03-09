# NeuralNetworkFromScratch

## Project Title & Description
A simple 2-layered neural network implemented from scratch using NumPy. This project demonstrates the core concepts of neural networks, including forward propagation, backpropagation, and gradient descent. It's designed for educational purposes and can classify non-linear datasets like spirals.

## Features
- **2-Layer Neural Network**: One hidden layer with ReLU activation and an output layer with softmax activation.
- **Kaiming Initialization**: Weights initialized using Kaiming initialization for better convergence.
- **Regularization**: L2 regularization to prevent overfitting.
- **Visualization**: Plots decision boundaries and dataset for intuitive understanding.
- **Accuracy Tracking**: Monitors training accuracy and loss over epochs.

## Tech Stack
- **Python**
- **NumPy**: For numerical computations.
- **Matplotlib**: For data visualization.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/aliii-codes/NeuralNetworkFromScratch.git
   cd NeuralNetworkFromScratch
   ```
2. Install required dependencies:
   ```bash
   pip install numpy matplotlib
   ```

## Usage
1. Run the neural network on a synthetic spiral dataset:
   ```bash
   python neural_net.py
   ```
2. The script will train the network and save the decision boundary plot as `spiral_classification.png`.

## Project Structure
```
NeuralNetworkFromScratch/
│
├── neural_net.py          # Main neural network implementation
├── utils.py               # Utility functions for dataset creation and plotting
└── test.ipynb             # Jupyter notebook for testing and experimentation
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```
