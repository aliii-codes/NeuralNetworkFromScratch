""""
Simple 2 layered Neural Network from Scratch
"""

import time
import numpy as np
from utils import create_dataset, plot_counter


class NeuralNetwork:
    def __init__(self, X, y):
        self.m, self.n = X.shape
        
        self.lambd = 1e-3
        self.learning_rate = 0.1

        # size of our first hidden-layer and second hidden layer 
        self.h1 = 25
        self.h2 = len(np.unique(y))

    def init_kaiming_weights(self, l0, l1):
        w = np.random.randn(l0, l1) * np.sqrt(2.0 / l0)
        b = np.zeros((1, l1))
        return w, b

    def forward_prop(self, X, parameters):
        W2 = parameters["W2"]
        W1 = parameters["W1"]
        b2 = parameters["b2"]
        b1 = parameters["b1"]

        # forward prop
        a0 = X
        z1 = np.dot(a0, W1) + b1

        # relu
        a1 = np.maximum(0, z1)
        z2 = np.dot(a1, W2) + b2

        # softmax
        exp_scores = np.exp(z2 - np.max(z2, axis=1, keepdims=True))  # numerical stability
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        cache = {"a0": X, "probs": probs, "a1": a1}
        return cache, probs

    def compute_cost(self, y, probs, parameters):
        W2 = parameters["W2"]
        W1 = parameters["W1"]

        y = y.astype(int)
        data_loss = np.sum(-np.log(probs[np.arange(self.m), y] + 1e-8)) / self.m  # add epsilon
        reg_loss = 0.5 * self.lambd * (np.sum(W1 * W1) + np.sum(W2 * W2))
        
        total_cost = data_loss + reg_loss
        return total_cost

    def compute_accuracy(self, X, y, parameters):
        _, probs = self.forward_prop(X, parameters)
        predictions = np.argmax(probs, axis=1)
        accuracy = np.mean(predictions == y)
        return accuracy

    def back_prop(self, cache, parameters, y):
        W2 = parameters["W2"]
        W1 = parameters["W1"]
        b2 = parameters["b2"]
        b1 = parameters["b1"]
        
        a0 = cache["a0"]
        a1 = cache["a1"]
        probs = cache["probs"]

        dz2 = probs.copy()
        dz2[np.arange(self.m), y] -= 1
        dz2 /= self.m

        dW2 = np.dot(a1.T, dz2) + self.lambd * W2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        dz1 = np.dot(dz2, W2.T)
        dz1 = dz1 * (a1 > 0)

        dW1 = np.dot(a0.T, dz1) + self.lambd * W1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return {"dW1": dW1, "dW2": dW2, "db1": db1, "db2": db2}

    def update_parameters(self, parameters, grads):
        learning_rate = self.learning_rate
        
        parameters["W2"] -= learning_rate * grads["dW2"]
        parameters["W1"] -= learning_rate * grads["dW1"]
        parameters["b2"] -= learning_rate * grads["db2"]
        parameters["b1"] -= learning_rate * grads["db1"]
        
        return parameters

    def main(self, X, y, num_epochs=10000):
        W1, b1 = self.init_kaiming_weights(self.n, self.h1)
        W2, b2 = self.init_kaiming_weights(self.h1, self.h2)
        
        parameters = {"W1": W1, "W2": W2, "b1": b1, "b2": b2}
        
        print(f"Training for {num_epochs} epochs...")
        time.sleep(1)
        print("-" * 50)
        time.sleep(1)
        print(f"{'Epoch':<10} {'Loss':<15} {'Accuracy':<10}")
        time.sleep(1)
        print("-" * 50)
        
        for epoch in range(num_epochs + 1):
            # Forward propagation
            cache, probs = self.forward_prop(X, parameters)
            
            # Calculate loss
            loss = self.compute_cost(y, probs, parameters)
            
            # Calculate accuracy
            accuracy = self.compute_accuracy(X, y, parameters)
            
            # Print progress
            if epoch % (num_epochs // 10) == 0 or epoch == num_epochs:
                print(f"{epoch:<10} {loss:<15.6f} {accuracy:<10.4f}")
            
            # Backward propagation and update
            grads = self.back_prop(cache, parameters, y)
            parameters = self.update_parameters(parameters, grads)
        
        print("-" * 50)
        final_acc = self.compute_accuracy(X, y, parameters)
        print(f"Final Accuracy: {final_acc:.4f}")
        
        return parameters


if __name__ == "__main__":
    # Generate dataset
    X, y = create_dataset(300, K=3)
    y = y.astype(int)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Unique classes: {np.unique(y)}")
    print("-" * 50)
    
    # Train network
    NN = NeuralNetwork(X, y)
    trained_parameters = NN.main(X, y, num_epochs=10000)
    
    # Plot the decision boundary
    plot_counter(X, y, NN, trained_parameters)
    
    # Save figure
    import matplotlib.pyplot as plt
    plt.savefig("spiral_classification.png", dpi=300, bbox_inches='tight')
    print("Decision boundary saved as 'spiral_classification.png'")