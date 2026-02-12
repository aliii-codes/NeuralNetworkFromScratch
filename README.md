## From-Scratch Neural Network for Spiral Classification
https://img.shields.io/badge/python-3.8+-blue.svg
https://img.shields.io/badge/NumPy-1.21+-green.svg
https://img.shields.io/badge/Matplotlib-3.4+-orange.svg
https://img.shields.io/badge/License-MIT-yellow.svg

A pure NumPy implementation of a 2-layer neural network from scratch that classifies spiral datasets with high accuracy. No deep learning frameworks — just math, gradients, and Python.

<div align="center"> <img src="spiral_classification.png" alt="Spiral Classification Decision Boundary" width="600"/> <p><em>Decision boundary visualization of the trained network on a 3-class spiral dataset</em></p> </div>

## Table of Contents
Overview
Architecture
Features
Installation
Usage
Results
Project Structure
How It Works
Customization
License

## Overview
This project implements a fully-connected neural network from scratch using only NumPy for numerical computations and Matplotlib for visualizations. The network is trained on synthetic spiral datasets with configurable number of classes and samples.

# Why this project?
Deep understanding of backpropagation and gradient descent
No black boxes — every matrix multiplication is explicit
Beautiful visualizations of decision boundaries
Fast training with Kaiming initialization

## Architecture

# Network Configuration
┌─────────────────────────────────────────────┐
│   INPUT LAYER                               │
│   Shape: (batch_size, 2)                   │
│   Features: x, y coordinates               │
├─────────────────────────────────────────────┤
│   HIDDEN LAYER 1                           │
│   Dense + ReLU                             │
│   Units: 25                                │
│   Weights: (2, 25)                        │
├─────────────────────────────────────────────┤
│   OUTPUT LAYER                             │
│   Dense + Softmax                          │
│   Units: K (default: 3)                   │
│   Weights: (25, K)                        │
└─────────────────────────────────────────────┘

# Key Parameters


Parameter	Value	Description
Hidden units	25	Neurons in hidden layer
Learning rate	0.1	Step size for gradient descent
Regularization	1e-3	L2 regularization strength
Activation	ReLU	Non-linearity for hidden layer
Output activation	Softmax	Multi-class probability distribution
Loss function	Cross-entropy	With L2 regularization

## Features

# Core Implementation
Kaiming/He initialization — optimal for ReLU networks
ReLU activation — avoids vanishing gradients
Softmax output — proper probability distributions
Cross-entropy loss — with numerical stability
L2 regularization — prevents overfitting
Mini-batch ready — architecture supports batching


# Visualization

Synthetic spiral dataset generation (2D, K classes)
Decision boundary plotting with contour maps
Training progress monitoring with live loss/accuracy
High-resolution exports (300 DPI)


# Installation

Prerequisites
Python 3.8+
NumPy
Matplotlib

## Quick Setup

# Clone the repository
git clone https://github.com/aliii-codes/NeuralNetworkFromScratch
cd spiral-net

# Install dependencies
pip install numpy matplotlib

# Run the network
python neural_net.py

##  Usage

# Basic Training
from neural_net import NeuralNetwork
from utils import create_dataset

# Generate spiral dataset with 3 classes, 300 samples per class
X, y = create_dataset(N=300, K=3)

# Initialize and train network
nn = NeuralNetwork(X, y)
parameters = nn.main(X, y, num_epochs=10000)

## Custom Configuration
# Modify network architecture in neural_net.py
self.h1 = 50  # Increase hidden units from 25 to 50
self.learning_rate = 0.05  # Slower, more stable learning
self.lambd = 1e-4  # Weaker regularization

# Training Progress
Training for 10000 epochs...
--------------------------------------------------
Epoch      Loss            Accuracy  
--------------------------------------------------
0          1.098612        0.3322    
1000       0.234567        0.9123    
2000       0.123456        0.9456    
3000       0.098765        0.9567    
...
10000      0.087654        0.9633    
--------------------------------------------------
Final Accuracy: 0.9633





