## From-Scratch Neural Network for Spiral Classification


A pure NumPy implementation of a 2-layer neural network from scratch that classifies spiral datasets with high accuracy. No deep learning frameworks — just math, gradients, and Python.



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

## Custom Configuration
# Modify network architecture in neural_net.py
