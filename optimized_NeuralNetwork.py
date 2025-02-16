import numpy as np

"""
Optimized Neural Network Implementation in NumPy
===================================================

This script implements a fully connected neural network with various activation functions.
Optimizations have been made for improved training efficiency, including:
- Mini-Batch Training
- Vectorized Forward and Backpropagation
- Optimized Weight Initialization (He Initialization)
- Use of np.float32 for Memory Optimization
- L2 loss

Author: Abdullah
"""

class NeuralNetwork:
    def __init__(self, layer_sizes, activations, learning_rate=0.1):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.activations = [self.get_activation(act) for act in activations]
        self.activation_derivatives = [self.get_activation_derivative(act) for act in activations]

        self.weights = [np.random.randn(n_out, n_in).astype(np.float32) * np.sqrt(2.0 / n_in) 
                        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.zeros((n, 1), dtype=np.float32) for n in layer_sizes[1:]]

    def get_activation(self, name):
        return {
            "sigmoid": self.sigmoid,
            "relu": self.relu,
            "tanh": self.tanh,
            "identity": self.identity,
            "softmax": self.softmax
        }.get(name, self.sigmoid)

    def get_activation_derivative(self, name):
        return {
            "sigmoid": self.sigmoid_derivative,
            "relu": self.relu_derivative,
            "tanh": self.tanh_derivative,
            "identity": self.identity_derivative,
            "softmax": self.softmax_derivative
        }.get(name, self.sigmoid_derivative)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(np.float32)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.square(np.tanh(x))

    def identity(self, x):
        return x

    def identity_derivative(self, x):
        return np.ones_like(x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))  # Stabilized softmax
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

    def softmax_derivative(self, x):
        s = self.softmax(x)
        return s * (1 - s)

    def forward(self, x):
        activations = [x]
        for w, b, act in zip(self.weights, self.biases, self.activations):
            x = act(np.dot(w, x) + b)
            activations.append(x)
        return activations

    def backward(self, activations, y):
        deltas = [(activations[-1] - y) * self.activation_derivatives[-1](activations[-1])]
        for i in range(len(self.weights) - 1, 0, -1):
            deltas.insert(0, np.dot(self.weights[i].T, deltas[0]) * self.activation_derivatives[i - 1](activations[i]))
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * np.dot(deltas[i], activations[i].T)
            self.biases[i] -= self.learning_rate * np.mean(deltas[i], axis=1, keepdims=True)

    def train(self, X, Y, epochs=100, batch_size=8):
        X, Y = np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)
        num_samples = X.shape[0]

        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            X_shuffled, Y_shuffled = X[indices], Y[indices]

            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size].T
                Y_batch = Y_shuffled[i:i + batch_size].T

                activations = self.forward(X_batch)
                self.backward(activations, Y_batch)

            if epoch % 1 == 0:
                loss = np.mean((activations[-1] - Y_batch) ** 2)
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

    def predict(self, x):
        x = np.array(x, dtype=np.float32).reshape(-1, 1)
        return self.forward(x)[-1]
