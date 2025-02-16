import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.1):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = [np.random.randn(n_out, n_in) * 0.1 
                        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.random.randn(n, 1) * 0.1 for n in layer_sizes[1:]]
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, x):
        activations = [x]
        for w, b in zip(self.weights, self.biases):
            x = self.sigmoid(np.dot(w, x) + b)
            activations.append(x)
        return activations
    
    def backward(self, activations, y):
        deltas = [(activations[-1] - y) * self.sigmoid_derivative(activations[-1])]
        
        for i in range(len(self.weights) - 1, 0, -1):
            deltas.insert(0, np.dot(self.weights[i].T, deltas[0]) * self.sigmoid_derivative(activations[i]))
        
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * np.dot(deltas[i], activations[i].T)
            self.biases[i] -= self.learning_rate * deltas[i]
    
    def train(self, X, Y, epochs=10000):
        for epoch in range(epochs):
            for x, y in zip(X, Y):
                x = x.reshape(-1, 1)
                y = y.reshape(-1, 1)
                activations = self.forward(x)
                self.backward(activations, y)
            if epoch % 1000 == 0:
                loss = np.mean((activations[-1] - y) ** 2)
                print(f"Epoch {epoch}: Loss = {loss:.4f}")
    
    def predict(self, x):
        x = x.reshape(-1, 1)
        return self.forward(x)[-1]

