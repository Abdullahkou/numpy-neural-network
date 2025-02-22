import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes: list[int], activations: list[str], learning_rate: float = 0.1):
        """
            A fully connected neural network implemented in NumPy.
            
            Parameters:
            - layer_sizes: list of integers specifying the number of neurons per layer.
            - learning_rate: float, determines the step size during gradient descent.
            - activations: list of strings specifying the activation functions per layer.
            - Loss: L2 loss
            
            Available activations: 'sigmoid', 'relu', 'tanh', 'identity', 'softmax'
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.activations = [self.get_activation(act) for act in activations]
        self.activation_derivatives = [self.get_activation_derivative(act) for act in activations]
        self.weights = [np.random.randn(n_out, n_in) * 0.1 for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.random.randn(n, 1) * 0.1 for n in layer_sizes[1:]]

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
        return (x > 0).astype(float)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

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
            self.biases[i] -= self.learning_rate * deltas[i]

    def train(self, X, Y, epochs=100):
        for epoch in range(epochs):
            for x, y in zip(X, Y):
                x = x.reshape(-1, 1)
                y = y.reshape(-1, 1)
                activations = self.forward(x)
                self.backward(activations, y)
            if epoch % 1 == 0:
                loss = np.mean((activations[-1] - y) ** 2)
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

    def predict(self, x):
        x = x.reshape(-1, 1)
        return self.forward(x)[-1]
