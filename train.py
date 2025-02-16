from NeuralNetwork import NeuralNetwork
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


digits = load_digits() # Load digits
X = digits.data / 16.0  # -> (0-1)
Y = digits.target.reshape(-1, 1)


encoder = OneHotEncoder(sparse=False) # One-Hot-Encoding

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

print("Traindata: ", len(X_train))
print("Validdata: ", len(X_test))

nn = NeuralNetwork([64, 32, 10], learning_rate=0.1)
nn.train(X_train, Y_train, epochs=5000)


for x, y in zip(X_test[:5], Y_test[:5]):
    pred = nn.predict(x).flatten()
    print(f"Actual: {y.argmax()}, Prediction: {pred.argmax()}")
