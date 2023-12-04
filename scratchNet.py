#building a neural network from scratch solving the xor problem

import numpy as np
import json

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))

    def forward(self, inputs):
        # Forward-Pass
        self.hidden_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)
        return self.final_output

    def backward(self, inputs, targets, learning_rate):
        # Calculating the error and gradients
        error = targets - self.final_output
        delta_output = error * sigmoid_derivative(self.final_output)
        error_hidden = delta_output.dot(self.weights_hidden_output.T)
        delta_hidden = error_hidden * sigmoid_derivative(self.hidden_output)

        # Updating the weightings and biases
        self.weights_hidden_output += learning_rate * self.hidden_output.T.dot(delta_output)
        self.bias_output += learning_rate * np.sum(delta_output, axis=0, keepdims=True)
        self.weights_input_hidden += learning_rate * np.array([inputs]).T.dot(delta_hidden)
        self.bias_hidden += learning_rate * np.sum(delta_hidden, axis=0, keepdims=True)

        return delta_hidden


    def train(self, inputs, targets, epochs, learning_rate):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(inputs)):
                x = inputs[i]
                y = targets[i]

                # Forward-Pass
                output = self.forward(x)

                # Calculating Loss
                loss = np.mean((y - output) ** 2)
                total_loss += loss

                # Backward-Pass
                self.backward(x, y, learning_rate)

            average_loss = total_loss / len(inputs)
            print(f"Epoch {epoch + 1}, Loss: {average_loss:.4f}")

    def predict(self, inputs):
        results = []

        for i, x in enumerate(inputs):
            output = self.forward(x)
            prediction = float(output[0][0])
            result = {"Input": x.tolist(), "Prediction": prediction}
            results.append(result)

        # print prediction into a json table
        print("Testergebnisse:")
        for result in results:
            print(f"Input: {result['Input']}, Prediction: {result['Prediction']:.6f}")


# XOR-Problem
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

# train the neural network 
input_size = inputs.shape[1]
hidden_size = 2
output_size = targets.shape[1]

nn = NeuralNetwork(input_size, hidden_size, output_size)
nn.train(inputs, targets, epochs=20000, learning_rate=0.001)

# test the neural network
print("Testergebnisse:")
nn.predict(inputs)
