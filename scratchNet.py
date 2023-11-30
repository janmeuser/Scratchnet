#Building a neural network from scratch

import numpy as np
import time
import matplotlib.pyplot as plt

class DifferentiableFunction:
    def derivative(self, net_input):
        pass

    def __call__(self, net_input):
        pass

class Sigmoid(DifferentiableFunction):
    def derivative(self, net_input):
        return self(net_input) * (1 - self(net_input))

    def __call__(self, net_input):
        return 1 / (1 + np.exp(-net_input))

class SquaredError:
    def __call__(self, predictions, targets):
        return np.mean((predictions - targets) ** 2)

    def gradient(self, predictions, targets):
        return 2 * (predictions - targets) / len(predictions)

class DenseLayer:
    def __init__(self, neuron_count, activation=None):
        self.neuron_count = neuron_count
        self.activation_func = activation or Sigmoid()
        self.weights = None
        self.biases = None
        self.layer_inputs = None
        self.activation_vec = None
        self.weight_gradients = None
        self.bias_gradients = None

    def initialize_parameters(self, input_size):
        self.weights = np.random.randn(self.neuron_count, input_size)
        self.biases = np.random.randn(self.neuron_count, 1)

    def compute_cost_gradients(self, targets, cost_func):
        cost = cost_func(self.activation_vec, targets)
        delta = cost_func.gradient(self.activation_vec, targets)
        self.bias_gradients = delta.mean(axis=0).reshape(-1, 1)
        self.weight_gradients = np.dot(delta, self.layer_inputs.T)
        return delta.dot(self.weights)

    def feed_backwards(self, prev_input_gradients):
        return prev_input_gradients.dot(self.weights)

    def feed_forward_layer(self, input_activations):
        self.layer_inputs = np.dot(self.weights, input_activations) + self.biases
        self.activation_vec = self.activation_func(self.layer_inputs)
        return self.activation_vec

class ScratchNet:
    def __init__(self, layers):
        self.learning_rate = 0.5
        self.cost_func = SquaredError()
        self.layers = layers
        for index, layer in enumerate(self.layers):
            input_size = 2 if index == 0 else self.layers[index - 1].neuron_count
            layer.initialize_parameters(input_size)

    def _update_parameters(self, input_samples):
        weight_gradients = [np.zeros(layer.weights.shape) for layer in self.layers]
        bias_gradients = [np.zeros(layer.biases.shape) for layer in self.layers]

        for sample in input_samples:
            sample_weight_gradients, sample_bias_gradients = self._backpropagate(sample)
            weight_gradients = np.add(weight_gradients, sample_weight_gradients)
            bias_gradients = np.add(bias_gradients, sample_bias_gradients)

        for layer, layer_weight_gradients, layer_bias_gradients in zip(self.layers, weight_gradients, bias_gradients):
            layer.weights -= self.learning_rate * layer_weight_gradients / len(input_samples)
            layer.biases -= self.learning_rate * layer_bias_gradients / len(input_samples)

    def _backpropagate(self, training_sample):
        train_input, train_output = training_sample
        self._feed_forward(train_input)
        gradients = self.layers[-1].compute_cost_gradients(train_output, cost_func=self.cost_func)

        for layer in reversed(self.layers[1:]):
            gradients = layer.feed_backwards(gradients)

        weight_gradients = [layer.weight_gradients for layer in self.layers]
        bias_gradients = [layer.bias_gradients for layer in self.layers]
        return weight_gradients, bias_gradients

    def _feed_forward(self, input_sample):
        for layer in self.layers:
            input_sample = layer.feed_forward_layer(input_sample)
        return input_sample

    def fit(self, train_inputs, train_labels, epochs=1):
        training_data = list(zip(train_inputs, train_labels))
        losses = []
        for epoch in range(epochs):
            self._update_parameters(training_data)
            loss = self._calculate_loss(training_data)
            losses.append(loss)
            print(f"Epoch {epoch + 1}: loss={loss:.3f}")
        return losses

    def _calculate_loss(self, input_samples):
        predictions = []
        targets = []
        for sample in input_samples:
            train_input, train_output = sample
            prediction = self._feed_forward(train_input)
            predictions.append(prediction)
            targets.append(train_output)
        return self.cost_func(np.array(predictions), np.array(targets))

    def predict(self, model_inputs):
        predictions = []
        for model_input in model_inputs:
            prediction = self._feed_forward(model_input)
            predictions.append(prediction)
        return np.array(predictions)

def xor():
    xor_train_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    xor_train_labels = np.array([[0], [1], [1], [0]])

    xor_net = ScratchNet([
        DenseLayer(neuron_count=2, activation=Sigmoid()),
        DenseLayer(neuron_count=1, activation=Sigmoid())
    ])

    losses = xor_net.fit(xor_train_inputs, xor_train_labels, epochs=10000)

    plt.plot(losses, label="loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    xor_test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    predictions = xor_net.predict(xor_test_inputs)
    print("Predictions:")
    print(predictions)

# Call the xor function to execute the code
xor()
