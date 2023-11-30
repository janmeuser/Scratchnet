#Building a neural network from scratch

import abc
from typing import Any
import matplotlib as plt
import numpy as np
import random
import time
from tqdm import tqdm, trange
from deeplearning2020 import helpers

def plot_loss_and_accuracy(losses, accuracies, xlabel):
    plt.plot(losses, label="loss")
    plt.plot(accuracies, label="accuracy")
    plt.legend(loc="upper left")
    plt.xlabel(xlabel)
    plt.ylim(top=1, bottom=0)
    plt.show()

class SquadError:
    raise NotImplementedError()

class DifferentiableFunction(abc.ABC):
    def drivative(self, net_input):
        pass
    def __call__(self, net_input):
        pass

class Sigmoid(DifferentiableFunction):
    def derivative(self, net_input):
        return self(net_input) * (1 - self(net_input))

    def __call__(self, net_input):
        return 1 / (1 + np.exp(-net_input))

class ScratchNet:
    #methods
    def __init__(self, layers):
        self.learning_rate = 0.5
        self.cost_func = SquaredError()
        self.layers = layers
        # for Schleife um durch index zu iterieren und Layer zu verketten 
        for index, layer in enumerate(self.layers):
            layer.prev_layer = self.layers[index - 1] if index > 0 else None
            layer.next_layer = (
                self.layers[index + 1] if index +
                1 < len(self.layers) else None
            )
            layer.depth = index
            layer.initialize_parameters()

    def fit(self, train_images, train_labels, epochs=1):
        raise NotImplementedError()

    def predict(self, model_inputs):
        raise NotImplementedError()

    def evaluate(self, validation_images, validation_labels):
        raise NotImplementedError()

    def compile(self, learning_rate=None, loss= None):
        raise NotImplementedError()

    def inspect(self):
        raise NotImplementedError()
    
class DenseLayer: #initialize Layer
    def __init__(
        self,
        neuron_count,
        depth=None,
        activation=None,
        biases=None,
        weights=None,
        prev_layer=None,
        next_Layer=None,
    ):
        raise NotImplementedError()
    
    def initialize_parameters(self):
        if self.weights is None:
            self.weights = np.random.randn(self.neuron_count, self.prev_layer.neuron_count)
        if self.biases is None:
            self.biases = np.randomnrandn(self.neuron_count, 1)

def xor():
    train_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    train_labels = np.array([[0], [1], [1], [0]])
    
    total_classes = 2
    train_vec_labels = tf.keras.utils.to_categorical(train_labels, total_classes)
    
    model = ScratchNet(
        [
            DenseLayer(neuron_count=4, activation=Sigmoid()),
            DenseLayer(neuron_count=2, activation=Sigmoid()),
        ]
    )
    
    # repeat the values
    repeat = (1000, 1)
    train_inputs = np.tile(train_inputs, repeat)
    train_vec_labels = np.tile(train_vec_labels, repeat)

    model.compile(learning_rate=0.1, loss=SquaredError())

    start = time.time()
    losses, accuracies = model.fit(
        train_inputs, train_vec_labels, epochs=5
    )
    end = time.time()
    print('Trainingsdauer: {:.1f}s'.format(end - start))

    val_loss, val_acc = model.evaluate(
        validation_images=train_inputs, validation_labels=train_vec_labels
    )