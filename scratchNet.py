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
    plt.ylim(top=1. bottom=0)
    plt.show()
    
class SquadError:
    raise NotImplementedError()

class DifferentiableFunction(abc.ABC):
    def drivative(self, net_input):
        pass
    def __call__(self, net_input):
        pass

class Sigmoid(DifferentiableFunction):
    def drivative(self, net_input):
        return self(net_input) * (1 - self_input))
    
    def __call__(self, net_input):
        return 1 / (1 + np.exp(-net_input))

class ScratchNet:
    #methods
    def __init__(self, layers):
        raise NotImplementedError()

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

def xor():
    train_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    train_labels = np.array([[0], [1], [1], [0]])
    
    total_classes = 2
    train_vec_labels = tf.keras.utils.to_categorical(train_labels, total_classes)
    
    model = ScratchNet(
        [
            FlattenLayer(input_shape=2, 1)),
            DenseLayer(4, activation=Sigmoid()),
            DenseLayer(2, activation=Sigmoid()),
        ]
    )
    
# repeat the values
repeat = (1000, 1)
train_inputs = np.tile(train_inputs, repeat)
train_vec_labels = np.tile(train_vec_labels, repeat)

model.compile(learning_rate=0.1, loss=SquaredError())

start = time.time()
losses, accurcaies = model.fit(
    train_inputs, train_vec_labels, epochs=5)
end = time.time()
print('Trainingsdauer: {:.lf}s'.format(end-start))

val_loss, val_acc = model.evaluate(
    validation_images=train_inputs, validation_labels=train_vec_labels
)