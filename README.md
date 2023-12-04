# Scratchnet
Building a Neural Network from Scratch using Python

## The XOR Problem

The XOR problem is a classic problem in the field of artificial intelligence and machine learning. XOR, which stands for exclusive or, is a logical operation that takes two binary inputs and produces a binary output. The operation returns true (1) if the number of true inputs is odd, and false (0) if it's even.

Here's the truth table for XOR:

| A     | B     | A XOR B |
|-------|-------|---------|
| false | false | false   |
| false | true  | true    |
| true  | false | true    |
| true  | true  | false   |


The challenge with the XOR problem arises when trying to find a linear decision boundary to separate the two classes (0 and 1). Linear models, like single-layer perceptrons, fail to solve the XOR problem because they can only learn linearly separable functions. In the XOR case, the data points are not linearly separable in the input space.

To solve the XOR problem, more complex models with non-linear activation functions and multiple layers, such as neural networks, are needed. Neural networks with hidden layers can learn to capture the non-linear relationships between inputs and produce the correct XOR output.

The following script scratchNet.py displays a table showing the input and predicted results.