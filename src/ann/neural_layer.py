"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

import numpy as np
from .activations import (sigmoid, sigmoid_derivative, tanh, tanh_derivative, relu, relu_derivative, softmax)


class NeuralLayer:
    def __init__(self, input_size, output_size, activation="relu", weight_init="random"):

        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation

        # Weight Initialization 
        if weight_init == "random":
            self.W = np.random.randn(input_size, output_size) * 0.01

        elif weight_init == "xavier":
            variance = 2.0 / (input_size + output_size)
            self.W = np.random.randn(input_size, output_size) * np.sqrt(variance)


        else:
            raise ValueError("weight_init must be 'random' or 'xavier'")

        self.b = np.zeros((1, output_size))

        # Cache
        self.X = None
        self.Z = None
        self.A = None

        # Gradients
        self.grad_W = None
        self.grad_b = None

    # FORWARD PASS

    def forward(self, X):

        self.X = X

        # Z = XW + b
        self.Z = np.dot(X, self.W) + self.b

        # Apply activation
        if self.activation_name == "relu":
            self.A = relu(self.Z)

        elif self.activation_name == "sigmoid":
            self.A = sigmoid(self.Z)

        elif self.activation_name == "tanh":
            self.A = tanh(self.Z)

        elif self.activation_name == "softmax":
            self.A = softmax(self.Z)

        else:
            raise ValueError("Unsupported activation")

        return self.A


    # BACKWARD PASS
    def backward(self, dA):

        # Activation derivative
        if self.activation_name == "relu":
            dZ = dA *relu_derivative(self.Z)

        elif self.activation_name == "sigmoid":
            dZ = dA *sigmoid_derivative(self.A)

        elif self.activation_name == "tanh":
            dZ = dA *tanh_derivative(self.A)

        elif self.activation_name == "softmax":
            s = self.A  # softmax output
            dZ = s * (dA - np.sum(dA * s, axis=1, keepdims=True))

        else:
            raise ValueError("Unsupported activation")

        # Gradient w.r.t weights
        self.grad_W = np.dot(self.X.T, dZ)

        # Gradient w.r.t bias
        self.grad_b = np.sum(dZ, axis=0, keepdims=True)

        #Gradient passed backward
        dX = np.dot(dZ, self.W.T)

        return dX