"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

import numpy as np
from .activations import (
    sigmoid,
    sigmoid_derivative,
    tanh,
    tanh_derivative,
    relu,
    relu_derivative,
    softmax
)


class NeuralLayer:
    def __init__(self, input_size, output_size, activation="relu", weight_init="random"):
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.activation_name = activation

        # Weight initialization
        if weight_init == "xavier":
            limit = np.sqrt(6.0 / (self.input_size + self.output_size))
            self.W = np.random.uniform(-limit, limit, (self.input_size, self.output_size))
        elif weight_init == "random":
            self.W = 0.01 * np.random.randn(self.input_size, self.output_size)
        else:
            raise ValueError("weight_init must be 'random' or 'xavier'")

        self.b = np.zeros((1, self.output_size))

        # Cache
        self.X = None
        self.Z = None
        self.A = None

        # Gradients
        self.grad_W = None
        self.grad_b = None

    def forward(self, X):
        self.X = X
        self.Z = X @ self.W + self.b

        if self.activation_name == "relu":
            self.A = relu(self.Z)
        elif self.activation_name == "sigmoid":
            self.A = sigmoid(self.Z)
        elif self.activation_name == "tanh":
            self.A = tanh(self.Z)
        elif self.activation_name == "softmax":
            self.A = softmax(self.Z)
        elif self.activation_name in [None, "linear"]:
            self.A = self.Z
        else:
            raise ValueError(f"Unsupported activation: {self.activation_name}")

        return self.A

    def backward(self, grad_output):
        """
        grad_output means:
        - for hidden layers: dL/dA
        - for softmax output layer: already dL/dZ
          (because objective_functions returns the ideal-case gradient)
        """

        if self.activation_name == "relu":
            dZ = grad_output * relu_derivative(self.Z)

        elif self.activation_name == "sigmoid":
            dZ = grad_output * sigmoid_derivative(self.Z)

        elif self.activation_name == "tanh":
            dZ = grad_output * tanh_derivative(self.Z)

        elif self.activation_name == "softmax":
            # IMPORTANT:
            # For softmax + CE / softmax + MSE in our setup,
            # the loss derivative already returns dL/dZ.
            dZ = grad_output

        elif self.activation_name in [None, "linear"]:
            dZ = grad_output

        else:
            raise ValueError(f"Unsupported activation: {self.activation_name}")

        self.grad_W = self.X.T @ dZ
        self.grad_b = np.sum(dZ, axis=0, keepdims=True)
        grad_input = dZ @ self.W.T

        return grad_input