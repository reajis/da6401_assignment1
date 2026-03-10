import numpy as np
from ann.activations import sigmoid, tanh, relu
from ann.activations import sigmoid_derivative, tanh_derivative, relu_derivative


class DenseLayer:
    def __init__(self, input_dim, output_dim, activation=None, weight_init="xavier"):
        if weight_init == "xavier":
            limit = np.sqrt(6.0 / (input_dim + output_dim))
            self.W = np.random.uniform(-limit, limit, (input_dim, output_dim))
        elif weight_init == "zeros":
            self.W = np.zeros((input_dim, output_dim))
        else:
            self.W = 0.01 * np.random.randn(input_dim, output_dim)

        self.b = np.zeros((1, output_dim))
        self.activation = activation
        self.input = None
        self.z = None
        self.output = None
        self.grad_W = None
        self.grad_b = None

    def activate(self, z):
        if self.activation == "sigmoid":
            return sigmoid(z)
        if self.activation == "tanh":
            return tanh(z)
        if self.activation == "relu":
            return relu(z)
        return z

    def activation_grad(self, z):
        if self.activation == "sigmoid":
            return sigmoid_derivative(z)
        if self.activation == "tanh":
            return tanh_derivative(z)
        if self.activation == "relu":
            return relu_derivative(z)
        return np.ones_like(z)

    def forward(self, x):
        self.input = x
        self.z = x @ self.W + self.b
        self.output = self.activate(self.z)
        return self.output

    def backward(self, grad_output, weight_decay=0.0):
        grad_z = grad_output * self.activation_grad(self.z)
        self.grad_W = self.input.T @ grad_z + weight_decay * self.W
        self.grad_b = np.sum(grad_z, axis=0, keepdims=True)
        grad_input = grad_z @ self.W.T
        return grad_input

    def backward_linear(self, grad_z, weight_decay=0.0):
        self.grad_W = self.input.T @ grad_z + weight_decay * self.W
        self.grad_b = np.sum(grad_z, axis=0, keepdims=True)
        grad_input = grad_z @ self.W.T
        return grad_input