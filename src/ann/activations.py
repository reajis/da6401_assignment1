"""
Activation Functions and Their Derivatives
ReLU, Sigmoid, Tanh, Softmax
"""
import numpy as np


def sigmoid(x):
    # Clip values to avoid overflow in exp
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(z):
    # Derivative using sigmoid output
    s = sigmoid(z)
    return s * (1.0 - s)


def tanh(x):
    # Hyperbolic tangent activation
    return np.tanh(x)


def tanh_derivative(z):
    # Derivative of tanh
    t = np.tanh(z)
    return 1.0 - t * t


def relu(x):
    # ReLU activation
    return np.maximum(0.0, x)


def relu_derivative(z):
    # Gradient is 1 for positive inputs, else 0
    return (z > 0).astype(float)


def softmax(x):
    # Shift values for numerical stability
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)