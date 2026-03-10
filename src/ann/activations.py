"""
Activation Functions and Their Derivatives
ReLU, Sigmoid, Tanh, Softmax
"""
import numpy as np


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1.0 - s)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(z):
    t = np.tanh(z)
    return 1.0 - t * t


def relu(x):
    return np.maximum(0.0, x)


def relu_derivative(z):
    return (z > 0).astype(float)


def softmax(x):
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)