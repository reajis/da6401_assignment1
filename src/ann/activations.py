"""
Activation Functions and Their Derivatives
ReLU, Sigmoid, Tanh, Softmax
"""
import numpy as np

# SIGMOID

def sigmoid(x):
    """
    f(x) = 1 / (1 + e^(-x))
    """
    # Prevent overflow in exp
    x = np.clip(x, -500, 500)

    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(output):
    """
    f'(x) = f(x) * (1 - f(x))
    """

    return output * (1 - output)

# HYPERBOLIC TANGENT (TANH)
def tanh(x):
    return np.tanh(x)


def tanh_derivative(output):
    """
    f'(x) = 1 - (f(x))^2
    """

    return 1 - output**2



# RELU

def relu(x):
    """
    f(x) = max(0, x)
    """

    return np.maximum(0, x)


def relu_derivative(x):
    """
    f'(x) = 1 if x > 0 else 0
    """

    return (x > 0).astype(float)

# SOFTMAX

def softmax(x):
    """
    e^x_i / sum(e^x_j)
    """
    # Shift x for numerical stability (prevents overflow)
    shifted_x = x - np.max(x, axis=1, keepdims=True)
    exps = np.exp(shifted_x)
    return exps / np.sum(exps, axis=1, keepdims=True)