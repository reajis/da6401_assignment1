"""
Loss/Objective Functions and Their Derivatives
Compatible with train.py:
- y_true is one-hot encoded
- y_pred is softmax probability output
"""
import numpy as np


def mean_squared_error(y_true, y_pred):
    """
    MSE between one-hot labels and predicted probabilities.
    """
    return np.mean((y_pred - y_true) ** 2)


def mean_squared_error_derivative(y_true, y_pred):
    """
    Returns dL/dZ for softmax output layer using MSE loss,
    not merely dL/dA. This matches the ideal-case behavior.

    y_true: one-hot labels
    y_pred: softmax probabilities
    """
    batch_size = y_true.shape[0]

    # dL/dA
    dA = 2.0 * (y_pred - y_true) / batch_size

    # softmax Jacobian contraction: dZ = s * (dA - sum(dA*s))
    dot = np.sum(dA * y_pred, axis=1, keepdims=True)
    dZ = y_pred * (dA - dot)

    return dZ


def cross_entropy(y_true, y_pred):
    """
    Cross-entropy loss using one-hot labels and predicted probabilities.
    """
    y_pred = np.clip(y_pred, 1e-12, 1.0)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def cross_entropy_derivative(y_true, y_pred):
    """
    Returns dL/dZ directly for softmax + cross-entropy.
    This is the ideal-case simplified gradient.

    y_true: one-hot labels
    y_pred: softmax probabilities
    """
    batch_size = y_true.shape[0]
    return (y_pred - y_true) / batch_size