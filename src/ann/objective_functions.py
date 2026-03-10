import numpy as np
from ann.activations import softmax


def one_hot(y, num_classes):
    # Convert labels to one-hot form
    y = y.astype(int)
    encoded = np.zeros((y.shape[0], num_classes))
    encoded[np.arange(y.shape[0]), y] = 1
    return encoded


def cross_entropy_loss(logits, y_true):
    # Cross-entropy on softmax probabilities
    probs = softmax(logits)
    probs = np.clip(probs, 1e-12, 1.0)
    return -np.mean(np.log(probs[np.arange(len(y_true)), y_true]))


def cross_entropy_grad(logits, y_true):
    # Gradient w.r.t. logits
    probs = softmax(logits)
    probs[np.arange(len(y_true)), y_true] -= 1
    return probs / len(y_true)


def mse_loss(logits, y_true):
    # MSE on softmax outputs
    probs = softmax(logits)
    y_encoded = one_hot(y_true, probs.shape[1])
    return np.mean((probs - y_encoded) ** 2)


def mse_grad(logits, y_true):
    # MSE gradient through softmax
    probs = softmax(logits)
    y_encoded = one_hot(y_true, probs.shape[1])
    dA = 2.0 * (probs - y_encoded) / len(y_true)
    dot_term = np.sum(dA * probs, axis=1, keepdims=True)
    dZ = probs * (dA - dot_term)
    return dZ