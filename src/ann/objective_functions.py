import numpy as np
from ann.activations import softmax


def one_hot(y, num_classes):
    y = y.astype(int)
    out = np.zeros((y.shape[0], num_classes))
    out[np.arange(y.shape[0]), y] = 1
    return out


def cross_entropy_loss(logits, y_true):
    probs = softmax(logits)
    probs = np.clip(probs, 1e-12, 1.0)
    return -np.mean(np.log(probs[np.arange(len(y_true)), y_true]))


def cross_entropy_grad(logits, y_true):
    probs = softmax(logits)
    probs[np.arange(len(y_true)), y_true] -= 1
    return probs / len(y_true)


def mse_loss(logits, y_true):
    probs = softmax(logits)
    y_one = one_hot(y_true, probs.shape[1])
    return np.mean((probs - y_one) ** 2)


def mse_grad(logits, y_true):
    probs = softmax(logits)
    y_one = one_hot(y_true, probs.shape[1])
    dA = 2.0 * (probs - y_one) / len(y_true)
    dot = np.sum(dA * probs, axis=1, keepdims=True)
    dZ = probs * (dA - dot)
    return dZ