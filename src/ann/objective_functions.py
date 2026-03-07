"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np

# MEAN SQUARED ERROR 

def mean_squared_error(y_true, y_pred):
    """
    Computes Mean Squared Error.
    Returns a scalar.
    """
    return np.mean((y_true - y_pred) ** 2)

def mean_squared_error_derivative(y_true, y_pred):
    """
    Computes derivative of MSE w.r.t y_pred.
    Returns matrix same shape as y_pred.
    """
    return 2 * (y_pred - y_true) / y_pred.size


# categorical cross entropy
def cross_entropy(y_true, y_pred):
    """
    Computes categorical cross-entropy loss.
    Returns scalar loss.
    """
    # Prevent log(0)
    y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
    
    N = y_pred.shape[0]
    
    loss = -np.sum(y_true * np.log(y_pred_clipped)) / N
    return loss

def cross_entropy_derivative(y_true, y_pred):
    """
    Computes derivative of categorical cross-entropy.
    Returns gradient matrix.
    """
    # Prevent division by zero
    y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
    
    N = y_pred.shape[0]
    
    #scale the derivative by batch size (N)
    return -(y_true / y_pred_clipped) / N