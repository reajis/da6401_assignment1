"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""
import numpy as np
from keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(dataset_name="mnist", val_split=0.1):
    """
    Loads the specified dataset, flattens the images, normalizes the pixel values, 
    one-hot encodes the labels, and creates a validation split.
    """

    # Load the raw data

    if dataset_name.lower() == "mnist":
        (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = mnist.load_data()
    elif dataset_name.lower() == "fashion_mnist":
        (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = fashion_mnist.load_data()
    else:
        raise ValueError("dataset_name must be 'mnist' or 'fashion_mnist'")


    # Flatten and Normalize the Images
    X_train_flat = X_train_raw.reshape(X_train_raw.shape[0], -1).astype('float32') / 255.0
    X_test_flat = X_test_raw.reshape(X_test_raw.shape[0], -1).astype('float32') / 255.0


    # one-Hot Encode the Labels
    num_classes = 10
    y_train_encoded = np.eye(num_classes)[y_train_raw]
    y_test_encoded = np.eye(num_classes)[y_test_raw]

    # 4. Create the Validation Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_flat, 
        y_train_encoded, 
        test_size=val_split, 
        random_state=42,
        stratify=y_train_raw
    )

    return (X_train, y_train), (X_val, y_val), (X_test_flat, y_test_encoded)


def get_batches(X, y, batch_size):
    """
    A generator function that yields mini-batches of data.
    """
    num_samples = X.shape[0]
    
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        
        yield X[batch_indices], y[batch_indices]
