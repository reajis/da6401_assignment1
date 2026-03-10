import os
import gzip
import urllib.request

import numpy as np
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(dataset_name):
    # Load dataset and prepare train/val/test splits
    if dataset_name == "mnist":
        if not os.path.exists("mnist.npz"):
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
                "mnist.npz",
            )

        data = np.load("mnist.npz")

        X_train_full = data["x_train"].reshape(-1, 784).astype(np.float32) / 255.0
        y_train_full = data["y_train"].astype(int)
        X_test = data["x_test"].reshape(-1, 784).astype(np.float32) / 255.0
        y_test = data["y_test"].astype(int)

    elif dataset_name == "fashion_mnist":
        if not os.path.exists("train-images-idx3-ubyte.gz"):
            urllib.request.urlretrieve(
                "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
                "train-images-idx3-ubyte.gz",
            )

        if not os.path.exists("train-labels-idx1-ubyte.gz"):
            urllib.request.urlretrieve(
                "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
                "train-labels-idx1-ubyte.gz",
            )

        if not os.path.exists("t10k-images-idx3-ubyte.gz"):
            urllib.request.urlretrieve(
                "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
                "t10k-images-idx3-ubyte.gz",
            )

        if not os.path.exists("t10k-labels-idx1-ubyte.gz"):
            urllib.request.urlretrieve(
                "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
                "t10k-labels-idx1-ubyte.gz",
            )

        with gzip.open("train-images-idx3-ubyte.gz", "rb") as f:
            X_train_full = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784).astype(np.float32) / 255.0

        with gzip.open("train-labels-idx1-ubyte.gz", "rb") as f:
            y_train_full = np.frombuffer(f.read(), np.uint8, offset=8).astype(int)

        with gzip.open("t10k-images-idx3-ubyte.gz", "rb") as f:
            X_test = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784).astype(np.float32) / 255.0

        with gzip.open("t10k-labels-idx1-ubyte.gz", "rb") as f:
            y_test = np.frombuffer(f.read(), np.uint8, offset=8).astype(int)

    else:
        raise ValueError("dataset must be mnist or fashion_mnist")

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.1,
        random_state=42,
        stratify=y_train_full,
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_batches(X, y, batch_size, shuffle=True):
    # Yield mini-batches
    indices = np.arange(len(X))

    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, len(X), batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        yield X[batch_indices], y[batch_indices]