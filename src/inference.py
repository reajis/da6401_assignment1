"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ann.neural_network import NeuralNetwork
from ann.objective_functions import cross_entropy, mean_squared_error
from utils.data_loader import load_and_preprocess_data


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(SRC_DIR, "best_model.npy")


def parse_arguments():
    """
    Parse command-line arguments for inference.
    """
    parser = argparse.ArgumentParser(description="Run inference on test set")

    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to saved model weights"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fashion_mnist"]
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Number of hidden layers"
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        nargs="+",
        default=[128, 64],
        help="List of hidden layer sizes"
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "sigmoid", "tanh"]
    )
    parser.add_argument(
        "--weight_init",
        type=str,
        default="xavier",
        choices=["random", "xavier"]
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="cross_entropy",
        choices=["cross_entropy", "mean_squared_error"]
    )

    return parser.parse_args()


def normalize_hidden_sizes(hidden_size_list, num_layers):
    """
    Ensures hidden_size list matches num_layers.
    """
    if num_layers == 0:
        return []

    if isinstance(hidden_size_list, int):
        return [hidden_size_list] * num_layers

    if len(hidden_size_list) == num_layers:
        return hidden_size_list

    if len(hidden_size_list) == 1:
        return hidden_size_list * num_layers

    raise ValueError(
        "Length of hidden_size list must match num_layers, "
        "or provide a single integer to be repeated."
    )


def load_model(model_path, input_size, hidden_sizes, num_layers, output_size, activation, weight_init):
    """
    Load trained model from disk by rebuilding the architecture and injecting weights.
    """
    model = NeuralNetwork(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        num_layers=num_layers,
        output_size=output_size,
        activation=activation,
        weight_init=weight_init
    )

    weights = np.load(model_path, allow_pickle=True).item()
    model.set_weights(weights)

    return model


def evaluate_model(model, X_test, y_test, loss_type="cross_entropy"):
    """
    Evaluate model on test data and return required metrics.
    """
    y_pred_probs, _ = model.forward(X_test)

    if loss_type == "cross_entropy":
        loss_val = cross_entropy(y_test, y_pred_probs)
    else:
        loss_val = mean_squared_error(y_test, y_pred_probs)

    y_true_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)

    metrics = {
        "probabilities": y_pred_probs,
        "loss": loss_val,
        "accuracy": accuracy_score(y_true_labels, y_pred_labels),
        "f1": f1_score(y_true_labels, y_pred_labels, average="macro", zero_division=0),
        "precision": precision_score(y_true_labels, y_pred_labels, average="macro", zero_division=0),
        "recall": recall_score(y_true_labels, y_pred_labels, average="macro", zero_division=0)
    }

    return metrics


def main():
    """
    Main inference function.
    """
    args = parse_arguments()
    args.hidden_size = normalize_hidden_sizes(args.hidden_size, args.num_layers)

    _, _, (X_test, y_test) = load_and_preprocess_data(dataset_name=args.dataset)

    model = load_model(
        model_path=args.model_path,
        input_size=X_test.shape[1],
        hidden_sizes=args.hidden_size,
        num_layers=args.num_layers,
        output_size=y_test.shape[1],
        activation=args.activation,
        weight_init=args.weight_init
    )

    results = evaluate_model(model, X_test, y_test, loss_type=args.loss)

    print("\nEvaluation Results:")
    print(f"Loss: {results['loss']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1: {results['f1']:.4f}")

    return results


if __name__ == "__main__":
    main()