import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_and_preprocess_data


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference with a saved model")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("--model_path", type=str, default="src/best_model.npy")
    parser.add_argument("--config_path", type=str, default="src/best_config.json")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--save_cm_path", type=str, default="")
    parser.add_argument("--save_failures_path", type=str, default="")
    return parser.parse_args()


def resolve_hidden_layers(config):
    hidden_size = config["hidden_size"]
    num_layers = int(config["num_layers"])

    if isinstance(hidden_size, int):
        return [hidden_size] * num_layers

    if isinstance(hidden_size, list):
        if len(hidden_size) == num_layers:
            return hidden_size
        if len(hidden_size) == 1:
            return hidden_size * num_layers
        raise ValueError("hidden_size list length must match num_layers")

    raise ValueError("hidden_size must be int or list")


def load_model(model_path, config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    hidden_layers = resolve_hidden_layers(config)

    model = NeuralNetwork(
        input_dim=784,
        hidden_layers=hidden_layers,
        output_dim=10,
        activation=config["activation"],
        loss=config["loss"],
        weight_init=config["weight_init"],
        learning_rate=config["learning_rate"],
        optimizer_name=config["optimizer"],
        weight_decay=config["weight_decay"],
    )

    saved_weights = np.load(model_path, allow_pickle=True).item()
    model.set_weights(saved_weights)
    return model


def main():
    args = parse_arguments()

    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data(args.dataset)

    if args.split == "train":
        X_data, y_data = X_train, y_train
    elif args.split == "val":
        X_data, y_data = X_val, y_val
    else:
        X_data, y_data = X_test, y_test

    model = load_model(args.model_path, args.config_path)

    if hasattr(model, "predict"):
        predictions = model.predict(X_data)
    else:
        y_pred, _ = model.forward(X_data)
        predictions = np.argmax(y_pred, axis=1)

    accuracy = accuracy_score(y_data, predictions)
    precision = precision_score(y_data, predictions, average="macro", zero_division=0)
    recall = recall_score(y_data, predictions, average="macro", zero_division=0)
    f1 = f1_score(y_data, predictions, average="macro", zero_division=0)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)

    if args.save_cm_path:
        os.makedirs(os.path.dirname(args.save_cm_path) or ".", exist_ok=True)
        cm = confusion_matrix(y_data, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(ax=ax, colorbar=False)
        plt.tight_layout()
        plt.savefig(args.save_cm_path)
        plt.close()

    if args.save_failures_path:
        wrong_indices = np.where(predictions != y_data)[0][:16]
        if len(wrong_indices) > 0:
            os.makedirs(os.path.dirname(args.save_failures_path) or ".", exist_ok=True)
            fig, axes = plt.subplots(4, 4, figsize=(8, 8))

            for ax, idx in zip(axes.flat, wrong_indices):
                ax.imshow(X_data[idx].reshape(28, 28), cmap="gray")
                ax.set_title(f"t:{y_data[idx]} p:{predictions[idx]}")
                ax.axis("off")

            for ax in axes.flat[len(wrong_indices):]:
                ax.axis("off")

            plt.tight_layout()
            plt.savefig(args.save_failures_path)
            plt.close()


if __name__ == "__main__":
    main()