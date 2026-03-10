import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

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
    if isinstance(hidden_size, int):
        return [hidden_size] * config["num_layers"]
    return list(hidden_size)


def load_model(model_path, config_path):
    """Load a trained NeuralNetwork from saved weights and config."""
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

    weights = np.load(model_path, allow_pickle=True).item()
    model.set_weights(weights)
    return model


def main():
    args = parse_arguments()
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data(args.dataset)

    if args.split == "train":
        X, y = X_train, y_train
    elif args.split == "val":
        X, y = X_val, y_val
    else:
        X, y = X_test, y_test

    model = load_model(args.model_path, args.config_path)
    preds = model.predict(X)

    accuracy = accuracy_score(y, preds)
    precision = precision_score(y, preds, average="macro", zero_division=0)
    recall = recall_score(y, preds, average="macro", zero_division=0)
    f1 = f1_score(y, preds, average="macro", zero_division=0)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)

    if args.save_cm_path:
        os.makedirs(os.path.dirname(args.save_cm_path) or ".", exist_ok=True)
        cm = confusion_matrix(y, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(ax=ax, colorbar=False)
        plt.tight_layout()
        plt.savefig(args.save_cm_path)
        plt.close()

    if args.save_failures_path:
        wrong = np.where(preds != y)[0][:16]
        if len(wrong) > 0:
            os.makedirs(os.path.dirname(args.save_failures_path) or ".", exist_ok=True)
            fig, axes = plt.subplots(4, 4, figsize=(8, 8))
            for ax, idx in zip(axes.flat, wrong):
                ax.imshow(X[idx].reshape(28, 28), cmap="gray")
                ax.set_title(f"t:{y[idx]} p:{preds[idx]}")
                ax.axis("off")
            for ax in axes.flat[len(wrong):]:
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(args.save_failures_path)
            plt.close()


if __name__ == "__main__":
    main()