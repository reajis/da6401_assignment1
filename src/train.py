"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import json
import os
import numpy as np
import wandb
from sklearn.metrics import f1_score

from ann.r_neural_network import NeuralNetwork
from ann.optimizers import Optimizer
from ann.r_objective_functions import (
    mean_squared_error,
    mean_squared_error_derivative,
    cross_entropy,
    cross_entropy_derivative
)
from utils.data_loader import load_and_preprocess_data, get_batches


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(SRC_DIR, "best_model.npy")
DEFAULT_CONFIG_PATH = os.path.join(SRC_DIR, "best_config.json")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a neural network")

    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fashion_mnist"]
    )
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=10
    )
    parser.add_argument(
        "-b", "--batch_size",
        type=int,
        default=64
    )
    parser.add_argument(
        "-l", "--loss",
        type=str,
        default="cross_entropy",
        choices=["mean_squared_error", "cross_entropy"]
    )
    parser.add_argument(
        "-o", "--optimizer",
        type=str,
        default="adam",
        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]
    )
    parser.add_argument(
        "-lr", "--learning_rate",
        type=float,
        default=0.001
    )
    parser.add_argument(
        "-wd", "--weight_decay",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "-nhl", "--num_layers",
        type=int,
        default=2
    )
    parser.add_argument(
        "-sz", "--hidden_size",
        type=int,
        nargs="+",
        default=[128, 64]
    )
    parser.add_argument(
        "-a", "--activation",
        type=str,
        default="relu",
        choices=["sigmoid", "tanh", "relu"]
    )
    parser.add_argument(
        "-w_i", "--weight_init",
        type=str,
        default="xavier",
        choices=["random", "xavier"]
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default="da6401_assignment_1"
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default=DEFAULT_MODEL_PATH
    )
    parser.add_argument(
        "--config_save_path",
        type=str,
        default=DEFAULT_CONFIG_PATH
    )

    return parser.parse_args()


def one_hot_encode(y, num_classes=10):
    return np.eye(num_classes)[y]


def calculate_accuracy(y_true, y_pred):
    pred_labels = np.argmax(y_pred, axis=1)
    return np.mean(y_true == pred_labels)


def calculate_macro_f1(y_true, y_pred):
    pred_labels = np.argmax(y_pred, axis=1)
    return f1_score(y_true, pred_labels, average="macro")


def evaluate_split(model, X, y, loss_fn, num_classes=10):
    y_pred, _ = model.forward(X)
    y_true_onehot = one_hot_encode(y, num_classes)

    metrics = {
        "loss": loss_fn(y_true_onehot, y_pred),
        "accuracy": calculate_accuracy(y, y_pred),
        "f1": calculate_macro_f1(y, y_pred)
    }
    return metrics


def ensure_parent_dir(filepath):
    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)


def normalize_hidden_sizes(hidden_size_list, num_layers):
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


def main():
    args = parse_arguments()

    ensure_parent_dir(args.model_save_path)
    ensure_parent_dir(args.config_save_path)

    args.hidden_size = normalize_hidden_sizes(args.hidden_size, args.num_layers)

    wandb_mode = os.environ.get("WANDB_MODE", "disabled")
    wandb.init(
        project=args.wandb_project,
        config=vars(args),
        name=f"{args.optimizer}_lr{args.learning_rate}_{args.activation}",
        mode=wandb_mode
    )

    save_outputs = not (wandb.run is not None and wandb.run.sweep_id is not None)

    print(f"Loading {args.dataset} dataset")
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data(
        dataset_name=args.dataset
    )

    input_size = X_train.shape[1]
    output_size = 10   # integer labels now, so fix output classes explicitly
    num_classes = output_size

    model = NeuralNetwork(
        input_size=input_size,
        hidden_sizes=args.hidden_size,
        num_layers=args.num_layers,
        output_size=output_size,
        activation=args.activation,
        weight_init=args.weight_init
    )

    optimizer = Optimizer(
        layers=model.get_layers(),
        optimizer_type=args.optimizer,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    if args.loss == "cross_entropy":
        loss_fn = cross_entropy
        loss_derivative_fn = cross_entropy_derivative
    else:
        loss_fn = mean_squared_error
        loss_derivative_fn = mean_squared_error_derivative

    best_test_f1 = -1.0
    best_epoch = -1
    best_val_accuracy = -1.0

    print("Starting training loop...")

    for epoch in range(args.epochs):
        train_loss_total = 0.0
        train_acc_total = 0.0
        num_batches = 0

        for X_batch, y_batch in get_batches(X_train, y_train, args.batch_size):
            y_batch_onehot = one_hot_encode(y_batch, num_classes)

            y_pred, _ = model.forward(X_batch)

            batch_loss = loss_fn(y_batch_onehot, y_pred)
            batch_acc = calculate_accuracy(y_batch, y_pred)

            train_loss_total += batch_loss
            train_acc_total += batch_acc
            num_batches += 1

            dA = loss_derivative_fn(y_batch_onehot, y_pred)
            model.backward(dA)

            optimizer.step()

        avg_train_loss = train_loss_total / num_batches
        avg_train_acc = train_acc_total / num_batches

        val_metrics = evaluate_split(model, X_val, y_val, loss_fn, num_classes)
        test_metrics = evaluate_split(model, X_test, y_test, loss_fn, num_classes)

        best_val_accuracy = max(best_val_accuracy, val_metrics["accuracy"])

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train Acc: {avg_train_acc:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Test Acc: {test_metrics['accuracy']:.4f} | "
            f"Test F1: {test_metrics['f1']:.4f}"
        )

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_accuracy": avg_train_acc,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "best_val_accuracy": best_val_accuracy,
            "val_f1": val_metrics["f1"],
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "test_f1": test_metrics["f1"]
        })

        if test_metrics["f1"] > best_test_f1:
            best_test_f1 = test_metrics["f1"]
            best_epoch = epoch + 1

            if save_outputs:
                model_weights = {}
                for i, layer in enumerate(model.get_layers(), start=1):
                    model_weights[f"W{i}"] = layer.W.copy()
                    model_weights[f"b{i}"] = layer.b.copy()

                np.save(args.model_save_path, model_weights, allow_pickle=True)

                best_config = vars(args).copy()
                best_config["best_epoch"] = best_epoch
                best_config["best_test_f1"] = float(best_test_f1)
                best_config["best_test_accuracy"] = float(test_metrics["accuracy"])
                best_config["best_val_f1_at_save"] = float(val_metrics["f1"])
                best_config["best_val_accuracy_at_save"] = float(val_metrics["accuracy"])

                with open(args.config_save_path, "w") as f:
                    json.dump(best_config, f, indent=4)

                print(
                    f"--> Best model saved at epoch {best_epoch} "
                    f"with Test F1: {best_test_f1:.4f}"
                )

    wandb.finish()

    print("\nTraining complete!")
    print(f"Best model epoch: {best_epoch}")
    print(f"Best Test F1: {best_test_f1:.4f}")

    if save_outputs:
        print(f"Saved model: {args.model_save_path}")
        print(f"Saved config: {args.config_save_path}")
    else:
        print("Sweep run detected: skipped saving model/config files.")


if __name__ == "__main__":
    main()