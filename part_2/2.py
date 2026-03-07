import os
import sys
import argparse
import numpy as np
import wandb
from sklearn.metrics import f1_score

# Make src/ importable when running from repo root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from ann.neural_network import NeuralNetwork
from ann.optimizers import Optimizer
from ann.objective_functions import (
    mean_squared_error,
    mean_squared_error_derivative,
    cross_entropy,
    cross_entropy_derivative,
)
from utils.data_loader import load_and_preprocess_data, get_batches


def calculate_accuracy(y_true, y_pred):
    true_labels = np.argmax(y_true, axis=1)
    pred_labels = np.argmax(y_pred, axis=1)
    return np.mean(true_labels == pred_labels)


def calculate_macro_f1(y_true, y_pred):
    true_labels = np.argmax(y_true, axis=1)
    pred_labels = np.argmax(y_pred, axis=1)
    return f1_score(true_labels, pred_labels, average="macro", zero_division=0)


def evaluate_split(model, X, y, loss_fn):
    y_pred = model.forward(X)
    return {
        "loss": loss_fn(y, y_pred),
        "accuracy": calculate_accuracy(y, y_pred),
        "f1": calculate_macro_f1(y, y_pred),
    }


def build_hidden_sizes(num_layers, hidden_size):
    if num_layers == 0:
        return []
    return [hidden_size] * num_layers


def train_one_run():
    wandb.init()

    config = wandb.config

    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_preprocess_data(
        dataset_name=config.dataset
    )

    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    hidden_sizes = build_hidden_sizes(config.num_layers, config.hidden_size)

    # Build model
    model = NeuralNetwork(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        num_layers=config.num_layers,
        output_size=output_size,
        activation=config.activation,
        weight_init=config.weight_init,
    )

    # Optimizer
    optimizer = Optimizer(
        layers=model.get_layers(),
        optimizer_type=config.optimizer,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Loss selection
    if config.loss == "cross_entropy":
        loss_fn = cross_entropy
        loss_derivative_fn = cross_entropy_derivative
    else:
        loss_fn = mean_squared_error
        loss_derivative_fn = mean_squared_error_derivative

    best_val_accuracy = -1.0
    best_val_f1 = -1.0
    best_test_accuracy = -1.0
    best_test_f1 = -1.0
    best_epoch = -1

    for epoch in range(config.epochs):
        train_loss_total = 0.0
        train_acc_total = 0.0
        num_batches = 0

        for X_batch, y_batch in get_batches(X_train, y_train, config.batch_size):
            # Forward
            y_pred = model.forward(X_batch)

            # Metrics
            batch_loss = loss_fn(y_batch, y_pred)
            batch_acc = calculate_accuracy(y_batch, y_pred)

            train_loss_total += batch_loss
            train_acc_total += batch_acc
            num_batches += 1

            # Backward
            dA = loss_derivative_fn(y_batch, y_pred)
            model.backward(dA)

            # Update
            optimizer.step()

        avg_train_loss = train_loss_total / num_batches
        avg_train_accuracy = train_acc_total / num_batches

        val_metrics = evaluate_split(model, X_val, y_val, loss_fn)
        test_metrics = evaluate_split(model, X_test, y_test, loss_fn)

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_val_f1 = val_metrics["f1"]
            best_test_accuracy = test_metrics["accuracy"]
            best_test_f1 = test_metrics["f1"]
            best_epoch = epoch + 1

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_accuracy": avg_train_accuracy,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "test_f1": test_metrics["f1"],
            "best_val_accuracy": best_val_accuracy,
            "best_val_f1": best_val_f1,
            "best_test_accuracy_at_best_val": best_test_accuracy,
            "best_test_f1_at_best_val": best_test_f1,
        })

    wandb.run.summary["best_epoch"] = best_epoch
    wandb.run.summary["best_val_accuracy"] = best_val_accuracy
    wandb.run.summary["best_val_f1"] = best_val_f1
    wandb.run.summary["best_test_accuracy_at_best_val"] = best_test_accuracy
    wandb.run.summary["best_test_f1_at_best_val"] = best_test_f1

    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Run W&B hyperparameter sweep for Q2.2")
    parser.add_argument("--project", type=str, default="da6401_assignment_1")
    parser.add_argument("--count", type=int, default=100, help="Number of sweep runs")
    args = parser.parse_args()

    sweep_config = {
        "method": "random",
        "metric": {
            "name": "best_val_accuracy",
            "goal": "maximize"
        },
        "parameters": {
            "dataset": {
                "values": ["mnist"]
            },
            "epochs": {
                "values": [10]
            },
            "batch_size": {
                "values": [16, 32, 64, 128]
            },
            "loss": {
                "values": ["cross_entropy", "mean_squared_error"]
            },
            "optimizer": {
                "values": ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]
            },
            "learning_rate": {
                "values": [1e-4, 5e-4, 1e-3, 3e-3, 1e-2]
            },
            "weight_decay": {
                "values": [0.0, 1e-4, 5e-4, 1e-3]
            },
            "num_layers": {
                "values": [1, 2, 3, 4]
            },
            "hidden_size": {
                "values": [32, 64, 128]
            },
            "activation": {
                "values": ["sigmoid", "tanh", "relu"]
            },
            "weight_init": {
                "values": ["random", "xavier"]
            },
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_config, project=args.project)
    print(f"Sweep created: {sweep_id}")
    wandb.agent(sweep_id, function=train_one_run, count=args.count)


if __name__ == "__main__":
    main()