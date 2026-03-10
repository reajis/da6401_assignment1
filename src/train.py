import argparse
import json
import os

import numpy as np
import wandb
from sklearn.metrics import f1_score, precision_score, recall_score

from ann.neural_network import NeuralNetwork
from utils.data_loader import get_batches, load_and_preprocess_data


def parse_arguments():
    # Command-line arguments for training
    parser = argparse.ArgumentParser(description="Train a neural network")
    parser.add_argument("-d", "--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e", "--epochs", type=int, default=5)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument(
        "-o",
        "--optimizer",
        type=str,
        default="adam",
        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],
    )
    parser.add_argument("-nhl", "--num_layers", type=int, default=2)
    parser.add_argument("-sz", "--hidden_size", type=int, nargs="+", default=[64])
    parser.add_argument("-a", "--activation", type=str, default="relu", choices=["relu", "sigmoid", "tanh"])
    parser.add_argument("-l", "--loss", type=str, default="cross_entropy", choices=["cross_entropy", "mse"])
    parser.add_argument("-wi", "--weight_init", type=str, default="xavier", choices=["random", "xavier", "zeros"])
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    parser.add_argument("--wandb_project", type=str, default="da6401_assignment_1")
    parser.add_argument("--model_save_path", type=str, default="src/best_model.npy")
    parser.add_argument("--config_save_path", type=str, default="src/best_config.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_activations", action="store_true")
    return parser.parse_args()


def metrics_dict(y_true, y_pred, prefix):
    # Standard classification metrics
    accuracy = float(np.mean(y_true == y_pred))
    precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    return {
        f"{prefix}_accuracy": accuracy,
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": f1,
    }


def resolve_hidden_layers(hidden_size_arg, num_layers):
    # Match hidden size config to number of layers
    if isinstance(hidden_size_arg, int):
        return [hidden_size_arg] * num_layers

    hidden_layers = list(hidden_size_arg)
    if len(hidden_layers) == 1:
        return hidden_layers * num_layers
    return hidden_layers


def main():
    # Setup
    args = parse_arguments()
    wandb.init(project=args.wandb_project, config=vars(args))
    config = wandb.config
    np.random.seed(config.seed)

    # Data
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data(config.dataset)
    hidden_layers = resolve_hidden_layers(config.hidden_size, config.num_layers)

    # Model
    model = NeuralNetwork(
        input_dim=784,
        hidden_layers=hidden_layers,
        output_dim=10,
        activation=config.activation,
        loss=config.loss,
        weight_init=config.weight_init,
        learning_rate=config.learning_rate,
        optimizer_name=config.optimizer,
        weight_decay=config.weight_decay,
    )
    model.optimizer.name = config.optimizer
    model.optimizer.learning_rate = config.learning_rate

    # Output paths
    model_path = config.model_save_path
    config_path = config.config_save_path
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(config_path) or ".", exist_ok=True)

    best_val_accuracy = -1.0
    best_epoch = 0
    global_step = 0

    for epoch_idx in range(config.epochs):
        epoch_batch_losses = []

        for X_batch, y_batch in get_batches(X_train, y_train, config.batch_size, shuffle=True):
            batch_loss = model.train_batch(X_batch, y_batch)
            epoch_batch_losses.append(batch_loss)
            global_step += 1

            # Log first layer gradient norms
            if len(model.layers) > 0 and model.layers[0].grad_W is not None:
                grad_w = model.layers[0].grad_W
                grad_norm = float(np.linalg.norm(grad_w))

                grad_log = {
                    "global_step": global_step,
                    "first_layer_grad_norm": grad_norm,
                }

                if global_step <= 50:
                    max_neurons = min(5, grad_w.shape[1])
                    for neuron_idx in range(max_neurons):
                        grad_log[f"grad_neuron_{neuron_idx}"] = float(
                            np.linalg.norm(grad_w[:, neuron_idx])
                        )

                wandb.log(grad_log)

        # Full-set evaluation
        train_loss, _, train_preds = model.evaluate(X_train, y_train)
        val_loss, val_accuracy, val_preds = model.evaluate(X_val, y_val)
        test_loss, test_accuracy, test_preds = model.evaluate(X_test, y_test)

        epoch_log = {
            "epoch": epoch_idx + 1,
            "batch_train_loss": float(np.mean(epoch_batch_losses)),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "test_loss": float(test_loss),
            "best_val_accuracy_so_far": float(max(best_val_accuracy, val_accuracy)),
        }
        epoch_log.update(metrics_dict(y_train, train_preds, "train"))
        epoch_log.update(metrics_dict(y_val, val_preds, "val"))
        epoch_log.update(metrics_dict(y_test, test_preds, "test"))

        # Optional activation logging
        if config.log_activations and model.last_hidden_output is not None:
            activations = model.last_hidden_output
            epoch_log["layer1_zero_fraction"] = float(np.mean(activations == 0))
            epoch_log["layer1_dead_fraction"] = float(np.mean(np.all(activations == 0, axis=0)))
            epoch_log["layer1_activations"] = wandb.Histogram(activations.flatten())

        wandb.log(epoch_log)

        # Save best checkpoint
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = float(val_accuracy)
            best_epoch = epoch_idx + 1
            np.save(model_path, model.get_weights(), allow_pickle=True)

            best_config = {
                "dataset": config.dataset,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "optimizer": config.optimizer,
                "num_layers": len(hidden_layers),
                "hidden_size": hidden_layers,
                "activation": config.activation,
                "loss": config.loss,
                "weight_init": config.weight_init,
                "weight_decay": config.weight_decay,
                "seed": config.seed,
            }

            with open(config_path, "w") as f:
                json.dump(best_config, f, indent=2)

    # Run summary
    wandb.summary["best_val_accuracy"] = best_val_accuracy
    wandb.summary["best_epoch"] = best_epoch
    wandb.summary["saved_model_path"] = model_path
    wandb.summary["saved_config_path"] = config_path

    print("Best val accuracy:", best_val_accuracy)
    print("Best epoch:", best_epoch)
    print("Saved model:", model_path)
    print("Saved config:", config_path)

    wandb.finish()


if __name__ == "__main__":
    main()