import argparse
import json
import os
import numpy as np
import wandb
from sklearn.metrics import precision_score, recall_score, f1_score

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_and_preprocess_data, get_batches


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a neural network")
    parser.add_argument("-d", "--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e", "--epochs", type=int, default=5)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-o", "--optimizer", type=str, default="adam", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])
    parser.add_argument("-nhl", "--num_layers", type=int, default=2)
    parser.add_argument("-sz", "--hidden_size", type=int, nargs="+", default=[64])
    parser.add_argument("-a", "--activation", type=str, default="relu", choices=["relu", "sigmoid", "tanh"])
    parser.add_argument("-l", "--loss", type=str, default="cross_entropy", choices=["cross_entropy", "mse"])
    parser.add_argument("-wi", "--weight_init", type=str, default="xavier", choices=["random", "xavier", "zeros"])
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    parser.add_argument("--wandb_project", type=str, default="da6401_assignment_1")
    parser.add_argument("--model_save_path", type=str, default="src/trial_model.npy")
    parser.add_argument("--config_save_path", type=str, default="src/trial_config.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_activations", action="store_true")
    return parser.parse_args()


def metrics_dict(y_true, y_pred, prefix):
    return {
        f"{prefix}_accuracy": float(np.mean(y_true == y_pred)),
        f"{prefix}_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        f"{prefix}_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        f"{prefix}_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def resolve_hidden_layers(hidden_size_arg, num_layers):
    if isinstance(hidden_size_arg, int):
        return [hidden_size_arg] * num_layers

    hidden_sizes = list(hidden_size_arg)
    if len(hidden_sizes) == 1:
        return hidden_sizes * num_layers
    return hidden_sizes


def main():
    args = parse_arguments()
    wandb.init(project=args.wandb_project, config=vars(args))
    config = wandb.config
    np.random.seed(config.seed)

    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data(config.dataset)
    hidden_layers = resolve_hidden_layers(config.hidden_size, config.num_layers)

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

    run_model_path = config.model_save_path
    run_config_path = config.config_save_path
    os.makedirs(os.path.dirname(run_model_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(run_config_path) or ".", exist_ok=True)

    best_val_accuracy = -1.0
    best_epoch = 0
    global_step = 0

    for epoch in range(config.epochs):
        batch_losses = []

        for X_batch, y_batch in get_batches(X_train, y_train, config.batch_size, shuffle=True):
            loss = model.train_batch(X_batch, y_batch)
            batch_losses.append(loss)
            global_step += 1

            if len(model.layers) > 0 and model.layers[0].grad_W is not None:
                first_layer_grad = model.layers[0].grad_W
                first_layer_grad_norm = float(np.linalg.norm(first_layer_grad))
                log_dict = {
                    "global_step": global_step,
                    "first_layer_grad_norm": first_layer_grad_norm,
                }

                if global_step <= 50:
                    for j in range(min(5, first_layer_grad.shape[1])):
                        log_dict[f"grad_neuron_{j}"] = float(np.linalg.norm(first_layer_grad[:, j]))

                wandb.log(log_dict)

        train_loss, _, train_preds = model.evaluate(X_train, y_train)
        val_loss, val_accuracy, val_preds = model.evaluate(X_val, y_val)
        test_loss, test_accuracy, test_preds = model.evaluate(X_test, y_test)

        log_data = {
            "epoch": epoch + 1,
            "batch_train_loss": float(np.mean(batch_losses)),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "test_loss": float(test_loss),
            "best_val_accuracy_so_far": float(max(best_val_accuracy, val_accuracy)),
        }
        log_data.update(metrics_dict(y_train, train_preds, "train"))
        log_data.update(metrics_dict(y_val, val_preds, "val"))
        log_data.update(metrics_dict(y_test, test_preds, "test"))

        if config.log_activations and model.last_hidden_output is not None:
            a = model.last_hidden_output
            zero_fraction = float(np.mean(a == 0))
            dead_fraction = float(np.mean(np.all(a == 0, axis=0)))
            log_data["layer1_zero_fraction"] = zero_fraction
            log_data["layer1_dead_fraction"] = dead_fraction
            log_data["layer1_activations"] = wandb.Histogram(a.flatten())

        wandb.log(log_data)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = float(val_accuracy)
            best_epoch = epoch + 1
            np.save(run_model_path, model.get_weights(), allow_pickle=True)

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

            with open(run_config_path, "w") as f:
                json.dump(best_config, f, indent=2)

    wandb.summary["best_val_accuracy"] = best_val_accuracy
    wandb.summary["best_epoch"] = best_epoch
    wandb.summary["saved_model_path"] = run_model_path
    wandb.summary["saved_config_path"] = run_config_path

    print("Best val accuracy:", best_val_accuracy)
    print("Best epoch:", best_epoch)
    print("Saved model:", run_model_path)
    print("Saved config:", run_config_path)
    wandb.finish()


if __name__ == "__main__":
    main()