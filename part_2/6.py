import os
import sys
import numpy as np
import wandb
import matplotlib.pyplot as plt

# Robust path handling
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from ann.r_neural_network import NeuralNetwork
from ann.optimizers import Optimizer
from ann.r_objective_functions import (
    mean_squared_error,
    mean_squared_error_derivative,
    cross_entropy,
    cross_entropy_derivative,
)
from utils.r_data_loader import load_and_preprocess_data, get_batches


def set_seed(seed=42):
    np.random.seed(seed)


def clone_model_weights(model):
    return [{"W": layer.W.copy(), "b": layer.b.copy()} for layer in model.get_layers()]


def load_model_weights(model, weights):
    for layer, saved_params in zip(model.get_layers(), weights):
        layer.W = saved_params["W"].copy()
        layer.b = saved_params["b"].copy()


def run_loss_comparison():
    print("Loading MNIST dataset...")
    (X_train, y_train), _, _ = load_and_preprocess_data(dataset_name="mnist")

    losses_to_test = ["mean_squared_error", "cross_entropy"]
    epochs = 15
    batch_size = 64
    learning_rate = 0.01

    training_curves = {}

    # Create one base model so both loss runs start from identical weights
    set_seed(42)
    base_model = NeuralNetwork(
        input_size=784,
        hidden_sizes=[64, 64],
        num_layers=2,
        output_size=10,
        activation="relu",
        weight_init="xavier"
    )
    base_weights = clone_model_weights(base_model)

    for loss_name in losses_to_test:
        print(f"\n--- Training with {loss_name.upper()} ---")

        wandb.init(
            project="da6401_assignment_1_q2_6",
            group="part2_q2_6",
            job_type="loss_comparison_run",
            tags=["part2", "q2_6", loss_name],
            name=f"q2_6_loss_comparison_{loss_name}",
            reinit=True,
            config={
                "dataset": "mnist",
                "loss": loss_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "optimizer": "adam",
                "hidden_sizes": [64, 64],
                "num_layers": 2,
                "activation": "relu",
                "weight_init": "xavier"
            }
        )

        model = NeuralNetwork(
            input_size=784,
            hidden_sizes=[64, 64],
            num_layers=2,
            output_size=10,
            activation="relu",
            weight_init="xavier"
        )
        load_model_weights(model, base_weights)

        optimizer = Optimizer(
            model.get_layers(),
            optimizer_type="adam",
            lr=learning_rate
        )

        if loss_name == "cross_entropy":
            loss_fn = cross_entropy
            loss_grad_fn = cross_entropy_derivative
        else:
            loss_fn = mean_squared_error
            loss_grad_fn = mean_squared_error_derivative

        run_losses = []

        # Reset batch shuffle seed so both runs see same batch order
        set_seed(123)

        for epoch in range(epochs):
            train_loss_total = 0.0
            num_batches = 0

            for X_batch, y_batch in get_batches(X_train, y_train, batch_size):
                y_pred = model.forward(X_batch)
                batch_loss = loss_fn(y_batch, y_pred)

                train_loss_total += batch_loss
                num_batches += 1

                dA = loss_grad_fn(y_batch, y_pred)
                model.backward(dA)
                optimizer.step()

            avg_train_loss = train_loss_total / num_batches
            run_losses.append(avg_train_loss)

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss
            })

            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_train_loss:.4f}")

        training_curves[loss_name] = run_losses
        wandb.finish()

    # Combined comparison plot
    plt.figure(figsize=(10, 6))
    for loss_name, losses in training_curves.items():
        plt.plot(
            range(1, epochs + 1),
            losses,
            label=loss_name.replace("_", " ").title(),
            marker="o",
            linewidth=2
        )

    plt.title("Training Curve Comparison: MSE vs Cross-Entropy", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    wandb.init(
        project="da6401_assignment_1_q2_6",
        group="part2_q2_6",
        job_type="loss_comparison_plot",
        tags=["part2", "q2_6", "summary_plot"],
        name="q2_6_loss_comparison_plot",
        reinit=True
    )
    wandb.log({"q2_6_loss_comparison_plot": wandb.Image(plt)})
    wandb.finish()

    plt.show()


if __name__ == "__main__":
    run_loss_comparison()