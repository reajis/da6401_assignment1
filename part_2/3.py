import os
import sys
import copy
import numpy as np
import wandb
import matplotlib.pyplot as plt

# path handling
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from ann.neural_network import NeuralNetwork
from ann.optimizers import Optimizer
from ann.objective_functions import cross_entropy, cross_entropy_derivative
from utils.r_data_loader import load_and_preprocess_data, get_batches


def set_seed(seed=42):
    np.random.seed(seed)


def clone_model_weights(model):
    return [{"W": layer.W.copy(), "b": layer.b.copy()} for layer in model.get_layers()]


def load_model_weights(model, weights):
    for layer, saved_params in zip(model.get_layers(), weights):
        layer.W = saved_params["W"].copy()
        layer.b = saved_params["b"].copy()


def run_optimizer_showdown():
    
    optimizers = ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]
    epochs = 5
    batch_size = 64
    learning_rate = 0.001

    print("Loading MNIST dataset...")
    (X_train, y_train), (X_val, y_val), _ = load_and_preprocess_data(dataset_name="mnist")

    # Create one reference model so every optimizer starts from identical weights
    set_seed(42)
    base_model = NeuralNetwork(
        input_size=784,
        hidden_sizes=[128, 128, 128],
        num_layers=3,
        output_size=10,
        activation="relu",
        weight_init="xavier"
    )
    base_weights = clone_model_weights(base_model)

    loss_history = {}

    for opt_name in optimizers:
        print(f"\n--- Training with {opt_name.upper()} ---")

        wandb.init(
            project="da6401_assignment_1_q2_3",
            group="part2_q2_3",
            job_type="optimizer_showdown",
            tags=["part2", "q2_3", opt_name],
            name=f"q2_3_optimizer_{opt_name}",
            reinit=True,
            config={
                "dataset": "mnist",
                "optimizer": opt_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_layers": 3,
                "hidden_sizes": [128, 128, 128],
                "activation": "relu",
                "weight_init": "xavier",
                "loss": "cross_entropy"
            }
        )

        # Rebuild model and load same starting weights
        model = NeuralNetwork(
            input_size=784,
            hidden_sizes=[128, 128, 128],
            num_layers=3,
            output_size=10,
            activation="relu",
            weight_init="xavier"
        )
        load_model_weights(model, base_weights)

        optimizer = Optimizer(
            model.get_layers(),
            optimizer_type=opt_name,
            lr=learning_rate
        )

        run_losses = []

        # Reset seed so batch shuffling is the same across optimizers
        set_seed(123)

        for epoch in range(epochs):
            train_loss_total = 0.0
            num_batches = 0

            for X_batch, y_batch in get_batches(X_train, y_train, batch_size):
                y_pred = model.forward(X_batch)
                batch_loss = cross_entropy(y_batch, y_pred)

                train_loss_total += batch_loss
                num_batches += 1

                dA = cross_entropy_derivative(y_batch, y_pred)
                model.backward(dA)
                optimizer.step()

            avg_train_loss = train_loss_total / num_batches
            run_losses.append(avg_train_loss)

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss
            })

            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_train_loss:.4f}")

        loss_history[opt_name] = run_losses
        wandb.finish()

    # Make one combined comparison plot
    plt.figure(figsize=(10, 6))
    for opt_name, losses in loss_history.items():
        plt.plot(
            range(1, epochs + 1),
            losses,
            label=opt_name.upper(),
            marker="o",
            linewidth=2
        )

    plt.title("Optimizer Convergence (First 5 Epochs)", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

   # Log combined plot to W&B as a final run
    wandb.init(
        project="da6401_assignment_1_q2_3",
        group="part2_q2_3",
        job_type="optimizer_showdown_plot",
        tags=["part2", "q2_3", "summary_plot"],
        name="q2_3_optimizer_comparison_plot",
        reinit=True
    )
    wandb.log({"optimizer_convergence_plot": wandb.Image(plt)})
    wandb.finish()

    plt.show()


if __name__ == "__main__":
    run_optimizer_showdown()