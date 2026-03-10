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
from ann.r_optimizers import Optimizer
from ann.r_objective_functions import cross_entropy_derivative
from utils.r_data_loader import load_and_preprocess_data, get_batches


def set_seed(seed=42):
    np.random.seed(seed)


def run_vanishing_gradient_analysis():
    print("Loading MNIST dataset...")
    (X_train, y_train), _, _ = load_and_preprocess_data(dataset_name="mnist")

    activations = ["sigmoid", "relu"]
    batch_size = 64
    learning_rate = 0.001
    max_steps = 200

    # different network configurations as required
    configs = {
        "shallow_2x64": [64, 64],
        "deep_4x64": [64, 64, 64, 64]
    }

    gradient_norms_history = {}

    for config_name, hidden_sizes in configs.items():
        num_layers = len(hidden_sizes)

        for act in activations:
            print(f"\n--- Config: {config_name} | Activation: {act.upper()} ---")

            
            wandb.init(
                project="da6401_assignment_1_q2_4",
                group="part2_q2_4",
                job_type="vanishing_gradient_run",
                tags=["part2", "q2_4", config_name, act],
                name=f"q2_4_{config_name}_{act}",
                reinit=True,
                config={
                    "dataset": "mnist",
                    "optimizer": "adam",
                    "activation": act,
                    "hidden_sizes": hidden_sizes,
                    "num_layers": num_layers,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "tracked_layer": "first_hidden_layer",
                    "max_steps": max_steps,
                    "weight_init": "xavier",
                    "loss": "cross_entropy"
                }
            )

            # reset seed so runs are comparable
            set_seed(42)

            model = NeuralNetwork(
                input_size=784,
                hidden_sizes=hidden_sizes,
                num_layers=num_layers,
                output_size=10,
                activation=act,
                weight_init="xavier"
            )

            optimizer = Optimizer(
                model.get_layers(),
                optimizer_type="adam",
                lr=learning_rate
            )

            norms = []
            step = 0

            # reset again so batch order stays the same across runs
            set_seed(123)

            for X_batch, y_batch in get_batches(X_train, y_train, batch_size):
                if step >= max_steps:
                    break

                y_pred = model.forward(X_batch)
                dA = cross_entropy_derivative(y_batch, y_pred)
                model.backward(dA)

                # gradient norm of first hidden layer
                first_layer_grad_W = model.get_layers()[0].grad_W
                grad_norm = np.linalg.norm(first_layer_grad_W)

                norms.append(grad_norm)

                wandb.log({
                    "step": step + 1,
                    "first_layer_grad_norm": grad_norm
                })

                optimizer.step()
                step += 1

            gradient_norms_history[(config_name, act)] = norms
            wandb.finish()

    # Combined comparison plot
    plt.figure(figsize=(11, 6))

    for (config_name, act), norms in gradient_norms_history.items():
        plt.plot(
            range(1, len(norms) + 1),
            norms,
            label=f"{config_name} | {act.upper()}",
            linewidth=2
        )

    plt.title("Vanishing Gradient Analysis: First Hidden Layer Gradient Norms", fontsize=14)
    plt.xlabel("Training Step")
    plt.ylabel("L2 Norm of Gradients (||dW||)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


    # log combined plot
    wandb.init(
        project="da6401_assignment_1_q2_4",
        group="part2_q2_4",
        job_type="vanishing_gradient_plot",
        tags=["part2", "q2_4", "summary_plot"],
        name="q2_4_combined_gradient_plot",
        reinit=True
    )
    wandb.log({"vanishing_gradient_comparison_plot": wandb.Image(plt)})
    wandb.finish()

    plt.show()


if __name__ == "__main__":
    run_vanishing_gradient_analysis()