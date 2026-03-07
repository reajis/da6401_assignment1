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

from ann.neural_network import NeuralNetwork
from ann.optimizers import Optimizer
from ann.objective_functions import cross_entropy_derivative
from utils.data_loader import load_and_preprocess_data, get_batches


def set_seed(seed=42):
    np.random.seed(seed)


def run_symmetry_experiment():
    print("Loading MNIST dataset...")
    (X_train, y_train), _, _ = load_and_preprocess_data(dataset_name="mnist")

    iterations = 50
    batch_size = 64
    learning_rate = 0.01

    inits_to_test = ["zeros", "xavier"]

    # Store gradient norms for first 5 neurons in the first hidden layer
    neuron_gradients = {
        "zeros": {i: [] for i in range(5)},
        "xavier": {i: [] for i in range(5)}
    }

    for init_type in inits_to_test:
        print(f"\n--- Running {init_type.upper()} Initialization ---")

        wandb.init(
            project="da6401_assignment_1_q2_9",
            group="part2_q2_9",
            job_type="symmetry_run",
            tags=["part2", "q2_9", init_type],
            name=f"q2_9_symmetry_{init_type}",
            reinit=True,
            config={
                "dataset": "mnist",
                "initialization": init_type,
                "iterations": iterations,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "hidden_sizes": [128, 64],
                "num_layers": 2,
                "activation": "relu",
                "optimizer": "sgd",
                "tracked_layer": "first_hidden_layer",
                "tracked_neurons": [0, 1, 2, 3, 4],
                "loss": "cross_entropy"
            }
        )

        # Reset model seed for reproducibility
        set_seed(42)

        model = NeuralNetwork(
            input_size=784,
            hidden_sizes=[128, 64],
            num_layers=2,
            output_size=10,
            activation="relu",
            weight_init="xavier"  # overwritten for zero init below
        )

        # Force all weights and biases to zero
        if init_type == "zeros":
            for layer in model.get_layers():
                layer.W = np.zeros_like(layer.W)
                layer.b = np.zeros_like(layer.b)

        optimizer = Optimizer(
            model.get_layers(),
            optimizer_type="sgd",
            lr=learning_rate
        )

        iteration_count = 0

        # Reset again so both runs see same batch order
        set_seed(123)

        for X_batch, y_batch in get_batches(X_train, y_train, batch_size):
            if iteration_count >= iterations:
                break

            y_pred = model.forward(X_batch)
            dA = cross_entropy_derivative(y_batch, y_pred)
            model.backward(dA)

            # Gradient matrix of first hidden layer: shape (784, 128)
            first_layer_grad = model.get_layers()[0].grad_W

            log_dict = {"iteration": iteration_count + 1}

            # Track L2 norm of gradient column for neurons 0..4
            for i in range(5):
                neuron_grad_norm = np.linalg.norm(first_layer_grad[:, i])
                neuron_gradients[init_type][i].append(neuron_grad_norm)
                log_dict[f"neuron_{i+1}_grad_norm"] = neuron_grad_norm

            wandb.log(log_dict)

            optimizer.step()
            iteration_count += 1

        wandb.finish()

    # Combined comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Symmetry Breaking: Gradients of 5 Neurons over First 50 Iterations", fontsize=16)

    # Zeros initialization
    for i in range(5):
        ax1.plot(
            range(1, iterations + 1),
            neuron_gradients["zeros"][i],
            label=f"Neuron {i+1}",
            linewidth=max(1, 5 - i),
            alpha=0.8
        )
    ax1.set_title("Zeros Initialization", color="red")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Gradient Norm")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Xavier initialization
    for i in range(5):
        ax2.plot(
            range(1, iterations + 1),
            neuron_gradients["xavier"][i],
            label=f"Neuron {i+1}",
            linewidth=2,
            alpha=0.8
        )
    ax2.set_title("Xavier Initialization", color="green")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Gradient Norm")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Log final comparison plot in a fresh run
    wandb.init(
        project="da6401_assignment_1_q2_9",
        group="part2_q2_9",
        job_type="symmetry_summary",
        tags=["part2", "q2_9", "summary_plot"],
        name="q2_9_symmetry_comparison_plot",
        reinit=True
    )
    wandb.log({"symmetry_comparison": wandb.Image(plt)})
    wandb.finish()

    plt.show()


if __name__ == "__main__":
    run_symmetry_experiment()