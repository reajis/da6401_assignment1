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


def calculate_accuracy(y_true, y_pred):
    true_labels = np.argmax(y_true, axis=1)
    pred_labels = np.argmax(y_pred, axis=1)
    return np.mean(true_labels == pred_labels)


def analyze_hidden_layers(model, X_sample, activation_name):
    """
    Runs a forward pass and analyzes hidden-layer activations.
    Excludes the final softmax output layer.
    """
    model.forward(X_sample)

    layer_stats = {}
    hidden_layers = model.get_layers()[:-1]

    for i, layer in enumerate(hidden_layers):
        A = layer.A

        stats = {
            "activations": A.copy(),
            "mean_activation": float(np.mean(A)),
            "std_activation": float(np.std(A)),
            "min_activation": float(np.min(A)),
            "max_activation": float(np.max(A)),
        }

        if activation_name == "relu":
            firing_counts = np.sum(A > 0, axis=0)
            dead_neurons = int(np.sum(firing_counts == 0))
            stats["dead_neurons"] = dead_neurons
            stats["firing_counts"] = firing_counts

        elif activation_name == "tanh":
            saturation_fraction = float(np.mean(np.abs(A) > 0.99))
            near_zero_fraction = float(np.mean(np.abs(A) < 1e-3))
            stats["saturation_fraction"] = saturation_fraction
            stats["near_zero_fraction"] = near_zero_fraction

        layer_stats[f"hidden_layer_{i+1}"] = stats

    return layer_stats


def run_dead_neuron_investigation():
    print("Loading MNIST dataset...")
    (X_train, y_train), (X_val, y_val), _ = load_and_preprocess_data(dataset_name="mnist")

    high_lr = 0.1
    activations = ["relu", "tanh"]
    epochs = 5
    batch_size = 64

    X_analysis = X_val[:1000]

    # hidden layers to monitor
    hidden_sizes = [128, 128]
    num_layers = len(hidden_sizes)

    results = {}

    for act in activations:
        print(f"\n--- Investigating {act.upper()} with LR={high_lr} ---")

        wandb.init(
            project="da6401_assignment_1_q2_5",
            group="part2_q2_5",
            job_type="dead_neuron_run",
            tags=["part2", "q2_5", act],
            name=f"q2_5_dead_neurons_{act}",
            reinit=True,
            config={
                "dataset": "mnist",
                "activation": act,
                "learning_rate": high_lr,
                "optimizer": "sgd",
                "epochs": epochs,
                "batch_size": batch_size,
                "hidden_sizes": hidden_sizes,
                "num_layers": num_layers,
                "weight_init": "random",
                "loss": "cross_entropy"
            }
        )

        # same initialization pattern for fair comparison
        set_seed(42)
        model = NeuralNetwork(
            input_size=784,
            hidden_sizes=hidden_sizes,
            num_layers=num_layers,
            output_size=10,
            activation=act,
            weight_init="random"
        )

        optimizer = Optimizer(
            model.get_layers(),
            optimizer_type="sgd",
            lr=high_lr
        )

        val_acc_history = []
        grad_norm_history = {f"hidden_layer_{i+1}": [] for i in range(num_layers)}
        final_layer_stats = None

        # same batch order across runs
        set_seed(123)

        for epoch in range(epochs):
            for X_batch, y_batch in get_batches(X_train, y_train, batch_size):
                y_pred = model.forward(X_batch)
                dA = cross_entropy_derivative(y_batch, y_pred)
                model.backward(dA)

                for i, layer in enumerate(model.get_layers()[:-1]):
                    grad_norm = np.linalg.norm(layer.grad_W)
                    grad_norm_history[f"hidden_layer_{i+1}"].append(grad_norm)

                optimizer.step()

            y_val_pred = model.forward(X_val)
            val_acc = calculate_accuracy(y_val, y_val_pred)
            val_acc_history.append(val_acc)

            layer_stats = analyze_hidden_layers(model, X_analysis, act)

            log_dict = {
                "epoch": epoch + 1,
                "val_accuracy": val_acc
            }

            for layer_name, stats in layer_stats.items():
                log_dict[f"{layer_name}_mean_activation"] = stats["mean_activation"]
                log_dict[f"{layer_name}_std_activation"] = stats["std_activation"]
                log_dict[f"{layer_name}_min_activation"] = stats["min_activation"]
                log_dict[f"{layer_name}_max_activation"] = stats["max_activation"]

                if act == "relu":
                    log_dict[f"{layer_name}_dead_neurons"] = stats["dead_neurons"]
                else:
                    log_dict[f"{layer_name}_saturation_fraction"] = stats["saturation_fraction"]
                    log_dict[f"{layer_name}_near_zero_fraction"] = stats["near_zero_fraction"]

            wandb.log(log_dict)
            print(f"Epoch {epoch + 1}/{epochs} - Val Acc: {val_acc:.4f}")

            final_layer_stats = layer_stats

        results[act] = {
            "val_acc_history": val_acc_history,
            "grad_norm_history": grad_norm_history,
            "final_layer_stats": final_layer_stats
        }

        # Final visualizations per run
        for layer_name, stats in final_layer_stats.items():
            # Activation distribution histogram
            plt.figure(figsize=(8, 4))
            plt.hist(stats["activations"].flatten(), bins=50, alpha=0.85)
            plt.title(f"{act.upper()} | {layer_name} activation distribution", fontsize=13)
            plt.xlabel("Activation value")
            plt.ylabel("Frequency")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            wandb.log({f"{layer_name}_activation_distribution": wandb.Image(plt)})
            plt.show()
            plt.close()

            if act == "relu":
                firing_counts = stats["firing_counts"]
                dead_neurons = stats["dead_neurons"]

                plt.figure(figsize=(10, 4))
                plt.bar(range(len(firing_counts)), firing_counts, alpha=0.8)
                plt.title(
                    f"{act.upper()} | {layer_name} firing counts\nDead neurons: {dead_neurons}",
                    fontsize=13
                )
                plt.xlabel("Neuron Index")
                plt.ylabel("Number of inputs with activation > 0")
                plt.grid(axis="y", alpha=0.3)
                plt.tight_layout()
                wandb.log({f"{layer_name}_firing_counts_plot": wandb.Image(plt)})
                plt.show()
                plt.close()

            else:
                plt.figure(figsize=(6, 4))
                plt.bar(
                    ["saturation_fraction", "near_zero_fraction"],
                    [stats["saturation_fraction"], stats["near_zero_fraction"]],
                    alpha=0.8
                )
                plt.title(f"{act.upper()} | {layer_name} activation behavior", fontsize=13)
                plt.ylabel("Fraction")
                plt.grid(axis="y", alpha=0.3)
                plt.tight_layout()
                wandb.log({f"{layer_name}_tanh_behavior_plot": wandb.Image(plt)})
                plt.show()
                plt.close()

        wandb.finish()

    # Validation accuracy comparison
    plt.figure(figsize=(10, 5))
    for act in activations:
        plt.plot(
            range(1, epochs + 1),
            results[act]["val_acc_history"],
            marker="o",
            linewidth=2,
            label=act.upper()
        )
    plt.title("Validation Accuracy: ReLU vs Tanh at High Learning Rate", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    wandb.init(
        project="da6401_assignment_1_q2_5",
        group="part2_q2_5",
        job_type="dead_neuron_summary",
        tags=["part2", "q2_5", "summary_plot"],
        name="q2_5_val_accuracy_comparison",
        reinit=True
    )
    wandb.log({"q2_5_val_accuracy_comparison": wandb.Image(plt)})
    wandb.finish()
    plt.show()
    plt.close()

    # Gradient comparison using first hidden layer
    plt.figure(figsize=(10, 5))
    for act in activations:
        grads = results[act]["grad_norm_history"]["hidden_layer_1"]
        plt.plot(
            range(1, len(grads) + 1),
            grads,
            linewidth=2,
            label=act.upper()
        )
    plt.title("First Hidden Layer Gradient Norms: ReLU vs Tanh", fontsize=14)
    plt.xlabel("Training Step")
    plt.ylabel("||dW||")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    wandb.init(
        project="da6401_assignment_1_q2_5",
        group="part2_q2_5",
        job_type="dead_neuron_summary",
        tags=["part2", "q2_5", "summary_plot"],
        name="q2_5_gradient_comparison",
        reinit=True
    )
    wandb.log({"q2_5_gradient_comparison": wandb.Image(plt)})
    wandb.finish()
    plt.show()
    plt.close()


if __name__ == "__main__":
    run_dead_neuron_investigation()