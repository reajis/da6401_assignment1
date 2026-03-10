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


def clone_model_weights(model):
    return [{"W": layer.W.copy(), "b": layer.b.copy()} for layer in model.get_layers()]


def load_model_weights(model, weights):
    for layer, saved_params in zip(model.get_layers(), weights):
        layer.W = saved_params["W"].copy()
        layer.b = saved_params["b"].copy()


def run_fashion_mnist_challenge():
    print("Loading FASHION-MNIST dataset...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_preprocess_data(
        dataset_name="fashion_mnist"
    )

    epochs = 15
    batch_size = 64

    
    configurations = [
        {
            "name": "Config_1_MNIST_Best",
            "optimizer": "adam",
            "lr": 0.001,
            "activation": "relu",
            "num_layers": 2,
            "hidden_sizes": [128, 128],
            "weight_init": "xavier"
        },
        {
            "name": "Config_2_Deeper_ReLU",
            "optimizer": "nadam",
            "lr": 0.001,
            "activation": "relu",
            "num_layers": 3,
            "hidden_sizes": [128, 128, 128],
            "weight_init": "xavier"
        },
        {
            "name": "Config_3_Tanh_Control",
            "optimizer": "adam",
            "lr": 0.0005,
            "activation": "tanh",
            "num_layers": 2,
            "hidden_sizes": [128, 128],
            "weight_init": "xavier"
        }
    ]

    challenge_results = {}

    # Shared starting point for fair comparison within each config
    set_seed(42)
    base_models = {}
    for cfg in configurations:
        model = NeuralNetwork(
            input_size=784,
            hidden_sizes=cfg["hidden_sizes"],
            num_layers=cfg["num_layers"],
            output_size=10,
            activation=cfg["activation"],
            weight_init=cfg["weight_init"]
        )
        base_models[cfg["name"]] = clone_model_weights(model)

    for cfg in configurations:
        print(f"\n--- Running {cfg['name']} ---")

        wandb.init(
            project="da6401_assignment_1_q2_10",
            group="part2_q2_10",
            job_type="fashion_transfer_run",
            tags=["part2", "q2_10", cfg["name"]],
            name=f"q2_10_fashion_{cfg['name']}",
            config={
                "dataset": "fashion_mnist",
                "epochs": epochs,
                "batch_size": batch_size,
                **cfg
            },
            reinit=True
        )

        model = NeuralNetwork(
            input_size=784,
            hidden_sizes=cfg["hidden_sizes"],
            num_layers=cfg["num_layers"],
            output_size=10,
            activation=cfg["activation"],
            weight_init=cfg["weight_init"]
        )
        load_model_weights(model, base_models[cfg["name"]])

        optimizer = Optimizer(
            model.get_layers(),
            optimizer_type=cfg["optimizer"],
            lr=cfg["lr"]
        )

        val_accs = []
        test_accs = []
        best_val_acc = -1.0
        best_test_acc_at_best_val = -1.0
        best_epoch = -1

        # same batch order across runs
        set_seed(123)

        for epoch in range(epochs):
            for X_batch, y_batch in get_batches(X_train, y_train, batch_size):
                y_pred = model.forward(X_batch)
                dA = cross_entropy_derivative(y_batch, y_pred)
                model.backward(dA)
                optimizer.step()

            y_val_pred = model.forward(X_val)
            val_acc = calculate_accuracy(y_val, y_val_pred)
            val_accs.append(val_acc)

            y_test_pred = model.forward(X_test)
            test_acc = calculate_accuracy(y_test, y_test_pred)
            test_accs.append(test_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc_at_best_val = test_acc
                best_epoch = epoch + 1

            wandb.log({
                "epoch": epoch + 1,
                "val_accuracy": val_acc,
                "test_accuracy": test_acc,
                "best_val_accuracy": best_val_acc,
                "best_test_accuracy_at_best_val": best_test_acc_at_best_val
            })

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Test Acc: {test_acc:.4f}"
            )

        challenge_results[cfg["name"]] = {
            "val_accs": val_accs,
            "test_accs": test_accs,
            "best_val_accuracy": best_val_acc,
            "best_test_accuracy_at_best_val": best_test_acc_at_best_val,
            "best_epoch": best_epoch
        }

        wandb.run.summary["best_epoch"] = best_epoch
        wandb.run.summary["best_val_accuracy"] = best_val_acc
        wandb.run.summary["best_test_accuracy_at_best_val"] = best_test_acc_at_best_val

        wandb.finish()

    # Validation plot
    plt.figure(figsize=(10, 6))
    for name, result in challenge_results.items():
        plt.plot(
            range(1, epochs + 1),
            result["val_accs"],
            label=name.replace("_", " "),
            marker="o",
            linewidth=2
        )

    plt.title("Fashion-MNIST Challenge: Validation Accuracy of 3 Chosen Configurations", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    wandb.init(
        project="da6401_assignment_1_q2_10",
        group="part2_q2_10",
        job_type="fashion_transfer_summary",
        tags=["part2", "q2_10", "summary_plot"],
        name="q2_10_fashion_val_plot",
        reinit=True
    )
    wandb.log({"q2_10_fashion_val_plot": wandb.Image(plt)})
    wandb.finish()
    plt.show()
    plt.close()

    # Test plot
    plt.figure(figsize=(10, 6))
    for name, result in challenge_results.items():
        plt.plot(
            range(1, epochs + 1),
            result["test_accs"],
            label=name.replace("_", " "),
            marker="o",
            linewidth=2
        )

    plt.title("Fashion-MNIST Challenge: Test Accuracy of 3 Chosen Configurations", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    wandb.init(
        project="da6401_assignment_1_q2_10",
        group="part2_q2_10",
        job_type="fashion_transfer_summary",
        tags=["part2", "q2_10", "summary_plot"],
        name="q2_10_fashion_test_plot",
        reinit=True
    )
    wandb.log({"q2_10_fashion_test_plot": wandb.Image(plt)})
    wandb.finish()
    plt.show()
    plt.close()

    print("\nFinal summary:")
    for name, result in challenge_results.items():
        print(
            f"{name}: "
            f"Best Val Acc = {result['best_val_accuracy']:.4f}, "
            f"Test Acc at Best Val = {result['best_test_accuracy_at_best_val']:.4f}, "
            f"Best Epoch = {result['best_epoch']}"
        )


if __name__ == "__main__":
    run_fashion_mnist_challenge()