import os
import sys
import json
import numpy as np
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Robust path handling
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from ann.r_neural_network import NeuralNetwork
from utils.r_data_loader import load_and_preprocess_data


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


def run_error_analysis():
    wandb.init(
        project="da6401_assignment_1_q2_8",
        group="part2_q2_8",
        job_type="error_analysis",
        tags=["part2", "q2_8"],
        name="q2_8_error_analysis",
        reinit=True
    )

    config_path = os.path.join(ROOT_DIR, "best_config.json")
    model_path = os.path.join(ROOT_DIR, "best_model.npy")

    # Load best config
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Could not find {config_path}. Run train.py first.")
        wandb.finish()
        return

    config["hidden_size"] = normalize_hidden_sizes(config["hidden_size"], config["num_layers"])

    print(f"Loading {config['dataset']} test data...")
    _, _, (X_test, y_test) = load_and_preprocess_data(dataset_name=config["dataset"])

    # Rebuild model
    model = NeuralNetwork(
        input_size=X_test.shape[1],
        hidden_sizes=config["hidden_size"],
        num_layers=config["num_layers"],
        output_size=y_test.shape[1],
        activation=config["activation"],
        weight_init=config["weight_init"]
    )

    # Load weights
    try:
        weights_list = np.load(model_path, allow_pickle=True)
        if len(weights_list) != len(model.get_layers()):
            raise ValueError(
                f"Mismatch between saved weights ({len(weights_list)} layers) and "
                f"constructed model ({len(model.get_layers())} layers)."
            )

        for layer, saved_params in zip(model.get_layers(), weights_list):
            layer.W = saved_params["W"].copy()
            layer.b = saved_params["b"].copy()

        print("Best model weights loaded successfully!")
    except FileNotFoundError:
        print(f"Could not find {model_path}. Run train.py first.")
        wandb.finish()
        return

    # Inference
    y_pred_probs = model.forward(X_test)
    y_true_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)


    # PART A: Confusion Matrix
 
    cm = confusion_matrix(y_true_labels, y_pred_labels)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"Confusion Matrix (Test Set: {config['dataset']})", fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    # Annotate counts
    threshold = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black"
            )

    plt.tight_layout()
    wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.show()
    plt.close()


    # Creative Visualization of Failures

    incorrect_indices = np.where(y_pred_labels != y_true_labels)[0]

    if len(incorrect_indices) == 0:
        print("No incorrect predictions found.")
        wandb.finish()
        return

    wrong_confidences = np.max(y_pred_probs[incorrect_indices], axis=1)
    most_confident_wrong_idx = incorrect_indices[np.argsort(wrong_confidences)[::-1]]

    print("\nVisualizing Top 6 Most Confident Mistakes")

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle("Highly Confident Misclassifications", fontsize=16)

    mistake_table = wandb.Table(columns=["Image", "True Label", "Predicted Label", "Confidence"])

    for i, ax in enumerate(axes.flat):
        if i < len(most_confident_wrong_idx):
            idx = most_confident_wrong_idx[i]
            true_label = int(y_true_labels[idx])
            pred_label = int(y_pred_labels[idx])
            confidence = float(np.max(y_pred_probs[idx]) * 100.0)

            img = X_test[idx].reshape(28, 28)

            ax.imshow(img, cmap="gray")
            ax.set_title(
                f"True: {true_label} | Pred: {pred_label}\nConfidence: {confidence:.1f}%",
                color="red"
            )
            ax.axis("off")

            mistake_table.add_data(
                wandb.Image(img, caption=f"T:{true_label}, P:{pred_label}"),
                true_label,
                pred_label,
                confidence
            )
        else:
            ax.axis("off")

    plt.tight_layout()
    wandb.log({
        "confident_failures": wandb.Image(plt),
        "confident_failure_table": mistake_table
    })
    plt.show()
    plt.close()

    wandb.finish()


if __name__ == "__main__":
    run_error_analysis()