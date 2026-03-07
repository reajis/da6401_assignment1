import numpy as np
import wandb
import matplotlib.pyplot as plt
from keras.datasets import mnist, fashion_mnist


def load_dataset(dataset_name="mnist"):
    if dataset_name == "mnist":
        return mnist.load_data()
    elif dataset_name == "fashion_mnist":
        return fashion_mnist.load_data()
    else:
        raise ValueError("dataset_name must be 'mnist' or 'fashion_mnist'")


def log_data_exploration(dataset_name="mnist", project_name="da6401_assignment_1"):
    np.random.seed(42)

    wandb.init(
        project="da6401_assignment_1_q2_1",
        group="part2_q2_1",
        job_type="show_sample",
        name=f"q2_1_data_exploration_{dataset_name}",
        config={"dataset": dataset_name}
    )

    print(f"Loading {dataset_name} dataset...")
    (X_train, y_train), _ = load_dataset(dataset_name)

    columns = ["Class Label", "Sample 1", "Sample 2", "Sample 3", "Sample 4", "Sample 5"]
    sample_table = wandb.Table(columns=columns)

    num_classes = 10
    samples_per_class = 5
    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(8, 12))
    fig.suptitle(f"{dataset_name.upper()} Data Exploration", fontsize=16)

    for class_label in range(num_classes):
        class_indices = np.where(y_train == class_label)[0]
        selected_indices = np.random.choice(class_indices, samples_per_class, replace=False)

        table_row = [str(class_label)]

        for i, idx in enumerate(selected_indices):
            img_array = X_train[idx]

            table_row.append(wandb.Image(img_array, caption=f"class {class_label}, idx {idx}"))

            ax = axes[class_label, i]
            ax.imshow(img_array, cmap="gray")
            ax.axis("off")
            if i == 0:
                ax.set_title(f"Class {class_label}")

        sample_table.add_data(*table_row)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    wandb.log({
        f"{dataset_name}_class_samples": sample_table,
        f"{dataset_name}_sample_grid": wandb.Image(fig)
    })

    plt.close(fig)
    wandb.finish()
    print("Data exploration logged successfully to W&B!")


if __name__ == "__main__":
    log_data_exploration()