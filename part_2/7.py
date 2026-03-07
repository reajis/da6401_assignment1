import argparse
import wandb
import matplotlib.pyplot as plt
import numpy as np


def plot_global_performance(sweep_path, min_train_acc=0.90, gap_threshold=0.10):
    """
    Creates an overlay plot of Training vs Test Accuracy across all finished runs
    in a W&B sweep, sorted by test accuracy.

    Also prints runs with high train accuracy but poor test accuracy.
    """
    print("Connecting to W&B API...")
    api = wandb.Api()

    try:
        sweep = api.sweep(sweep_path)
    except Exception as e:
        print(f"Error connecting to sweep: {e}")
        return

    run_data = []

    print(f"Fetching data from sweep: {sweep.id} ...")
    for run in sweep.runs:
        if run.state != "finished":
            continue

        train_acc = run.summary.get("train_accuracy")
        test_acc = run.summary.get("test_accuracy")

        # Skip incomplete runs
        if train_acc is None or test_acc is None:
            continue

        gap = train_acc - test_acc

        run_data.append({
            "name": run.name,
            "id": run.id,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "gap": gap
        })

    if not run_data:
        print("No finished runs with both train_accuracy and test_accuracy found.")
        return

    # Sort by test accuracy for a cleaner overlay plot
    run_data = sorted(run_data, key=lambda x: x["test_accuracy"])

    train_accs = np.array([r["train_accuracy"] for r in run_data])
    test_accs = np.array([r["test_accuracy"] for r in run_data])
    gaps = np.array([r["gap"] for r in run_data])

    # Identify potentially overfitting runs
    suspicious_runs = [
        r for r in run_data
        if r["train_accuracy"] >= min_train_acc and r["gap"] >= gap_threshold
    ]

    x = np.arange(len(run_data))

    # ---------------- Overlay plot ----------------
    plt.figure(figsize=(12, 6))
    plt.plot(x, train_accs, label="Training Accuracy", linewidth=2)
    plt.plot(x, test_accs, label="Test Accuracy", linewidth=2)

    plt.fill_between(
        x,
        test_accs,
        train_accs,
        where=(train_accs > test_accs),
        alpha=0.2,
        label="Train-Test Gap"
    )

    plt.title("Global Performance Analysis: Training vs Test Accuracy Across Sweep Runs", fontsize=14)
    plt.xlabel("Run Index (sorted by Test Accuracy)")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    overlay_fig = plt.gcf()
    plt.show()

    # Gap plot 
    plt.figure(figsize=(12, 4))
    plt.plot(x, gaps, linewidth=2, marker="o")
    plt.axhline(gap_threshold, linestyle="--", label=f"Gap threshold = {gap_threshold:.2f}")
    plt.title("Generalization Gap Across Sweep Runs", fontsize=14)
    plt.xlabel("Run Index (sorted by Test Accuracy)")
    plt.ylabel("Train Accuracy - Test Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    gap_fig = plt.gcf()
    plt.show()

    # Print suspicious runs for report writing
    print("\nRuns with high training accuracy but relatively poor test accuracy:")
    if not suspicious_runs:
        print("None found using the current thresholds.")
    else:
        suspicious_runs = sorted(suspicious_runs, key=lambda x: x["gap"], reverse=True)
        for r in suspicious_runs[:10]:
            print(
                f"Run: {r['name']} ({r['id']}) | "
                f"Train Acc: {r['train_accuracy']:.4f} | "
                f"Test Acc: {r['test_accuracy']:.4f} | "
                f"Gap: {r['gap']:.4f}"
            )

    # Log summary plots/table to a separate W&B project
    wandb.init(
        project="da6401_assignment_1_q2_7",
        group="part2_q2_7",
        job_type="global_performance_summary",
        tags=["part2", "q2_7", "summary_plot"],
        name="q2_7_global_performance_analysis",
        reinit=True,
        config={
            "sweep_path": sweep_path,
            "min_train_acc": min_train_acc,
            "gap_threshold": gap_threshold
        }
    )

    suspicious_table = wandb.Table(columns=["Run Name", "Run ID", "Train Accuracy", "Test Accuracy", "Gap"])
    for r in sorted(suspicious_runs, key=lambda x: x["gap"], reverse=True):
        suspicious_table.add_data(
            r["name"],
            r["id"],
            r["train_accuracy"],
            r["test_accuracy"],
            r["gap"]
        )

    wandb.log({
        "q2_7_overlay_plot": wandb.Image(overlay_fig),
        "q2_7_gap_plot": wandb.Image(gap_fig),
        "q2_7_suspicious_runs": suspicious_table
    })

    wandb.finish()
    plt.close("all")


def main():
    parser = argparse.ArgumentParser(description="Q2.7 Global Performance Analysis from W&B Sweep")
    parser.add_argument(
        "--sweep_path",
        type=str,
        required=True,
        help="W&B sweep path in the form entity/project/sweep_id"
    )
    parser.add_argument(
        "--min_train_acc",
        type=float,
        default=0.90,
        help="Minimum training accuracy to flag a run"
    )
    parser.add_argument(
        "--gap_threshold",
        type=float,
        default=0.10,
        help="Minimum train-test gap to flag a run"
    )

    args = parser.parse_args()

    plot_global_performance(
        sweep_path=args.sweep_path,
        min_train_acc=args.min_train_acc,
        gap_threshold=args.gap_threshold
    )


if __name__ == "__main__":
    main()