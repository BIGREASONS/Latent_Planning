import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_training_curves(log_csv_path: str, output_img_path: str):
    """Plots training and evaluation loss from a CSV log file."""
    if not os.path.exists(log_csv_path):
        print(f"File not found: {log_csv_path}")
        return

    df = pd.read_csv(log_csv_path)

    plt.figure(figsize=(10, 6))

    if "loss" in df.columns and "step" in df.columns:
        train_data = df.dropna(subset=["loss"])
        plt.plot(
            train_data["step"],
            train_data["loss"],
            label="Train Loss",
            color="blue",
            alpha=0.7,
        )

    if "eval_loss" in df.columns and "step" in df.columns:
        eval_data = df.dropna(subset=["eval_loss"])
        plt.plot(
            eval_data["step"],
            eval_data["eval_loss"],
            label="Eval Loss",
            color="orange",
            marker="o",
        )

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training and Evaluation Curves")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    os.makedirs(os.path.dirname(os.path.abspath(output_img_path)), exist_ok=True)
    plt.savefig(output_img_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_evaluation_metric(
    csv_path: str, metric_name: str, x_axis: str, output_img_path: str
):
    """Plots an arbitrary evaluation metric against an x-axis (e.g., layer, step)."""
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    if metric_name not in df.columns or x_axis not in df.columns:
        print(f"Columns {metric_name} or {x_axis} not in CSV.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(df[x_axis], df[metric_name], marker="s", color="green")
    plt.xlabel(x_axis.capitalize())
    plt.ylabel(metric_name.capitalize())
    plt.title(f"{metric_name.capitalize()} vs {x_axis.capitalize()}")
    plt.grid(True, linestyle="--", alpha=0.5)

    os.makedirs(os.path.dirname(os.path.abspath(output_img_path)), exist_ok=True)
    plt.savefig(output_img_path, dpi=300, bbox_inches="tight")
    plt.close()
