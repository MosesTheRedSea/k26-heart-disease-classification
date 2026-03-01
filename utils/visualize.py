import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def save_run_outputs(base_directory):
    os.makedirs(base_directory, exist_ok=True)

    run_nums = []
    for name in os.listdir(base_directory):
        full_path = os.path.join(base_directory, name)
        if os.path.isdir(full_path) and re.fullmatch(r"\d+", name):
            run_nums.append(int(name))

    next_run = max(run_nums, default=0) + 1
    run_dir = os.path.join(base_directory, str(next_run))

    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def plot_and_save_data(X_train, y_train, base_dir="outputs/train"):
    run_dir = save_run_outputs(base_dir)

    plt.figure()
    unique, counts = np.unique(y_train, return_counts=True)
    plt.bar(["Absence (0)", "Presence (1)"], counts)
    plt.title("Heart Disease Class Distribution")
    plt.ylabel("Count")
    plt.savefig(os.path.join(run_dir, "class_distribution.png"))
    plt.close()

    n_features = X_train.shape[1]

    for i in range(n_features):
        plt.figure()
        plt.hist(X_train[y_train == 0, i], bins=30, alpha=0.5, label="Absence")
        plt.hist(X_train[y_train == 1, i], bins=30, alpha=0.5, label="Presence")
        plt.title(f"Feature {i} Distribution by Class")
        plt.xlabel(f"Feature {i}")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig(os.path.join(run_dir, f"feature_{i}_hist.png"))
        plt.close()

    plt.figure(figsize=(10, 8))
    corr = np.corrcoef(X_train.T)
    sns.heatmap(corr, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.savefig(os.path.join(run_dir, "correlation_heatmap.png"))
    plt.close()

    print(f"All plots saved in directory: {run_dir}")