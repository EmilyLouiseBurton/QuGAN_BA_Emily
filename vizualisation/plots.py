import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from training.config import MODEL_DATA, HYPERPARAMS

# Set global plot style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": (6, 4)
})

MODEL_COLORS = {
    36: 'blue',
    66: 'orange',
    72: 'green',
    132: 'red'
}

EPOCHS = HYPERPARAMS["epochs"]

def plot_valid_graphs():
    from training.config import EVAL_SETTINGS
    max_graphs = EVAL_SETTINGS["num_graphs"]

    plt.figure()
    for model_size, color in MODEL_COLORS.items():
        data = MODEL_DATA.get(model_size, {}).get("valid_graphs", [])
        if not data:
            continue

        arr = np.array(data, dtype=np.float32)
        x = np.arange(1, len(arr) + 1)

        plt.plot(x, arr, label=f"QuGAN({model_size})", color=color)

    plt.axhline(y=max_graphs * 0.279, color='black', linestyle='--', label='Random Sample (27.9%)')
    plt.xlabel('Epoch')
    plt.ylabel(f'# Valid Graphs (out of {max_graphs})')
    plt.ylim(0, max_graphs)
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_std_dev():
    plt.figure()
    for model_size, color in MODEL_COLORS.items():
        arr = np.array(MODEL_DATA.get(model_size, {}).get("std_dev", []), dtype=np.float32)
        if arr.size == 0:
            continue

        x = np.arange(1, len(arr) + 1)
        plt.plot(x, arr, label=f"QuGAN({model_size})", color=color)

        std = np.nanstd(arr)
        plt.fill_between(x, arr - std, arr + std, color=color, alpha=0.2)

    plt.axhline(y=0.0826, color='red', linestyle='--', label='Training Data STD')
    plt.xlabel('Epoch')
    plt.ylabel('Standard Deviation')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_edge_weight_distribution():
    plt.figure()
    for model_size, color in MODEL_COLORS.items():
        weights = MODEL_DATA.get(model_size, {}).get("edge_weights", [])
        if weights:
            sns.kdeplot(weights, label=f"QuGAN({model_size})", color=color, fill=False, linewidth=1.5)

    training_data_weights = MODEL_DATA.get("train", {}).get("edge_weights", [])
    if training_data_weights:
        sns.kdeplot(training_data_weights, color='red', linestyle='--', label="Training Data")

    plt.xlabel('Weight')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_generator_loss():
    plt.figure()
    for model_size, color in MODEL_COLORS.items():
        losses = MODEL_DATA.get(model_size, {}).get("gen_loss", [])
        if not losses:
            continue

        losses = np.array(losses, dtype=np.float32)
        epochs = np.arange(1, len(losses) + 1)
        plt.plot(epochs, losses, label=f"QuGAN({model_size})", color=color)

    plt.xlabel('Epoch')
    plt.ylabel('Generator BCE Loss')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()

def run_plots():
    print("→ Plotting valid graph counts")
    plot_valid_graphs()
    print("→ Plotting standard deviation")
    plot_std_dev()
    print("→ Plotting edge weight distributions")
    plot_edge_weight_distribution()
    print("→ Plotting generator loss curves")
    plot_generator_loss()
