import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def run_plots(model_data, max_graphs=1000):
    sns.set(style="whitegrid", font_scale=1.2)

    MODEL_COLORS = {
        36: 'blue',
        66: 'orange',
        72: 'green',
        132: 'red',
        180: 'purple',
        90: 'teal',
    }

    TRAIN_STD = 0.08
    KDE_STD = 0.06

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()

    def plot_avg_epochs(ax, key, ylabel, title, baselines=False):
        for size, color in MODEL_COLORS.items():
            vals = model_data.get(size, {}).get(key, [])
            if not vals:
                continue
            vals = [np.array(v) for v in vals if isinstance(v, (list, np.ndarray)) and len(v) > 0]
            if not vals or not all(v.shape == vals[0].shape for v in vals):
                continue
            arr = np.array(vals)
            mean, std = arr.mean(axis=0), arr.std(axis=0)
            x = np.arange(1, len(mean) + 1)
            ax.plot(x, mean, label=f"QuGAN({size})", color=color)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
        if baselines:
            ax.axhline(y=TRAIN_STD, color='red', linestyle='--')
            ax.axhline(y=KDE_STD, color='purple', linestyle='--')
        ax.set_xlabel("Epoch", fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

    # (a) Valid Graphs
    plot_avg_epochs(axs[0], "valid_graphs_all_seeds", "#valid graphs", "(a) Average Number of Valid Graphs")

    # (b) Standard Deviation of Edge-Weights
    plot_avg_epochs(axs[1], "std_dev_all_seeds", "standard deviation", "(b) Average Standard Deviation of Edge-Weights", baselines=True)

    # (c) KDE Edge-Weight Distribution
    ax = axs[2]
    for size, color in MODEL_COLORS.items():
        weights = model_data.get(size, {}).get("all_edge_weights_all_seeds", [])
        flat = np.array(weights).flatten()
        if flat.size > 1 and np.std(flat) > 0:
            sns.kdeplot(flat, ax=ax, color=color, lw=2, label=f"QuGAN({size})", bw_adjust=0.5)
    real = model_data.get("train", {}).get("edge_weights", [])
    if real:
        sns.kdeplot(real, ax=ax, color="black", lw=2, linestyle="--", label="Real Data")
    ax.set_xlabel("weight", fontweight='bold')
    ax.set_ylabel("density", fontweight='bold')
    ax.set_title("(c) Edge-Weight Distribution")
    ax.legend()
    ax.grid(True)

    # (d) Generator Loss
    ax = axs[3]
    for size, color in MODEL_COLORS.items():
        losses = model_data.get(size, {}).get("gen_loss_all_seeds", [])
        steps = model_data.get(size, {}).get("steps_all_seeds", [])
        for loss, step in zip(losses, steps):
            ax.plot(step, loss, color=color, alpha=0.5)
    ax.set_xlabel("step", fontweight='bold')
    ax.set_ylabel("BCE loss", fontweight='bold')
    ax.set_title("(d) Average Loss of Generator")
    ax.grid(True)

    plt.tight_layout()
    plt.show()