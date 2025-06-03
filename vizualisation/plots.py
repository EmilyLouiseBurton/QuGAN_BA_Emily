import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="whitegrid", font_scale=1.2)

MODEL_COLORS = {
    36: 'blue',
    66: 'orange',
    72: 'green',
    132: 'red',
}

def run_plots(model_data, max_graphs=1000):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # --- (a) Average Number of Valid Graphs ---
    ax = axs[0, 0]
    for model_size, color in MODEL_COLORS.items():
        vals = model_data.get(model_size, {}).get("valid_graphs_all_seeds", [])
        if not vals:
            continue
        arr = np.array(vals) * max_graphs
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        x = np.arange(1, len(mean) + 1)
        ax.plot(x, mean, label=f"QuGAN({model_size})", color=color)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
    ax.axhline(y=max_graphs * 0.279, color='black', linestyle='--', label='Random Sample (27.9%)')
    ax.set_xlabel("epoch", fontweight='bold')
    ax.set_ylabel("#valid graphs", fontweight='bold')
    ax.set_title("(a) Average Number of Valid Graphs")
    ax.legend()
    ax.grid(True)

    # --- (b) Average Standard Deviation of Edge-Weights ---
    ax = axs[0, 1]
    for model_size, color in MODEL_COLORS.items():
        vals = model_data.get(model_size, {}).get("std_dev_all_seeds", [])
        if not vals:
            continue
        arr = np.array(vals)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        x = np.arange(1, len(mean) + 1)
        ax.plot(x, mean, label=f"QuGAN({model_size})", color=color)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
    ax.axhline(y=0.08, color='red', linestyle='--')
    ax.axhline(y=0.06, color='purple', linestyle='--')
    ax.set_xlabel("epoch", fontweight='bold')
    ax.set_ylabel("standard deviation", fontweight='bold')
    ax.set_title("(b) Average Standard Deviation of Edge-Weights")
    ax.legend()
    ax.grid(True)

    # --- (c) Edge Weight Distribution ---
    ax = axs[1, 0]
    for model_size, color in MODEL_COLORS.items():
        weights = model_data.get(model_size, {}).get("edge_weights", [])
        if weights:
            sns.kdeplot(weights, ax=ax, label=f"QuGAN({model_size})", color=color, linewidth=1.5)
    training_weights = model_data.get("train", {}).get("edge_weights", [])
    if training_weights:
        sns.kdeplot(training_weights, ax=ax, color='red', linestyle='--', label="Training Data", linewidth=1.5)
    ax.set_xlabel("weight", fontweight='bold')
    ax.set_ylabel("density", fontweight='bold')
    ax.set_title("(c) Edge-Weight Distribution")
    ax.legend()

    # --- (d) Average Loss of Generator ---
    ax = axs[1, 1]
    for model_size, color in MODEL_COLORS.items():
        losses = model_data.get(model_size, {}).get("gen_loss_all_seeds", [])
        if not losses:
            continue
        arr = np.array(losses)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        x = np.arange(1, len(mean) + 1)
        ax.plot(x, mean, label=f"QuGAN({model_size})", color=color)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
    ax.set_xlabel("step", fontweight='bold')
    ax.set_ylabel("BCE loss", fontweight='bold')
    ax.set_title("(d) Average Loss of Generator")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()