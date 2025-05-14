import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from training.config import HYPERPARAMS, EVAL_SETTINGS  # Removed MODEL_DATA

sns.set(style="whitegrid", font_scale=1.2)

MODEL_COLORS = {
    36: 'blue',
    66: 'orange',
    72: 'green',
    132: 'red'
}

EPOCHS = HYPERPARAMS["epochs"]
max_graphs = EVAL_SETTINGS["num_graphs"]

def run_plots(model_data):  # <-- Accept model_data as argument
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # --- (a) Valid Graphs ---
    ax = axs[0, 0]
    for model_size, color in MODEL_COLORS.items():
        data = model_data.get(model_size, {}).get("valid_graphs", [])
        if not data:
            print(f"[Plot] No valid_graphs data for model {model_size}")
            continue
        arr = np.array(data, dtype=np.float32) * max_graphs
        x = np.arange(1, len(arr) + 1)
        ax.plot(x, arr, label=f"QuGAN({model_size})", color=color)

    ax.axhline(y=max_graphs * 0.279, color='black', linestyle='--', label='Random Sample (27.9%)')
    ax.set_xlabel("epoch")
    ax.set_ylabel("#valid graphs")
    ax.set_title("(a) Average Number of Valid Graphs")
    ax.legend()
    ax.grid(True)

    # --- (b) Std Dev ---
    ax = axs[0, 1]
    for model_size, color in MODEL_COLORS.items():
        std_devs = model_data.get(model_size, {}).get("std_dev", [])
        if not std_devs or isinstance(std_devs, float):
            print(f"[Plot] No std_dev data for model {model_size}")
            continue
        std_devs = np.array(std_devs)
        x = np.arange(1, len(std_devs) + 1)
        ax.plot(x, std_devs, label=f"QuGAN({model_size})", color=color)

    ax.set_xlabel("epoch")
    ax.set_ylabel("standard deviation")
    ax.set_title("(b) Average Standard Deviation of Edge-Weights")
    ax.grid(True)

    # --- (c) Edge Weight Distribution ---
    ax = axs[1, 0]
    for model_size, color in MODEL_COLORS.items():
        weights = model_data.get(model_size, {}).get("edge_weights", [])
        if weights:
            sns.kdeplot(weights, ax=ax, label=f"QuGAN({model_size})", color=color, linewidth=1.5)
        else:
            print(f"[Plot] No edge_weights data for model {model_size}")

    training_weights = model_data.get("train", {}).get("edge_weights", [])
    if training_weights:
        sns.kdeplot(training_weights, ax=ax, color='red', linestyle='--', label="Training Data")
    else:
        print("[Plot] No edge_weights for training data")

    ax.set_xlabel("weight")
    ax.set_ylabel("density")
    ax.set_title("(c) Edge-Weight Distribution")
    ax.legend()

    # --- (d) Generator Loss ---
    ax = axs[1, 1]
    for model_size, color in MODEL_COLORS.items():
        losses = model_data.get(model_size, {}).get("gen_loss", [])
        if not losses:
            print(f"[Plot] No gen_loss data for model {model_size}")
            continue
        losses = np.array(losses, dtype=np.float32)
        x = np.arange(1, len(losses) + 1)
        ax.plot(x, losses, label=f"QuGAN({model_size})", color=color)

    ax.set_xlabel("step")
    ax.set_ylabel("BCE loss")
    ax.set_title("(d) Average Loss of Generator")
    ax.legend()
    ax.grid(True)

    # --- Tight layout ---
    plt.tight_layout()
    plt.show()