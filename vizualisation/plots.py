import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="whitegrid", font_scale=1.2)

MODEL_COLORS = {
    36: 'blue',
    66: 'orange',
    72: 'green',
    132: 'red',
    180: 'purple',
    90: 'teal',
}

def run_plots(model_data, max_graphs=1000):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # --- (a) Average Number of Valid Graphs ---
    ax = axs[0, 0]
    for model_size, color in MODEL_COLORS.items():
        vals = model_data.get(model_size, {}).get("valid_graphs_all_seeds", [])
        if not vals:
            continue
        vals = [np.array(v) for v in vals if isinstance(v, (list, np.ndarray))]
        if not vals or not all(v.shape == vals[0].shape for v in vals):
            print(f"Skipping model {model_size} due to inconsistent shape in valid_graphs_all_seeds")
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
        vals = [np.array(v) for v in vals if isinstance(v, (list, np.ndarray))]
        if not vals or not all(v.shape == vals[0].shape for v in vals):
            print(f"Skipping model {model_size} due to inconsistent shape in std_dev_all_seeds")
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
        all_weights_seeds = model_data.get(model_size, {}).get("edge_weights_all_seeds", [])
        flat_weights = [w for seed_list in all_weights_seeds for w in seed_list]
        if flat_weights:
            sns.kdeplot(flat_weights, ax=ax, label=f"QuGAN({model_size})", color=color, linewidth=1.5)

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
        losses = [np.array(l) for l in losses if isinstance(l, (list, np.ndarray))]
        if not losses or not all(l.shape == losses[0].shape for l in losses):
            print(f"Skipping model {model_size} due to inconsistent shape in gen_loss_all_seeds")
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
    
    # --- (e) Validity–Diversity Scatter Plot ---
    plt.figure(figsize=(7, 5))
    for model_size, color in MODEL_COLORS.items():
        validity_seeds = model_data.get(model_size, {}).get("valid_graphs_all_seeds", [])
        std_seeds = model_data.get(model_size, {}).get("std_dev_all_seeds", [])
        if not validity_seeds or not std_seeds:
            continue
        for idx, (val_list, std_list) in enumerate(zip(validity_seeds, std_seeds)):
            val_arr = np.array(val_list)
            std_arr = np.array(std_list)
            if len(val_arr) >= 10 and len(std_arr) >= 10:
                final_val = np.mean(val_arr[-10:]) * max_graphs
                final_std = np.mean(std_arr[-10:])
                label = f"QuGAN({model_size})" if idx == 0 else None
                plt.scatter(final_std, final_val, color=color, label=label, alpha=0.8)

    plt.xlabel("Edge-Weight Std Dev (last 10 epochs)", fontweight="bold")
    plt.ylabel("#Valid Graphs (last 10 epochs)", fontweight="bold")
    plt.title("(e) Validity–Diversity Scatter", fontweight="bold")
    plt.grid(True)
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.show()

    # --- (f) KL Divergence: Expressibility Measure ---
    kl_scores = []
    model_labels = []

    for model_size, color in MODEL_COLORS.items():
        kl = model_data.get(model_size, {}).get("expressibility_kl", None)
        if kl is not None:
            model_labels.append(f"QuGAN({model_size})")
            kl_scores.append(kl)
        else:
            print(f"[Expressibility Plot] Missing KL score for model {model_size}, skipping.")

    if kl_scores:
        import pandas as pd
        df = pd.DataFrame({"Model": model_labels, "KL": kl_scores})
        df = df.sort_values("KL", ascending=True)

        palette = [MODEL_COLORS[int(label.split("(")[1].rstrip(")"))] for label in df["Model"]]

        plt.figure(figsize=(8, 5))
        ax = sns.barplot(x="KL", y="Model", data=df, palette=palette)

        # Add value labels to bars
        for i, val in enumerate(df["KL"]):
            ax.text(val + 0.00005, i, f"{val:.4f}", va="center", fontsize=10)

        plt.xlabel("KL Divergence from Haar", fontweight="bold")
        plt.ylabel("Model", fontweight="bold")
        plt.title("(f) Circuit Expressibility (Lower = More Expressive)", fontweight="bold")
        plt.tight_layout()
        plt.show()