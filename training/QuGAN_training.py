import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from quantum.graph_generation import generate_graph_from_qugan
from quantum.circuit import create_qugan_generator
from data.ports import real_graphs_tensor
from training.config import MODEL_CONFIGS, HYPERPARAMS, EVAL_SETTINGS, MODEL_ARCH, MODEL_DATA
from models.Discriminator import Discriminator
from training.evaluation import evaluate_generator
from training.evaluation import compute_kl_fidelity_distribution
from collections import defaultdict
import json
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_discriminator(discriminator, real_graphs_tensor, qnode, param_tensor,
                        latent_dim, optimizer_disc,
                        batch_size=32, device=device, epoch=0):
    criterion = nn.BCELoss()
    discriminator.train()

    batch_real_indices = np.random.choice(len(real_graphs_tensor), batch_size, replace=True)
    batch_real = real_graphs_tensor[batch_real_indices].to(device)
    batch_real_labels = torch.full((batch_size, 1), 0.9, device=device)

    optimizer_disc.zero_grad()
    real_out = discriminator(batch_real)
    disc_loss_real = criterion(real_out, batch_real_labels)

    shared_latent = None
    batch_fake = []
    triu_indices = torch.triu_indices(4, 4, offset=1)
    for _ in range(batch_size):
        with torch.no_grad():
            adj_matrix, _, _ = generate_graph_from_qugan(qnode, param_tensor, latent_dim=latent_dim, fixed_latent=shared_latent)
        fake = adj_matrix[triu_indices[0], triu_indices[1]]
        batch_fake.append(fake)

    if not batch_fake:
        print("[Discriminator] No fake samples!")
        return None

    batch_fake = torch.stack(batch_fake).to(device).float()
    fake_out = discriminator(batch_fake)
    batch_fake_labels = torch.full((batch_size, 1), 0.1, device=device)

    if torch.isnan(fake_out).any():
        print("NaNs in discriminator output!")
        return None

    disc_loss_fake = criterion(fake_out, batch_fake_labels)

    real_pred = (real_out > 0.5).sum().item()
    fake_pred = (fake_out > 0.5).sum().item()

    total_loss = disc_loss_real + disc_loss_fake
    total_loss.backward()
    optimizer_disc.step()

    print(f"[Discriminator] Loss: {total_loss.item():.4f} (Real: {disc_loss_real.item():.4f}, Fake: {disc_loss_fake.item():.4f})")
    print(f"[Prediction Counts] Real→Real: {real_pred}/{batch_size}, Fake→Real: {fake_pred}/{batch_size}")
    return total_loss.item()

def train_generator(discriminator, qnode, param_tensor, latent_dim, epochs, optimizer_gen, device=device):
    criterion = nn.BCELoss()
    gen_losses = []
    triu_indices = torch.triu_indices(4, 4, offset=1)

    for epoch in range(epochs):
        generator_batch = []

        if HYPERPARAMS.get("share_latent_vector", False):
            shared_latent = torch.randn(latent_dim, device=param_tensor.device)
        else:
            shared_latent = None

        for _ in range(HYPERPARAMS["batch_size"]):
            adj_matrix, valid, _ = generate_graph_from_qugan(qnode, param_tensor, latent_dim=latent_dim, fixed_latent=shared_latent)
            edge_tensor = adj_matrix[triu_indices[0], triu_indices[1]]
            generator_batch.append((edge_tensor.view(-1), valid))

        batch, valid_flags = zip(*generator_batch)
        batch = torch.stack(batch).to(device).float()
        batch_labels = torch.full((len(batch), 1), 0.9, device=device)

        optimizer_gen.zero_grad()
        fake_out = discriminator(batch)
        loss = criterion(fake_out, batch_labels)

        loss.backward()
        optimizer_gen.step()
        gen_losses.append(loss.item())

        print(f"[Generator] Epoch {epoch + 1}, Loss: {loss.item():.4f}")
    return gen_losses

KL_RESULTS = defaultdict(dict)

def train_model():
    epochs = HYPERPARAMS["epochs"]
    batch_size = HYPERPARAMS["batch_size"]
    num_seeds = HYPERPARAMS["num_seeds"]

    CHECKPOINT_DIR = "checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    MODEL_DATA["train"] = {"edge_weights": real_graphs_tensor.flatten().tolist()}

    for config in MODEL_CONFIGS:
        import re
        model_key = int(re.findall(r"\d+", config["model_type"])[0])
        print(f"\n=== Training Model {model_key} ===")

        MODEL_DATA[model_key] = {
            "valid_graphs_all_seeds": [],
            "std_dev_all_seeds": [],
            "intra_graph_std_all_seeds": [],
            "graph_means_all_seeds": [],
            "gen_loss_all_seeds": [],
            "edge_weights_all_seeds": [],
            "all_edge_weights_all_seeds": [],
            "steps_all_seeds": [],
            "expressibility_kl": None
        }
        model_data = MODEL_DATA[model_key]
        combined_all_edge_weights = []

        for seed in range(num_seeds):
            print(f"\n-- Seed {seed + 1}/{num_seeds} --")
            torch.manual_seed(seed)
            np.random.seed(seed)

            qnode, num_params = create_qugan_generator(
                num_qubits=config["num_qubits"],
                layers=config["layers"],
                gate_type=config["gate_type"]
            )

            if seed == 0:
                kl_div = compute_kl_fidelity_distribution(
                    qnode=qnode,
                    param_count=num_params,
                    num_qubits=config["num_qubits"],
                    num_samples=100,
                    torch_device=device
                )
                model_data["expressibility_kl"] = kl_div
                print(f"[Expressibility] KL Divergence from Haar: {kl_div:.6f}")

            param_tensor = torch.nn.Parameter(
                2 * np.pi * torch.rand(num_params, device=device), requires_grad=True
            )
            discriminator = Discriminator(input_size=MODEL_ARCH["discriminator_input_size"]).to(device)
            optimizer_disc = optim.Adam(discriminator.parameters(), lr=HYPERPARAMS["learning_rate_disc"])
            optimizer_gen = optim.Adam([param_tensor], lr=HYPERPARAMS["learning_rate_gen"])
            latent_dim = config["num_qubits"]

            valid_graphs = []
            std_dev = []
            intra_graph_std_distribution = []
            graph_means = []
            edge_weights = []
            gen_loss = []
            steps = []
            step_counter = 0

            for epoch in range(epochs):
                print(f"\n[Epoch {epoch + 1}/{epochs}]")
                train_discriminator(
                    discriminator, real_graphs_tensor, qnode, param_tensor,
                    latent_dim=latent_dim, optimizer_disc=optimizer_disc,
                    batch_size=batch_size, device=device, epoch=epoch
                )
                g_losses = train_generator(
                    discriminator, qnode, param_tensor, latent_dim,
                    epochs=1, optimizer_gen=optimizer_gen, device=device
                )
                (_, mean_valid, std_across_graph_means, mean_intra_graph_std,
                 graph_means_epoch, graph_stds_epoch, diagnostics) = evaluate_generator(
                    qnode, param_tensor, latent_dim,
                    num_graphs=EVAL_SETTINGS["num_graphs"],
                )

                edge_weights.extend(graph_means_epoch)
                print(f"[Evaluation] STD (valid only): {std_across_graph_means:.4f}, Intra STD: {mean_intra_graph_std:.4f}, Valid Graphs: {mean_valid:.2f}")

                valid_graphs.append(mean_valid)
                std_dev.append(std_across_graph_means)
                intra_graph_std_distribution.append(mean_intra_graph_std)
                graph_means.extend(graph_means_epoch)
                gen_loss.extend(g_losses)
                steps.extend(range(step_counter, step_counter + len(g_losses)))
                step_counter += len(g_losses)

                all_edges_epoch = []
                if HYPERPARAMS.get("share_latent_vector", False):
                    shared_latent_eval = torch.randn(latent_dim, device=param_tensor.device)
                else:
                    shared_latent_eval = None

                for _ in range(EVAL_SETTINGS["num_graphs"]):
                    adj_matrix, _, edge_weights_tensor = generate_graph_from_qugan(
                        qnode, param_tensor, latent_dim, fixed_latent=shared_latent_eval)
                    all_edges_epoch.extend(edge_weights_tensor.detach().cpu().numpy().tolist())

                combined_all_edge_weights.extend(all_edges_epoch)

        print(f"\n[Summary for Model {model_key}] Accumulated valid graphs so far: {valid_graphs}")
        model_data["valid_graphs_all_seeds"].append(valid_graphs)
        model_data["std_dev_all_seeds"].append(std_dev)
        model_data["intra_graph_std_all_seeds"].append(intra_graph_std_distribution)
        model_data["graph_means_all_seeds"].append(graph_means)
        model_data["gen_loss_all_seeds"].append(gen_loss)
        model_data["edge_weights_all_seeds"].append(edge_weights)
        model_data["all_edge_weights_all_seeds"].append(combined_all_edge_weights)
        model_data["steps_all_seeds"].append(steps)

        # Save summary for this model
        
        summary_data = {
            "valid_graphs_all_seeds": model_data["valid_graphs_all_seeds"],
            "std_dev_all_seeds": model_data["std_dev_all_seeds"],
            "intra_graph_std_all_seeds": model_data["intra_graph_std_all_seeds"],
            "gen_loss_all_seeds": model_data["gen_loss_all_seeds"],
            "steps_all_seeds": model_data["steps_all_seeds"],
            "expressibility_kl": model_data["expressibility_kl"]
        }

        save_path = os.path.join(CHECKPOINT_DIR, f"both_model_{model_key}_summary.json")
        with open(save_path, "w") as f:
            json.dump(summary_data, f)

        print(f"[✔] Saved summary to {save_path}")