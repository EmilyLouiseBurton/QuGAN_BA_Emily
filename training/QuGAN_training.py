import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from quantum.graph_generation import generate_graph_from_qugan
from quantum.circuit import create_qugan_generator
from data.ports import real_graphs_tensor
from training.config import MODEL_CONFIGS, HYPERPARAMS, EVAL_SETTINGS, MODEL_ARCH, MODEL_DATA
from models.Discriminator import Discriminator
from training.evaluation import evaluate_generator, calculate_standard_deviation_and_edges_from_qugan, triangle_inequality_penalty

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_discriminator(discriminator, real_graphs_tensor, qnode, param_tensor,
                        latent_dim, optimizer_disc,
                        batch_size=32, device= device, epoch=0):
    criterion = nn.BCELoss()
    discriminator.train()

    batch_real_indices = np.random.choice(len(real_graphs_tensor), batch_size, replace=True)
    batch_real = real_graphs_tensor[batch_real_indices].to(device)


    batch_real_labels = torch.ones((batch_size, 1), device=device)

    optimizer_disc.zero_grad()
    real_out = discriminator(batch_real)
    disc_loss_real = criterion(real_out, batch_real_labels)

    batch_fake = []
    triu_indices = torch.triu_indices(4, 4, offset=1)

    for _ in range(batch_size):
        with torch.no_grad():
            adj_matrix, _, _ = generate_graph_from_qugan(
                qnode, param_tensor,
                latent_dim=latent_dim,
            )
        fake = adj_matrix[triu_indices[0], triu_indices[1]]
        batch_fake.append(fake)

    if not batch_fake:
        print("[Discriminator] No fake samples!")
        return None

    batch_fake = torch.stack(batch_fake).to(device).float()

    fake_out = discriminator(batch_fake)
    batch_fake_labels = torch.zeros_like(fake_out)

    if torch.isnan(fake_out).any():
        print("NaNs in discriminator output!")
        return None

    disc_loss_fake = criterion(fake_out, batch_fake_labels)

    total_loss = disc_loss_real + disc_loss_fake
    total_loss.backward()
    optimizer_disc.step()

    print(f"[Discriminator] Loss: {total_loss.item():.4f} (Real: {disc_loss_real.item():.4f}, Fake: {disc_loss_fake.item():.4f})")
    return total_loss.item()

def train_generator(discriminator, qnode, param_tensor, latent_dim, epochs, optimizer_gen, device = device):
    criterion = nn.BCELoss()
    gen_losses = []
    triu_indices = torch.triu_indices(4, 4, offset=1)

    for epoch in range(epochs):
        generator_batch = []

        for _ in range(HYPERPARAMS["batch_size"]):
            adj_matrix, valid, _ = generate_graph_from_qugan(
                qnode, param_tensor,
                latent_dim=latent_dim
            )
            adj_matrix = torch.clamp(adj_matrix, 0.0, 1.0)

            edge_tensor = adj_matrix[triu_indices[0], triu_indices[1]]
            generator_batch.append((edge_tensor.view(-1), valid))

        batch, valid_flags = zip(*generator_batch)
        batch = torch.stack(batch).to(device).float()
        batch_labels = torch.full((len(batch), 1), 0.9, device=device)

        optimizer_gen.zero_grad()
        fake_out = discriminator(batch)
        loss = criterion(fake_out, batch_labels)

        loss.backward()

        if param_tensor.grad is not None:
            grad_norm = param_tensor.grad.norm().item()
            print(f"[Gradient] param_tensor grad norm: {grad_norm:.6f}")
        else:
            print("[Gradient] param_tensor has no gradient!")

        optimizer_gen.step()
        gen_losses.append(loss.item())

        print(f"[Generator] Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    return gen_losses

def train_model():
    epochs = HYPERPARAMS["epochs"]
    batch_size = HYPERPARAMS["batch_size"]
    num_seeds = HYPERPARAMS["num_seeds"]

    MODEL_DATA["train"] = {
        "edge_weights": [e.tolist() for graph in real_graphs_tensor for e in graph]
    }

    for config in MODEL_CONFIGS:
        import re
        model_key = int(re.search(r"\d+", config["model_type"]).group())
        print(f"\n=== Training Model {model_key} ===")

        model_data = MODEL_DATA.setdefault(model_key, {
            "valid_graphs": [], "std_dev": [], "std_dev_per_step": [],
            "edge_weights": [], "gen_loss": [], "steps": [], "losses": []
        })

        for seed in range(num_seeds):
            print(f"\n-- Seed {seed + 1}/{num_seeds} --")
            torch.manual_seed(seed)
            np.random.seed(seed)

            qnode, num_params = create_qugan_generator(
                num_qubits=config["num_qubits"],
                layers=config["layers"],
                gate_type=config.get("gate_type", HYPERPARAMS["gate_type"])
            )

            param_tensor = torch.nn.Parameter(torch.empty(num_params, device=device).uniform_(0, 2 * np.pi))
            discriminator = Discriminator(input_size=MODEL_ARCH["discriminator_input_size"]).to(device)

            optimizer_disc = optim.Adam(discriminator.parameters(), lr=0.3)
            optimizer_gen = optim.Adam([param_tensor], lr=0.001)

            latent_dim = config["num_qubits"]
            prev_param = param_tensor.clone().detach()

            for epoch in range(epochs):
                print(f"\n[Epoch {epoch + 1}/{epochs}]")
                HYPERPARAMS["param_noise_std"] = 2.0 if 8 <= epoch <= 12 else 0.2

                train_discriminator(
                    discriminator, real_graphs_tensor, qnode, param_tensor,
                    latent_dim=latent_dim, optimizer_disc=optimizer_disc,
                    batch_size=batch_size, device=device, epoch=epoch
                )

                g_losses = train_generator(
                    discriminator, qnode, param_tensor, latent_dim,
                    epochs=1, optimizer_gen=optimizer_gen, device=device
                )

                std_dev_epoch, _ = calculate_standard_deviation_and_edges_from_qugan(
                    qnode, param_tensor, latent_dim,
                    num_samples=EVAL_SETTINGS.get("std_eval_samples", 100)
                )

                current_loss = g_losses[-1] if g_losses else float('nan')
                param_change = torch.norm(param_tensor.detach() - prev_param).item()
                prev_param = param_tensor.clone().detach()

                valid_graphs, mean_valid, std_dev, all_weights, diagnostics = evaluate_generator(
                    qnode, param_tensor, latent_dim, EVAL_SETTINGS["num_graphs"]
                )

                print(f"[Evaluation] STD: {std_dev:.4f}, Valid Graphs: {mean_valid:.2f}")
                print(f"[Diagnostics Summary] {diagnostics[-1]}")

                model_data["valid_graphs"].append(mean_valid)
                model_data["std_dev"].append(std_dev)
                model_data["std_dev_per_step"].extend([std_dev_epoch] * len(g_losses))
                model_data["edge_weights"].extend(all_weights)
                model_data["gen_loss"].append(current_loss)
                model_data["steps"].extend([len(model_data["steps"]) + i for i in range(len(g_losses))])
                model_data["losses"].extend(g_losses)

            print(f"\n[Summary for Model {model_key}] Accumulated valid graphs so far: {model_data['valid_graphs']}")

    # === KDE baseline evaluation ===
    print("\n[Baseline] KDE Sampling from Real Edge Weights")
    import scipy.stats as stats
    all_real_edges = np.array(MODEL_DATA["train"]["edge_weights"]).flatten()
    kde = stats.gaussian_kde(all_real_edges)
    num_kde_graphs = EVAL_SETTINGS["num_graphs"]

    triangle_violations = 0
    for _ in range(num_kde_graphs):
        sampled_edges = torch.tensor(kde.resample(6).flatten(), dtype=torch.float32)
        adj = torch.zeros((4, 4), dtype=torch.float32)
        triu_indices = torch.triu_indices(4, 4, offset=1)
        adj[triu_indices[0], triu_indices[1]] = sampled_edges
        adj = adj + adj.T

        if not triangle_inequality_penalty(adj) == 0.0:
            triangle_violations += 1

    violation_ratio = triangle_violations / num_kde_graphs
    print(f"[KDE Baseline] Triangle inequality violation rate: {violation_ratio:.2%}")