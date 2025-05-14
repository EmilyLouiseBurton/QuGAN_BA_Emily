import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from quantum.graph_generation import generate_graph_from_qugan
from quantum.circuit import create_qugan_circuit
from data.ports import real_graphs_tensor
from training.config import MODEL_CONFIGS, HYPERPARAMS, EVAL_SETTINGS, MODEL_ARCH, MODEL_DATA
from models.Discriminator import Discriminator
from training.evaluation import evaluate_generator
from training.evaluation import calculate_standard_deviation_and_edges_from_qugan


def train_discriminator(discriminator, real_graphs_tensor, qnode, param_tensor, batch_size=32, device='cpu', epoch=0):
    criterion = nn.BCELoss()
    optimizer_disc = optim.Adam(
        discriminator.parameters(),
        lr=HYPERPARAMS["learning_rate_disc"],
        weight_decay=1e-3  # L2 regularization strength
    )
    discriminator.train()

    batch_real_indices = np.random.choice(len(real_graphs_tensor), batch_size, replace=True)
    batch_real = real_graphs_tensor[batch_real_indices].to(device)
    batch_real_labels = torch.full((batch_size, 1), 0.9, device=device)

    optimizer_disc.zero_grad()
    real_out = discriminator(batch_real)
    disc_loss_real = criterion(real_out, batch_real_labels)
    disc_loss_real.backward()

    batch_fake = []
    triu_indices = torch.triu_indices(4, 4, offset=1)

    for _ in range(batch_size):
        with torch.no_grad():
            noise = torch.normal(mean=0, std=HYPERPARAMS["noise_std"] * 2.0, size=param_tensor.shape, device=param_tensor.device)
            noisy_params = param_tensor + noise
            adj_matrix, _, _ = generate_graph_from_qugan(qnode, noisy_params)

        fake = adj_matrix[triu_indices[0], triu_indices[1]]
        batch_fake.append(fake)

    print(f"[Discriminator] Fake samples generated: {len(batch_fake)} / {batch_size}")

    if len(batch_fake) == 0:
        print("[Discriminator] No fake samples — using 1 fallback sample.")
        with torch.no_grad():
            adj_matrix, _, _ = generate_graph_from_qugan(qnode, param_tensor)
        fake = adj_matrix[triu_indices[0], triu_indices[1]]
        batch_fake = [fake]

    batch_fake = torch.stack(batch_fake).to(device).float()
    batch_fake_labels = torch.full((len(batch_fake), 1), 0.1, device=device)

    fake_out = discriminator(batch_fake)

    if torch.isnan(fake_out).any():
        print(f"Discriminator output contains NaNs!")
        print(f"Sample fake_out: {fake_out}")
        return None

    print(f"[Discriminator] Fake_out stats - Mean: {fake_out.mean().item():.4f}, "
          f"Std: {fake_out.std().item():.4f}, Min: {fake_out.min().item():.4f}, Max: {fake_out.max().item():.4f}")

    disc_loss_fake = criterion(fake_out, batch_fake_labels)
    disc_loss_fake.backward()

    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
    optimizer_disc.step()

    total_loss = disc_loss_real.item() + disc_loss_fake.item()
    print(f"[Discriminator] Total Loss: {total_loss:.4f}")

    return total_loss

def train_qugan(discriminator, qnode, param_tensor, epochs, device='cpu'):
    criterion = nn.BCELoss()
    optimizer_gen = optim.Adam([param_tensor], lr=HYPERPARAMS["learning_rate_gen"])
    gen_losses = []

    for epoch in range(epochs):
        generator_batch = []
        triu_indices = torch.triu_indices(4, 4, offset=1)

        for _ in range(HYPERPARAMS["batch_size"]):
            noise = torch.normal(mean=0, std=HYPERPARAMS["noise_std"], size=param_tensor.shape, device=param_tensor.device)
            noisy_params = (param_tensor + noise).requires_grad_()
            adj_matrix, valid, _ = generate_graph_from_qugan(qnode, noisy_params)


            edge_tensor = adj_matrix[triu_indices[0], triu_indices[1]]
            generator_batch.append(edge_tensor.view(-1))

        if len(generator_batch) == 0:
            dummy = qnode(param_tensor)
            dummy_tensor = torch.stack([
                torch.tensor(d, dtype=torch.float32, device=param_tensor.device, requires_grad=True)
                if not isinstance(d, torch.Tensor) else d for d in dummy])
            fallback_loss = dummy_tensor.sum()

            optimizer_gen.zero_grad()
            fallback_loss.backward()
            torch.nn.utils.clip_grad_norm_([param_tensor], max_norm=1.0)
            optimizer_gen.step()
            continue

        batch = torch.stack(generator_batch).to(device).float()
        batch_labels = torch.full((len(batch), 1), 0.9, device=device)

        optimizer_gen.zero_grad()
        fake_out = discriminator(batch)
        loss = criterion(fake_out, batch_labels)

        if torch.isnan(loss):
            print(f"NaN detected in generator loss at epoch {epoch + 1}")
            continue

        loss.backward()

        with torch.no_grad():
            grad_norm = param_tensor.grad.norm().item() if param_tensor.grad is not None else 0.0
            grad_min = param_tensor.grad.min().item() if param_tensor.grad is not None else 0.0
            grad_max = param_tensor.grad.max().item() if param_tensor.grad is not None else 0.0
            print(f"[Generator] Grad Norm: {grad_norm:.6f}, Min: {grad_min:.6f}, Max: {grad_max:.6f}")

        torch.nn.utils.clip_grad_norm_([param_tensor], max_norm=1.0)

        with torch.no_grad():
            print(f"[Generator] Param Mean (Before): {param_tensor.mean().item():.6f}, Std: {param_tensor.std().item():.6f}")

        optimizer_gen.step()

        with torch.no_grad():
            print(f"[Generator] Param Mean (After): {param_tensor.mean().item():.6f}, Std: {param_tensor.std().item():.6f}")

        gen_losses.append(loss.item())

        print(f"[Generator] Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    return gen_losses


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = HYPERPARAMS["epochs"]

        # === Store training edge weights ===
    triu_indices = torch.triu_indices(4, 4, offset=1)
    training_edge_weights = []

    for real_graph in real_graphs_tensor:
        edges = real_graph  
        training_edge_weights.extend(edges.tolist())

    MODEL_DATA["train"] = {
        "edge_weights": training_edge_weights
    }

    for config in MODEL_CONFIGS:
        import re
        model_key = int(re.search(r"\d+", config["model_type"]).group())
        print(f"\n=== Training {model_key} ===")

        num_qubits = config["num_qubits"]
        layers = config["layers"]
        gate_type = config.get("gate_type", HYPERPARAMS["gate_type"])

        qnode, num_params = create_qugan_circuit(
            num_qubits=num_qubits,
            layers=layers,
            gate_type=gate_type
        )

        param_tensor = torch.nn.Parameter(
            torch.full((num_params,), np.pi, dtype=torch.float32, device=device, requires_grad=True)
        )
        discriminator = Discriminator(input_size=MODEL_ARCH["discriminator_input_size"]).to(device)

        gen_losses = []
        prev_param = param_tensor.clone().detach()
        prev_loss = None

        for epoch in range(epochs):
            print(f"\n[Epoch {epoch + 1}/{epochs}]")

            train_discriminator(
                discriminator=discriminator,
                real_graphs_tensor=real_graphs_tensor,
                qnode=qnode,
                param_tensor=param_tensor,
                batch_size=HYPERPARAMS["batch_size"],
                device=device
            )

            g_losses = train_qugan(
                discriminator=discriminator,
                qnode=qnode,
                param_tensor=param_tensor,
                epochs=1,
                device=device
            )
            gen_losses.extend(g_losses)

            std_dev_epoch, _ = calculate_standard_deviation_and_edges_from_qugan(
                qnode, param_tensor,
                num_samples=EVAL_SETTINGS.get("std_eval_samples", 20),
                param_noise_std=HYPERPARAMS["noise_std"],
                output_noise_std=HYPERPARAMS["output_noise_std"]
            )

            current_loss = g_losses[-1] if g_losses else float('nan')
            if prev_loss is not None and abs(current_loss - prev_loss) < 1e-5:
                print("Generator loss has barely changed — possible training stall.")
            if np.isnan(current_loss):
                print("Generator loss is NaN!")
            prev_loss = current_loss

            param_change = torch.norm(param_tensor.detach() - prev_param).item()
            print(f"[Param Change] {param_change:.6f}")
            if param_change < 1e-5:
                print("Param change is very small — Generator might not be updating.")
            prev_param = param_tensor.clone().detach()

            valid_graphs, mean_valid, std_dev, all_weights, diagnostics = evaluate_generator(
                param_tensor=param_tensor,
                qnode=qnode,
                num_graphs=EVAL_SETTINGS["num_graphs"],
                param_noise_std=HYPERPARAMS["noise_std"],
                output_noise_std=HYPERPARAMS["output_noise_std"]
            )

            print(f"[Evaluation] STD: {std_dev:.4f}, Valid Graphs (struct + triangle): {mean_valid:.2f}")
            print(f"[Diagnostics Summary] {diagnostics[-1]}")

            if model_key not in MODEL_DATA:
                MODEL_DATA[model_key] = {
                    "valid_graphs": [],
                    "std_dev": [],
                    "edge_weights": [],
                    "gen_loss": [],
                    "steps": [],
                    "losses": []
                }

            MODEL_DATA[model_key]["valid_graphs"].append(mean_valid)
            MODEL_DATA[model_key]["std_dev"].append(std_dev_epoch)
            MODEL_DATA[model_key]["edge_weights"].extend(all_weights)
            MODEL_DATA[model_key]["gen_loss"].append(current_loss)
            MODEL_DATA[model_key]["steps"].extend([len(MODEL_DATA[model_key]["steps"]) + i for i in range(len(g_losses))])
            MODEL_DATA[model_key]["losses"].extend(g_losses)

        print(f"Model {model_key} valid graph means: {MODEL_DATA[model_key]['valid_graphs']}")