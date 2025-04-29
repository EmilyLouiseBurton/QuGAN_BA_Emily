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


def train_discriminator(discriminator, real_graphs_tensor, qnode, param_tensor, batch_size=32, device='cpu'):
    criterion = nn.BCELoss()
    optimizer_disc = optim.Adam(discriminator.parameters(), lr=HYPERPARAMS["learning_rate_disc"])
    discriminator.train()

    batch_real_indices = np.random.choice(len(real_graphs_tensor), batch_size, replace=True)
    batch_real = real_graphs_tensor[batch_real_indices].to(device)
    batch_real_labels = torch.ones((batch_size, 1), device=device)

    optimizer_disc.zero_grad()
    real_out = discriminator(batch_real)
    disc_loss_real = criterion(real_out, batch_real_labels)
    disc_loss_real.backward()

    batch_fake = []
    for _ in range(batch_size):
        with torch.no_grad():
            adj_matrix, valid, _ = generate_graph_from_qugan(qnode, param_tensor)
        if valid:
            triu_indices = torch.triu_indices(4, 4, offset=1)
            fake = adj_matrix[triu_indices[0], triu_indices[1]]
            batch_fake.append(fake)

    if len(batch_fake) == 0:
        print("[Discriminator] No valid fake samples; skipping step.")
        return None

    batch_fake = torch.stack(batch_fake).to(device)
    batch_fake_labels = torch.zeros((len(batch_fake), 1), device=device)

    # === Fake output through discriminator ===
    fake_out = discriminator(batch_fake)

    # === NaN detection on fake_out ===
    if torch.isnan(fake_out).any():
        print(f"⚠️ Discriminator output contains NaNs!")
        print(f"Sample fake_out: {fake_out}")
        return None  # Abort this training step

    # === Optional: Print stats of discriminator outputs ===
    print(f"[Discriminator] Fake_out stats - Mean: {fake_out.mean().item():.4f}, Std: {fake_out.std().item():.4f}, Min: {fake_out.min().item():.4f}, Max: {fake_out.max().item():.4f}")

    disc_loss_fake = criterion(fake_out, batch_fake_labels)
    disc_loss_fake.backward()

    # === Gradient Clipping ===
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

        for _ in range(HYPERPARAMS["batch_size"]):
            with torch.no_grad():
                noise = torch.normal(mean=0, std=HYPERPARAMS["noise_std"], size=param_tensor.shape, device=param_tensor.device)
                noisy_params = param_tensor + noise
                adj_matrix, valid, _ = generate_graph_from_qugan(qnode, noisy_params)

            if valid:
                triu_indices = torch.triu_indices(4, 4, offset=1)
                edge_tensor = adj_matrix[triu_indices[0], triu_indices[1]]
                generator_batch.append(edge_tensor.view(-1))

        if len(generator_batch) == 0:
            dummy = qnode(param_tensor)
            fallback_loss = torch.stack([
                d if isinstance(d, torch.Tensor) and d.requires_grad
                else torch.tensor(d, device=param_tensor.device, dtype=torch.float32, requires_grad=True)
                for d in dummy
            ]).sum()

            fallback_loss.backward()
            torch.nn.utils.clip_grad_norm_([param_tensor], max_norm=1.0)
            optimizer_gen.step()
            continue

        batch = torch.stack(generator_batch).to(device)
        batch_labels = torch.ones((len(batch), 1), device=device)

        optimizer_gen.zero_grad()
        fake_out = discriminator(batch)
        loss = criterion(fake_out, batch_labels)

        # NaN detection
        if torch.isnan(loss):
            print(f"NaN detected in generator loss at epoch {epoch + 1}")
            for name, param in discriminator.named_parameters():
                if torch.isnan(param).any():
                    print(f"NaN found in discriminator param: {name}")
            if torch.isnan(param_tensor).any():
                print("NaN found in generator param_tensor")
            continue

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_([param_tensor], max_norm=1.0)

        optimizer_gen.step()
        gen_losses.append(loss.item())

        print(f"[Generator] Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    return gen_losses

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for config in MODEL_CONFIGS:
        model_key = str(config["model_type"])
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
            torch.randn(num_params, dtype=torch.float32, device=device, requires_grad=True)
        )

        # uniform initialization between -0.1 and 0.1
        torch.nn.init.uniform_(param_tensor, -0.1, 0.1)

        discriminator = Discriminator(input_size=MODEL_ARCH["discriminator_input_size"]).to(device)

        gen_losses = []
        prev_param = param_tensor.clone().detach()

        for epoch in range(HYPERPARAMS["epochs"]):
            print(f"\n[Epoch {epoch + 1}/{HYPERPARAMS['epochs']}]")

            d_loss = train_discriminator(
                discriminator=discriminator,
                real_graphs_tensor=real_graphs_tensor,
                qnode=qnode,
                param_tensor=param_tensor,
                batch_size=HYPERPARAMS["batch_size"],
                device=device
            )
            if d_loss is not None:
                print(f"[Discriminator] Loss: {d_loss:.4f}")

            g_losses = train_qugan(
                discriminator=discriminator,
                qnode=qnode,
                param_tensor=param_tensor,
                epochs=1,
                device=device
            )
            gen_losses.extend(g_losses)

            # Track parameter change
            param_change = torch.norm(param_tensor.detach() - prev_param).item()
            print(f"[Epoch {epoch + 1}] Param Change: {param_change:.6f}")
            if param_change < 1e-5:
                print("Param change is very small — Generator might not be updating.")
            prev_param = param_tensor.clone().detach()

            # === Evaluation ===
            valid_counts, _, _ = evaluate_generator(
                param_tensor=param_tensor,
                qnode=qnode,
                num_graphs=EVAL_SETTINGS["num_graphs"],
                param_noise_std=HYPERPARAMS["noise_std"],
                output_noise_std=HYPERPARAMS["output_noise_std"]
            )

            mean_valid = np.mean(valid_counts)

            std_qugan, edge_weights = calculate_standard_deviation_and_edges_from_qugan(
                qnode=qnode,
                param_tensor=param_tensor,
                num_samples=EVAL_SETTINGS["num_graphs"],
                param_noise_std=HYPERPARAMS["noise_std"],
                output_noise_std=HYPERPARAMS["output_noise_std"]
            )

            print("Edge Weights STD this epoch:", np.std(edge_weights))

            if g_losses:
                print(f"[Epoch {epoch + 1}] Gen Loss: {g_losses[-1]:.4f}, "
                    f"Valid Graphs: {mean_valid:.2f}, STD: {std_qugan:.4f}")
            else:
                print(f"[Epoch {epoch + 1}] Gen Loss: fallback used (no valid batch), "
                    f"Valid Graphs: {mean_valid:.2f}, STD: {std_qugan:.4f}")


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
            MODEL_DATA[model_key]["std_dev"].append(std_qugan)
            MODEL_DATA[model_key]["edge_weights"].extend(edge_weights)
            MODEL_DATA[model_key]["gen_loss"].append(g_losses[-1])
            MODEL_DATA[model_key]["steps"] = list(range(len(gen_losses)))
            MODEL_DATA[model_key]["losses"] = gen_losses

    # === Summary ===
    for model_key in MODEL_DATA:
        print(f"Model {model_key} valid graph means: {MODEL_DATA[model_key]['valid_graphs']}")


if __name__ == "__main__":
    train_model()