import torch
import numpy as np
from quantum.graph_generation import generate_graph_from_qugan
from .config import HYPERPARAMS
from collections import Counter
import torch.autograd as autograd
from quantum.utils import check_triangle_inequality

def evaluate_generator(qnode, param_tensor, latent_dim, num_graphs=1000, param_noise_std=0.1):
    device = param_tensor.device
    valid_count = 0
    all_weights = []
    diagnostics = []

    for _ in range(num_graphs):
        noise = torch.normal(mean=0, std=param_noise_std, size=param_tensor.shape, device=device)
        noisy_params = param_tensor + noise

        adj_matrix, valid_struct, edge_weights = generate_graph_from_qugan(qnode, noisy_params, latent_dim=latent_dim)
        is_triangle_valid = check_triangle_inequality(adj_matrix.detach().cpu().numpy())
        is_valid = valid_struct and is_triangle_valid

        if is_valid:
            valid_count += 1

        all_weights.extend(edge_weights.detach().cpu().tolist())
        diagnostics.append({
            "is_valid": is_valid,
            "structural_valid": valid_struct,
            "triangle_valid": is_triangle_valid,
            "edge_std": edge_weights.std().item()
        })

    std_qugan = np.std(all_weights)
    mean_valid = valid_count / num_graphs
    summary = Counter()
    for d in diagnostics:
        summary["valid"] += int(d["is_valid"])
        summary["struct"] += int(d["structural_valid"])
        summary["triangle"] += int(d["triangle_valid"])

    print(f"[Diagnostics] Valid: {summary['valid']}, Struct: {summary['struct']}, Triangle: {summary['triangle']}")
    return valid_count, mean_valid, std_qugan, all_weights, diagnostics


def calculate_standard_deviation_and_edges_from_qugan(qnode, param_tensor, latent_dim, num_samples=20):
    edge_weights = []
    for _ in range(num_samples):
        noise = torch.normal(mean=0, std=HYPERPARAMS.get("param_noise_std", 0.1), size=param_tensor.shape, device=param_tensor.device)
        noisy_params = (param_tensor + noise).detach()
        _, _, weights = generate_graph_from_qugan(qnode, noisy_params, latent_dim=latent_dim)
        if weights is not None:
            edge_weights.extend(weights.tolist())

    edge_weights = np.array(edge_weights)
    std_dev = np.std(edge_weights) if len(edge_weights) > 0 else 0.0
    return std_dev, edge_weights


def triangle_inequality_penalty(adj_matrix, threshold=1e-4):
    n = adj_matrix.shape[0]
    penalty = 0.0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if i != j and j != k and i != k:
                    lhs = adj_matrix[i, j]
                    rhs = adj_matrix[i, k] + adj_matrix[k, j]
                    violation = lhs - rhs - threshold
                    penalty += torch.relu(violation)
    return penalty / (n * (n - 1) * (n - 2))


def compute_gradient_penalty(discriminator, real_samples, fake_samples, device='cpu'):
    alpha = torch.rand(real_samples.size(0), 1, device=device)
    alpha = alpha.expand_as(real_samples)

    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates = interpolates.requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    ones = torch.ones_like(d_interpolates)

    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    return gradient_penalty


def compute_gradient_penalty(D, real_samples, fake_samples, device='cpu', lambda_gp=1):
    """Calculates the gradient penalty loss for WGAN GP"""
    real_samples = real_samples.to(device)
    fake_samples = fake_samples.to(device)
    
    batch_size = real_samples.size(0)
    
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand_as(real_samples)
    
    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
    interpolates = interpolates.requires_grad_(True)

    d_interpolates = D(interpolates)
    
    fake = torch.ones_like(d_interpolates, device=device, requires_grad=False)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean() * lambda_gp
    
    return gradient_penalty