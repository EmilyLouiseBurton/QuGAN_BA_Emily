import torch
import numpy as np
from quantum.graph_generation import generate_graph_from_qugan
from .config import HYPERPARAMS



def evaluate_generator(qnode, param_tensor, num_graphs=1000, param_noise_std=0.1, output_noise_std=0.01):
    from quantum.graph_generation import generate_graph_from_qugan
    from quantum.utils import check_triangle_inequality

    device = param_tensor.device
    valid_count = 0
    all_weights = []
    diagnostics = []

    for _ in range(num_graphs):
        noise = torch.normal(mean=0, std=param_noise_std, size=param_tensor.shape, device=device)
        noisy_params = param_tensor + noise

        adj_matrix, valid_struct, edge_weights = generate_graph_from_qugan(qnode, noisy_params, output_noise_std)

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

    return valid_count, mean_valid, std_qugan, all_weights, diagnostics

def calculate_standard_deviation_and_edges_from_qugan(qnode, param_tensor,
                                                       num_samples=100,
                                                       param_noise_std=0.01,
                                                       output_noise_std=0.0):
    edge_weights = []

    for _ in range(num_samples):
        noise = torch.normal(mean=0, std=param_noise_std, size=param_tensor.shape, device=param_tensor.device)
        noisy_params = (param_tensor + noise).detach()

        _, _, weights = generate_graph_from_qugan(qnode, noisy_params)

        if weights is not None:
            edge_weights.extend(weights.tolist())

    edge_weights = np.array(edge_weights)
    std_dev = np.std(edge_weights) if len(edge_weights) > 0 else 0.0

    return std_dev, edge_weights
