import torch
import numpy as np
from quantum.graph_generation import generate_graph_from_qugan
from .config import HYPERPARAMS
from quantum.utils import check_triangle_inequality


def evaluate_generator(qnode, param_tensor, latent_dim, num_graphs=1000):
    device = param_tensor.device
    valid_count = 0
    graph_stds = []
    diagnostics = []
    all_valid_edges = []

    for _ in range(num_graphs):
        params = param_tensor.detach()
        adj_matrix, _, edge_weights = generate_graph_from_qugan(qnode, params, latent_dim=latent_dim)

        is_valid = check_triangle_inequality(adj_matrix.detach().cpu().numpy())

        if is_valid:
            valid_count += 1
            weights_np = edge_weights.detach().cpu().numpy()
            std_weight = weights_np.std()

            graph_stds.append(std_weight)
            all_valid_edges.append(weights_np)

            diagnostics.append({
                "is_valid": is_valid,
                "edge_mean": weights_np.mean(),
                "edge_std": std_weight
            })

    if all_valid_edges:
        edge_matrix = np.stack(all_valid_edges)  # shape = (num_valid_graphs, 6)
        std_across_graphs = np.std(edge_matrix, axis=0).mean()  # mean std across edge positions
    else:
        std_across_graphs = 0.0

    mean_intra_graph_std = np.mean(graph_stds) if graph_stds else 0.0
    mean_valid = valid_count / num_graphs

    print(f"[Diagnostics] Valid Graphs: {valid_count}/{num_graphs}")
    print(f"[Inter-Graph STD] (of Edge Weights) {std_across_graphs:.6f} across {valid_count} valid graphs")
    print(f"[Intra-Graph STD] (Mean across Graphs) {mean_intra_graph_std:.6f} across {valid_count} valid graphs")

    return valid_count, mean_valid, std_across_graphs, mean_intra_graph_std, [], graph_stds, diagnostics

def calculate_standard_deviation_and_edges_from_qugan(qnode, param_tensor, latent_dim, num_samples=20):
    graph_means = []

    for _ in range(num_samples):
        params = param_tensor.detach()
        adj_matrix, _, edge_weights = generate_graph_from_qugan(qnode, params, latent_dim=latent_dim)

        is_valid = check_triangle_inequality(adj_matrix.detach().cpu().numpy())

        if is_valid:
            mean_weight = edge_weights.detach().cpu().numpy().mean()
            graph_means.append(mean_weight)

    std_across_graph_means = np.std(graph_means) if graph_means else 0.0
    return std_across_graph_means, graph_means


def compute_kl_fidelity_distribution(qnode, param_count, num_qubits, torch_device, num_samples=100):
    params = torch.randn(param_count, device=torch_device)
    latent1 = torch.randn(num_qubits, device=torch_device)
    latent2 = torch.randn(num_qubits, device=torch_device)
    state1 = qnode(params, latent1).detach().cpu().numpy()
    state2 = qnode(params, latent2).detach().cpu().numpy()
    eps = 1e-10
    state1 = np.clip(state1, eps, 1.0)
    state2 = np.clip(state2, eps, 1.0)
    kl_div = np.sum(state1 * np.log(state1 / state2))
    return kl_div