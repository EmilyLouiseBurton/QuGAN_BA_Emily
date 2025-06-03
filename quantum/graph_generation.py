import torch
import numpy as np
from training.config import HYPERPARAMS
from quantum.utils import check_triangle_inequality

def generate_graph_from_qugan(qnode, params, latent_dim=6):
    # Sample latent vector
    latent_input = torch.empty(latent_dim, device=params.device).uniform_(0, 2 * np.pi)

    # Run quantum circuit
    raw_outputs = qnode(params, latent_input)

    if isinstance(raw_outputs, list):
        raw_outputs = torch.stack([
            r if isinstance(r, torch.Tensor) else torch.tensor(r, dtype=torch.float32, device=params.device)
            for r in raw_outputs
        ])
    elif not isinstance(raw_outputs, torch.Tensor):
        raw_outputs = torch.tensor(raw_outputs, dtype=torch.float32, device=params.device)

    # Take first 6 outputs and map to [0, 1] using sigmoid
    edge_weights = (1 + raw_outputs[:6]) / 2

    # === normalize to target mean ===
    target_mean = HYPERPARAMS.get("target_edge_mean", 0.25)
    edge_mean = edge_weights.mean().clamp(min=1e-4)
    edge_weights = edge_weights / edge_mean * target_mean

    # === Build symmetric 4x4 adjacency matrix ===
    num_nodes = 4
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=edge_weights.dtype, device=edge_weights.device)
    triu_indices = torch.triu_indices(num_nodes, num_nodes, offset=1)
    adj_matrix[triu_indices[0], triu_indices[1]] = edge_weights
    adj_matrix = adj_matrix + adj_matrix.T

    # === Check triangle inequality ===
    valid = check_triangle_inequality(adj_matrix.detach().cpu().numpy())

    return adj_matrix, valid, edge_weights