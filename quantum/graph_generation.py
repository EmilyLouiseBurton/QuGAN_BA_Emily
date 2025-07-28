import torch
import numpy as np
from training.config import HYPERPARAMS
from quantum.utils import check_triangle_inequality

def generate_graph_from_qugan(qnode, params, latent_dim=6, fixed_latent=None):
    # Sample fixed or random latent input
    latent_input = fixed_latent.detach().clone() if fixed_latent is not None else torch.rand(latent_dim, device=params.device)
    raw_outputs = qnode(params, latent_input)

    if isinstance(raw_outputs, list):
        raw_outputs = torch.stack([
            r if isinstance(r, torch.Tensor) else torch.tensor(r, dtype=torch.float32, device=params.device)
            for r in raw_outputs
        ])
    elif not isinstance(raw_outputs, torch.Tensor):
        raw_outputs = torch.tensor(raw_outputs, dtype=torch.float32, device=params.device)

    edge_weights = raw_outputs[:6]
    edge_weights = edge_weights / edge_weights.sum()

    # Build symmetric adjacency matrix
    num_nodes = 4
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=edge_weights.dtype, device=edge_weights.device)
    triu_indices = torch.triu_indices(num_nodes, num_nodes, offset=1)
    adj_matrix[triu_indices[0], triu_indices[1]] = edge_weights
    adj_matrix = adj_matrix + adj_matrix.T

    is_valid = check_triangle_inequality(adj_matrix.detach().cpu().numpy())

    return adj_matrix, is_valid, edge_weights