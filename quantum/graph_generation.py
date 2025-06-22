import torch
import numpy as np
from training.config import HYPERPARAMS
from quantum.utils import check_triangle_inequality

def generate_graph_from_qugan(qnode, params, latent_dim=6, fixed_latent=None):
    # Fixed latent input 
    latent_input = fixed_latent if fixed_latent is not None else torch.randn(latent_dim, device=params.device)

    # Run quantum circuit
    raw_outputs = qnode(params, latent_input)

    if isinstance(raw_outputs, list):
        raw_outputs = torch.stack([
            r if isinstance(r, torch.Tensor) else torch.tensor(r, dtype=torch.float32, device=params.device)
            for r in raw_outputs
        ])
    elif not isinstance(raw_outputs, torch.Tensor):
        raw_outputs = qnode(params, latent_input)

    raw_edge_weights = raw_outputs[:6]

    # === Build symmetric 4x4 adjacency matrix with raw weights ===
    num_nodes = 4
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=raw_edge_weights.dtype, device=raw_edge_weights.device)
    triu_indices = torch.triu_indices(num_nodes, num_nodes, offset=1)
    adj_matrix[triu_indices[0], triu_indices[1]] = raw_edge_weights
    adj_matrix = adj_matrix + adj_matrix.T

    # Triangle inequality check 
    is_valid = check_triangle_inequality(adj_matrix.detach().cpu().numpy())

    # Normalize so edge weights sum to 1 
    total = adj_matrix.sum()
    if total > 0:
        adj_matrix /= total
        edge_weights = adj_matrix[triu_indices[0], triu_indices[1]]
    else:
        edge_weights = raw_edge_weights  # fallback

    return adj_matrix, is_valid, edge_weights