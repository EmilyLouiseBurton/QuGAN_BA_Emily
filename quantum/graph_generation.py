import pennylane as qml
import torch
import numpy as np
from .utils import check_triangle_inequality

def generate_graph_from_qugan(qnode, params, noise_std=0.0, verbose=False):
    # Inject Gaussian noise into parameters
    if noise_std > 0:
        noise = torch.randn_like(params) * noise_std
        noisy_params = params + noise
    else:
        noisy_params = params

    raw_outputs = qnode(noisy_params)

    # Ensure tensor output
    if isinstance(raw_outputs, list):
        raw_outputs = torch.stack([
            r if isinstance(r, torch.Tensor)
            else torch.tensor(r, dtype=torch.float32, device=params.device)
            for r in raw_outputs
        ])
    elif not isinstance(raw_outputs, torch.Tensor):
        raw_outputs = torch.tensor(raw_outputs, dtype=torch.float32, device=params.device)

    # First 6 outputs are used for the 6 edges of a 4-node graph
    raw_edges = raw_outputs[:6]

    raw_edges_scaled = 2.0 * raw_edges  
    edge_weights = 3.0 * torch.tanh(raw_edges_scaled) + 3.0
    edge_weights += torch.randn_like(edge_weights) * 0.8  # more noise
    edge_weights = edge_weights.clamp(0, 6)
    
    # Build symmetric adjacency matrix
    num_nodes = 4
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=edge_weights.dtype, device=edge_weights.device)
    idx = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            adj_matrix[i, j] = adj_matrix[j, i] = edge_weights[idx]
            idx += 1

    valid = check_triangle_inequality(adj_matrix.detach().cpu().numpy())

    return adj_matrix, valid, edge_weights