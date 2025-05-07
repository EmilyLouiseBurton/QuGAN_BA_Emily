import pennylane as qml
import torch
import numpy as np
from .utils import check_triangle_inequality
def generate_graph_from_qugan(qnode, params, output_noise_std=0.0, verbose=False):
    raw_outputs = qnode(params)

    if isinstance(raw_outputs, list):
        raw_outputs = torch.stack([
            r if isinstance(r, torch.Tensor)
            else torch.tensor(r, dtype=torch.float32, device=params.device)
            for r in raw_outputs
        ])
    elif not isinstance(raw_outputs, torch.Tensor):
        raw_outputs = torch.tensor(raw_outputs, dtype=torch.float32, device=params.device)

    # Shuffle outputs to break positional correlation
    raw_outputs = raw_outputs[torch.randperm(raw_outputs.shape[0])]

    edge_weights = torch.sigmoid(raw_outputs.float()[:6])  # Only first 6 for 4-node graph
    edge_weights = torch.clamp(edge_weights, min=1e-3)

    if verbose:
        print("Processed Edge Weights:", edge_weights)

    num_nodes = 4
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=raw_outputs.dtype, device=raw_outputs.device)
    idx = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            adj_matrix[i, j] = adj_matrix[j, i] = edge_weights[idx]
            idx += 1

    valid = check_triangle_inequality(adj_matrix.detach().cpu().numpy())
    return adj_matrix, valid, edge_weights
