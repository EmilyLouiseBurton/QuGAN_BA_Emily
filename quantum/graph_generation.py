import pennylane as qml
import torch
import numpy as np
from .utils import check_triangle_inequality

def generate_graph_from_qugan(qnode, params, output_noise_std=0.0, verbose=False):
    raw_outputs = qnode(params)
    if isinstance(raw_outputs, list):
        raw_outputs = torch.tensor(raw_outputs, dtype=torch.float32, device=params.device)
    else:
        raw_outputs = raw_outputs.float()

    if verbose:
        print("Raw Outputs from Quantum Circuit:", raw_outputs)

    if output_noise_std > 0.0:
        noise = torch.normal(0, output_noise_std, size=raw_outputs.shape, device=raw_outputs.device)
        raw_outputs = raw_outputs + noise

    edge_weights = torch.sigmoid(raw_outputs)
    edge_weights = torch.clamp(edge_weights, min=1e-3)
    edge_weights = edge_weights / edge_weights.sum()

    if verbose:
        print("Processed Edge Weights:", edge_weights)

    num_nodes = 4
    expected_edges = num_nodes * (num_nodes - 1) // 2
    assert len(edge_weights) == expected_edges, f"Expected {expected_edges} edge weights, got {len(edge_weights)}"

    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=raw_outputs.dtype, device=raw_outputs.device)
    idx = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            adj_matrix[i, j] = adj_matrix[j, i] = edge_weights[idx]
            idx += 1

    valid = check_triangle_inequality(adj_matrix.detach().cpu().numpy())
    if verbose:
        print("Graph Validity (Triangle Inequality Check):", valid)

    return adj_matrix, valid, edge_weights