import numpy as np
import torch

def compute_noise_std(epoch, max_epochs, param_noise_min, param_noise_max, output_noise_min, output_noise_max, adj_noise_min, adj_noise_max):
    if epoch is not None and max_epochs:
        factor = max(0.0, (1.0 - epoch / max_epochs)**0.5)
        current_param_noise_std = param_noise_min + factor * (param_noise_max - param_noise_min)
        current_output_noise_std = output_noise_min + factor * (output_noise_max - output_noise_min)
        current_adj_noise_std = adj_noise_min + factor * (adj_noise_max - adj_noise_min)
    else:
        current_param_noise_std = param_noise_max
        current_output_noise_std = output_noise_max
        current_adj_noise_std = adj_noise_max
    return current_param_noise_std, current_output_noise_std, current_adj_noise_std

def check_triangle_inequality(adj_matrix, tol=1e-5):
    # Ensure symmetry
    if not np.allclose(adj_matrix, adj_matrix.T, atol=tol):
        raise ValueError("Adjacency matrix must be symmetric.")

    n = adj_matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = adj_matrix[i, j]
                b = adj_matrix[j, k]
                c = adj_matrix[i, k]

                if a + b < c - tol or a + c < b - tol or b + c < a - tol:
                    return False
    return True