import numpy as np
import torch

def check_triangle_inequality(adj_matrix, tol=0):
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