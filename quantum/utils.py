import numpy as np
import torch
from itertools import combinations

def check_triangle_inequality(adj_matrix, tol=0.1):
    adj_matrix = (adj_matrix + adj_matrix.T) / 2
    triplets = list(combinations(range(adj_matrix.shape[0]), 3))
    i, j, k = triplets[np.random.randint(len(triplets))]
    a, b, c = adj_matrix[i, j], adj_matrix[j, k], adj_matrix[i, k]
    return (a + b + tol >= c) and (a + c + tol >= b) and (b + c + tol >= a) 