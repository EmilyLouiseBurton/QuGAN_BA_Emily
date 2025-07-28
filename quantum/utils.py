import numpy as np
import torch
from itertools import combinations


def check_all_triangle_inequalities(adj_matrix, tol=0.01):
    adj_matrix = (adj_matrix + adj_matrix.T) / 2
    triplets = combinations(range(adj_matrix.shape[0]), 3)
    for i, j, k in triplets:
        a, b, c = adj_matrix[i, j], adj_matrix[j, k], adj_matrix[i, k]
        if not ((a + b + tol >= c) and (a + c + tol >= b) and (b + c + tol >= a)):
            return False 
    return True  # All triplets satisfy the inequality