import numpy as np

def sample_latent_variables(num_params):
    return np.random.normal(0, 1, num_params)

def check_triangle_inequality(adjacency_matrix):
    for i in range(4):
        for j in range(i + 1, 4):
            for k in range(j + 1, 4):
                if adjacency_matrix[i, j] > adjacency_matrix[i, k] + adjacency_matrix[k, j]:
                    return False
    return True