import torch
import numpy as np
from quantum.graph_generation import generate_graph_from_qugan
from quantum.utils import check_triangle_inequality


def evaluate_generator(param_tensor, qnode, num_graphs=1000, num_seeds=5,
                       param_noise_std=0.01, output_noise_std=0.0):
    """
    Evaluates the generator by adding noise to parameters and outputs.

    Args:
        param_tensor: Torch tensor of generator parameters.
        qnode: Quantum node used to generate graphs.
        num_graphs: Number of graphs per seed to generate.
        num_seeds: Number of random seeds to try.
        param_noise_std: Standard deviation of noise added to parameters.
        output_noise_std: Standard deviation of noise added to outputs.

    Returns:
        Tuple of (valid_counts per seed, mean valid graphs, std dev of all edge weights, all edge weights)
    """
    valid_counts = []
    all_weights = []

    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        valid_count = 0

        for _ in range(num_graphs):
            # Add noise to parameters
            noise = torch.normal(mean=0, std=param_noise_std, size=param_tensor.shape, device=param_tensor.device)
            noisy_params = (param_tensor + noise).detach()

            adjacency_matrix, valid, edge_weights = generate_graph_from_qugan(
                qnode, noisy_params, output_noise_std=output_noise_std
            )

            if valid and check_triangle_inequality(adjacency_matrix):
                valid_count += 1

            if edge_weights is not None:
                all_weights.extend(edge_weights)

        # Log valid count for the current seed
        print(f"[Seed {seed + 1}] Valid Graphs: {valid_count}/{num_graphs}")

        valid_counts.append(valid_count)

    # Handle empty valid_counts and avoid NaN during mean calculation
    if valid_counts:
        mean_valid = np.mean(valid_counts)
        std_dev = np.std(all_weights) if all_weights else 0.0
    else:
        mean_valid = 0.0
        std_dev = 0.0
        print("[Evaluation] No valid graphs across all seeds.")

    std_dev = np.std(all_weights) if all_weights else 0.0

    # Return the results with better clarity
    return valid_counts, mean_valid, std_dev, all_weights


def calculate_standard_deviation_and_edges_from_qugan(qnode, param_tensor,
                                                       num_samples=100,
                                                       param_noise_std=0.01,
                                                       output_noise_std=0.0):
    """
    Computes the standard deviation of edge weights using noisy parameter inputs.

    Args:
        qnode: Quantum node used to generate graphs.
        param_tensor: Torch tensor of generator parameters.
        num_samples: Number of samples to generate.
        param_noise_std: Standard deviation of parameter noise.
        output_noise_std: Standard deviation of output noise.

    Returns:
        Tuple of (standard deviation of edge weights, all edge weights)
    """
    edge_weights = []

    for sample_idx in range(num_samples):
        noise = torch.normal(mean=0, std=param_noise_std, size=param_tensor.shape, device=param_tensor.device)
        noisy_params = (param_tensor + noise).detach()

        _, _, weights = generate_graph_from_qugan(qnode, noisy_params, output_noise_std=output_noise_std)

        if weights is not None:
            edge_weights.extend(weights)

        # Logging progress for edge weights calculation
        if (sample_idx + 1) % 10 == 0:
            print(f"[Sample {sample_idx + 1}/{num_samples}] Collected {len(edge_weights)} edge weights.")

    # Handle empty edge weights and avoid NaN during std dev calculation
    if edge_weights:
        edge_weights = np.array(edge_weights)
        std_dev = np.std(edge_weights)
    else:
        edge_weights = np.array([])
        std_dev = 0.0

    return std_dev, edge_weights