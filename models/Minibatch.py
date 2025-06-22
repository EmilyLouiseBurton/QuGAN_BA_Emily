import torch
import torch.nn as nn

class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims):
        super(MinibatchDiscrimination, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims

        # Tensors for projecting input features into a 3D tensor for comparison
        self.T = nn.Parameter(torch.randn(in_features, out_features, kernel_dims))

    def forward(self, x):
        # Compute M = x * T, shape: [batch_size, out_features, kernel_dims]
        M = x @ self.T  # Shape: [N, out_features, kernel_dims]

        # Compute L1 distance between samples in minibatch
        M_exp1 = M.unsqueeze(0)  # Shape: [1, N, out_features, kernel_dims]
        M_exp2 = M.unsqueeze(1)  # Shape: [N, 1, out_features, kernel_dims]
        abs_diff = torch.abs(M_exp1 - M_exp2)  # Shape: [N, N, out_features, kernel_dims]
        L1_dist = abs_diff.sum(dim=3)  # Shape: [N, N, out_features]

        # Apply negative exponential
        c = torch.exp(-L1_dist)  # Shape: [N, N, out_features]

        # Zero self-similarity (i ≠ j)
        mask = 1 - torch.eye(x.size(0), device=x.device).unsqueeze(2)
        c = c * mask

        # Sum over minibatch
        o_b = c.sum(dim=1)  # Shape: [N, out_features]

        # Concatenate to input
        return torch.cat([x, o_b], dim=1)