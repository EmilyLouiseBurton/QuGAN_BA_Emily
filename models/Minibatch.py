import torch
import torch.nn as nn
import torch.nn.functional as F
class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims):
        super().__init__()
        self.T = nn.Parameter(torch.randn(in_features, out_features, kernel_dims))
        self.minibatch_weight = 0.5

    def forward(self, x):
        batch_size = x.size(0)
        M = torch.einsum('bi,ijk->bjk', x, self.T)
        M1 = M.unsqueeze(0)
        M2 = M.unsqueeze(1)
        L1 = torch.abs(M1 - M2).sum(dim=3)
        c = torch.exp(-L1)

        mask = 1 - torch.eye(batch_size, device=x.device).unsqueeze(2)
        c = c * mask

        o_b = self.minibatch_weight * c.mean(dim=1)

        return torch.cat([x, o_b], dim=1)
        