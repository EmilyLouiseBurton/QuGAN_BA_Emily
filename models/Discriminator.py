import torch
import torch.nn as nn
from training.config import HYPERPARAMS

from models.Minibatch import MinibatchDiscrimination

class Discriminator(nn.Module):
    def __init__(self, input_size=6):
        super(Discriminator, self).__init__()

        self.use_minibatch = HYPERPARAMS.get("use_minibatch_discriminator", False)

        if self.use_minibatch:
            self.mbd = MinibatchDiscrimination(
                in_features=input_size,
                out_features=5,          # number of similarity features
                kernel_dims=3            # dimension of each kernel
            )
            adjusted_input_size = input_size + 5
        else:
            self.mbd = None
            adjusted_input_size = input_size

        self.model = nn.Sequential(
            nn.Linear(adjusted_input_size, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if self.use_minibatch and self.mbd is not None:
            x = self.mbd(x)
        return self.model(x)