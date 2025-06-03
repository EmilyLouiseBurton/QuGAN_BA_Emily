# Discriminator
import torch.nn as nn

# 3 layers  
import torch.nn as nn

import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_size=6):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 16),  # Hidden layer with 16 neurons
            nn.LeakyReLU(0.2),          # LeakyReLU 
            nn.Linear(16, 1),           # Output layer
            nn.Sigmoid()                # Sigmoid activation for binary classification
        )

    def forward(self, x):
        return self.model(x)