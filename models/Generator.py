import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_size=10, hidden_size=8, output_size=6):
        super(Generator, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x