# Discriminator
import torch.nn as nn

# 3 layers  
class Discriminator(nn.Module):
    def __init__(self, input_size=6, hidden_size=16):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input layer
        self.fc2 = nn.Linear(hidden_size, 1)  # Output layer 
        self.leaky_relu = nn.LeakyReLU(0.2)  # Leaky ReLU for hidden layer
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))  # Apply Leaky ReLU to hidden layer
        x = self.sigmoid(self.fc2(x))  # Apply Sigmoid to output layer
        return x