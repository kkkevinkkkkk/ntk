from torch import nn
import numpy as np
import torch.nn.functional as F

class DNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        #         self.alpha = 1
        #         self.alpha = 1 / hidden_dim
        self.alpha = 1 / np.sqrt(hidden_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        prediction = self.fc2(x)

        prediction = self.alpha * prediction
        # prediction = [batch size, output dim]
        return prediction