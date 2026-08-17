import torch
import torch.nn as nn


class ToolPolicy(nn.Module):
    def __init__(self, input_dim=384, num_actions=4):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.network(x)