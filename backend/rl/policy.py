import torch.nn as nn


class ToolSelectionPolicy(nn.Module):

    def __init__(
        self,
        input_size,
        num_actions,
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_size,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                num_actions
            ),
        )


    def forward(self, x):

        return self.network(x)