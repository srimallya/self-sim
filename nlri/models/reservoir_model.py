import torch
from torch import nn


class ReservoirModel(nn.Module):
    def __init__(self, belief_dim: int = 64, reservoir_dim: int = 6):
        super().__init__()
        self.next_head = nn.Sequential(
            nn.Linear(belief_dim, 64),
            nn.ReLU(),
            nn.Linear(64, reservoir_dim),
            nn.Sigmoid(),
        )
        self.star_head = nn.Sequential(
            nn.Linear(belief_dim, 64),
            nn.ReLU(),
            nn.Linear(64, reservoir_dim),
            nn.Sigmoid(),
        )

    def forward(self, belief: torch.Tensor):
        reservoir_next = self.next_head(belief)
        reservoir_star = self.star_head(belief)
        leakage = torch.relu(reservoir_star - reservoir_next)
        return reservoir_next, reservoir_star, leakage
