import torch
from torch import nn


class LatentRouter(nn.Module):
    def __init__(self, belief_dim: int = 64, z_dim: int = 32):
        super().__init__()
        self.z_head = nn.Sequential(
            nn.Linear(belief_dim, 64),
            nn.ReLU(),
            nn.Linear(64, z_dim),
            nn.Tanh(),
        )
        self.compute_budget_head = nn.Sequential(
            nn.Linear(belief_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, belief: torch.Tensor, compute_floor: float = 0.0):
        z = self.z_head(belief)
        raw_compute_budget = self.compute_budget_head(belief)
        compute_budget = compute_floor + (1.0 - compute_floor) * raw_compute_budget
        return z, compute_budget
