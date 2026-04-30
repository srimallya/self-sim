import torch
from torch import nn


class PolicyHead(nn.Module):
    def __init__(self, belief_dim: int = 64, z_dim: int = 32, action_dim: int = 21):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(belief_dim + z_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, belief: torch.Tensor, z: torch.Tensor, compute_budget: torch.Tensor):
        return self.net(torch.cat([belief, z, compute_budget], dim=1))


class ValueHead(nn.Module):
    def __init__(self, belief_dim: int = 64, z_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(belief_dim + z_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, belief: torch.Tensor, z: torch.Tensor, compute_budget: torch.Tensor):
        return self.net(torch.cat([belief, z, compute_budget], dim=1))
