from typing import Dict

import torch
from torch import nn


class NLRIEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 128, latent_dim: int = 64):
        super().__init__()
        self.perception_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 180, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gradient_net = nn.Sequential(
            nn.Linear(360, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.scalar_net = nn.Sequential(
            nn.Linear(1 + 1 + 2 + 1 + (8 * 180), hidden_dim),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, obs: Dict[str, torch.Tensor]):
        perception_feat = self.perception_net(obs["perception_cone"])
        gradient_feat = self.gradient_net(obs["raw_energy_gradient"])
        scalar_input = torch.cat(
            [
                obs["angle"],
                obs["energy"],
                obs["last_movement"],
                obs["last_action"],
                obs["shared_signal"].flatten(start_dim=1),
            ],
            dim=1,
        )
        scalar_feat = self.scalar_net(scalar_input)
        return self.fusion(torch.cat([perception_feat, gradient_feat, scalar_feat], dim=1))
