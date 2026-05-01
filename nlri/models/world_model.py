import torch
from torch import nn


class WorldModel(nn.Module):
    def __init__(self, latent_dim: int = 64, belief_dim: int = 64):
        super().__init__()
        self.transition = nn.Sequential(
            nn.Linear(latent_dim + belief_dim, 128),
            nn.ReLU(),
            nn.Linear(128, belief_dim),
        )
        self.obs_head = nn.Sequential(
            nn.Linear(belief_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(belief_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )
        self.transition_quality_head = nn.Sequential(
            nn.Linear(belief_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, encoded_obs: torch.Tensor, belief: torch.Tensor):
        inputs = torch.cat([encoded_obs, belief], dim=1)
        next_belief = self.transition(inputs)
        obs_embedding = self.obs_head(next_belief)
        uncertainty = self.uncertainty_head(next_belief)
        return next_belief, obs_embedding, uncertainty

    def predict_transition_quality(self, belief: torch.Tensor):
        raw = self.transition_quality_head(belief)
        return {
            "collision_logit": raw[:, 0:1],
            "movement_cost": torch.nn.functional.softplus(raw[:, 1:2]),
            "progress": torch.sigmoid(raw[:, 2:3]),
        }
