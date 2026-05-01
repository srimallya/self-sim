import torch
from torch import nn


class FeedbackEncoder(nn.Module):
    def __init__(self, feedback_dim: int = 10, context_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feedback_dim, 64),
            nn.ReLU(),
            nn.Linear(64, context_dim),
            nn.Tanh(),
        )

    def forward(self, feedback: torch.Tensor):
        return self.net(feedback)
