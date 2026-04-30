from typing import Dict

import torch
import torch.nn.functional as F


def compute_nlri_loss(
    pred_next_belief: torch.Tensor,
    pred_obs_embedding: torch.Tensor,
    target_obs_embedding: torch.Tensor,
    reservoir_next_pred: torch.Tensor,
    reservoir_next_target: torch.Tensor,
    reservoir_star_pred: torch.Tensor,
    reservoir_star_target: torch.Tensor,
    leakage_pred: torch.Tensor,
    leakage_target: torch.Tensor,
    policy_logits: torch.Tensor,
    actions: torch.Tensor,
    uncertainty: torch.Tensor,
    z: torch.Tensor,
    compute_budget: torch.Tensor,
    world_weight: float = 1.0,
    reservoir_weight: float = 1.0,
    alpha: float = 1.0,
    beta: float = 0.1,
    lambda_op: float = 0.01,
    mu_search: float = 0.01,
    eta_uncertainty: float = 0.01,
    xi_latent: float = 0.001,
) -> Dict[str, torch.Tensor]:
    world_prediction_loss = F.mse_loss(pred_next_belief, target_obs_embedding) + F.mse_loss(
        pred_obs_embedding, target_obs_embedding
    )
    reservoir_loss = (
        F.mse_loss(reservoir_next_pred, reservoir_next_target)
        + F.mse_loss(reservoir_star_pred, reservoir_star_target)
        + F.mse_loss(leakage_pred, leakage_target)
    )
    policy_loss = F.cross_entropy(policy_logits, actions)
    operating_cost = compute_budget.mean()
    search_cost = leakage_pred.mean()
    uncertainty_loss = uncertainty.mean()
    latent_regularization = z.pow(2).mean()

    total = (
        world_weight * world_prediction_loss
        + reservoir_weight * alpha * reservoir_loss
        + beta * policy_loss
        + lambda_op * operating_cost
        + mu_search * search_cost
        + eta_uncertainty * uncertainty_loss
        + xi_latent * latent_regularization
    )
    return {
        "loss": total,
        "world_prediction_loss": world_prediction_loss,
        "reservoir_loss": reservoir_loss,
        "policy_loss": policy_loss,
        "operating_cost": operating_cost,
        "search_cost": search_cost,
        "uncertainty_loss": uncertainty_loss,
        "latent_regularization": latent_regularization,
    }
