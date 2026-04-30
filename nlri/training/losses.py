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
    fallback_actions: torch.Tensor,
    bc_mask: torch.Tensor,
    utility_scores: torch.Tensor,
    uncertainty: torch.Tensor,
    z: torch.Tensor,
    compute_budget: torch.Tensor,
    world_weight: float = 1.0,
    reservoir_weight: float = 1.0,
    alpha: float = 1.0,
    beta: float = 0.1,
    bc_weight: float = 1.0,
    entropy_weight: float = 0.01,
    lambda_op: float = 0.01,
    mu_search: float = 0.01,
    eta_uncertainty: float = 0.01,
    xi_latent: float = 0.001,
    latent_loss_weight: float = 0.01,
    compute_loss_weight: float = 0.05,
    utility_aux_weight: float = 0.1,
    compute_cost_weight: float = 0.01,
    compute_uncertainty_weight: float = 0.05,
    compute_leakage_weight: float = 0.05,
) -> Dict[str, torch.Tensor]:
    world_prediction_loss = F.smooth_l1_loss(pred_next_belief, target_obs_embedding) + F.smooth_l1_loss(
        pred_obs_embedding, target_obs_embedding
    )
    reservoir_loss = (
        F.smooth_l1_loss(reservoir_next_pred, reservoir_next_target)
        + F.smooth_l1_loss(reservoir_star_pred, reservoir_star_target)
        + F.smooth_l1_loss(leakage_pred, leakage_target)
    )
    policy_loss = F.cross_entropy(policy_logits, actions)
    bc_ce = F.cross_entropy(policy_logits, fallback_actions, reduction="none")
    bc_loss = (bc_ce * bc_mask).sum() / torch.clamp(bc_mask.sum(), min=1.0)
    probs = torch.softmax(policy_logits, dim=1)
    log_probs = torch.log_softmax(policy_logits, dim=1)
    entropy_bonus = -(probs * log_probs).sum(dim=1).mean()
    chosen_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    utility_scores = torch.clamp(utility_scores, -5.0, 5.0)
    utility_aux_loss = -(chosen_log_probs * utility_scores.detach()).mean()
    operating_cost = compute_budget.mean()
    search_cost = leakage_pred.mean()
    uncertainty_loss = uncertainty.mean()
    latent_regularization = z.pow(2).mean()
    detached_uncertainty = uncertainty.detach().squeeze(1)
    detached_leakage = leakage_target.detach().mean(dim=1)
    compute_variance_penalty = torch.relu(0.02 - compute_budget.std(unbiased=False))
    compute_loss = (
        compute_cost_weight * compute_budget.mean()
        - compute_uncertainty_weight * (detached_uncertainty * compute_budget.squeeze(1)).mean()
        - compute_leakage_weight * (detached_leakage * compute_budget.squeeze(1)).mean()
        + compute_variance_penalty
    )

    total = (
        world_weight * world_prediction_loss
        + reservoir_weight * alpha * reservoir_loss
        + beta * policy_loss
        + bc_weight * bc_loss
        - entropy_weight * entropy_bonus
        + lambda_op * operating_cost
        + mu_search * search_cost
        + eta_uncertainty * uncertainty_loss
        + latent_loss_weight * xi_latent * latent_regularization
        + compute_loss_weight * compute_loss
        + utility_aux_weight * utility_aux_loss
    )
    return {
        "loss": total,
        "world_prediction_loss": world_prediction_loss,
        "reservoir_loss": reservoir_loss,
        "policy_loss": policy_loss,
        "bc_loss": bc_loss,
        "entropy_bonus": entropy_bonus,
        "operating_cost": operating_cost,
        "search_cost": search_cost,
        "uncertainty_loss": uncertainty_loss,
        "latent_regularization": latent_regularization,
        "compute_loss": compute_loss,
        "utility_aux_loss": utility_aux_loss,
    }
