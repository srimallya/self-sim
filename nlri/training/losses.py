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
    clean_utility_scores: torch.Tensor | None,
    value_pred: torch.Tensor,
    value_target: torch.Tensor,
    imagined_actions: torch.Tensor,
    imagined_mask: torch.Tensor,
    action_histogram: torch.Tensor,
    uncertainty: torch.Tensor,
    z: torch.Tensor,
    compute_budget: torch.Tensor,
    teacher_logits: torch.Tensor | None = None,
    teacher_value: torch.Tensor | None = None,
    teacher_z: torch.Tensor | None = None,
    teacher_compute_budget: torch.Tensor | None = None,
    transition_quality_pred: Dict[str, torch.Tensor] | None = None,
    collision_targets: torch.Tensor | None = None,
    movement_cost_targets: torch.Tensor | None = None,
    progress_targets: torch.Tensor | None = None,
    world_weight: float = 1.0,
    reservoir_weight: float = 1.0,
    alpha: float = 1.0,
    beta: float = 0.1,
    bc_weight: float = 1.0,
    entropy_weight: float = 0.01,
    value_loss_weight: float = 0.5,
    entropy_target: float = 1.0,
    entropy_target_weight: float = 0.05,
    action_diversity_weight: float = 0.02,
    imagined_weight: float = 0.2,
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
    compute_target_weight: float = 0.05,
    compute_target_floor: float = 0.05,
    compute_target_ceil: float = 0.8,
    advantage_clip: float = 5.0,
    value_huber_delta: float = 1.0,
    policy_loss_clip: float = 10.0,
    value_loss_clip: float = 10.0,
    total_loss_clip: float = 100.0,
    max_entropy_bonus: float = 0.2,
    log_prob_clip: float = 20.0,
    min_action_prob: float = 1e-6,
    self_distill_weight: float = 1.0,
    self_distill_temperature: float = 2.0,
    collision_loss_weight: float = 0.2,
    movement_cost_loss_weight: float = 0.1,
    progress_loss_weight: float = 0.1,
    clean_utility_weight: float = 0.2,
) -> Dict[str, torch.Tensor]:
    world_prediction_loss = F.smooth_l1_loss(pred_next_belief, target_obs_embedding) + F.smooth_l1_loss(
        pred_obs_embedding, target_obs_embedding
    )
    reservoir_loss = (
        F.smooth_l1_loss(reservoir_next_pred, reservoir_next_target)
        + F.smooth_l1_loss(reservoir_star_pred, reservoir_star_target)
        + F.smooth_l1_loss(leakage_pred, leakage_target)
    )
    value_target = value_target.detach()
    value_pred_flat = value_pred.squeeze(1)
    raw_value_loss = F.smooth_l1_loss(value_pred_flat, value_target, beta=value_huber_delta)
    value_loss = torch.clamp(raw_value_loss, max=value_loss_clip)
    safe_logits = torch.nan_to_num(policy_logits, nan=0.0, posinf=log_prob_clip, neginf=-log_prob_clip)
    safe_logits = torch.clamp(safe_logits, min=-log_prob_clip, max=log_prob_clip)
    bc_ce = F.cross_entropy(safe_logits, fallback_actions, reduction="none")
    bc_loss = (bc_ce * bc_mask).sum() / torch.clamp(bc_mask.sum(), min=1.0)
    probs = torch.softmax(safe_logits, dim=1)
    probs = torch.clamp(probs, min=min_action_prob)
    probs = probs / probs.sum(dim=1, keepdim=True)
    log_probs = torch.log(probs)
    entropy_bonus = -(probs * log_probs).sum(dim=1).mean()
    chosen_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    chosen_log_probs = torch.clamp(chosen_log_probs, min=-log_prob_clip, max=0.0)
    action_probs = probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    valid_action_mask = (action_probs >= min_action_prob).float()
    raw_advantage = torch.clamp((value_target - value_pred_flat).detach(), -advantage_clip, advantage_clip)
    advantage = raw_advantage
    advantage = (advantage - advantage.mean()) / torch.clamp(advantage.std(unbiased=False), min=1e-3)
    advantage = torch.clamp(advantage, -advantage_clip, advantage_clip).detach()
    raw_policy_loss = -((chosen_log_probs * advantage) * valid_action_mask).sum() / torch.clamp(
        valid_action_mask.sum(), min=1.0
    )
    policy_loss = torch.clamp(raw_policy_loss, min=-policy_loss_clip, max=policy_loss_clip)
    utility_scores = torch.clamp(utility_scores, -5.0, 5.0)
    utility_aux_loss = -(chosen_log_probs * utility_scores.detach()).mean()
    clean_utility_aux_loss = torch.zeros((), device=policy_logits.device)
    if clean_utility_scores is not None:
        clean_utility_scores = torch.clamp(clean_utility_scores, -5.0, 5.0)
        clean_utility_aux_loss = -(chosen_log_probs * clean_utility_scores.detach()).mean()
    entropy_gap = torch.relu(torch.as_tensor(entropy_target, device=policy_logits.device) - entropy_bonus)
    entropy_target_loss = entropy_gap.pow(2)
    uniform = torch.full_like(probs, 1.0 / probs.shape[1])
    uniform_kl_loss = (probs * (torch.log(torch.clamp(probs, min=1e-8)) - torch.log(uniform))).sum(dim=1).mean()
    histogram = torch.clamp(action_histogram.to(policy_logits.device), min=1e-6)
    histogram = histogram / histogram.sum()
    action_diversity_loss = torch.sum(histogram * torch.log(histogram * histogram.numel()))
    imagined_ce = F.cross_entropy(safe_logits, imagined_actions, reduction="none")
    imagined_policy_loss = (imagined_ce * imagined_mask).sum() / torch.clamp(imagined_mask.sum(), min=1.0)
    operating_cost = compute_budget.mean()
    search_cost = leakage_pred.mean()
    uncertainty_loss = uncertainty.mean()
    latent_regularization = z.pow(2).mean()
    detached_uncertainty = uncertainty.detach().squeeze(1)
    detached_leakage = leakage_target.detach().mean(dim=1)
    entropy_gap_detached = torch.relu(torch.as_tensor(entropy_target, device=policy_logits.device) - (-(probs.detach() * torch.log(torch.clamp(probs.detach(), min=1e-8))).sum(dim=1)))
    raw_compute_target = 0.35 * detached_uncertainty + 0.45 * detached_leakage + 0.20 * entropy_gap_detached
    compute_target = torch.clamp(raw_compute_target, min=compute_target_floor, max=compute_target_ceil)
    compute_target_loss = F.smooth_l1_loss(compute_budget.squeeze(1), compute_target)
    compute_variance_penalty = torch.relu(0.02 - compute_budget.std(unbiased=False))
    compute_loss = (
        compute_cost_weight * compute_budget.mean()
        - compute_uncertainty_weight * (detached_uncertainty * compute_budget.squeeze(1)).mean()
        - compute_leakage_weight * (detached_leakage * compute_budget.squeeze(1)).mean()
        + compute_variance_penalty
    )
    distill_loss = torch.zeros((), device=policy_logits.device)
    teacher_entropy = torch.zeros((), device=policy_logits.device)
    teacher_student_kl = torch.zeros((), device=policy_logits.device)
    value_distill_loss = torch.zeros((), device=policy_logits.device)
    compute_distill_loss = torch.zeros((), device=policy_logits.device)
    z_distill_loss = torch.zeros((), device=policy_logits.device)
    if teacher_logits is not None and teacher_value is not None and teacher_compute_budget is not None:
        safe_teacher_logits = torch.nan_to_num(teacher_logits.detach(), nan=0.0, posinf=log_prob_clip, neginf=-log_prob_clip)
        safe_teacher_logits = torch.clamp(safe_teacher_logits, -log_prob_clip, log_prob_clip)
        temp = max(float(self_distill_temperature), 1e-3)
        teacher_probs = torch.softmax(safe_teacher_logits / temp, dim=1)
        student_log_probs = torch.log_softmax(safe_logits / temp, dim=1)
        teacher_probs = torch.clamp(teacher_probs, min=min_action_prob)
        teacher_probs = teacher_probs / teacher_probs.sum(dim=1, keepdim=True)
        teacher_entropy = -(teacher_probs * torch.log(torch.clamp(teacher_probs, min=1e-8))).sum(dim=1).mean()
        teacher_student_kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temp * temp)
        value_distill_loss = F.smooth_l1_loss(value_pred_flat, teacher_value.detach().squeeze(1), beta=value_huber_delta)
        compute_distill_loss = F.mse_loss(compute_budget, teacher_compute_budget.detach())
        if teacher_z is not None:
            z_distill_loss = F.mse_loss(z, teacher_z.detach())
        distill_loss = torch.clamp(
            teacher_student_kl + value_distill_loss + compute_distill_loss + 0.1 * z_distill_loss,
            max=10.0,
        )
    collision_bce_loss = torch.zeros((), device=policy_logits.device)
    movement_cost_huber_loss = torch.zeros((), device=policy_logits.device)
    progress_huber_loss = torch.zeros((), device=policy_logits.device)
    if transition_quality_pred is not None:
        if collision_targets is not None:
            collision_bce_loss = F.binary_cross_entropy_with_logits(
                transition_quality_pred["collision_logit"].squeeze(1),
                collision_targets.detach(),
            )
        if movement_cost_targets is not None:
            movement_cost_huber_loss = F.smooth_l1_loss(
                transition_quality_pred["movement_cost"].squeeze(1),
                movement_cost_targets.detach(),
                beta=value_huber_delta,
            )
        if progress_targets is not None:
            progress_huber_loss = F.smooth_l1_loss(
                transition_quality_pred["progress"].squeeze(1),
                progress_targets.detach(),
                beta=value_huber_delta,
            )

    entropy_contribution = torch.clamp(entropy_weight * entropy_bonus, min=0.0, max=max_entropy_bonus)
    raw_total = (
        world_weight * world_prediction_loss
        + reservoir_weight * alpha * reservoir_loss
        + beta * policy_loss
        + bc_weight * bc_loss
        - entropy_contribution
        + self_distill_weight * distill_loss
        + value_loss_weight * value_loss
        + entropy_target_weight * (entropy_target_loss + 0.25 * uniform_kl_loss)
        + action_diversity_weight * action_diversity_loss
        + imagined_weight * imagined_policy_loss
        + lambda_op * operating_cost
        + mu_search * search_cost
        + eta_uncertainty * uncertainty_loss
        + latent_loss_weight * xi_latent * latent_regularization
        + compute_loss_weight * compute_loss
        + compute_target_weight * compute_target_loss
        + utility_aux_weight * utility_aux_loss
        + clean_utility_weight * clean_utility_aux_loss
        + collision_loss_weight * collision_bce_loss
        + movement_cost_loss_weight * movement_cost_huber_loss
        + progress_loss_weight * progress_huber_loss
    )
    total = torch.clamp(raw_total, min=-total_loss_clip, max=total_loss_clip)
    return {
        "loss": total,
        "raw_total_loss": raw_total,
        "world_prediction_loss": world_prediction_loss,
        "reservoir_loss": reservoir_loss,
        "policy_loss": policy_loss,
        "raw_policy_loss": raw_policy_loss,
        "clipped_policy_loss": policy_loss,
        "value_loss": value_loss,
        "raw_value_loss": raw_value_loss,
        "clipped_value_loss": value_loss,
        "value_mean": value_pred_flat.mean(),
        "value_std": value_pred_flat.std(unbiased=False),
        "value_target_mean": value_target.mean(),
        "value_target_std": value_target.std(unbiased=False),
        "advantage_mean": advantage.mean(),
        "advantage_std": advantage.std(unbiased=False),
        "advantage_max_abs": advantage.abs().max(),
        "bc_loss": bc_loss,
        "entropy_bonus": entropy_bonus,
        "student_entropy": entropy_bonus,
        "teacher_entropy": teacher_entropy,
        "teacher_student_kl": teacher_student_kl,
        "distill_loss": distill_loss,
        "value_distill_loss": value_distill_loss,
        "compute_distill_loss": compute_distill_loss,
        "z_distill_loss": z_distill_loss,
        "entropy_contribution": entropy_contribution,
        "entropy_target_loss": entropy_target_loss,
        "uniform_kl_loss": uniform_kl_loss,
        "action_diversity_loss": action_diversity_loss,
        "imagined_policy_loss": imagined_policy_loss,
        "operating_cost": operating_cost,
        "search_cost": search_cost,
        "uncertainty_loss": uncertainty_loss,
        "latent_regularization": latent_regularization,
        "compute_loss": compute_loss,
        "compute_target_loss": compute_target_loss,
        "compute_target": compute_target.mean(),
        "utility_aux_loss": utility_aux_loss,
        "clean_utility_aux_loss": clean_utility_aux_loss,
        "collision_bce_loss": collision_bce_loss,
        "movement_cost_huber_loss": movement_cost_huber_loss,
        "progress_huber_loss": progress_huber_loss,
    }
