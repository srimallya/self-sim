from __future__ import annotations

import copy
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from .losses import compute_nlri_loss
from .normalizer import MultiNormalizer
from .demo_memory import DemoMemory
from .trajectory_feedback import TrajectoryFeedbackBuilder


class OnlineNLRITrainer:
    def __init__(self, agents, replay_buffer, config):
        self.agents = agents
        self.replay_buffer = replay_buffer
        self.config = dict(config)
        self.device = agents[0].device if agents else torch.device("cpu")
        self.training_step = 0
        self.env_step = 0
        self.eval_mode = bool(self.config.get("eval_mode", False))
        self.ablation = self.config.get("ablation", "full")
        self.grad_clip = float(self.config.get("grad_clip", 1.0))
        self.total_loss_abs_guard = float(self.config.get("total_loss_abs_guard", 1000.0))
        self.grad_abs_guard = float(self.config.get("grad_abs_guard", 100.0))
        self.nan_recovery_checkpoint = self.config.get("nan_recovery_checkpoint", "checkpoints/nlri/last_safe.pt")
        self.loss_ema_beta = float(self.config.get("loss_ema_beta", 0.98))
        self.normalizer = MultiNormalizer(clamp=10.0)
        self.teacher_ema_rate = float(self.config.get("teacher_ema_rate", 0.01))
        self.feedback_builder = TrajectoryFeedbackBuilder(
            window=int(self.config.get("feedback_window", 100)),
            energy_delta_weight=float(self.config.get("clean_energy_delta_weight", 0.2)),
            collision_weight=float(self.config.get("demo_collision_weight", 0.5)),
            movement_weight=float(self.config.get("demo_movement_weight", 0.2)),
            novelty_weight=float(self.config.get("demo_novelty_weight", 0.2)),
            leakage_weight=float(self.config.get("demo_leakage_weight", 1.0)),
        )
        self.demo_memory = DemoMemory(
            capacity=int(self.config.get("demo_memory_size", 128)),
            min_score=float(self.config.get("demo_min_score", 0.0)),
        )
        self.teacher_agents = [copy.deepcopy(agent).eval() for agent in agents]
        for teacher in self.teacher_agents:
            for param in teacher.parameters():
                param.requires_grad_(False)
        self.loss_ema = [
            {"world_loss_ema": None, "total_loss_ema": None}
            for _ in agents
        ]
        self.loss_history = [
            {
                "world_prediction_loss": None,
                "world_loss_ema": None,
                "reservoir_loss": None,
                "policy_loss": None,
                "raw_policy_loss": None,
                "clipped_policy_loss": None,
                "bc_loss": None,
                "value_loss": None,
                "raw_value_loss": None,
                "clipped_value_loss": None,
                "value_mean": None,
                "value_std": None,
                "value_target_mean": None,
                "value_target_std": None,
                "advantage_mean": None,
                "advantage_std": None,
                "advantage_max_abs": None,
                "compute_loss": None,
                "utility_aux_loss": None,
                "entropy_bonus": None,
                "entropy_target_loss": None,
                "action_diversity_loss": None,
                "compute_target": None,
                "distill_loss": None,
                "hybrid_distill_loss": None,
                "student_teacher_kl": None,
                "fallback_student_kl": None,
                "fallback_action_agreement": None,
                "no_fallback_action_entropy": None,
                "collision_bce_loss": None,
                "movement_cost_huber_loss": None,
                "progress_huber_loss": None,
                "clean_utility_aux_loss": None,
                "teacher_entropy": None,
                "student_entropy": None,
                "teacher_student_kl": None,
                "feedback_score": 0.0,
                "demo_memory_size": 0,
                "demo_score_mean": 0.0,
                "loss": None,
                "raw_total_loss": None,
                "nonfinite_update_skipped": False,
                "guard_update_skipped": False,
                "skipped_updates": 0,
                "nan_recoveries": 0,
            }
            for _ in agents
        ]
        self.live_stats = [self._make_live_stats() for _ in agents]
        self.fallback_windows = [deque(maxlen=100) for _ in agents]
        self.leakage_windows = [deque(maxlen=100) for _ in agents]
        self.movement_cost_windows = [deque(maxlen=100) for _ in agents]
        self.collision_windows = [deque(maxlen=100) for _ in agents]
        self.utility_windows = [deque(maxlen=100) for _ in agents]
        self.learned_action_windows = [deque(maxlen=100) for _ in agents]
        self.position_windows = [deque(maxlen=100) for _ in agents]
        self.last_action_seen = [None for _ in agents]
        self.repeated_action_counts = [0 for _ in agents]
        self.steps_since_food = [0 for _ in agents]
        self.prev_episode_counters = [
            {"food_eaten": 0, "collision_count": 0, "wait_count": 0}
            for _ in agents
        ]
        self.optimizers = [torch.optim.Adam(self._optimizer_param_groups(agent)) for agent in agents]
        for agent in agents:
            agent.compute_floor = float(self.config.get("compute_floor", 0.05))
            agent.obs_normalizer = self.normalizer
        self.reset_live_tracking()

    def observe_transition(
        self,
        agent_id: int,
        obs: Dict[str, Any],
        action: int,
        next_obs: Dict[str, Any],
        reward: float,
        terminated: bool,
        agent_info: Dict[str, Any],
        debug: Dict[str, Any],
        collision: bool = False,
        movement_cost: float = 0.0,
    ):
        self._update_normalizers(obs, next_obs, agent_info)

        leakage_values = list((agent_info.get("leakage") or {}).values())
        prev = self.prev_episode_counters[agent_id]
        food_now = int(agent_info.get("food_eaten", 0))
        collision_now = int(agent_info.get("collision_count", 0))
        wait_now = int(agent_info.get("wait_count", 0))
        food_delta = food_now if food_now < prev["food_eaten"] else food_now - prev["food_eaten"]
        collision_delta = collision_now if collision_now < prev["collision_count"] else collision_now - prev["collision_count"]
        wait_delta = wait_now if wait_now < prev["wait_count"] else wait_now - prev["wait_count"]
        self.prev_episode_counters[agent_id] = {
            "food_eaten": food_now,
            "collision_count": collision_now,
            "wait_count": wait_now,
        }

        energy_delta = (float(next_obs["energy"][0]) - float(obs["energy"][0])) * 1000.0
        useful_transition_score = food_delta + energy_delta - (1.0 if collision else 0.0) - float(movement_cost)
        leakage_mean = float(np.mean(leakage_values) if leakage_values else 0.0)
        fallback_flag = 1.0 if debug.get("fallback_used", False) else 0.0
        action_entropy = self._entropy(debug.get("action_probs"))
        self.fallback_windows[agent_id].append(fallback_flag)
        self.leakage_windows[agent_id].append(leakage_mean)
        self.movement_cost_windows[agent_id].append(float(movement_cost))
        self.collision_windows[agent_id].append(float(collision_delta))
        self.utility_windows[agent_id].append(float(useful_transition_score))
        position = tuple(agent_info.get("position", ()))
        if position:
            self.position_windows[agent_id].append(position)
        learned_action = not bool(debug.get("fallback_used", False))
        if learned_action:
            self.learned_action_windows[agent_id].append(int(action))
        if self.last_action_seen[agent_id] == int(action):
            self.repeated_action_counts[agent_id] += 1
        else:
            self.repeated_action_counts[agent_id] = 1
            self.last_action_seen[agent_id] = int(action)
        self.steps_since_food[agent_id] = 0 if food_delta > 0 else self.steps_since_food[agent_id] + 1
        position_novelty = self._position_novelty(self.position_windows[agent_id])
        local_loop_score = self._local_loop_score(self.position_windows[agent_id])
        clean_utility = self._clean_utility(food_delta, energy_delta, collision_delta, movement_cost, position_novelty)
        feedback_transition = {
            "energy": float(agent_info.get("energy", 0.0)),
            "energy_delta": float(energy_delta / 1000.0),
            "food_delta": float(food_delta),
            "collision": float(collision),
            "movement_cost": float(movement_cost),
            "action_entropy": float(action_entropy),
            "position_novelty": position_novelty,
            "local_loop_score": local_loop_score,
            "compute_budget": float(np.asarray(debug.get("compute_budget", [0.0])).reshape(-1)[0]),
            "z_std": float(np.std(debug.get("z", np.zeros(1)))),
            "steps_since_food": int(self.steps_since_food[agent_id]),
            "useful_transition_score": float(useful_transition_score),
            "leakage": float(leakage_mean),
        }
        self.feedback_builder.add(agent_id, feedback_transition)
        feedback_snapshot = self.feedback_builder.snapshot(agent_id)
        self.demo_memory.maybe_add(feedback_snapshot["feedback_vector"], feedback_snapshot["feedback_score"])
        demo_context = self.demo_memory.context()

        self.replay_buffer.push(
            agent_id=agent_id,
            obs=obs,
            action=int(action),
            next_obs=next_obs,
            reward=float(reward),
            reservoir=agent_info.get("reservoir") or {},
            next_reservoir=agent_info.get("reservoir_next") or {},
            reservoir_star=agent_info.get("reservoir_star") or {},
            leakage=agent_info.get("leakage") or {},
            movement_cost=float(movement_cost),
            collision=bool(collision),
            done=bool(terminated),
            info=agent_info,
            fallback_used=bool(debug.get("fallback_used", False)),
            fallback_action=int(debug.get("fallback_action", action)),
            useful_transition_score=float(useful_transition_score),
            clean_utility=float(clean_utility),
            food_delta=int(food_delta),
            collision_delta=int(collision_delta),
            energy_delta=float(energy_delta),
            progress=float(np.linalg.norm(np.asarray(next_obs.get("last_movement", [0.0, 0.0]), dtype=np.float32))),
            learned_action=learned_action,
            feedback_vector=feedback_snapshot["feedback_vector"],
            feedback_score=float(feedback_snapshot["feedback_score"]),
            demo_context=demo_context,
        )

        live = self.live_stats[agent_id]
        live["energy"] = float(agent_info.get("energy", 0.0))
        live["food_eaten"] += int(food_delta)
        live["movement_cost"] = float(np.mean(self.movement_cost_windows[agent_id]))
        live["collision_count"] += int(collision_delta)
        live["wait_count"] += int(wait_delta)
        live["leakage"] = leakage_mean
        live["leakage_mean_100"] = float(np.mean(self.leakage_windows[agent_id]))
        live["compute_budget"] = float(np.asarray(debug.get("compute_budget", [0.0])).reshape(-1)[0])
        live["z_mean"] = float(np.mean(debug.get("z", np.zeros(1))))
        live["z_std"] = float(np.std(debug.get("z", np.zeros(1))))
        live["action_entropy"] = action_entropy
        live["fallback_used_rate"] = float(np.mean(self.fallback_windows[agent_id]))
        live["selected_action"] = int(debug.get("selected_action", action))
        live["fallback_used"] = bool(debug.get("fallback_used", False))
        live["uncertainty"] = float(np.asarray(debug.get("uncertainty", [0.0])).reshape(-1)[0])
        for key in ("compute_budget", "z_mean", "z_std", "uncertainty", "action_entropy"):
            live[key] = float(np.nan_to_num(live[key], nan=0.0, posinf=0.0, neginf=0.0))
        live["useful_transition_score"] = float(np.mean(self.utility_windows[agent_id]))
        live["action_histogram_max_fraction"] = self._histogram_max_fraction(self.learned_action_windows[agent_id])
        live["position_novelty"] = position_novelty
        live["local_loop_score"] = local_loop_score
        live["wall_contact_rate"] = float(np.mean([1.0 if v else 0.0 for v in list(self.collision_windows[agent_id])]))
        live["food_per_collision"] = live_food_ratio(live["food_eaten"], live["collision_count"])
        live["food_per_100_steps"] = 100.0 * live["food_eaten"] / max(1, live["energy_count"] + 1)
        live["collisions_per_100_steps"] = 100.0 * live["collision_count"] / max(1, live["energy_count"] + 1)
        live["movement_cost_per_food"] = float(np.sum(self.movement_cost_windows[agent_id])) / max(1.0, float(live["food_eaten"]))
        live["useful_score_per_100_steps"] = 100.0 * live["useful_transition_score"]
        live["energy_slope"] = float(energy_delta / 1000.0)
        live["clean_utility"] = clean_utility
        live["steps_since_food"] = int(self.steps_since_food[agent_id])
        live["repeated_same_action_count"] = int(self.repeated_action_counts[agent_id])
        live["feedback_score"] = float(feedback_snapshot["feedback_score"])
        live["demo_memory_size"] = len(self.demo_memory)
        live["demo_score_mean"] = self.demo_memory.score_mean()

        live["energy_sum"] += live["energy"]
        live["energy_count"] += 1
        live["leakage_sum"] += leakage_mean
        live["movement_cost_sum"] += float(movement_cost)
        live["compute_budget_sum"] += live["compute_budget"]
        live["fallback_sum"] += fallback_flag
        live["entropy_sum"] += action_entropy
        live["uncertainty_sum"] += live["uncertainty"]
        live["useful_transition_sum"] += useful_transition_score

    def train_step(self, batch_size: int | None = None):
        if self.eval_mode or self.ablation in {"random-policy", "legacy-fallback-only"}:
            return {}

        batch_size = batch_size or self.config.get("batch_size", 64)
        if len(self.replay_buffer) < batch_size:
            return {}

        batch = self.replay_buffer.sample(batch_size)
        grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for item in batch:
            grouped[int(item["agent_id"])].append(item)

        summaries = {}
        world_weight = 0.0 if self.ablation == "no-world" else float(self.config.get("world_loss_weight", 0.2))
        reservoir_weight = 0.0 if self.ablation == "no-reservoir" else float(self.config.get("reservoir_loss_weight", 1.0))
        policy_weight = float(self.config.get("actor_critic_weight", self.config.get("policy_loss_weight", 0.1)))
        latent_weight = float(self.config.get("latent_loss_weight", 0.01))
        compute_loss_weight = float(self.config.get("compute_loss_weight", 0.05))
        utility_aux_weight = float(self.config.get("utility_aux_weight", 0.1))
        entropy_weight = self._scheduled_value(
            "entropy_weight",
            "entropy_decay",
            "min_entropy_weight",
            default_start=0.01,
            default_decay=0.0001,
            default_min=0.001,
        )
        bc_weight = self._scheduled_value(
            "bc_weight",
            "bc_decay",
            "min_bc_weight",
            default_start=1.0,
            default_decay=0.0005,
            default_min=0.05,
        )
        hybrid_distill_weight = 0.0 if self.ablation == "no-distill" else self._scheduled_value(
            "hybrid_distill_weight",
            "hybrid_distill_decay",
            "min_hybrid_distill_weight",
            default_start=0.5,
            default_decay=0.00005,
            default_min=0.1,
        )

        for agent_id, samples in grouped.items():
            agent = self.agents[agent_id]
            optimizer = self.optimizers[agent_id]
            obs_batch = self._stack_obs([sample["obs"] for sample in samples])
            next_obs_batch = self._stack_obs([sample["next_obs"] for sample in samples])
            normalized_obs_batch = self._normalize_obs_batch(obs_batch)
            normalized_next_obs_batch = self._normalize_obs_batch(next_obs_batch)
            if not self._finite_batch(normalized_obs_batch) or not self._finite_batch(normalized_next_obs_batch):
                self._mark_skip(agent_id)
                continue

            obs_t = agent._tensorize_obs(normalized_obs_batch, normalize=False)
            next_obs_t = agent._tensorize_obs(normalized_next_obs_batch, normalize=False)
            encoded = agent.encoder(obs_t)
            target_next_embedding = agent.encoder(next_obs_t).detach()
            self.normalizer.update("world_target", target_next_embedding.detach().cpu().numpy())
            normalized_world_target = self.normalizer.normalize_tensor("world_target", target_next_embedding)
            zero_belief = torch.zeros(encoded.shape[0], agent.belief_dim, device=agent.device)
            pred_next_belief, pred_obs_embedding, uncertainty = agent.world_model(encoded, zero_belief)
            normalized_pred_belief = self.normalizer.normalize_tensor("world_target", pred_next_belief)
            normalized_pred_obs = self.normalizer.normalize_tensor("world_target", pred_obs_embedding)

            if self.ablation == "no-router":
                z = torch.zeros(encoded.shape[0], agent.z_dim, device=agent.device)
                compute_budget = torch.full((encoded.shape[0], 1), 0.5, device=agent.device)
            else:
                z, compute_budget = agent.latent_router(
                    pred_next_belief, compute_floor=float(self.config.get("compute_floor", 0.05))
                )
            z = torch.nan_to_num(z, nan=0.0, posinf=1.0, neginf=-1.0)
            compute_budget = torch.nan_to_num(
                compute_budget,
                nan=float(self.config.get("compute_floor", 0.05)),
                posinf=1.0,
                neginf=float(self.config.get("compute_floor", 0.05)),
            ).clamp(float(self.config.get("compute_floor", 0.05)), 1.0)
            reservoir_next_pred, reservoir_star_pred, leakage_pred = agent.reservoir_model(pred_next_belief)
            transition_quality_pred = agent.world_model.predict_transition_quality(pred_next_belief)
            policy_logits = torch.nan_to_num(agent.policy(pred_next_belief, z, compute_budget), nan=0.0, posinf=20.0, neginf=-20.0)
            value_pred = torch.nan_to_num(agent.value_head(pred_next_belief, z, compute_budget), nan=0.0, posinf=10.0, neginf=-10.0)
            feedback_context = self._stack_feedback(samples, "feedback_vector", agent.device)
            demo_context = self._stack_feedback(samples, "demo_context", agent.device)
            teacher_outputs = self._teacher_outputs(agent_id, obs_t, feedback_context, demo_context)
            with torch.no_grad():
                next_encoded = agent.encoder(next_obs_t)
                next_zero_belief = torch.zeros(next_encoded.shape[0], agent.belief_dim, device=agent.device)
                next_belief, _next_obs_embedding, _next_uncertainty = agent.world_model(next_encoded, next_zero_belief)
                if self.ablation == "no-router":
                    next_z = torch.zeros(next_encoded.shape[0], agent.z_dim, device=agent.device)
                    next_compute_budget = torch.full((next_encoded.shape[0], 1), 0.5, device=agent.device)
                else:
                    next_z, next_compute_budget = agent.latent_router(
                        next_belief, compute_floor=float(self.config.get("compute_floor", 0.05))
                    )
                next_z = torch.nan_to_num(next_z, nan=0.0, posinf=1.0, neginf=-1.0)
                next_compute_budget = torch.nan_to_num(
                    next_compute_budget,
                    nan=float(self.config.get("compute_floor", 0.05)),
                    posinf=1.0,
                    neginf=float(self.config.get("compute_floor", 0.05)),
                ).clamp(float(self.config.get("compute_floor", 0.05)), 1.0)
                next_value = torch.nan_to_num(agent.value_head(next_belief, next_z, next_compute_budget).squeeze(1), nan=0.0)
                next_value = torch.clamp(next_value, -float(self.config.get("return_clip", 10.0)), float(self.config.get("return_clip", 10.0)))

            reservoir_next_target = self._stack_reservoir(samples, "next_reservoir", agent.device)
            reservoir_star_target = self._stack_reservoir(samples, "reservoir_star", agent.device)
            leakage_target = self._stack_reservoir(samples, "leakage", agent.device)
            utility_scores = torch.tensor(
                [sample.get("useful_transition_score", 0.0) for sample in samples],
                dtype=torch.float32,
                device=agent.device,
            )
            clean_utility_scores = torch.tensor(
                [sample.get("clean_utility", sample.get("useful_transition_score", 0.0)) for sample in samples],
                dtype=torch.float32,
                device=agent.device,
            )
            utility_scores = torch.clamp(
                torch.nan_to_num(utility_scores, nan=0.0),
                -float(self.config.get("utility_clip", 5.0)),
                float(self.config.get("utility_clip", 5.0)),
            )
            clean_utility_scores = torch.clamp(
                torch.nan_to_num(clean_utility_scores, nan=0.0),
                -float(self.config.get("clean_utility_clip", 5.0)),
                float(self.config.get("clean_utility_clip", 5.0)),
            )
            reservoir_leakage_penalty = leakage_target.mean(dim=1)
            utility_target = torch.clamp(
                utility_scores - reservoir_leakage_penalty,
                -float(self.config.get("utility_clip", 5.0)),
                float(self.config.get("utility_clip", 5.0)),
            )
            done_mask = torch.tensor(
                [0.0 if sample.get("done") else 1.0 for sample in samples],
                dtype=torch.float32,
                device=agent.device,
            )
            value_base = utility_target + float(self.config.get("clean_utility_weight", 0.2)) * clean_utility_scores
            value_target = torch.clamp(
                value_base + float(self.config.get("gamma", 0.99)) * done_mask * next_value,
                -float(self.config.get("value_target_clip", 10.0)),
                float(self.config.get("value_target_clip", 10.0)),
            )
            collision_targets = torch.tensor(
                [1.0 if sample.get("collision") else 0.0 for sample in samples],
                dtype=torch.float32,
                device=agent.device,
            )
            movement_cost_targets = torch.clamp(
                torch.tensor([sample.get("movement_cost", 0.0) for sample in samples], dtype=torch.float32, device=agent.device),
                0.0,
                5.0,
            )
            progress_targets = torch.clamp(
                torch.tensor([sample.get("progress", 0.0) for sample in samples], dtype=torch.float32, device=agent.device),
                0.0,
                1.0,
            )
            self.normalizer.update("reservoir_target", reservoir_next_target.detach().cpu().numpy())
            normalized_reservoir_next_pred = self.normalizer.normalize_tensor("reservoir_target", reservoir_next_pred)
            normalized_reservoir_next_target = self.normalizer.normalize_tensor("reservoir_target", reservoir_next_target)
            normalized_reservoir_star_pred = self.normalizer.normalize_tensor("reservoir_target", reservoir_star_pred)
            normalized_reservoir_star_target = self.normalizer.normalize_tensor("reservoir_target", reservoir_star_target)
            normalized_leakage_pred = self.normalizer.normalize_tensor("reservoir_target", leakage_pred)
            normalized_leakage_target = self.normalizer.normalize_tensor("reservoir_target", leakage_target)
            actions = torch.tensor([sample["action"] for sample in samples], dtype=torch.long, device=agent.device)
            fallback_actions = torch.tensor(
                [sample.get("fallback_action", sample["action"]) for sample in samples],
                dtype=torch.long,
                device=agent.device,
            )
            bc_mask = torch.ones(len(samples), dtype=torch.float32, device=agent.device)
            imagined_actions, imagined_mask = self._imagined_actions(
                agent_id, agent, pred_next_belief, z, compute_budget, policy_logits
            )
            action_histogram = self._action_histogram_tensor(agent_id, agent.device)
            fallback_teacher_probs = self._fallback_distribution(
                fallback_actions,
                action_dim=policy_logits.shape[1],
                confidence=float(self.config.get("fallback_target_confidence", 0.7)),
                neighbor_mass=float(self.config.get("fallback_neighbor_mass", 0.15)),
                device=agent.device,
            )
            fallback_used_mask = torch.tensor(
                [1.0 if sample.get("fallback_used") else 0.0 for sample in samples],
                dtype=torch.float32,
                device=agent.device,
            )
            hybrid_teacher_probs = self._hybrid_teacher_distribution(
                policy_logits=policy_logits,
                fallback_teacher_probs=fallback_teacher_probs,
                fallback_used_mask=fallback_used_mask,
                imagined_actions=imagined_actions,
                imagined_mask=imagined_mask,
                temperature=float(self.config.get("hybrid_distill_temperature", 1.5)),
            )

            if not self._finite_tensor(
                [
                    normalized_world_target,
                    normalized_reservoir_next_target,
                    normalized_reservoir_star_target,
                    normalized_leakage_target,
                    utility_scores,
                    clean_utility_scores,
                    value_target,
                    policy_logits,
                    value_pred,
                    z,
                    compute_budget,
                    teacher_outputs["logits"],
                    teacher_outputs["value"],
                    teacher_outputs["z"],
                    teacher_outputs["compute_budget"],
                    fallback_teacher_probs,
                    hybrid_teacher_probs,
                ]
            ):
                self._mark_skip(agent_id)
                continue

            losses = compute_nlri_loss(
                pred_next_belief=normalized_pred_belief,
                pred_obs_embedding=normalized_pred_obs,
                target_obs_embedding=normalized_world_target,
                reservoir_next_pred=normalized_reservoir_next_pred,
                reservoir_next_target=normalized_reservoir_next_target,
                reservoir_star_pred=normalized_reservoir_star_pred,
                reservoir_star_target=normalized_reservoir_star_target,
                leakage_pred=normalized_leakage_pred,
                leakage_target=normalized_leakage_target,
                policy_logits=policy_logits,
                actions=actions,
                fallback_actions=fallback_actions,
                bc_mask=bc_mask,
                utility_scores=utility_scores,
                clean_utility_scores=clean_utility_scores,
                value_pred=value_pred,
                value_target=value_target,
                imagined_actions=imagined_actions,
                imagined_mask=imagined_mask,
                action_histogram=action_histogram,
                uncertainty=uncertainty,
                z=z,
                compute_budget=compute_budget,
                teacher_logits=teacher_outputs["logits"],
                teacher_value=teacher_outputs["value"],
                teacher_z=teacher_outputs["z"],
                teacher_compute_budget=teacher_outputs["compute_budget"],
                hybrid_teacher_probs=hybrid_teacher_probs,
                fallback_teacher_probs=fallback_teacher_probs,
                transition_quality_pred=transition_quality_pred,
                collision_targets=collision_targets,
                movement_cost_targets=movement_cost_targets,
                progress_targets=progress_targets,
                world_weight=world_weight,
                reservoir_weight=reservoir_weight,
                beta=policy_weight,
                bc_weight=bc_weight,
                entropy_weight=entropy_weight,
                latent_loss_weight=latent_weight,
                compute_loss_weight=compute_loss_weight,
                utility_aux_weight=utility_aux_weight,
                compute_cost_weight=float(self.config.get("compute_cost_weight", 0.01)),
                compute_uncertainty_weight=float(self.config.get("compute_uncertainty_weight", 0.05)),
                compute_leakage_weight=float(self.config.get("compute_leakage_weight", 0.05)),
                value_loss_weight=float(self.config.get("value_loss_weight", 0.5)),
                entropy_target=float(self.config.get("entropy_target", 1.0)),
                entropy_target_weight=float(self.config.get("entropy_target_weight", 0.05)),
                action_diversity_weight=float(self.config.get("action_diversity_weight", 0.02)),
                imagined_weight=float(self.config.get("imagined_weight", 0.2)),
                compute_target_weight=float(self.config.get("compute_target_weight", 0.05)),
                compute_target_floor=float(self.config.get("compute_target_floor", 0.05)),
                compute_target_ceil=float(self.config.get("compute_target_ceil", 0.8)),
                advantage_clip=float(self.config.get("advantage_clip", 5.0)),
                value_huber_delta=float(self.config.get("value_huber_delta", 1.0)),
                policy_loss_clip=float(self.config.get("policy_loss_clip", 10.0)),
                value_loss_clip=float(self.config.get("value_loss_clip", 10.0)),
                total_loss_clip=float(self.config.get("total_loss_clip", 100.0)),
                max_entropy_bonus=float(self.config.get("max_entropy_bonus", 0.2)),
                log_prob_clip=float(self.config.get("log_prob_clip", 20.0)),
                min_action_prob=float(self.config.get("min_action_prob", 1e-6)),
                self_distill_weight=float(self.config.get("self_distill_weight", 1.0)),
                self_distill_temperature=float(self.config.get("self_distill_temperature", 2.0)),
                hybrid_distill_weight=hybrid_distill_weight,
                hybrid_distill_temperature=float(self.config.get("hybrid_distill_temperature", 1.5)),
                collision_loss_weight=float(self.config.get("collision_loss_weight", 0.2)),
                movement_cost_loss_weight=float(self.config.get("movement_cost_loss_weight", 0.1)),
                progress_loss_weight=float(self.config.get("progress_loss_weight", 0.1)),
                clean_utility_weight=float(self.config.get("clean_utility_weight", 0.2)),
            )

            total_loss = losses["loss"]
            raw_total = losses.get("raw_total_loss", total_loss)
            if not torch.isfinite(total_loss) or not torch.isfinite(raw_total):
                self._mark_skip(agent_id, reason="nonfinite")
                self._recover_last_safe(agent_id)
                continue
            if abs(float(raw_total.detach().cpu().item())) > self.total_loss_abs_guard:
                self._mark_skip(agent_id, reason="guard")
                continue

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), self.grad_clip)
            if not torch.isfinite(grad_norm) or float(grad_norm.detach().cpu().item()) > self.grad_abs_guard:
                optimizer.zero_grad(set_to_none=True)
                self._mark_skip(agent_id, reason="guard")
                continue
            optimizer.step()
            if not self._finite_parameters(agent):
                self._mark_skip(agent_id, reason="nonfinite")
                self._recover_last_safe(agent_id)
                continue
            self._save_last_safe(agent_id)

            summary = {name: float(value.detach().cpu().item()) for name, value in losses.items()}
            ema_state = self.loss_ema[agent_id]
            ema_state["world_loss_ema"] = self._ema(ema_state["world_loss_ema"], summary["world_prediction_loss"])
            ema_state["total_loss_ema"] = self._ema(ema_state["total_loss_ema"], summary["loss"])
            summary["world_loss_ema"] = ema_state["world_loss_ema"]
            summary["nonfinite_update_skipped"] = False
            summary["guard_update_skipped"] = False
            summary["skipped_updates"] = int(self.loss_history[agent_id].get("skipped_updates", 0) or 0)
            summary["nan_recoveries"] = int(self.loss_history[agent_id].get("nan_recoveries", 0) or 0)
            self.loss_history[agent_id] = summary
            summaries[agent_id] = summary
            self._update_teacher_ema(agent_id)

        self.training_step += 1
        return summaries

    def save_checkpoint(self, checkpoint_dir, step: int, periodic: bool = True):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "training_step": self.training_step,
            "env_step": self.env_step,
            "config": self.config,
            "loss_history": self.loss_history,
            "loss_ema": self.loss_ema,
            "live_stats": self.live_stats,
            "normalizer": self.normalizer.state_dict(),
            "agents": [agent.state_dict() for agent in self.agents],
            "teacher_agents": [agent.state_dict() for agent in self.teacher_agents],
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
        }
        latest_path = checkpoint_dir / "latest.pt"
        torch.save(state, latest_path)
        saved_paths = [str(latest_path)]
        if periodic:
            periodic_path = checkpoint_dir / f"step_{step:07d}.pt"
            torch.save(state, periodic_path)
            saved_paths.append(str(periodic_path))
        return saved_paths

    def load_checkpoint(self, checkpoint_path_or_dir):
        path = Path(checkpoint_path_or_dir)
        checkpoint_path = path / "latest.pt" if path.is_dir() else path
        if not checkpoint_path.exists():
            return False
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        for agent, agent_state in zip(self.agents, state.get("agents", [])):
            self._load_compatible_state(agent, agent_state)
        teacher_states = state.get("teacher_agents", state.get("agents", []))
        for teacher, teacher_state in zip(self.teacher_agents, teacher_states):
            self._load_compatible_state(teacher, teacher_state)
        optimizer_states = state.get("optimizers", [])
        if not self.eval_mode:
            for optimizer, optimizer_state in zip(self.optimizers, optimizer_states):
                try:
                    optimizer.load_state_dict(optimizer_state)
                except ValueError:
                    pass
        self.training_step = int(state.get("training_step", 0))
        self.env_step = int(state.get("env_step", 0))
        self.config.update(state.get("config", {}))
        if "loss_history" in state:
            self.loss_history = state["loss_history"]
        if "loss_ema" in state:
            self.loss_ema = state["loss_ema"]
        if "live_stats" in state:
            self.live_stats = state["live_stats"]
        if "normalizer" in state:
            self.normalizer.load_state_dict(state["normalizer"])
        for agent in self.agents:
            agent.obs_normalizer = self.normalizer
            agent.compute_floor = float(self.config.get("compute_floor", 0.05))
        return str(checkpoint_path)

    def reset_live_tracking(self):
        self.live_stats = [self._make_live_stats() for _ in self.agents]
        self.fallback_windows = [deque(maxlen=100) for _ in self.agents]
        self.leakage_windows = [deque(maxlen=100) for _ in self.agents]
        self.movement_cost_windows = [deque(maxlen=100) for _ in self.agents]
        self.collision_windows = [deque(maxlen=100) for _ in self.agents]
        self.utility_windows = [deque(maxlen=100) for _ in self.agents]
        self.learned_action_windows = [deque(maxlen=100) for _ in self.agents]
        self.position_windows = [deque(maxlen=100) for _ in self.agents]
        self.last_action_seen = [None for _ in self.agents]
        self.repeated_action_counts = [0 for _ in self.agents]
        self.steps_since_food = [0 for _ in self.agents]
        self.prev_episode_counters = [
            {"food_eaten": 0, "collision_count": 0, "wait_count": 0}
            for _ in self.agents
        ]

    def fork_agent(
        self,
        winner_agent_id: int,
        loser_agent_id: int,
        mutation_std: float = 0.005,
        mutation_prob: float = 0.05,
        reset_optimizer: bool = False,
        mutate_policy: bool = True,
        mutate_router: bool = True,
        mutate_value: bool = True,
        mutate_world: bool = False,
        mutate_reservoir: bool = False,
        fork_diversity_noise: float = 0.01,
    ):
        """Copy a winning lifetime into a losing slot, then add bounded param noise."""
        if winner_agent_id == loser_agent_id:
            return False
        self._save_last_safe(loser_agent_id)
        winner = self.agents[winner_agent_id]
        loser = self.agents[loser_agent_id]
        loser.load_state_dict(copy.deepcopy(winner.state_dict()))
        self.teacher_agents[loser_agent_id].load_state_dict(copy.deepcopy(self.teacher_agents[winner_agent_id].state_dict()))
        if reset_optimizer:
            self.optimizers[loser_agent_id] = torch.optim.Adam(self._optimizer_param_groups(loser))
        else:
            try:
                self.optimizers[loser_agent_id].load_state_dict(copy.deepcopy(self.optimizers[winner_agent_id].state_dict()))
            except ValueError:
                self.optimizers[loser_agent_id] = torch.optim.Adam(self._optimizer_param_groups(loser))
        self.loss_history[loser_agent_id] = copy.deepcopy(self.loss_history[winner_agent_id])
        self.loss_ema[loser_agent_id] = copy.deepcopy(self.loss_ema[winner_agent_id])
        if not bool(self.config.get("preserve_demo_memory", False)):
            self.demo_memory.items.clear()
        self._mutate_agent(
            loser,
            mutation_std=mutation_std,
            mutation_prob=mutation_prob,
            mutate_policy=mutate_policy,
            mutate_router=mutate_router,
            mutate_value=mutate_value,
            mutate_world=mutate_world,
            mutate_reservoir=mutate_reservoir,
            fork_diversity_noise=fork_diversity_noise,
        )
        if not self._finite_parameters(loser):
            self._recover_last_safe(loser_agent_id)
            return False
        self._save_last_safe(loser_agent_id)
        return True

    def metrics(self):
        rows = []
        for agent_id, stats in enumerate(self.live_stats):
            row = dict(stats)
            row.update(self.loss_history[agent_id])
            row["warning_flags"] = self._warning_flags(row)
            rows.append(row)
        return {
            "training_step": self.training_step,
            "env_step": self.env_step,
            "per_agent": rows,
        }

    def _normalize_obs_batch(self, obs_batch: Dict[str, np.ndarray]):
        return {
            "perception_cone": self.normalizer.normalize_array("perception_cone", obs_batch["perception_cone"]),
            "raw_energy_gradient": self.normalizer.normalize_array("raw_energy_gradient", obs_batch["raw_energy_gradient"]),
            "energy": self.normalizer.normalize_array("energy", obs_batch["energy"]),
            "angle": self.normalizer.normalize_array("angle", obs_batch["angle"]),
            "last_movement": self.normalizer.normalize_array("last_movement", obs_batch["last_movement"]),
            "last_action": self.normalizer.normalize_array("last_action", obs_batch["last_action"]),
            "shared_signal": self.normalizer.normalize_array("shared_signal", obs_batch["shared_signal"]),
        }

    def _stack_obs(self, obs_list: List[Dict[str, Any]]):
        keys = obs_list[0].keys()
        stacked = {}
        for key in keys:
            stacked[key] = np.stack([np.asarray(obs[key], dtype=np.float32) for obs in obs_list], axis=0)
        return stacked

    def _stack_reservoir(self, samples: List[Dict[str, Any]], key: str, device):
        ordered_keys = [
            "self_energy",
            "visible_food_value",
            "reachable_food_value",
            "collision_safety",
            "time_budget",
            "attention_budget",
        ]
        rows = []
        for sample in samples:
            reservoir = sample.get(key) or {}
            rows.append([float(reservoir.get(name, 0.0)) for name in ordered_keys])
        return torch.tensor(rows, dtype=torch.float32, device=device)

    def _stack_feedback(self, samples: List[Dict[str, Any]], key: str, device):
        rows = []
        for sample in samples:
            value = sample.get(key)
            if value is None:
                value = np.zeros(self.feedback_builder.dim, dtype=np.float32)
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.size < self.feedback_builder.dim:
                arr = np.pad(arr, (0, self.feedback_builder.dim - arr.size))
            elif arr.size > self.feedback_builder.dim:
                arr = arr[: self.feedback_builder.dim]
            rows.append(arr)
        array = np.nan_to_num(np.stack(rows, axis=0), nan=0.0, posinf=1.0, neginf=-1.0)
        return torch.tensor(array, dtype=torch.float32, device=device)

    def _update_normalizers(self, obs, next_obs, agent_info):
        self.normalizer.update("perception_cone", obs["perception_cone"])
        self.normalizer.update("perception_cone", next_obs["perception_cone"])
        self.normalizer.update("raw_energy_gradient", obs["raw_energy_gradient"])
        self.normalizer.update("raw_energy_gradient", next_obs["raw_energy_gradient"])
        self.normalizer.update("energy", obs["energy"])
        self.normalizer.update("energy", next_obs["energy"])
        self.normalizer.update("angle", obs["angle"])
        self.normalizer.update("last_movement", obs["last_movement"])
        self.normalizer.update("last_action", obs["last_action"])
        self.normalizer.update("shared_signal", obs["shared_signal"])
        for key in ("reservoir", "reservoir_next", "reservoir_star", "leakage"):
            reservoir = agent_info.get(key) or {}
            ordered = [
                float(reservoir.get(name, 0.0))
                for name in (
                    "self_energy",
                    "visible_food_value",
                    "reachable_food_value",
                    "collision_safety",
                    "time_budget",
                    "attention_budget",
                )
            ]
            self.normalizer.update("reservoir_target", np.asarray(ordered, dtype=np.float32))

    def _warning_flags(self, row):
        flags = []
        if row.get("compute_budget", 0.0) < 0.01:
            flags.append("cb_low")
        if row.get("action_entropy", 0.0) < 0.05:
            flags.append("entropy_low")
        if (row.get("world_prediction_loss") or 0.0) > 1000.0:
            flags.append("world_spike")
        if row.get("z_std", 0.0) < 0.01:
            flags.append("z_collapse")
        early_handoff_threshold = self.config.get("warmup_steps", 0) + 1500
        min_fb = float(self.config.get("min_fallback_prob", 0.25))
        if self.env_step < early_handoff_threshold and row.get("fallback_used_rate", 1.0) <= min_fb + 0.02:
            flags.append("fb_early")
        if row.get("nonfinite_update_skipped"):
            flags.append("skip_nf")
        if row.get("guard_update_skipped"):
            flags.append("policy_loss_guard")
        if abs(row.get("value_mean", 0.0) or 0.0) > 20.0:
            flags.append("value_explosion")
        advantage_clip = float(self.config.get("advantage_clip", 5.0))
        if (row.get("advantage_max_abs") or 0.0) > advantage_clip + 1e-3:
            flags.append("advantage_explosion")
        if row.get("action_histogram_max_fraction", 0.0) > 0.8:
            flags.append("action_collapse")
        if row.get("steps_since_food", 0) >= 500:
            flags.append("stagnation")
        if row.get("position_novelty", 1.0) < 0.2:
            flags.append("low_position_novelty")
        return flags

    def _ema(self, current, new_value):
        if current is None:
            return float(new_value)
        return float(self.loss_ema_beta * current + (1.0 - self.loss_ema_beta) * new_value)

    def _scheduled_value(self, start_key, decay_key, min_key, default_start, default_decay, default_min):
        start = float(self.config.get(start_key, default_start))
        decay = float(self.config.get(decay_key, default_decay))
        min_value = float(self.config.get(min_key, default_min))
        steps_since_warmup = max(0, self.env_step - int(self.config.get("warmup_steps", 0)))
        return max(min_value, start - decay * steps_since_warmup)

    def _optimizer_param_groups(self, agent):
        model_lr = float(self.config.get("model_lr", self.config.get("learning_rate", 1e-4)))
        policy_lr = float(self.config.get("policy_lr", 3e-5))
        value_lr = float(self.config.get("value_lr", 3e-5))
        router_lr = float(self.config.get("router_lr", policy_lr))
        return [
            {
                "params": list(agent.encoder.parameters())
                + list(agent.feedback_encoder.parameters())
                + list(agent.world_model.parameters())
                + list(agent.reservoir_model.parameters()),
                "lr": model_lr,
            },
            {"params": agent.latent_router.parameters(), "lr": router_lr},
            {"params": agent.policy.parameters(), "lr": policy_lr},
            {"params": agent.value_head.parameters(), "lr": value_lr},
        ]

    def _mark_skip(self, agent_id, reason="nonfinite"):
        summary = dict(self.loss_history[agent_id])
        summary["nonfinite_update_skipped"] = reason == "nonfinite"
        summary["guard_update_skipped"] = reason == "guard"
        summary["skipped_updates"] = int(summary.get("skipped_updates", 0) or 0) + 1
        self.loss_history[agent_id] = summary

    def _finite_batch(self, batch: Dict[str, np.ndarray]):
        return all(np.isfinite(np.asarray(value)).all() for value in batch.values())

    def _finite_tensor(self, tensors: List[torch.Tensor]):
        return all(torch.isfinite(tensor).all().item() for tensor in tensors)

    def _teacher_outputs(self, agent_id, obs_t, feedback_context, demo_context):
        teacher = self.teacher_agents[agent_id]
        with torch.no_grad():
            outputs = teacher.teacher_forward(obs_t, feedback_context=feedback_context, demo_context=demo_context)
        return {
            "logits": torch.nan_to_num(outputs["logits"], nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0),
            "value": torch.nan_to_num(outputs["value"], nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0),
            "z": torch.nan_to_num(outputs["z"], nan=0.0, posinf=1.0, neginf=-1.0),
            "compute_budget": torch.nan_to_num(
                outputs["compute_budget"],
                nan=float(self.config.get("compute_floor", 0.05)),
                posinf=1.0,
                neginf=float(self.config.get("compute_floor", 0.05)),
            ).clamp(float(self.config.get("compute_floor", 0.05)), 1.0),
        }

    def _update_teacher_ema(self, agent_id):
        rate = float(self.teacher_ema_rate)
        teacher = self.teacher_agents[agent_id]
        student = self.agents[agent_id]
        with torch.no_grad():
            for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
                teacher_param.data.mul_(1.0 - rate).add_(student_param.data, alpha=rate)
            for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
                teacher_buffer.data.copy_(student_buffer.data)

    def _finite_parameters(self, agent):
        return all(torch.isfinite(param).all().item() for param in agent.parameters())

    def _load_compatible_state(self, module, state):
        current = module.state_dict()
        compatible = {
            key: value
            for key, value in state.items()
            if key in current and tuple(current[key].shape) == tuple(value.shape)
        }
        module.load_state_dict(compatible, strict=False)

    def _save_last_safe(self, agent_id):
        path = self._last_safe_path(agent_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "agent_id": agent_id,
            "training_step": self.training_step,
            "env_step": self.env_step,
            "agent": self.agents[agent_id].state_dict(),
            "optimizer": self.optimizers[agent_id].state_dict(),
            "normalizer": self.normalizer.state_dict(),
        }
        torch.save(payload, path)

    def _recover_last_safe(self, agent_id):
        path = self._last_safe_path(agent_id)
        summary = dict(self.loss_history[agent_id])
        summary["nan_recoveries"] = int(summary.get("nan_recoveries", 0) or 0) + 1
        self.loss_history[agent_id] = summary
        if not path.exists():
            return False
        state = torch.load(path, map_location=self.device, weights_only=False)
        if int(state.get("agent_id", agent_id)) != agent_id:
            return False
        self.agents[agent_id].load_state_dict(state["agent"])
        self.optimizers[agent_id].load_state_dict(state["optimizer"])
        if "normalizer" in state:
            self.normalizer.load_state_dict(state["normalizer"])
        return True

    def _last_safe_path(self, agent_id):
        path = Path(self.nan_recovery_checkpoint)
        if len(self.agents) <= 1:
            return path
        return path.with_name(f"{path.stem}_agent_{agent_id}{path.suffix}")

    def _mutate_agent(
        self,
        agent,
        mutation_std,
        mutation_prob,
        mutate_policy,
        mutate_router,
        mutate_value,
        mutate_world,
        mutate_reservoir,
        fork_diversity_noise,
    ):
        modules = []
        if mutate_policy:
            modules.append(agent.policy)
        if mutate_router:
            modules.append(agent.latent_router)
        if mutate_value:
            modules.append(agent.value_head)
        if mutate_world:
            modules.append(agent.world_model)
        if mutate_reservoir:
            modules.append(agent.reservoir_model)
        std = max(0.0, float(mutation_std) + float(fork_diversity_noise))
        prob = float(np.clip(mutation_prob, 0.0, 1.0))
        if std <= 0.0 or prob <= 0.0:
            return
        with torch.no_grad():
            for module in modules:
                for param in module.parameters():
                    if not param.requires_grad:
                        continue
                    mask = torch.rand_like(param) < prob
                    if not mask.any():
                        continue
                    param.add_(mask * torch.randn_like(param) * std)
                    param.nan_to_num_(nan=0.0, posinf=5.0, neginf=-5.0)
                    param.clamp_(-20.0, 20.0)

    def _imagined_actions(self, agent_id, agent, belief, z, compute_budget, policy_logits):
        candidates = max(1, int(self.config.get("imagined_candidates", 4)))
        if int(self.config.get("imagined_horizon", 3)) <= 0 or float(self.config.get("imagined_weight", 0.2)) <= 0:
            return torch.zeros(policy_logits.shape[0], dtype=torch.long, device=policy_logits.device), torch.zeros(
                policy_logits.shape[0], dtype=torch.float32, device=policy_logits.device
            )
        with torch.no_grad():
            probs = torch.softmax(policy_logits, dim=1)
            top_actions = torch.topk(probs, k=min(candidates, probs.shape[1]), dim=1).indices
            reservoir_next, _reservoir_star, leakage = agent.reservoir_model(belief)
            value = agent.value_head(belief, z, compute_budget).squeeze(1)
            base_score = value - leakage.mean(dim=1) - 0.02 * compute_budget.squeeze(1)
            histogram = self._action_histogram_tensor(agent_id, policy_logits.device)
            diversity_bonus = 1.0 - histogram[top_actions]
            candidate_scores = base_score.unsqueeze(1) + 0.05 * diversity_bonus
            best_idx = torch.argmax(candidate_scores, dim=1)
            imagined_actions = top_actions.gather(1, best_idx.unsqueeze(1)).squeeze(1)
        return imagined_actions, torch.ones(policy_logits.shape[0], dtype=torch.float32, device=policy_logits.device)

    def _fallback_distribution(self, fallback_actions, action_dim, confidence, neighbor_mass, device):
        confidence = float(np.clip(confidence, 0.0, 0.95))
        neighbor_mass = float(np.clip(neighbor_mass, 0.0, max(0.0, 1.0 - confidence)))
        wait_action = action_dim - 1
        rows = []
        for action in fallback_actions.detach().cpu().numpy().astype(np.int64):
            action = int(np.clip(action, 0, action_dim - 1))
            probs = torch.zeros(action_dim, dtype=torch.float32, device=device)
            probs[action] += confidence
            if action != wait_action and action_dim >= 21:
                left = (action - 1) % 20
                right = (action + 1) % 20
                probs[left] += neighbor_mass * 0.5
                probs[right] += neighbor_mass * 0.5
                remaining = max(0.0, 1.0 - confidence - neighbor_mass)
                wait_mass = min(0.01, remaining * 0.1)
                probs[wait_action] += wait_mass
                angular_remaining = remaining - wait_mass
                probs[:wait_action] += angular_remaining / max(1, wait_action)
            else:
                remaining = max(0.0, 1.0 - confidence)
                probs += remaining / max(1, action_dim)
            probs = probs / torch.clamp(probs.sum(), min=1e-8)
            rows.append(probs)
        return torch.stack(rows, dim=0)

    def _hybrid_teacher_distribution(
        self,
        policy_logits,
        fallback_teacher_probs,
        fallback_used_mask,
        imagined_actions,
        imagined_mask,
        temperature,
    ):
        with torch.no_grad():
            temp = max(float(temperature), 1e-3)
            student_probs = torch.softmax(torch.nan_to_num(policy_logits.detach(), nan=0.0).clamp(-20.0, 20.0) / temp, dim=1)
            fallback_weight = 0.15 + 0.65 * fallback_used_mask.view(-1, 1)
            hybrid = (1.0 - fallback_weight) * student_probs + fallback_weight * fallback_teacher_probs.detach()
            if imagined_actions is not None and imagined_mask is not None:
                imagined_probs = torch.full_like(hybrid, 1e-4)
                imagined_probs.scatter_(1, imagined_actions.detach().view(-1, 1), 1.0)
                imagined_probs = imagined_probs / torch.clamp(imagined_probs.sum(dim=1, keepdim=True), min=1e-8)
                imagined_weight = 0.10 * imagined_mask.detach().view(-1, 1)
                hybrid = (1.0 - imagined_weight) * hybrid + imagined_weight * imagined_probs
            hybrid = torch.nan_to_num(hybrid, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-8)
            return hybrid / torch.clamp(hybrid.sum(dim=1, keepdim=True), min=1e-8)

    def _action_histogram_tensor(self, agent_id, device):
        hist = torch.ones(21, dtype=torch.float32, device=device) * 1e-3
        for action in self.learned_action_windows[agent_id]:
            if 0 <= int(action) < hist.numel():
                hist[int(action)] += 1.0
        return hist / hist.sum()

    def _histogram_max_fraction(self, actions):
        if not actions:
            return 0.0
        counts = np.bincount(np.asarray(actions, dtype=np.int64), minlength=21)
        return float(counts.max() / max(1, counts.sum()))

    def _position_novelty(self, positions):
        if not positions:
            return 1.0
        return float(len(set(positions)) / len(positions))

    def _local_loop_score(self, positions):
        if not positions:
            return 0.0
        return float(1.0 - len(set(positions)) / len(positions))

    def _clean_utility(self, food_delta, energy_delta, collision_count, movement_cost, position_novelty):
        target = (
            float(food_delta)
            + 0.2 * (float(energy_delta) / 1000.0)
            - 0.5 * float(collision_count)
            - 0.1 * float(movement_cost)
            + 0.1 * float(position_novelty)
        )
        clip = float(self.config.get("clean_utility_clip", 5.0))
        return float(np.clip(target, -clip, clip))

    def _entropy(self, probs):
        if probs is None:
            return 0.0
        probs = np.asarray(probs, dtype=np.float32)
        return float(-(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum())

    def _make_live_stats(self):
        return {
            "energy": 0.0,
            "food_eaten": 0,
            "movement_cost": 0.0,
            "collision_count": 0,
            "wait_count": 0,
            "leakage": 0.0,
            "leakage_mean_100": 0.0,
            "compute_budget": 0.0,
            "z_mean": 0.0,
            "z_std": 0.0,
            "action_entropy": 0.0,
            "fallback_used_rate": 0.0,
            "selected_action": 0,
            "fallback_used": False,
            "uncertainty": 0.0,
            "useful_transition_score": 0.0,
            "value_loss": None,
            "raw_value_loss": None,
            "clipped_value_loss": None,
            "value_mean": 0.0,
            "value_std": 0.0,
            "value_target_mean": 0.0,
            "value_target_std": 0.0,
            "advantage_mean": 0.0,
            "advantage_std": 0.0,
            "advantage_max_abs": 0.0,
            "raw_policy_loss": None,
            "clipped_policy_loss": None,
            "entropy_target_loss": None,
            "action_diversity_loss": None,
            "action_histogram_max_fraction": 0.0,
            "position_novelty": 1.0,
            "steps_since_food": 0,
            "repeated_same_action_count": 0,
            "compute_target": 0.0,
            "distill_loss": None,
            "hybrid_distill_loss": None,
            "student_teacher_kl": 0.0,
            "fallback_student_kl": 0.0,
            "fallback_action_agreement": 0.0,
            "no_fallback_action_entropy": 0.0,
            "collision_bce_loss": None,
            "movement_cost_huber_loss": None,
            "progress_huber_loss": None,
            "clean_utility_aux_loss": None,
            "teacher_entropy": 0.0,
            "student_entropy": 0.0,
            "teacher_student_kl": 0.0,
            "feedback_score": 0.0,
            "demo_memory_size": 0,
            "demo_score_mean": 0.0,
            "food_per_collision": 0.0,
            "food_per_100_steps": 0.0,
            "collisions_per_100_steps": 0.0,
            "movement_cost_per_food": 0.0,
            "useful_score_per_100_steps": 0.0,
            "energy_slope": 0.0,
            "local_loop_score": 0.0,
            "wall_contact_rate": 0.0,
            "clean_utility": 0.0,
            "raw_total_loss": None,
            "guard_update_skipped": False,
            "nonfinite_update_skipped": False,
            "skipped_updates": 0,
            "nan_recoveries": 0,
            "energy_sum": 0.0,
            "energy_count": 0,
            "leakage_sum": 0.0,
            "movement_cost_sum": 0.0,
            "compute_budget_sum": 0.0,
            "fallback_sum": 0.0,
            "entropy_sum": 0.0,
            "uncertainty_sum": 0.0,
            "useful_transition_sum": 0.0,
        }


def live_food_ratio(food, collisions):
    return float(food) / max(1.0, float(collisions))
