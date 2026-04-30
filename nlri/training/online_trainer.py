from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from .losses import compute_nlri_loss
from .normalizer import MultiNormalizer


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
        self.loss_ema_beta = float(self.config.get("loss_ema_beta", 0.98))
        self.normalizer = MultiNormalizer(clamp=10.0)
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
                "bc_loss": None,
                "value_loss": None,
                "value_mean": None,
                "advantage_mean": None,
                "compute_loss": None,
                "utility_aux_loss": None,
                "entropy_bonus": None,
                "entropy_target_loss": None,
                "action_diversity_loss": None,
                "compute_target": None,
                "loss": None,
                "nonfinite_update_skipped": False,
            }
            for _ in agents
        ]
        self.live_stats = [self._make_live_stats() for _ in agents]
        self.fallback_windows = [deque(maxlen=100) for _ in agents]
        self.leakage_windows = [deque(maxlen=100) for _ in agents]
        self.movement_cost_windows = [deque(maxlen=100) for _ in agents]
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
        self.optimizers = [
            torch.optim.Adam(agent.parameters(), lr=self.config.get("learning_rate", 1e-3))
            for agent in agents
        ]
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
            learned_action=learned_action,
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
        live["useful_transition_score"] = float(np.mean(self.utility_windows[agent_id]))
        live["action_histogram_max_fraction"] = self._histogram_max_fraction(self.learned_action_windows[agent_id])
        live["position_novelty"] = self._position_novelty(self.position_windows[agent_id])
        live["steps_since_food"] = int(self.steps_since_food[agent_id])
        live["repeated_same_action_count"] = int(self.repeated_action_counts[agent_id])

        live["energy_sum"] += live["energy"]
        live["energy_count"] += 1
        live["leakage_sum"] += leakage_mean
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
        policy_weight = float(self.config.get("policy_loss_weight", 0.5))
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
            reservoir_next_pred, reservoir_star_pred, leakage_pred = agent.reservoir_model(pred_next_belief)
            policy_logits = agent.policy(pred_next_belief, z, compute_budget)
            value_pred = agent.value_head(pred_next_belief, z, compute_budget)
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
                next_value = agent.value_head(next_belief, next_z, next_compute_budget).squeeze(1)

            reservoir_next_target = self._stack_reservoir(samples, "next_reservoir", agent.device)
            reservoir_star_target = self._stack_reservoir(samples, "reservoir_star", agent.device)
            leakage_target = self._stack_reservoir(samples, "leakage", agent.device)
            utility_scores = torch.tensor(
                [sample.get("useful_transition_score", 0.0) for sample in samples],
                dtype=torch.float32,
                device=agent.device,
            )
            reservoir_leakage_penalty = leakage_target.mean(dim=1)
            utility_target = torch.clamp(utility_scores - reservoir_leakage_penalty, -10.0, 10.0)
            done_mask = torch.tensor(
                [0.0 if sample.get("done") else 1.0 for sample in samples],
                dtype=torch.float32,
                device=agent.device,
            )
            value_target = utility_target + float(self.config.get("gamma", 0.99)) * done_mask * next_value
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

            if not self._finite_tensor(
                [
                    normalized_world_target,
                    normalized_reservoir_next_target,
                    normalized_reservoir_star_target,
                    normalized_leakage_target,
                    utility_scores,
                    value_target,
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
                value_pred=value_pred,
                value_target=value_target,
                imagined_actions=imagined_actions,
                imagined_mask=imagined_mask,
                action_histogram=action_histogram,
                uncertainty=uncertainty,
                z=z,
                compute_budget=compute_budget,
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
            )

            total_loss = losses["loss"]
            if not torch.isfinite(total_loss):
                self._mark_skip(agent_id)
                continue

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), self.grad_clip)
            optimizer.step()

            summary = {name: float(value.detach().cpu().item()) for name, value in losses.items()}
            ema_state = self.loss_ema[agent_id]
            ema_state["world_loss_ema"] = self._ema(ema_state["world_loss_ema"], summary["world_prediction_loss"])
            ema_state["total_loss_ema"] = self._ema(ema_state["total_loss_ema"], summary["loss"])
            summary["world_loss_ema"] = ema_state["world_loss_ema"]
            summary["nonfinite_update_skipped"] = False
            self.loss_history[agent_id] = summary
            summaries[agent_id] = summary

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
            agent.load_state_dict(agent_state)
        optimizer_states = state.get("optimizers", [])
        if not self.eval_mode:
            for optimizer, optimizer_state in zip(self.optimizers, optimizer_states):
                optimizer.load_state_dict(optimizer_state)
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

    def _mark_skip(self, agent_id):
        summary = dict(self.loss_history[agent_id])
        summary["nonfinite_update_skipped"] = True
        self.loss_history[agent_id] = summary

    def _finite_batch(self, batch: Dict[str, np.ndarray]):
        return all(np.isfinite(np.asarray(value)).all() for value in batch.values())

    def _finite_tensor(self, tensors: List[torch.Tensor]):
        return all(torch.isfinite(tensor).all().item() for tensor in tensors)

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
            "value_mean": 0.0,
            "advantage_mean": 0.0,
            "entropy_target_loss": None,
            "action_diversity_loss": None,
            "action_histogram_max_fraction": 0.0,
            "position_novelty": 1.0,
            "steps_since_food": 0,
            "repeated_same_action_count": 0,
            "compute_target": 0.0,
            "energy_sum": 0.0,
            "energy_count": 0,
            "leakage_sum": 0.0,
            "compute_budget_sum": 0.0,
            "fallback_sum": 0.0,
            "entropy_sum": 0.0,
            "uncertainty_sum": 0.0,
            "useful_transition_sum": 0.0,
        }
