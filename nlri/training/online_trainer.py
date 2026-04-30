from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from .losses import compute_nlri_loss


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
        self.loss_history = [
            {
                "world_prediction_loss": None,
                "reservoir_loss": None,
                "policy_loss": None,
                "loss": None,
            }
            for _ in agents
        ]
        self.live_stats = [self._make_live_stats() for _ in agents]
        self.fallback_windows = [deque(maxlen=100) for _ in agents]
        self.leakage_windows = [deque(maxlen=100) for _ in agents]
        self.movement_cost_windows = [deque(maxlen=100) for _ in agents]
        self.prev_episode_counters = [
            {"food_eaten": 0, "collision_count": 0, "wait_count": 0}
            for _ in agents
        ]
        self.optimizers = [
            torch.optim.Adam(agent.parameters(), lr=self.config.get("learning_rate", 1e-3))
            for agent in agents
        ]
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
        leakage_values = list((agent_info.get("leakage") or {}).values())
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
        )

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

        fallback_flag = 1.0 if debug.get("fallback_used", False) else 0.0
        leakage_mean = float(np.mean(leakage_values) if leakage_values else 0.0)
        action_entropy = self._entropy(debug.get("action_probs"))
        self.fallback_windows[agent_id].append(fallback_flag)
        self.leakage_windows[agent_id].append(leakage_mean)
        self.movement_cost_windows[agent_id].append(float(movement_cost))

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

        live["energy_sum"] += live["energy"]
        live["energy_count"] += 1
        live["leakage_sum"] += leakage_mean
        live["compute_budget_sum"] += live["compute_budget"]
        live["fallback_sum"] += fallback_flag
        live["entropy_sum"] += action_entropy
        live["uncertainty_sum"] += live["uncertainty"]

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
        world_weight = 0.0 if self.ablation == "no-world" else 1.0
        reservoir_weight = 0.0 if self.ablation == "no-reservoir" else 1.0
        policy_weight = float(self.config.get("policy_weight", 0.1))

        for agent_id, samples in grouped.items():
            agent = self.agents[agent_id]
            optimizer = self.optimizers[agent_id]
            obs_batch = self._stack_obs([sample["obs"] for sample in samples])
            next_obs_batch = self._stack_obs([sample["next_obs"] for sample in samples])

            obs_t = agent._tensorize_obs(obs_batch)
            next_obs_t = agent._tensorize_obs(next_obs_batch)
            encoded = agent.encoder(obs_t)
            target_next_embedding = agent.encoder(next_obs_t).detach()
            zero_belief = torch.zeros(encoded.shape[0], agent.belief_dim, device=agent.device)
            pred_next_belief, pred_obs_embedding, uncertainty = agent.world_model(encoded, zero_belief)
            if self.ablation == "no-router":
                z = torch.zeros(encoded.shape[0], agent.z_dim, device=agent.device)
                compute_budget = torch.full((encoded.shape[0], 1), 0.5, device=agent.device)
            else:
                z, compute_budget = agent.latent_router(pred_next_belief)
            reservoir_next_pred, reservoir_star_pred, leakage_pred = agent.reservoir_model(pred_next_belief)
            policy_logits = agent.policy(pred_next_belief, z, compute_budget)

            reservoir_next_target = self._stack_reservoir(samples, "next_reservoir", agent.device)
            reservoir_star_target = self._stack_reservoir(samples, "reservoir_star", agent.device)
            leakage_target = self._stack_reservoir(samples, "leakage", agent.device)
            actions = torch.tensor([sample["action"] for sample in samples], dtype=torch.long, device=agent.device)

            losses = compute_nlri_loss(
                pred_next_belief=pred_next_belief,
                pred_obs_embedding=pred_obs_embedding,
                target_obs_embedding=target_next_embedding,
                reservoir_next_pred=reservoir_next_pred,
                reservoir_next_target=reservoir_next_target,
                reservoir_star_pred=reservoir_star_pred,
                reservoir_star_target=reservoir_star_target,
                leakage_pred=leakage_pred,
                leakage_target=leakage_target,
                policy_logits=policy_logits,
                actions=actions,
                uncertainty=uncertainty,
                z=z,
                compute_budget=compute_budget,
                world_weight=world_weight,
                reservoir_weight=reservoir_weight,
                beta=policy_weight,
            )

            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()

            summary = {name: float(value.detach().cpu().item()) for name, value in losses.items()}
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
            "live_stats": self.live_stats,
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
        state = torch.load(checkpoint_path, map_location=self.device)
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
        if "live_stats" in state:
            self.live_stats = state["live_stats"]
        return str(checkpoint_path)

    def reset_live_tracking(self):
        self.live_stats = [self._make_live_stats() for _ in self.agents]
        self.fallback_windows = [deque(maxlen=100) for _ in self.agents]
        self.leakage_windows = [deque(maxlen=100) for _ in self.agents]
        self.movement_cost_windows = [deque(maxlen=100) for _ in self.agents]
        self.prev_episode_counters = [
            {"food_eaten": 0, "collision_count": 0, "wait_count": 0}
            for _ in self.agents
        ]

    def metrics(self):
        rows = []
        for agent_id, stats in enumerate(self.live_stats):
            row = dict(stats)
            row.update(self.loss_history[agent_id])
            rows.append(row)
        return {
            "training_step": self.training_step,
            "env_step": self.env_step,
            "per_agent": rows,
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
            "energy_sum": 0.0,
            "energy_count": 0,
            "leakage_sum": 0.0,
            "compute_budget_sum": 0.0,
            "fallback_sum": 0.0,
            "entropy_sum": 0.0,
            "uncertainty_sum": 0.0,
        }
