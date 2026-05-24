from __future__ import annotations

import argparse
import csv
import json
import os
from collections import deque
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from nlri.envs.maze_reservoir_env import ACTION_DIM, MazeReservoirEnv
from nlri.models.dual_system import DualSystemAgent, RESERVOIR_KEYS
from nlri.training import ReplayBuffer
from nlri.training.latent_diagnostics import LatentDiagnosticsCollector
from nlri.training.normalizer import MultiNormalizer


CSV_COLUMNS = [
    "step",
    "agent_id",
    "energy",
    "food_eaten",
    "collisions_per_100_steps",
    "fallback_rate",
    "world_error",
    "reservoir_error",
    "collision_error",
    "energy_delta_error",
    "error_ema",
    "z_mean",
    "z_std",
    "z_norm",
    "slow_energy_scale",
    "slow_compute_budget",
    "slow_goal_entropy",
    "slow_tick",
    "policy_loss",
    "value_loss",
    "entropy",
    "total_loss",
    "warnings",
]


def build_parser():
    parser = argparse.ArgumentParser(description="Realtime dual-system embodied NLRI simulation.")
    parser.add_argument("--steps", type=int, default=0, help="0 means run until the pygame window closes.")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--slow-interval", type=int, default=25)
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--buffer-capacity", type=int, default=20000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--policy-lr", type=float, default=5e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/dual_system")
    parser.add_argument("--checkpoint-path", type=str, default="")
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--force-no-fallback", action="store_true")
    parser.add_argument("--fallback-prob", type=float, default=1.0)
    parser.add_argument("--fallback-decay", type=float, default=0.0002)
    parser.add_argument("--min-fallback-prob", type=float, default=0.25)
    parser.add_argument("--eval-fallback-prob", type=float, default=0.0)
    parser.add_argument("--policy-temperature", type=float, default=1.0)
    parser.add_argument("--eval-temperature", type=float, default=0.55)
    parser.add_argument("--deterministic-eval", action="store_true")
    parser.add_argument("--dummy-sdl", action="store_true")
    parser.add_argument("--render-mode", choices=["human", "none"], default="human")
    parser.add_argument("--quiet", action="store_true")
    return parser


def parse_args():
    return build_parser().parse_args()


class OnlineDualSystemTrainer:
    def __init__(self, agents: List[DualSystemAgent], replay_buffer: ReplayBuffer, config: Dict[str, Any]):
        self.agents = agents
        self.replay_buffer = replay_buffer
        self.config = dict(config)
        self.device = agents[0].device if agents else torch.device("cpu")
        self.normalizer = MultiNormalizer(clamp=10.0)
        self.env_step = 0
        self.training_step = 0
        self.eval_mode = bool(self.config.get("eval_mode", False))
        self.optimizers = [torch.optim.Adam(self._param_groups(agent)) for agent in agents]
        for agent in agents:
            agent.obs_normalizer = self.normalizer
        self.live_stats = [self._make_live_stats() for _ in agents]
        self.loss_history = [self._make_loss_history() for _ in agents]
        self.error_windows = [deque(maxlen=100) for _ in agents]
        self.fallback_windows = [deque(maxlen=100) for _ in agents]
        self.collision_windows = [deque(maxlen=100) for _ in agents]
        self.prev_counters = [{"food_eaten": 0, "collision_count": 0} for _ in agents]
        self.error_ema = [0.0 for _ in agents]

    def observe_transition(
        self,
        agent_id: int,
        obs: Dict[str, Any],
        action: int,
        next_obs: Dict[str, Any],
        reward: float,
        done: bool,
        agent_info: Dict[str, Any],
        debug: Dict[str, Any],
        collision: bool,
        movement_cost: float,
        slow_tick: bool,
    ):
        self._update_normalizers(obs, next_obs, agent_info)
        actual_reservoir = self._reservoir_array(agent_info.get("reservoir_next") or {})
        predicted_reservoir = np.asarray(debug.get("pred_reservoir", np.zeros(len(RESERVOIR_KEYS))), dtype=np.float32)
        reservoir_error = float(np.mean(np.abs(predicted_reservoir - actual_reservoir)))
        with torch.no_grad():
            agent = self.agents[agent_id]
            next_obs_t = agent._tensorize_obs(self.normalizer.normalize_obs(next_obs), normalize=False)
            actual_next_z = agent.shared_encoder(next_obs_t)[0].detach().cpu().numpy()
        pred_next_z = np.asarray(debug.get("pred_next_z", np.zeros_like(actual_next_z)), dtype=np.float32)
        world_error = float(np.mean(np.abs(pred_next_z - actual_next_z)))
        collision_target = 1.0 if collision else 0.0
        collision_error = abs(float(debug.get("pred_collision_prob", 0.0)) - collision_target)
        energy_delta = (float(next_obs["energy"][0]) - float(obs["energy"][0])) * 1000.0
        energy_delta_error = abs(float(debug.get("pred_energy_delta", 0.0)) - energy_delta)
        total_error = world_error + reservoir_error + collision_error + 0.05 * energy_delta_error
        self.error_ema[agent_id] = 0.98 * self.error_ema[agent_id] + 0.02 * total_error
        self.error_windows[agent_id].append(total_error)
        self.fallback_windows[agent_id].append(1.0 if debug.get("fallback_used") else 0.0)

        prev = self.prev_counters[agent_id]
        food_now = int(agent_info.get("food_eaten", 0))
        collision_now = int(agent_info.get("collision_count", 0))
        food_delta = food_now if food_now < prev["food_eaten"] else food_now - prev["food_eaten"]
        collision_delta = collision_now if collision_now < prev["collision_count"] else collision_now - prev["collision_count"]
        self.prev_counters[agent_id] = {"food_eaten": food_now, "collision_count": collision_now}
        self.collision_windows[agent_id].append(float(collision_delta))

        self.replay_buffer.push(
            agent_id=agent_id,
            obs=obs,
            action=int(action),
            next_obs=next_obs,
            reward=float(reward),
            done=bool(done),
            reservoir=agent_info.get("reservoir") or {},
            next_reservoir=agent_info.get("reservoir_next") or {},
            collision=bool(collision),
            movement_cost=float(movement_cost),
            progress=float(np.linalg.norm(np.asarray(next_obs.get("last_movement", [0.0, 0.0]), dtype=np.float32))),
            energy_delta=float(energy_delta),
            fallback_action=int(debug.get("fallback_action", action)),
            fallback_used=bool(debug.get("fallback_used", False)),
            error_summary=self.error_summary(agent_id),
            history_summary=self.history_summary(agent_id),
        )

        entropy_value = entropy(debug.get("action_probs"))
        goal_probs = np.asarray(debug.get("slow_goal_probs", np.ones(4) / 4.0), dtype=np.float32)
        goal_entropy = entropy(goal_probs)
        live = self.live_stats[agent_id]
        live.update(
            {
                "energy": float(agent_info.get("energy", 0.0)),
                "food_eaten": live["food_eaten"] + int(food_delta),
                "collision_count": live["collision_count"] + int(collision_delta),
                "collisions_per_100_steps": 100.0 * float(np.sum(self.collision_windows[agent_id])) / max(1, len(self.collision_windows[agent_id])),
                "fallback_rate": float(np.mean(self.fallback_windows[agent_id])) if self.fallback_windows[agent_id] else 0.0,
                "world_error": world_error,
                "reservoir_error": reservoir_error,
                "collision_error": collision_error,
                "energy_delta_error": energy_delta_error,
                "error_ema": float(self.error_ema[agent_id]),
                "z_mean": float(debug.get("z_mean", 0.0)),
                "z_std": float(debug.get("z_std", 0.0)),
                "z_norm": float(debug.get("z_norm", 0.0)),
                "slow_energy_scale": float(debug.get("slow_energy_scale", 1.0)),
                "slow_compute_budget": float(debug.get("slow_compute_budget", 0.5)),
                "slow_goal_entropy": goal_entropy,
                "slow_tick": bool(slow_tick),
                "selected_action": int(debug.get("selected_action", action)),
                "fallback_used": bool(debug.get("fallback_used", False)),
                "action_entropy": entropy_value,
                "slow_goal_probs": goal_probs,
                "energy_sum": live["energy_sum"] + float(agent_info.get("energy", 0.0)),
                "count": live["count"] + 1,
                "fallback_sum": live["fallback_sum"] + (1.0 if debug.get("fallback_used") else 0.0),
            }
        )

    def train_step(self, batch_size: int | None = None):
        if self.eval_mode or len(self.replay_buffer) < (batch_size or self.config.get("batch_size", 64)):
            return {}
        samples = self.replay_buffer.sample(batch_size or self.config.get("batch_size", 64))
        grouped: Dict[int, List[Dict[str, Any]]] = {}
        for sample in samples:
            grouped.setdefault(int(sample["agent_id"]), []).append(sample)

        summaries = {}
        for agent_id, rows in grouped.items():
            agent = self.agents[agent_id]
            optimizer = self.optimizers[agent_id]
            obs_batch = self._normalize_obs_batch(self._stack_obs([row["obs"] for row in rows]))
            next_obs_batch = self._normalize_obs_batch(self._stack_obs([row["next_obs"] for row in rows]))
            if not self._finite_batch(obs_batch) or not self._finite_batch(next_obs_batch):
                continue

            obs_t = agent._tensorize_obs(obs_batch, normalize=False)
            next_obs_t = agent._tensorize_obs(next_obs_batch, normalize=False)
            errors = torch.tensor(np.stack([row["error_summary"] for row in rows]), dtype=torch.float32, device=agent.device)
            reservoirs = torch.tensor(
                np.stack([self._reservoir_array(row.get("reservoir") or {}) for row in rows], axis=0),
                dtype=torch.float32,
                device=agent.device,
            )
            histories = torch.tensor(np.stack([row["history_summary"] for row in rows]), dtype=torch.float32, device=agent.device)
            output = agent.forward_tensors(
                obs_t,
                error_summary=errors,
                reservoir=reservoirs,
                history_summary=histories,
                force_slow_tick=True,
            )
            with torch.no_grad():
                next_z = agent.shared_encoder(next_obs_t)
                next_output = agent.forward_tensors(
                    next_obs_t,
                    error_summary=errors,
                    reservoir=torch.tensor(
                        np.stack([self._reservoir_array(row.get("next_reservoir") or {}) for row in rows], axis=0),
                        dtype=torch.float32,
                        device=agent.device,
                    ),
                    history_summary=histories,
                    force_slow_tick=True,
                )
                next_value = next_output.value.squeeze(1)

            actions = torch.tensor([row["action"] for row in rows], dtype=torch.long, device=agent.device)
            fallback_actions = torch.tensor([row.get("fallback_action", row["action"]) for row in rows], dtype=torch.long, device=agent.device)
            rewards = torch.tensor([row["reward"] for row in rows], dtype=torch.float32, device=agent.device).clamp(-10.0, 10.0)
            done_mask = torch.tensor([0.0 if row.get("done") else 1.0 for row in rows], dtype=torch.float32, device=agent.device)
            value_target = (rewards + 0.99 * done_mask * next_value).clamp(-10.0, 10.0)
            reservoir_target = torch.tensor(
                np.stack([self._reservoir_array(row.get("next_reservoir") or {}) for row in rows], axis=0),
                dtype=torch.float32,
                device=agent.device,
            )
            collision_target = torch.tensor([[1.0 if row.get("collision") else 0.0] for row in rows], dtype=torch.float32, device=agent.device)
            movement_target = torch.tensor([[row.get("movement_cost", 0.0)] for row in rows], dtype=torch.float32, device=agent.device).clamp(0.0, 5.0)
            progress_target = torch.tensor([[row.get("progress", 0.0)] for row in rows], dtype=torch.float32, device=agent.device).clamp(0.0, 1.0)
            energy_target = torch.tensor([[row.get("energy_delta", 0.0)] for row in rows], dtype=torch.float32, device=agent.device).clamp(-5.0, 5.0)

            log_probs = F.log_softmax(output.final_logits, dim=1)
            probs = torch.exp(log_probs)
            entropy_bonus = -(probs * log_probs).sum(dim=1).mean()
            advantage = (value_target - output.value.squeeze(1)).detach().clamp(-5.0, 5.0)
            policy_loss = -(log_probs.gather(1, actions[:, None]).squeeze(1) * advantage).mean()
            value_loss = F.huber_loss(output.value.squeeze(1), value_target, delta=1.0)
            bc_weight = max(0.05, 1.0 - 0.0005 * max(0, self.env_step - int(self.config.get("warmup_steps", 0))))
            bc_loss = F.cross_entropy(output.final_logits, fallback_actions)
            world_loss = F.huber_loss(output.fast.pred_next_z, next_z.detach(), delta=1.0)
            reservoir_loss = F.huber_loss(output.fast.pred_reservoir, reservoir_target, delta=1.0)
            collision_loss = F.binary_cross_entropy_with_logits(output.fast.collision_logit, collision_target)
            movement_loss = F.huber_loss(output.fast.movement_cost, movement_target, delta=1.0)
            progress_loss = F.binary_cross_entropy_with_logits(output.fast.progress_logit, progress_target)
            energy_loss = F.huber_loss(output.fast.energy_delta, energy_target, delta=1.0)
            compute_penalty = output.slow.compute_budget.mean() * 0.01
            total_loss = (
                0.5 * world_loss
                + reservoir_loss
                + 0.5 * collision_loss
                + 0.1 * movement_loss
                + 0.1 * progress_loss
                + 0.05 * energy_loss
                + 0.2 * policy_loss
                + 0.5 * value_loss
                + bc_weight * bc_loss
                - 0.01 * entropy_bonus
                + compute_penalty
            )
            if not torch.isfinite(total_loss):
                continue
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), float(self.config.get("grad_clip", 1.0)))
            optimizer.step()

            summary = {
                "world_error": float(world_loss.detach().cpu().item()),
                "reservoir_error": float(reservoir_loss.detach().cpu().item()),
                "collision_error": float(collision_loss.detach().cpu().item()),
                "energy_delta_error": float(energy_loss.detach().cpu().item()),
                "policy_loss": float(policy_loss.detach().cpu().item()),
                "value_loss": float(value_loss.detach().cpu().item()),
                "entropy": float(entropy_bonus.detach().cpu().item()),
                "total_loss": float(total_loss.detach().cpu().item()),
            }
            self.loss_history[agent_id].update(summary)
            summaries[agent_id] = summary
        self.training_step += 1
        return summaries

    def error_summary(self, agent_id: int):
        live = self.live_stats[agent_id]
        return np.asarray(
            [
                live.get("world_error", 0.0),
                live.get("reservoir_error", 0.0),
                live.get("collision_error", 0.0),
                live.get("energy_delta_error", 0.0) * 0.05,
                self.error_ema[agent_id],
            ],
            dtype=np.float32,
        )

    def history_summary(self, agent_id: int):
        live = self.live_stats[agent_id]
        return np.asarray(
            [
                live.get("fallback_rate", 0.0),
                live.get("collisions_per_100_steps", 0.0) / 100.0,
                live.get("action_entropy", 0.0) / max(1.0, np.log(ACTION_DIM)),
                live.get("energy", 0.0) / 1000.0,
            ],
            dtype=np.float32,
        )

    def metrics(self):
        rows = []
        for agent_id, live in enumerate(self.live_stats):
            row = dict(live)
            row.update(self.loss_history[agent_id])
            warnings = []
            if row.get("z_std", 0.0) < 0.01:
                warnings.append("z_collapse")
            if row.get("action_entropy", 0.0) < 0.05 and self.env_step > 100:
                warnings.append("action_collapse")
            row["warning_flags"] = warnings
            rows.append(row)
        return {"training_step": self.training_step, "env_step": self.env_step, "per_agent": rows}

    def save_checkpoint(self, checkpoint_dir, step: int, periodic: bool = True):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "training_step": self.training_step,
            "env_step": self.env_step,
            "config": self.config,
            "normalizer": self.normalizer.state_dict(),
            "agents": [agent.state_dict() for agent in self.agents],
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            "loss_history": self.loss_history,
            "live_stats": self.live_stats,
            "error_ema": self.error_ema,
        }
        latest = checkpoint_dir / "latest.pt"
        torch.save(state, latest)
        paths = [str(latest)]
        if periodic:
            periodic_path = checkpoint_dir / f"step_{step:07d}.pt"
            torch.save(state, periodic_path)
            paths.append(str(periodic_path))
        return paths

    def load_checkpoint(self, checkpoint_path_or_dir):
        path = Path(checkpoint_path_or_dir)
        checkpoint_path = path / "latest.pt" if path.is_dir() else path
        if not checkpoint_path.exists():
            return False
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        for agent, agent_state in zip(self.agents, state.get("agents", [])):
            agent.load_state_dict(agent_state, strict=False)
            agent.obs_normalizer = self.normalizer
            agent.reset_state()
        if not self.eval_mode:
            for optimizer, optimizer_state in zip(self.optimizers, state.get("optimizers", [])):
                try:
                    optimizer.load_state_dict(optimizer_state)
                except ValueError:
                    pass
        self.training_step = int(state.get("training_step", 0))
        self.env_step = int(state.get("env_step", 0))
        if "normalizer" in state:
            self.normalizer.load_state_dict(state["normalizer"])
        self.loss_history = state.get("loss_history", self.loss_history)
        self.live_stats = state.get("live_stats", self.live_stats)
        self.error_ema = state.get("error_ema", self.error_ema)
        return str(checkpoint_path)

    def reset_live_tracking(self):
        self.live_stats = [self._make_live_stats() for _ in self.agents]
        self.loss_history = [self._make_loss_history() for _ in self.agents]
        self.error_windows = [deque(maxlen=100) for _ in self.agents]
        self.fallback_windows = [deque(maxlen=100) for _ in self.agents]
        self.collision_windows = [deque(maxlen=100) for _ in self.agents]
        self.prev_counters = [{"food_eaten": 0, "collision_count": 0} for _ in self.agents]
        self.error_ema = [0.0 for _ in self.agents]

    def _param_groups(self, agent):
        return [
            {"params": list(agent.shared_encoder.parameters()) + list(agent.fast_system.parameters()), "lr": float(self.config.get("learning_rate", 1e-4))},
            {"params": agent.slow_system.parameters(), "lr": float(self.config.get("policy_lr", 5e-5))},
        ]

    def _stack_obs(self, obs_list: List[Dict[str, Any]]):
        return {key: np.stack([np.asarray(obs[key], dtype=np.float32) for obs in obs_list], axis=0) for key in obs_list[0].keys()}

    def _normalize_obs_batch(self, obs_batch: Dict[str, np.ndarray]):
        return {key: self.normalizer.normalize_array(key, value) for key, value in obs_batch.items()}

    def _finite_batch(self, batch: Dict[str, np.ndarray]):
        return all(np.isfinite(np.asarray(value)).all() for value in batch.values())

    def _update_normalizers(self, obs, next_obs, agent_info):
        for source in (obs, next_obs):
            for key, value in source.items():
                self.normalizer.update(key, value)
        for key in ("reservoir", "reservoir_next"):
            self.normalizer.update("reservoir_target", self._reservoir_array(agent_info.get(key) or {}))

    def _reservoir_array(self, reservoir: Dict[str, float]):
        return np.asarray([float((reservoir or {}).get(key, 0.0)) for key in RESERVOIR_KEYS], dtype=np.float32)

    def _make_live_stats(self):
        return {
            "energy": 0.0,
            "food_eaten": 0,
            "collision_count": 0,
            "collisions_per_100_steps": 0.0,
            "fallback_rate": 0.0,
            "world_error": 0.0,
            "reservoir_error": 0.0,
            "collision_error": 0.0,
            "energy_delta_error": 0.0,
            "error_ema": 0.0,
            "z_mean": 0.0,
            "z_std": 0.0,
            "z_norm": 0.0,
            "slow_energy_scale": 1.0,
            "slow_compute_budget": 0.5,
            "slow_goal_entropy": 0.0,
            "slow_tick": False,
            "selected_action": 0,
            "fallback_used": False,
            "action_entropy": 0.0,
            "slow_goal_probs": np.ones(4, dtype=np.float32) / 4.0,
            "energy_sum": 0.0,
            "count": 0,
            "fallback_sum": 0.0,
        }

    def _make_loss_history(self):
        return {
            "policy_loss": None,
            "value_loss": None,
            "entropy": None,
            "total_loss": None,
        }


def run_session(args):
    if args.dummy_sdl:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        args.render_mode = "none"
    run_dir = make_run_dir(args)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.csv"
    summary_path = run_dir / "final_summary.json"
    latent_path = run_dir / "latent_samples.npz"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump({key: _jsonable(value) for key, value in vars(args).items()}, handle, indent=2, sort_keys=True)

    max_steps = args.steps if args.steps > 0 else 1_000_000_000
    env = MazeReservoirEnv(render_mode=None if args.render_mode == "none" else "human", max_steps=max_steps)
    if env.viewer is not None:
        env.viewer.fps = args.fps
    observations, _ = env.reset(seed=args.seed)
    agents = [DualSystemAgent(use_legacy_fallback=True) for _ in observations]
    for agent in agents:
        agent.reset_state()
        agent.eval() if args.eval else agent.train()
    trainer = OnlineDualSystemTrainer(
        agents,
        ReplayBuffer(capacity=args.buffer_capacity),
        {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "policy_lr": args.policy_lr,
            "grad_clip": args.grad_clip,
            "warmup_steps": args.warmup_steps,
            "eval_mode": args.eval,
        },
    )
    latent_collector = LatentDiagnosticsCollector()
    loaded_checkpoint = None
    if args.resume:
        loaded_checkpoint = trainer.load_checkpoint(args.checkpoint_path or checkpoint_dir)
        if args.eval:
            trainer.reset_live_tracking()
        if not args.quiet:
            print(f"resume={'loaded' if loaded_checkpoint else 'not-found'} path={args.checkpoint_path or checkpoint_dir / 'latest.pt'}")

    rows = []
    last_checkpoint = ""
    step = 0
    try:
        while True:
            if env.viewer is not None and env.viewer.closed:
                break
            if args.steps > 0 and step >= args.steps:
                break

            fallback_prob = fallback_probability(step, args)
            actions = []
            debug_rows = []
            slow_tick = step % max(1, args.slow_interval) == 0
            for agent_id, (agent, obs) in enumerate(zip(agents, observations)):
                agent.use_legacy_fallback = not args.force_no_fallback
                action, debug = agent.act(
                    obs,
                    error_summary=trainer.error_summary(agent_id),
                    reservoir=env.last_info.get("agents", [{}])[agent_id].get("reservoir_next") if env.last_info else {},
                    history_summary=trainer.history_summary(agent_id),
                    slow_tick=slow_tick,
                    deterministic=args.eval and args.deterministic_eval,
                    fallback_probability=fallback_prob,
                    force_no_fallback=args.force_no_fallback,
                    temperature=args.eval_temperature if args.eval else args.policy_temperature,
                )
                actions.append(action)
                debug_rows.append(debug)

            next_observations, rewards, terminated, truncated, info = env.step(actions)
            done = bool(terminated or truncated)
            for agent_id, (obs, next_obs, reward, debug, agent_info) in enumerate(
                zip(observations, next_observations, rewards, debug_rows, info["agents"])
            ):
                trainer.observe_transition(
                    agent_id=agent_id,
                    obs=obs,
                    action=actions[agent_id],
                    next_obs=next_obs,
                    reward=float(reward),
                    done=done,
                    agent_info=agent_info,
                    debug=debug,
                    collision=bool(info["collisions"][agent_id]),
                    movement_cost=float(info["movement_costs"][agent_id]),
                    slow_tick=slow_tick,
                )
                latent_collector.add(
                    z=debug.get("z", []),
                    compute_budget=float(debug.get("slow_compute_budget", 0.0)),
                    energy=float(agent_info.get("energy", 0.0)),
                    leakage=float(np_mean_dict(agent_info.get("leakage") or {})),
                    visible_food_value=float((agent_info.get("reservoir_next") or {}).get("visible_food_value", 0.0)),
                    reachable_food_value=float((agent_info.get("reservoir_next") or {}).get("reachable_food_value", 0.0)),
                    collision_safety=float((agent_info.get("reservoir_next") or {}).get("collision_safety", 0.0)),
                    selected_action=int(debug.get("selected_action", 0)),
                    fallback_used=bool(debug.get("fallback_used", False)),
                    uncertainty=float(trainer.error_ema[agent_id]),
                    action_entropy=float(entropy(debug.get("action_probs"))),
                )
            trainer.env_step = step + 1

            if not args.eval and step >= args.warmup_steps and step % max(1, args.train_every) == 0:
                trainer.train_step(args.batch_size)

            metrics = trainer.metrics()
            env.overlay_stats = []
            countdown = (args.slow_interval - ((step + 1) % args.slow_interval)) % args.slow_interval
            for agent_id, (agent_info, metric_row) in enumerate(zip(info["agents"], metrics["per_agent"])):
                env.overlay_stats.append(
                    {
                        "mode": "DUAL SYSTEM",
                        "color": agent_info["color"],
                        "energy": agent_info["energy"],
                        "food_eaten": metric_row["food_eaten"],
                        "collisions_per_100_steps": metric_row["collisions_per_100_steps"],
                        "fallback_rate": metric_row["fallback_rate"],
                        "world_error": metric_row["error_ema"],
                        "slow_energy_scale": metric_row["slow_energy_scale"],
                        "slow_countdown": countdown,
                        "slow_goal_probs": metric_row.get("slow_goal_probs"),
                        "selected_action": metric_row["selected_action"],
                        "fallback_used": metric_row["fallback_used"],
                        "loss": metric_row.get("total_loss"),
                    }
                )

            step += 1
            observations = next_observations
            if step % 100 == 0:
                snapshot = snapshot_rows(step, metrics)
                rows.extend(snapshot)
                if not args.quiet:
                    for row in snapshot:
                        print(format_metrics(row))
            if not args.eval and step % max(1, args.checkpoint_every) == 0:
                last_checkpoint = trainer.save_checkpoint(checkpoint_dir, step, periodic=True)[0]
            if done:
                observations, _ = env.reset(seed=args.seed + step)
                for agent in agents:
                    agent.reset_state()
    finally:
        env.close()

    final_metrics = trainer.metrics()
    if step > 0 and (not rows or rows[-1]["step"] != step):
        rows.extend(snapshot_rows(step, final_metrics))
    if not args.eval:
        last_checkpoint = trainer.save_checkpoint(checkpoint_dir, step, periodic=False)[0]
    write_metrics_csv(metrics_path, rows)
    latent_collector.save(latent_path)
    summary = build_final_summary(final_metrics, latent_collector.summary(), last_checkpoint, loaded_checkpoint, step)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    if not args.quiet:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return {"run_dir": str(run_dir), "summary": summary}


def fallback_probability(step: int, args):
    if args.force_no_fallback:
        return 0.0
    if args.eval:
        return max(0.0, min(1.0, float(args.eval_fallback_prob)))
    if step < args.warmup_steps:
        return 1.0
    return max(float(args.min_fallback_prob), float(args.fallback_prob) - float(args.fallback_decay) * (step - args.warmup_steps))


def make_run_dir(args):
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = Path("runs/dual_system") / f"{stamp}_{'eval' if args.eval else 'train'}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def snapshot_rows(step, metrics):
    rows = []
    for agent_id, row in enumerate(metrics["per_agent"]):
        rows.append(
            {
                "step": step,
                "agent_id": agent_id,
                "energy": float(row.get("energy", 0.0)),
                "food_eaten": int(row.get("food_eaten", 0)),
                "collisions_per_100_steps": float(row.get("collisions_per_100_steps", 0.0)),
                "fallback_rate": float(row.get("fallback_rate", 0.0)),
                "world_error": float(row.get("world_error", 0.0)),
                "reservoir_error": float(row.get("reservoir_error", 0.0)),
                "collision_error": float(row.get("collision_error", 0.0)),
                "energy_delta_error": float(row.get("energy_delta_error", 0.0)),
                "error_ema": float(row.get("error_ema", 0.0)),
                "z_mean": float(row.get("z_mean", 0.0)),
                "z_std": float(row.get("z_std", 0.0)),
                "z_norm": float(row.get("z_norm", 0.0)),
                "slow_energy_scale": float(row.get("slow_energy_scale", 1.0)),
                "slow_compute_budget": float(row.get("slow_compute_budget", 0.0)),
                "slow_goal_entropy": float(row.get("slow_goal_entropy", 0.0)),
                "slow_tick": bool(row.get("slow_tick", False)),
                "policy_loss": row.get("policy_loss"),
                "value_loss": row.get("value_loss"),
                "entropy": row.get("entropy"),
                "total_loss": row.get("total_loss"),
                "warnings": "|".join(row.get("warning_flags", [])),
            }
        )
    return rows


def format_metrics(row):
    return (
        f"step={row['step']} agent={row['agent_id']} energy={row['energy']:.1f} "
        f"food={row['food_eaten']} coll/100={row['collisions_per_100_steps']:.1f} "
        f"fb={row['fallback_rate']:.2f} err_ema={row['error_ema']:.3f} "
        f"z_mean={row['z_mean']:.3f} z_std={row['z_std']:.3f} "
        f"slow_E={row['slow_energy_scale']:.2f} slow_C={row['slow_compute_budget']:.2f} "
        f"goal_H={row['slow_goal_entropy']:.2f} loss={fmt_loss(row['total_loss'])}"
    )


def write_metrics_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            for key in CSV_COLUMNS:
                value = serialized.get(key)
                if isinstance(value, float) and not np.isfinite(value):
                    serialized[key] = ""
            writer.writerow(serialized)


def build_final_summary(metrics, latent_summary, checkpoint_path, loaded_checkpoint, step):
    rows = metrics["per_agent"]
    return {
        "steps": int(step),
        "mean_energy": safe_mean([row.get("energy_sum", 0.0) / max(1, row.get("count", 0)) for row in rows]),
        "total_food_eaten": int(sum(row.get("food_eaten", 0) for row in rows)),
        "mean_fallback_rate": safe_mean([row.get("fallback_sum", 0.0) / max(1, row.get("count", 0)) for row in rows]),
        "mean_error_ema": safe_mean([row.get("error_ema", 0.0) for row in rows]),
        "mean_collisions_per_100_steps": safe_mean([row.get("collisions_per_100_steps", 0.0) for row in rows]),
        "checkpoint_path": checkpoint_path or "",
        "loaded_checkpoint": loaded_checkpoint or "",
        "latent_diagnostics": latent_summary,
    }


def entropy(probs):
    if probs is None:
        return 0.0
    probs = np.asarray(probs, dtype=np.float32)
    total = float(probs.sum())
    if total <= 0.0:
        return 0.0
    probs = probs / total
    return float(-(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum())


def np_mean_dict(values):
    return 0.0 if not values else float(np.mean(list(values.values())))


def safe_mean(values):
    values = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return 0.0 if not values else float(sum(values) / len(values))


def fmt_loss(value):
    return "n/a" if value is None or not np.isfinite(float(value)) else f"{float(value):.4f}"


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    return value


def build_runtime_args(overrides):
    defaults = vars(build_parser().parse_args([]))
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def main():
    run_session(parse_args())


if __name__ == "__main__":
    main()
