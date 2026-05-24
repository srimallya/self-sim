from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn

from nlri.envs.maze_reservoir_env import ACTION_DIM
from .encoder import NLRIEncoder


RESERVOIR_KEYS = (
    "self_energy",
    "visible_food_value",
    "reachable_food_value",
    "collision_safety",
    "time_budget",
    "attention_budget",
)


@dataclass
class FastOutput:
    logits: torch.Tensor
    value: torch.Tensor
    pred_next_z: torch.Tensor
    pred_reservoir: torch.Tensor
    collision_logit: torch.Tensor
    movement_cost: torch.Tensor
    progress_logit: torch.Tensor
    energy_delta: torch.Tensor


@dataclass
class SlowOutput:
    modulation: torch.Tensor
    energy_scale: torch.Tensor
    compute_budget: torch.Tensor
    action_bias: torch.Tensor
    goal_logits: torch.Tensor
    next_thought: torch.Tensor


@dataclass
class DualSystemOutput:
    z: torch.Tensor
    z_diagnostics: Dict[str, float]
    fast: FastOutput
    slow: SlowOutput
    final_logits: torch.Tensor
    value: torch.Tensor


class SharedEncoder(nn.Module):
    """Shared sensory root read by both fast and slow systems."""

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.encoder = NLRIEncoder(hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.latent_dim = latent_dim

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        z = self.encoder(obs)
        return torch.nan_to_num(z, nan=0.0, posinf=5.0, neginf=-5.0).clamp(-10.0, 10.0)

    @staticmethod
    def diagnostics(z: torch.Tensor) -> Dict[str, float]:
        detached = z.detach()
        z_mean = float(detached.mean().cpu().item())
        z_std = float(detached.std(unbiased=False).cpu().item())
        z_norm = float(detached.norm(dim=1).mean().cpu().item())
        return {
            "z_mean": z_mean,
            "z_std": z_std,
            "z_norm": z_norm,
            "z_collapse_warning": float(z_std < 0.01 or z_norm < 1e-3),
        }


class FastSystem(nn.Module):
    """System 1: every-step motor controller and near-term predictor."""

    def __init__(self, z_dim: int = 64, modulation_dim: int = 16, action_dim: int = ACTION_DIM):
        super().__init__()
        self.z_dim = z_dim
        self.modulation_dim = modulation_dim
        self.action_dim = action_dim
        input_dim = z_dim + modulation_dim + action_dim
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)
        self.next_z_head = nn.Linear(128, z_dim)
        self.reservoir_head = nn.Sequential(nn.Linear(128, len(RESERVOIR_KEYS)), nn.Sigmoid())
        self.cleanliness_head = nn.Linear(128, 4)

    def forward(self, z: torch.Tensor, modulation: torch.Tensor, prev_action_onehot: torch.Tensor) -> FastOutput:
        hidden = self.trunk(torch.cat([z, modulation, prev_action_onehot], dim=1))
        clean = self.cleanliness_head(hidden)
        return FastOutput(
            logits=torch.nan_to_num(self.policy_head(hidden), nan=0.0).clamp(-20.0, 20.0),
            value=torch.nan_to_num(self.value_head(hidden), nan=0.0).clamp(-20.0, 20.0),
            pred_next_z=torch.nan_to_num(self.next_z_head(hidden), nan=0.0).clamp(-20.0, 20.0),
            pred_reservoir=self.reservoir_head(hidden),
            collision_logit=clean[:, 0:1],
            movement_cost=torch.nn.functional.softplus(clean[:, 1:2]),
            progress_logit=clean[:, 2:3],
            energy_delta=torch.nan_to_num(clean[:, 3:4], nan=0.0).clamp(-5.0, 5.0),
        )


class SlowSystem(nn.Module):
    """System 2: slower deliberative modulation over fast policy style."""

    def __init__(
        self,
        z_dim: int = 64,
        error_dim: int = 5,
        reservoir_dim: int = len(RESERVOIR_KEYS),
        history_dim: int = 4,
        modulation_dim: int = 16,
        action_dim: int = ACTION_DIM,
        thought_dim: int = 32,
        goal_dim: int = 4,
    ):
        super().__init__()
        self.modulation_dim = modulation_dim
        input_dim = z_dim + error_dim + reservoir_dim + history_dim
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.modulation_head = nn.Linear(128, modulation_dim)
        self.energy_head = nn.Linear(128, 1)
        self.compute_head = nn.Linear(128, 1)
        self.bias_head = nn.Linear(128, action_dim)
        self.goal_head = nn.Linear(128, goal_dim)
        self.thought_head = nn.Linear(128, thought_dim)

    def forward(
        self,
        z: torch.Tensor,
        error_summary: torch.Tensor,
        reservoir: torch.Tensor,
        history_summary: torch.Tensor,
    ) -> SlowOutput:
        hidden = self.trunk(torch.cat([z, error_summary, reservoir, history_summary], dim=1))
        energy_scale = 0.35 + 1.65 * torch.sigmoid(self.energy_head(hidden))
        compute_budget = torch.sigmoid(self.compute_head(hidden))
        return SlowOutput(
            modulation=torch.tanh(self.modulation_head(hidden)),
            energy_scale=energy_scale,
            compute_budget=compute_budget,
            action_bias=torch.tanh(self.bias_head(hidden)) * 1.5,
            goal_logits=torch.nan_to_num(self.goal_head(hidden), nan=0.0).clamp(-20.0, 20.0),
            next_thought=torch.tanh(self.thought_head(hidden)),
        )


class DualSystemAgent(nn.Module):
    def __init__(
        self,
        action_dim: int = ACTION_DIM,
        z_dim: int = 64,
        modulation_dim: int = 16,
        device: Optional[str] = None,
        use_legacy_fallback: bool = True,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.z_dim = z_dim
        self.modulation_dim = modulation_dim
        self.use_legacy_fallback = use_legacy_fallback
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.obs_normalizer = None
        self.shared_encoder = SharedEncoder(latent_dim=z_dim)
        self.fast_system = FastSystem(z_dim=z_dim, modulation_dim=modulation_dim, action_dim=action_dim)
        self.slow_system = SlowSystem(z_dim=z_dim, modulation_dim=modulation_dim, action_dim=action_dim)

        self.register_buffer("last_modulation", torch.zeros(1, modulation_dim))
        self.register_buffer("last_energy_scale", torch.ones(1, 1))
        self.register_buffer("last_compute_budget", torch.full((1, 1), 0.5))
        self.register_buffer("last_action_bias", torch.zeros(1, action_dim))
        self.register_buffer("last_goal_logits", torch.zeros(1, 4))
        self.register_buffer("last_thought", torch.zeros(1, 32))
        self.last_debug: Dict[str, object] = {}
        self.to(self.device)

    def reset_state(self, batch_size: int = 1):
        self.last_modulation = torch.zeros(batch_size, self.modulation_dim, device=self.device)
        self.last_energy_scale = torch.ones(batch_size, 1, device=self.device)
        self.last_compute_budget = torch.full((batch_size, 1), 0.5, device=self.device)
        self.last_action_bias = torch.zeros(batch_size, self.action_dim, device=self.device)
        self.last_goal_logits = torch.zeros(batch_size, 4, device=self.device)
        self.last_thought = torch.zeros(batch_size, 32, device=self.device)

    def forward_tensors(
        self,
        obs_t: Dict[str, torch.Tensor],
        error_summary: Optional[torch.Tensor] = None,
        reservoir: Optional[torch.Tensor] = None,
        history_summary: Optional[torch.Tensor] = None,
        force_slow_tick: bool = False,
        detach_slow_state: bool = False,
    ) -> DualSystemOutput:
        z = self.shared_encoder(obs_t)
        batch_size = z.shape[0]
        if error_summary is None:
            error_summary = torch.zeros(batch_size, 5, device=self.device)
        if reservoir is None:
            reservoir = torch.zeros(batch_size, len(RESERVOIR_KEYS), device=self.device)
        if history_summary is None:
            history_summary = torch.zeros(batch_size, 4, device=self.device)

        if force_slow_tick or self.last_modulation.shape[0] != batch_size:
            slow = self.slow_system(z, error_summary, reservoir, history_summary)
            if batch_size == 1:
                stored = slow
                if detach_slow_state:
                    stored = SlowOutput(
                        modulation=slow.modulation.detach(),
                        energy_scale=slow.energy_scale.detach(),
                        compute_budget=slow.compute_budget.detach(),
                        action_bias=slow.action_bias.detach(),
                        goal_logits=slow.goal_logits.detach(),
                        next_thought=slow.next_thought.detach(),
                    )
                self.last_modulation = stored.modulation
                self.last_energy_scale = stored.energy_scale
                self.last_compute_budget = stored.compute_budget
                self.last_action_bias = stored.action_bias
                self.last_goal_logits = stored.goal_logits
                self.last_thought = stored.next_thought
        else:
            slow = SlowOutput(
                modulation=self._expand(self.last_modulation, batch_size),
                energy_scale=self._expand(self.last_energy_scale, batch_size),
                compute_budget=self._expand(self.last_compute_budget, batch_size),
                action_bias=self._expand(self.last_action_bias, batch_size),
                goal_logits=self._expand(self.last_goal_logits, batch_size),
                next_thought=self._expand(self.last_thought, batch_size),
            )

        prev_action = self._last_action_onehot(obs_t["last_action"])
        fast = self.fast_system(z, slow.modulation, prev_action)
        final_logits = torch.nan_to_num(fast.logits * slow.energy_scale + slow.action_bias, nan=0.0).clamp(-20.0, 20.0)
        return DualSystemOutput(
            z=z,
            z_diagnostics=SharedEncoder.diagnostics(z),
            fast=fast,
            slow=slow,
            final_logits=final_logits,
            value=fast.value,
        )

    @torch.no_grad()
    def act(
        self,
        obs: Dict[str, np.ndarray],
        error_summary: Optional[np.ndarray] = None,
        reservoir: Optional[Dict[str, float]] = None,
        history_summary: Optional[np.ndarray] = None,
        slow_tick: bool = False,
        deterministic: bool = False,
        fallback_probability: float = 1.0,
        force_no_fallback: bool = False,
        temperature: float = 1.0,
    ):
        obs_t = self._tensorize_obs(obs)
        error_t = self._tensorize_vector(error_summary, 5)
        reservoir_t = self._tensorize_reservoir(reservoir)
        history_t = self._tensorize_vector(history_summary, 4)
        output = self.forward_tensors(
            obs_t,
            error_summary=error_t,
            reservoir=reservoir_t,
            history_summary=history_t,
            force_slow_tick=slow_tick,
            detach_slow_state=True,
        )
        logits = output.final_logits[0]
        temp = max(float(temperature), 1e-3)
        probs = torch.softmax(logits / temp, dim=0)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        prob_sum = probs.sum()
        invalid = bool((not torch.isfinite(prob_sum)) or prob_sum.item() <= 0.0)
        if invalid:
            probs = torch.full_like(probs, 1.0 / probs.numel())
        else:
            probs = probs / prob_sum

        fallback_action = self._fallback_action(obs)
        fallback_used = False
        if invalid:
            action = fallback_action
            fallback_used = True
        elif deterministic:
            action = int(torch.argmax(probs).item())
        else:
            action = int(torch.distributions.Categorical(probs=probs).sample().item())

        if self.use_legacy_fallback and not force_no_fallback and np.random.random() < fallback_probability:
            action = fallback_action
            fallback_used = True

        goal_probs = torch.softmax(output.slow.goal_logits[0], dim=0)
        self.last_debug = {
            "z": output.z[0].detach().cpu().numpy(),
            "z_mean": output.z_diagnostics["z_mean"],
            "z_std": output.z_diagnostics["z_std"],
            "z_norm": output.z_diagnostics["z_norm"],
            "z_collapse_warning": bool(output.z_diagnostics["z_collapse_warning"]),
            "action_probs": probs.detach().cpu().numpy(),
            "selected_action": action,
            "fallback_action": fallback_action,
            "fallback_used": fallback_used,
            "slow_energy_scale": float(output.slow.energy_scale[0, 0].detach().cpu().item()),
            "slow_compute_budget": float(output.slow.compute_budget[0, 0].detach().cpu().item()),
            "slow_goal_logits": output.slow.goal_logits[0].detach().cpu().numpy(),
            "slow_goal_probs": goal_probs.detach().cpu().numpy(),
            "value": float(output.value[0, 0].detach().cpu().item()),
            "pred_next_z": output.fast.pred_next_z[0].detach().cpu().numpy(),
            "pred_reservoir": output.fast.pred_reservoir[0].detach().cpu().numpy(),
            "pred_collision_prob": float(torch.sigmoid(output.fast.collision_logit[0, 0]).detach().cpu().item()),
            "pred_movement_cost": float(output.fast.movement_cost[0, 0].detach().cpu().item()),
            "pred_progress_prob": float(torch.sigmoid(output.fast.progress_logit[0, 0]).detach().cpu().item()),
            "pred_energy_delta": float(output.fast.energy_delta[0, 0].detach().cpu().item()),
            "nan_action_recovery": invalid,
        }
        return action, self.last_debug

    def _tensorize_obs(self, obs: Dict[str, np.ndarray], normalize: bool = True):
        obs_t: Dict[str, torch.Tensor] = {}
        source_obs = self.obs_normalizer.normalize_obs(obs) if normalize and self.obs_normalizer is not None else obs
        for key, value in source_obs.items():
            arr = np.asarray(value, dtype=np.float32)
            if key in {"perception_cone", "shared_signal"} and arr.ndim == 2:
                arr = arr[None, :, :]
            elif key not in {"perception_cone", "shared_signal"} and arr.ndim == 1:
                arr = arr[None, :]
            obs_t[key] = torch.from_numpy(np.nan_to_num(arr, nan=0.0)).to(self.device)
        return obs_t

    def _tensorize_vector(self, value, dim: int):
        if value is None:
            arr = np.zeros(dim, dtype=np.float32)
        else:
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.size < dim:
                arr = np.pad(arr, (0, dim - arr.size))
            elif arr.size > dim:
                arr = arr[:dim]
        return torch.from_numpy(np.nan_to_num(arr[None, :], nan=0.0)).to(self.device)

    def _tensorize_reservoir(self, reservoir: Optional[Dict[str, float]]):
        row = [float((reservoir or {}).get(key, 0.0)) for key in RESERVOIR_KEYS]
        return torch.tensor([row], dtype=torch.float32, device=self.device)

    def _last_action_onehot(self, last_action: torch.Tensor):
        idx = torch.round(last_action.reshape(-1) * float(self.action_dim - 1)).long()
        idx = torch.clamp(idx, 0, self.action_dim - 1)
        return torch.nn.functional.one_hot(idx, num_classes=self.action_dim).to(dtype=torch.float32)

    def _fallback_action(self, obs: Dict[str, np.ndarray]):
        gradient = np.asarray(obs.get("raw_energy_gradient", []), dtype=np.float32)
        if gradient.shape[0] == 360 and float(np.max(gradient)) > 0.0:
            best_angle = int(np.argmax(gradient))
            return min(19, int(round(best_angle / 18.0)) % 20)
        return int(np.random.randint(0, self.action_dim))

    @staticmethod
    def _expand(tensor: torch.Tensor, batch_size: int):
        if tensor.shape[0] == batch_size:
            return tensor
        if tensor.shape[0] == 1:
            return tensor.expand(batch_size, -1)
        return tensor[:batch_size]
