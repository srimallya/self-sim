from typing import Dict, Optional

import numpy as np
import torch
from torch import nn

from nlri.envs.maze_reservoir_env import ACTION_DIM
from .encoder import NLRIEncoder
from .feedback_encoder import FeedbackEncoder
from .latent_router import LatentRouter
from .policy import PolicyHead, ValueHead
from .reservoir_model import ReservoirModel
from .world_model import WorldModel
from nlri.training.trajectory_feedback import FEEDBACK_KEYS


class NLRIAgent(nn.Module):
    def __init__(
        self,
        action_dim: int = ACTION_DIM,
        belief_dim: int = 64,
        z_dim: int = 32,
        use_legacy_fallback: bool = True,
        ablation_mode: str = "full",
        device: Optional[str] = None,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.belief_dim = belief_dim
        self.z_dim = z_dim
        self.use_legacy_fallback = use_legacy_fallback
        self.ablation_mode = ablation_mode
        self.compute_floor = 0.0
        self.temperature_jitter = 0.0
        self.z_noise = 0.0
        self.obs_normalizer = None
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.encoder = NLRIEncoder(latent_dim=belief_dim)
        self.feedback_encoder = FeedbackEncoder(feedback_dim=len(FEEDBACK_KEYS), context_dim=belief_dim)
        self.world_model = WorldModel(latent_dim=belief_dim, belief_dim=belief_dim)
        self.reservoir_model = ReservoirModel(belief_dim=belief_dim)
        self.latent_router = LatentRouter(belief_dim=belief_dim, z_dim=z_dim)
        self.policy = PolicyHead(belief_dim=belief_dim, z_dim=z_dim, action_dim=action_dim)
        self.value_head = ValueHead(belief_dim=belief_dim, z_dim=z_dim)
        self.planner = None

        self.register_buffer("belief", torch.zeros(1, belief_dim))
        self.last_debug: Dict[str, np.ndarray] = {}
        self.to(self.device)

    def reset_state(self, batch_size: int = 1):
        self.belief = torch.zeros(batch_size, self.belief_dim, device=self.device)

    def forward(self, obs: Dict[str, np.ndarray]):
        obs_t = self._tensorize_obs(obs)
        return self.forward_tensors(obs_t)

    def forward_tensors(self, obs_t: Dict[str, torch.Tensor], feedback_context=None, demo_context=None):
        encoded = self.encoder(obs_t)
        if self.belief.shape[0] != encoded.shape[0]:
            self.reset_state(encoded.shape[0])
        next_belief, obs_embedding, uncertainty = self.world_model(encoded, self.belief)
        next_belief = self._apply_feedback_context(next_belief, feedback_context, demo_context)
        self.belief = next_belief.detach()

        if self.ablation_mode == "no-router":
            z = torch.zeros(next_belief.shape[0], self.z_dim, device=self.device)
            compute_budget = torch.full((next_belief.shape[0], 1), 0.5, device=self.device)
        else:
            z, compute_budget = self.latent_router(next_belief, compute_floor=self.compute_floor)
        z = torch.nan_to_num(z, nan=0.0, posinf=1.0, neginf=-1.0)
        if float(getattr(self, "z_noise", 0.0)) > 0.0 and self.training:
            z = z + torch.randn_like(z) * float(self.z_noise)
            z = torch.nan_to_num(z, nan=0.0, posinf=1.0, neginf=-1.0)
        compute_budget = torch.nan_to_num(
            compute_budget,
            nan=float(self.compute_floor),
            posinf=1.0,
            neginf=float(self.compute_floor),
        ).clamp(float(self.compute_floor), 1.0)

        reservoir_next, reservoir_star, leakage = self.reservoir_model(next_belief)
        logits = torch.nan_to_num(self.policy(next_belief, z, compute_budget), nan=0.0, posinf=20.0, neginf=-20.0)
        logits = torch.clamp(logits, -20.0, 20.0)
        value = torch.nan_to_num(self.value_head(next_belief, z, compute_budget), nan=0.0, posinf=10.0, neginf=-10.0)
        return {
            "encoded": encoded,
            "belief": next_belief,
            "obs_embedding": obs_embedding,
            "uncertainty": uncertainty,
            "z": z,
            "compute_budget": compute_budget,
            "reservoir_next": reservoir_next,
            "reservoir_star": reservoir_star,
            "leakage": leakage,
            "logits": logits,
            "value": value,
        }

    @torch.no_grad()
    def teacher_forward(self, obs_t: Dict[str, torch.Tensor], feedback_context=None, demo_context=None):
        prior_belief = self.belief.detach().clone()
        outputs = self.forward_tensors(obs_t, feedback_context=feedback_context, demo_context=demo_context)
        self.belief = prior_belief
        return {key: value.detach() if torch.is_tensor(value) else value for key, value in outputs.items()}

    @torch.no_grad()
    def act(
        self,
        obs: Dict[str, np.ndarray],
        deterministic: bool = False,
        fallback_probability: float = 1.0,
        force_no_fallback: bool = False,
        temperature: float = 1.0,
    ):
        outputs = self.forward(obs)
        logits = outputs["logits"][0]
        temp = max(float(temperature), 1e-3)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        probs = torch.softmax(logits / temp, dim=0)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        prob_sum = probs.sum()
        invalid = bool((not torch.isfinite(prob_sum)) or prob_sum.item() <= 0.0)
        if invalid:
            probs = torch.full_like(probs, 1.0 / probs.numel())
        else:
            probs = probs / prob_sum
        fallback_used = False
        nan_action_recovery = invalid

        if self.ablation_mode == "legacy-fallback-only":
            action = self._fallback_action(obs)
            fallback_used = True
        elif self.ablation_mode == "random-policy":
            action = int(np.random.randint(0, self.action_dim))
            probs = torch.full_like(probs, 1.0 / probs.numel())
        elif invalid:
            action = self._fallback_action(obs)
            fallback_used = True
        elif deterministic:
            action = int(torch.argmax(probs).item())
        else:
            distribution = torch.distributions.Categorical(probs=probs)
            action = int(distribution.sample().item())

        fallback_action = self._fallback_action(obs)
        if (
            self.ablation_mode not in {"random-policy", "legacy-fallback-only"}
            and self.use_legacy_fallback
            and not force_no_fallback
            and np.random.random() < fallback_probability
        ):
            action = fallback_action
            fallback_used = True

        self.last_debug = {
            "z": outputs["z"][0].detach().cpu().numpy(),
            "compute_budget": outputs["compute_budget"][0].detach().cpu().numpy(),
            "action_probs": probs.detach().cpu().numpy(),
            "uncertainty": outputs["uncertainty"][0].detach().cpu().numpy(),
            "value": outputs["value"][0].detach().cpu().numpy(),
            "fallback_used": fallback_used,
            "nan_action_recovery": nan_action_recovery,
            "selected_action": action,
            "fallback_action": fallback_action,
        }
        return action, self.last_debug

    def _should_apply_legacy_fallback(self, obs: Dict[str, np.ndarray], probs: torch.Tensor):
        gradient = np.asarray(obs["raw_energy_gradient"])
        has_signal = float(np.max(gradient)) > 0.0
        entropy = -torch.sum(probs * torch.log(probs.clamp_min(1e-8))).item()
        return has_signal and entropy > 2.8

    def _fallback_action(self, obs: Dict[str, np.ndarray]):
        gradient = np.asarray(obs["raw_energy_gradient"])
        if gradient.shape[0] == 360 and float(np.max(gradient)) > 0.0:
            best_angle = int(np.argmax(gradient))
            return min(19, int(round(best_angle / 18.0)) % 20)
        return int(np.random.randint(0, self.action_dim))

    def _tensorize_obs(self, obs: Dict[str, np.ndarray], normalize: bool = True):
        obs_t: Dict[str, torch.Tensor] = {}
        source_obs = self.obs_normalizer.normalize_obs(obs) if normalize and self.obs_normalizer is not None else obs
        for key, value in source_obs.items():
            arr = np.asarray(value, dtype=np.float32)
            if key in {"perception_cone", "shared_signal"} and arr.ndim == 2:
                arr = arr[None, :, :]
            elif key not in {"perception_cone", "shared_signal"} and arr.ndim == 1:
                arr = arr[None, :]
            obs_t[key] = torch.from_numpy(arr).to(self.device)
        return obs_t

    def _apply_feedback_context(self, belief, feedback_context=None, demo_context=None):
        contexts = []
        for context in (feedback_context, demo_context):
            if context is None:
                continue
            if not torch.is_tensor(context):
                context = torch.as_tensor(context, dtype=torch.float32, device=self.device)
            context = context.to(self.device, dtype=torch.float32)
            if context.ndim == 1:
                context = context.unsqueeze(0)
            if context.shape[0] == 1 and belief.shape[0] > 1:
                context = context.expand(belief.shape[0], -1)
            contexts.append(self.feedback_encoder(context))
        if not contexts:
            return belief
        feedback = torch.stack(contexts, dim=0).mean(dim=0)
        return belief + 0.1 * feedback
