from __future__ import annotations

from typing import Dict

import numpy as np
import torch


class RunningStat:
    def __init__(self, eps: float = 1e-4):
        self.count = eps
        self.mean = None
        self.m2 = None

    def update(self, values):
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0:
            return
        flat = array.reshape(-1, array.shape[-1]) if array.ndim > 1 else array.reshape(-1, 1)
        batch_count = flat.shape[0]
        batch_mean = flat.mean(axis=0)
        batch_var = flat.var(axis=0)
        if self.mean is None:
            self.mean = batch_mean
            self.m2 = batch_var * batch_count
            self.count = float(batch_count)
            return

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total_count
        self.m2 = self.m2 + batch_var * batch_count + (delta**2) * self.count * batch_count / total_count
        self.count = float(total_count)

    def std(self):
        if self.mean is None:
            return None
        variance = np.maximum(self.m2 / max(self.count, 1.0), 1e-6)
        return np.sqrt(variance)

    def state_dict(self):
        return {"count": self.count, "mean": self.mean, "m2": self.m2}

    def load_state_dict(self, state):
        self.count = float(state.get("count", 1e-4))
        self.mean = state.get("mean")
        self.m2 = state.get("m2")


class MultiNormalizer:
    def __init__(self, clamp: float = 10.0):
        self.clamp = clamp
        self.stats: Dict[str, RunningStat] = {}

    def update(self, name: str, values):
        stat = self.stats.setdefault(name, RunningStat())
        stat.update(values)

    def normalize_array(self, name: str, values):
        array = np.asarray(values, dtype=np.float32)
        stat = self.stats.get(name)
        if stat is None or stat.mean is None:
            return np.clip(array, -self.clamp, self.clamp)
        mean = np.asarray(stat.mean, dtype=np.float32)
        std = np.asarray(stat.std(), dtype=np.float32)
        normalized = (array - mean) / np.maximum(std, 1e-6)
        return np.clip(normalized, -self.clamp, self.clamp)

    def normalize_tensor(self, name: str, tensor: torch.Tensor):
        stat = self.stats.get(name)
        if stat is None or stat.mean is None:
            return torch.clamp(tensor, -self.clamp, self.clamp)
        mean = torch.as_tensor(stat.mean, dtype=tensor.dtype, device=tensor.device)
        std = torch.as_tensor(stat.std(), dtype=tensor.dtype, device=tensor.device)
        normalized = (tensor - mean) / torch.clamp(std, min=1e-6)
        return torch.clamp(normalized, -self.clamp, self.clamp)

    def normalize_obs(self, obs: Dict[str, np.ndarray]):
        return {
            "perception_cone": self.normalize_array("perception_cone", obs["perception_cone"]),
            "raw_energy_gradient": self.normalize_array("raw_energy_gradient", obs["raw_energy_gradient"]),
            "energy": self.normalize_array("energy", obs["energy"]),
            "angle": self.normalize_array("angle", obs["angle"]),
            "last_movement": self.normalize_array("last_movement", obs["last_movement"]),
            "last_action": self.normalize_array("last_action", obs["last_action"]),
            "shared_signal": self.normalize_array("shared_signal", obs["shared_signal"]),
        }

    def state_dict(self):
        return {
            "clamp": self.clamp,
            "stats": {name: stat.state_dict() for name, stat in self.stats.items()},
        }

    def load_state_dict(self, state):
        self.clamp = float(state.get("clamp", 10.0))
        self.stats = {}
        for name, stat_state in state.get("stats", {}).items():
            stat = RunningStat()
            stat.load_state_dict(stat_state)
            self.stats[name] = stat
