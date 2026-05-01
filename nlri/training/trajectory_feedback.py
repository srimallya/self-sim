from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List

import numpy as np


FEEDBACK_KEYS = [
    "leakage",
    "energy_slope",
    "food_delta",
    "collision_rate",
    "action_entropy",
    "position_novelty",
    "compute_budget",
    "z_variance",
    "steps_since_food",
    "useful_transition_score",
]


class TrajectoryFeedbackBuilder:
    """Builds dense, non-semantic feedback vectors from recent rollout windows."""

    def __init__(self, window: int = 100):
        self.window = int(window)
        self.windows: Dict[int, Deque[Dict[str, Any]]] = {}

    @property
    def dim(self):
        return len(FEEDBACK_KEYS)

    def add(self, agent_id: int, transition: Dict[str, Any]):
        if agent_id not in self.windows:
            self.windows[agent_id] = deque(maxlen=self.window)
        self.windows[agent_id].append(dict(transition))

    def vector(self, agent_id: int):
        rows = list(self.windows.get(agent_id, []))
        if not rows:
            return np.zeros(self.dim, dtype=np.float32)
        energies = np.asarray([row.get("energy", 0.0) for row in rows], dtype=np.float32)
        food = np.asarray([row.get("food_delta", 0.0) for row in rows], dtype=np.float32)
        z_std = np.asarray([row.get("z_std", 0.0) for row in rows], dtype=np.float32)
        values = {
            "leakage": np.mean([row.get("leakage", 0.0) for row in rows]),
            "energy_slope": (energies[-1] - energies[0]) / max(1.0, 1000.0 * len(energies)),
            "food_delta": np.sum(food) / max(1.0, len(rows)),
            "collision_rate": np.mean([row.get("collision", 0.0) for row in rows]),
            "action_entropy": np.mean([row.get("action_entropy", 0.0) for row in rows]) / 3.1,
            "position_novelty": rows[-1].get("position_novelty", 0.0),
            "compute_budget": np.mean([row.get("compute_budget", 0.0) for row in rows]),
            "z_variance": float(np.mean(z_std * z_std)),
            "steps_since_food": min(1.0, rows[-1].get("steps_since_food", 0) / max(1.0, self.window)),
            "useful_transition_score": np.mean([row.get("useful_transition_score", 0.0) for row in rows]) / 5.0,
        }
        vector = np.asarray([values[key] for key in FEEDBACK_KEYS], dtype=np.float32)
        return np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=-1.0)

    def score(self, agent_id: int):
        v = self.vector(agent_id)
        lookup = dict(zip(FEEDBACK_KEYS, v))
        score = (
            2.0 * lookup["food_delta"]
            + lookup["energy_slope"]
            + lookup["position_novelty"]
            + 0.5 * lookup["action_entropy"]
            + lookup["useful_transition_score"]
            - lookup["leakage"]
            - lookup["collision_rate"]
            - 0.5 * lookup["steps_since_food"]
        )
        return float(np.nan_to_num(score, nan=0.0))

    def snapshot(self, agent_id: int):
        return {"feedback_vector": self.vector(agent_id), "feedback_score": self.score(agent_id)}

