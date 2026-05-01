from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List

import numpy as np


FEEDBACK_KEYS = [
    "leakage",
    "energy_slope",
    "energy_delta",
    "food_delta",
    "collision_rate",
    "movement_cost_per_food",
    "food_per_collision",
    "local_loop_score",
    "wall_contact_rate",
    "action_entropy",
    "position_novelty",
    "compute_budget",
    "z_variance",
    "steps_since_food",
    "useful_transition_score",
]


class TrajectoryFeedbackBuilder:
    """Builds dense, non-semantic feedback vectors from recent rollout windows."""

    def __init__(
        self,
        window: int = 100,
        energy_delta_weight: float = 0.2,
        collision_weight: float = 0.5,
        movement_weight: float = 0.2,
        novelty_weight: float = 0.2,
        leakage_weight: float = 1.0,
    ):
        self.window = int(window)
        self.energy_delta_weight = float(energy_delta_weight)
        self.collision_weight = float(collision_weight)
        self.movement_weight = float(movement_weight)
        self.novelty_weight = float(novelty_weight)
        self.leakage_weight = float(leakage_weight)
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
        collisions = np.asarray([row.get("collision", 0.0) for row in rows], dtype=np.float32)
        movement = np.asarray([row.get("movement_cost", 0.0) for row in rows], dtype=np.float32)
        z_std = np.asarray([row.get("z_std", 0.0) for row in rows], dtype=np.float32)
        energy_delta = (energies[-1] - energies[0]) / 1000.0
        food_sum = float(np.sum(food))
        collision_sum = float(np.sum(collisions))
        movement_sum = float(np.sum(movement))
        values = {
            "leakage": np.mean([row.get("leakage", 0.0) for row in rows]),
            "energy_slope": energy_delta / max(1.0, len(energies)),
            "energy_delta": energy_delta,
            "food_delta": food_sum / max(1.0, len(rows)),
            "collision_rate": collision_sum / max(1.0, len(rows)),
            "movement_cost_per_food": movement_sum / max(1.0, food_sum),
            "food_per_collision": food_sum / max(1.0, collision_sum),
            "local_loop_score": rows[-1].get("local_loop_score", 0.0),
            "wall_contact_rate": collision_sum / max(1.0, len(rows)),
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
            lookup["food_delta"]
            + self.energy_delta_weight * lookup["energy_delta"]
            + self.novelty_weight * lookup["position_novelty"]
            + 0.5 * lookup["action_entropy"]
            + lookup["useful_transition_score"]
            - self.leakage_weight * lookup["leakage"]
            - self.collision_weight * lookup["collision_rate"]
            - self.movement_weight * lookup["movement_cost_per_food"]
            - 0.25 * lookup["local_loop_score"]
            - 0.5 * lookup["steps_since_food"]
        )
        return float(np.nan_to_num(score, nan=0.0))

    def snapshot(self, agent_id: int):
        return {"feedback_vector": self.vector(agent_id), "feedback_score": self.score(agent_id)}
