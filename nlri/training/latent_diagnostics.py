from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np


class LatentDiagnosticsCollector:
    def __init__(self):
        self.rows: List[Dict[str, float]] = []

    def add(
        self,
        z,
        compute_budget,
        energy,
        leakage,
        visible_food_value,
        reachable_food_value,
        collision_safety,
        selected_action,
        fallback_used,
        uncertainty=0.0,
        action_entropy=0.0,
    ):
        self.rows.append(
            {
                "z": np.asarray(z, dtype=np.float32),
                "compute_budget": float(compute_budget),
                "energy": float(energy),
                "leakage": float(leakage),
                "visible_food_value": float(visible_food_value),
                "reachable_food_value": float(reachable_food_value),
                "collision_safety": float(collision_safety),
                "selected_action": int(selected_action),
                "fallback_used": float(bool(fallback_used)),
                "uncertainty": float(uncertainty),
                "action_entropy": float(action_entropy),
            }
        )

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.rows:
            np.savez_compressed(path, z=np.zeros((0, 0), dtype=np.float32))
            return
        z = np.stack([row["z"] for row in self.rows], axis=0)
        arrays = {
            "z": z,
            "compute_budget": np.asarray([row["compute_budget"] for row in self.rows], dtype=np.float32),
            "energy": np.asarray([row["energy"] for row in self.rows], dtype=np.float32),
            "leakage": np.asarray([row["leakage"] for row in self.rows], dtype=np.float32),
            "visible_food_value": np.asarray([row["visible_food_value"] for row in self.rows], dtype=np.float32),
            "reachable_food_value": np.asarray([row["reachable_food_value"] for row in self.rows], dtype=np.float32),
            "collision_safety": np.asarray([row["collision_safety"] for row in self.rows], dtype=np.float32),
            "selected_action": np.asarray([row["selected_action"] for row in self.rows], dtype=np.int32),
            "fallback_used": np.asarray([row["fallback_used"] for row in self.rows], dtype=np.float32),
            "uncertainty": np.asarray([row["uncertainty"] for row in self.rows], dtype=np.float32),
            "action_entropy": np.asarray([row["action_entropy"] for row in self.rows], dtype=np.float32),
        }
        np.savez_compressed(path, **arrays)

    def summary(self):
        if not self.rows:
            return {
                "z_variance": 0.0,
                "z_collapse_score": 1.0,
                "z_leakage_correlation_mean_abs": 0.0,
                "compute_budget_leakage_correlation": 0.0,
                "compute_budget_uncertainty_correlation": 0.0,
                "action_entropy_by_cluster": {},
            }

        z = np.stack([row["z"] for row in self.rows], axis=0)
        leakage = np.asarray([row["leakage"] for row in self.rows], dtype=np.float32)
        compute_budget = np.asarray([row["compute_budget"] for row in self.rows], dtype=np.float32)
        uncertainty = np.asarray([row["uncertainty"] for row in self.rows], dtype=np.float32)
        action_entropy = np.asarray([row["action_entropy"] for row in self.rows], dtype=np.float32)
        z_var = np.var(z, axis=0)
        z_corrs = [self._corr(z[:, idx], leakage) for idx in range(z.shape[1])]
        clusters = np.argmax(np.abs(z[:, : min(4, z.shape[1])]), axis=1)
        entropy_by_cluster = {}
        for cluster_id in np.unique(clusters):
            mask = clusters == cluster_id
            entropy_by_cluster[str(int(cluster_id))] = float(np.mean(action_entropy[mask]))
        return {
            "z_variance": float(np.mean(z_var)),
            "z_collapse_score": float(np.mean(z_var < 1e-3)),
            "z_leakage_correlation_mean_abs": float(np.mean(np.abs(z_corrs))),
            "compute_budget_leakage_correlation": float(self._corr(compute_budget, leakage)),
            "compute_budget_uncertainty_correlation": float(self._corr(compute_budget, uncertainty)),
            "action_entropy_by_cluster": entropy_by_cluster,
        }

    def _corr(self, a, b):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        if a.size < 2 or float(np.std(a)) < 1e-8 or float(np.std(b)) < 1e-8:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])
