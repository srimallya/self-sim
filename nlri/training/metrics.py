from typing import Dict, List

import numpy as np


class RunningMetrics:
    def __init__(self):
        self.rows: List[Dict[str, float]] = []

    def update(self, info: Dict[str, object], agent_debug: List[Dict[str, np.ndarray]]):
        energies = [agent["energy"] for agent in info["agents"]]
        food_eaten = [agent["food_eaten"] for agent in info["agents"]]
        leakage = []
        for agent in info["agents"]:
            if agent["leakage"]:
                leakage.extend(agent["leakage"].values())

        z_values = [np.asarray(debug.get("z", np.zeros(1)), dtype=np.float32) for debug in agent_debug]
        compute_budgets = [float(np.asarray(debug.get("compute_budget", [0.0])).reshape(-1)[0]) for debug in agent_debug]
        entropies = []
        for debug in agent_debug:
            probs = debug.get("action_probs")
            if probs is None:
                entropies.append(0.0)
            else:
                probs = np.asarray(probs, dtype=np.float32)
                entropies.append(float(-(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum()))

        self.rows.append(
            {
                "average_agent_energy": float(np.mean(energies)),
                "food_eaten_per_100_steps": float(sum(food_eaten)),
                "movement_cost": float(np.mean(info["movement_costs"])),
                "collision_count": float(sum(info["collisions"])),
                "stationary_wait_count": float(sum(info["waits"])),
                "reservoir_leakage": float(np.mean(leakage) if leakage else 0.0),
                "latent_z_mean": float(np.mean([z.mean() for z in z_values])),
                "latent_z_std": float(np.mean([z.std() for z in z_values])),
                "compute_budget_mean": float(np.mean(compute_budgets)),
                "action_entropy": float(np.mean(entropies)),
            }
        )

    def latest(self):
        return self.rows[-1] if self.rows else {}

    def format_status(self, step: int):
        row = self.latest()
        if not row:
            return f"step={step}"
        return (
            f"step={step} energy={row['average_agent_energy']:.2f} "
            f"food={row['food_eaten_per_100_steps']:.0f} move_cost={row['movement_cost']:.2f} "
            f"collisions={row['collision_count']:.0f} waits={row['stationary_wait_count']:.0f} "
            f"leakage={row['reservoir_leakage']:.3f} z_mean={row['latent_z_mean']:.3f} "
            f"z_std={row['latent_z_std']:.3f} budget={row['compute_budget_mean']:.3f} "
            f"entropy={row['action_entropy']:.3f}"
        )
