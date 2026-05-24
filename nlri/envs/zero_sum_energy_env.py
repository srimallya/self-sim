from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .maze_reservoir_env import (
    ACTION_DIM,
    CONSTANT_FOOD_COUNT,
    ENERGY_GAIN_FOOD,
    ENERGY_LOSS_STATIONARY,
    MAX_ENERGY,
    MazeReservoirEnv,
)


class ZeroSumEnergyEnv(MazeReservoirEnv):
    """Competitive two-agent energy duel on the existing NLRI maze.

    The visual mechanics stay close to MazeReservoirEnv, but food is a contested
    transfer: when one agent captures energy, the opponent loses the same amount.
    Rewards are centered to be strictly zero-sum each step.
    """

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_steps: int = 5000,
        food_count: int = 18,
        food_energy_gain: float = ENERGY_GAIN_FOOD,
        steal_on_contact: float = 6.0,
        scarcity_pressure: float = 0.0,
    ):
        super().__init__(render_mode=render_mode, max_steps=max_steps)
        self.target_food_count = max(1, int(food_count))
        self.food_energy_gain = max(0.0, float(food_energy_gain))
        self.steal_on_contact = max(0.0, float(steal_on_contact))
        self.scarcity_pressure = max(0.0, float(scarcity_pressure))
        self.capture_counts: List[int] = []
        self.steal_counts: List[int] = []
        self.energy_transferred: List[float] = []
        self.lead_changes = 0
        self.previous_leader = None

    def reset(self, seed: Optional[int] = None):
        observations, info = super().reset(seed=seed)
        if len(self.food_positions) > self.target_food_count:
            keep = list(self.food_positions.items())[: self.target_food_count]
            self.food_positions = dict(keep)
        self.capture_counts = [0 for _ in self.agents]
        self.steal_counts = [0 for _ in self.agents]
        self.energy_transferred = [0.0 for _ in self.agents]
        self.lead_changes = 0
        self.previous_leader = self._leader_id()
        info = self._build_info()
        self.last_info = info
        return observations, info

    def step(self, actions: Sequence[int]):
        if len(actions) != len(self.agents):
            raise ValueError(f"Expected {len(self.agents)} actions, got {len(actions)}")

        self.step_count += 1
        raw_rewards = []
        movement_costs = []
        collisions = []
        waits = []
        pre_energies = [float(agent.energy) for agent in self.agents]
        capture_events = [0 for _ in self.agents]
        steal_events = [0 for _ in self.agents]

        for agent, action in zip(self.agents, actions):
            action = int(action)
            prev_energy = float(agent.energy)
            prev_pos = agent.pos.copy()
            agent.last_action = action

            observation = self._compute_observation(agent)
            agent.perception = observation["perception_cone"]
            agent.raw_energy_gradient = observation["raw_energy_gradient"]
            agent.received_signal = observation["shared_signal"]
            agent.reservoir = self._compute_reservoir(agent)

            energy_cost, collision = self._apply_action(agent, action)
            reward = -energy_cost
            ate_food = False

            cell = tuple(agent.pos.tolist())
            if cell in self.food_positions:
                transfer = self._transfer_from_opponents(agent.agent_id, self.food_energy_gain)
                agent.energy = min(MAX_ENERGY, agent.energy + transfer)
                del self.food_positions[cell]
                agent.food_eaten += 1
                self.capture_counts[agent.agent_id] += 1
                self.energy_transferred[agent.agent_id] += transfer
                capture_events[agent.agent_id] += 1
                reward += transfer
                ate_food = True

            contact_transfer = self._contact_steal(agent.agent_id)
            if contact_transfer > 0.0:
                reward += contact_transfer
                self.steal_counts[agent.agent_id] += 1
                self.energy_transferred[agent.agent_id] += contact_transfer
                steal_events[agent.agent_id] += 1

            survival_tax = self.scarcity_pressure * max(0.0, 1.0 - len(self.food_positions) / float(self.target_food_count))
            total_cost = energy_cost + survival_tax
            agent.energy = max(0.0, agent.energy - total_cost)
            reward += 0.2 * (agent.energy - prev_energy)

            agent.last_movement = (agent.pos - prev_pos).astype(np.float32)
            agent.last_reward = float(reward)
            agent.movement_cost = float(total_cost)
            if collision:
                agent.collision_count += 1
            if action == ACTION_DIM - 1:
                agent.wait_count += 1

            agent.reservoir_next = self._compute_reservoir(agent)
            agent.reservoir_star = self._compute_reservoir_star(agent, observation, collision, ate_food)
            agent.leakage = self._compute_leakage(agent.reservoir_next, agent.reservoir_star)

            raw_rewards.append(float(reward))
            movement_costs.append(float(total_cost))
            collisions.append(bool(collision))
            waits.append(bool(action == ACTION_DIM - 1))

        self._add_food(self.target_food_count - len(self.food_positions))
        rewards = self._zero_sum_rewards(raw_rewards, pre_energies)
        leader = self._leader_id()
        if self.previous_leader is not None and leader is not None and leader != self.previous_leader:
            self.lead_changes += 1
        self.previous_leader = leader

        observations = self._collect_observations()
        terminated = any(agent.energy <= 0 for agent in self.agents)
        truncated = self.step_count >= self.max_steps
        info = self._build_info(
            rewards=list(rewards),
            movement_costs=movement_costs,
            collisions=collisions,
            waits=waits,
        )
        post_energies = [float(agent.energy) for agent in self.agents]
        info["zero_sum"] = {
            "raw_rewards": raw_rewards,
            "energy_gap": float(post_energies[0] - post_energies[1]) if len(post_energies) >= 2 else 0.0,
            "energy_gap_abs": float(abs(post_energies[0] - post_energies[1])) if len(post_energies) >= 2 else 0.0,
            "total_agent_energy": float(sum(post_energies)),
            "reward_sum": float(np.sum(rewards)),
            "capture_events": capture_events,
            "steal_events": steal_events,
            "capture_counts": list(self.capture_counts),
            "steal_counts": list(self.steal_counts),
            "energy_transferred": list(self.energy_transferred),
            "leader": leader,
            "lead_changes": int(self.lead_changes),
            "food_count": len(self.food_positions),
            "starvation_pressure": float(self.scarcity_pressure),
        }
        self.last_info = info

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminated, truncated, info

    def _add_food(self, count: int):
        target_count = self.target_food_count if hasattr(self, "target_food_count") else CONSTANT_FOOD_COUNT
        super()._add_food(min(max(0, int(count)), max(0, target_count - len(self.food_positions))))

    def _transfer_from_opponents(self, winner_id: int, amount: float):
        opponents = [agent for agent in self.agents if agent.agent_id != winner_id]
        if not opponents:
            return 0.0
        per_opponent = float(amount) / len(opponents)
        transferred = 0.0
        for opponent in opponents:
            taken = min(float(opponent.energy), per_opponent)
            opponent.energy = max(0.0, opponent.energy - taken)
            transferred += taken
        return transferred

    def _contact_steal(self, winner_id: int):
        if self.steal_on_contact <= 0.0:
            return 0.0
        winner = self.agents[winner_id]
        total = 0.0
        for opponent in self.agents:
            if opponent.agent_id == winner_id:
                continue
            distance = float(np.abs(winner.pos - opponent.pos).sum())
            if distance > 1.0:
                continue
            taken = min(float(opponent.energy), self.steal_on_contact)
            opponent.energy = max(0.0, opponent.energy - taken)
            winner.energy = min(MAX_ENERGY, winner.energy + taken)
            total += taken
        return total

    def _zero_sum_rewards(self, raw_rewards: Sequence[float], pre_energies: Sequence[float]):
        post_energies = [float(agent.energy) for agent in self.agents]
        energy_deltas = np.asarray(post_energies, dtype=np.float32) - np.asarray(pre_energies, dtype=np.float32)
        shaped = np.asarray(raw_rewards, dtype=np.float32) + energy_deltas
        centered = shaped - float(np.mean(shaped))
        if len(centered) == 2:
            advantage = 0.5 * float(centered[0] - centered[1])
            return np.asarray([advantage, -advantage], dtype=np.float32)
        return centered.astype(np.float32)

    def _leader_id(self):
        if not self.agents:
            return None
        energies = [float(agent.energy) for agent in self.agents]
        if len(energies) >= 2 and abs(energies[0] - energies[1]) < 1e-6:
            return None
        return int(np.argmax(energies))

