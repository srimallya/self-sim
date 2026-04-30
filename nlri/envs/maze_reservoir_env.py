import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


GRID_SIZE = 22
CELL_SIZE = 20
SCREEN_SIZE = GRID_SIZE * CELL_SIZE
N_AGENTS = 2
COLORS = ["cyan", "yellow"]
CONSTANT_FOOD_COUNT = 50
FOOD_ENERGY_RANGE = (100, 200)
MAX_ENERGY = 1000
ENERGY_LOSS_MOVE = 1.0
ENERGY_LOSS_STATIONARY = 0.1
ENERGY_GAIN_FOOD = 20
PERCEPTION_WINDOW = 180
PERCEPTION_RANGE = 8
COLLISION_PENALTY = 1.0
ACTION_DIM = 21
ROTATION_SPEED = 60
MAX_LOCOMOTION_STEPS = 10
MIN_OBSERVATION_STEPS = 1
DEFAULT_MAX_STEPS = 5_000

# Preserved directly from the legacy realtime simulator.
MAZE = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1],
    [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
], dtype=np.int8)


@dataclass
class AgentState:
    agent_id: int
    color: str
    pos: np.ndarray
    angle: float
    energy: float = MAX_ENERGY / 2
    locomotion_steps: int = 0
    observation_steps: int = 0
    last_action: int = ACTION_DIM - 1
    last_movement: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    perception: np.ndarray = field(default_factory=lambda: np.zeros((PERCEPTION_RANGE, PERCEPTION_WINDOW), dtype=np.float32))
    raw_energy_gradient: np.ndarray = field(default_factory=lambda: np.zeros(360, dtype=np.float32))
    received_signal: np.ndarray = field(default_factory=lambda: np.zeros((PERCEPTION_RANGE, PERCEPTION_WINDOW), dtype=np.float32))
    collision_count: int = 0
    wait_count: int = 0
    movement_cost: float = 0.0
    food_eaten: int = 0
    last_reward: float = 0.0
    reservoir: Optional[Dict[str, float]] = None
    reservoir_next: Optional[Dict[str, float]] = None
    reservoir_star: Optional[Dict[str, float]] = None
    leakage: Optional[Dict[str, float]] = None


class MazeReservoirEnv:
    """Realtime maze environment preserving the legacy pygame simulator mechanics."""

    metadata = {"render_modes": ["human", None], "fps": 24}

    def __init__(self, render_mode: Optional[str] = None, max_steps: int = DEFAULT_MAX_STEPS):
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.viewer = None
        self.rng = random.Random()
        self.np_rng = np.random.default_rng()
        self.step_count = 0
        self.agents: List[AgentState] = []
        self.food_positions: Dict[Tuple[int, int], int] = {}
        self.last_info: Dict[str, object] = {}
        if render_mode == "human":
            from nlri.viz.pygame_viewer import PygameViewer

            self.viewer = PygameViewer(self)

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng.seed(seed)
            self.np_rng = np.random.default_rng(seed)
        self.step_count = 0
        self.food_positions = {}
        self.agents = []

        occupied = set()
        for agent_id, color in enumerate(COLORS[:N_AGENTS]):
            while True:
                x = self.rng.randint(0, GRID_SIZE - 1)
                y = self.rng.randint(0, GRID_SIZE - 1)
                if MAZE[y, x] == 0 and (x, y) not in occupied:
                    occupied.add((x, y))
                    self.agents.append(
                        AgentState(
                            agent_id=agent_id,
                            color=color,
                            pos=np.array([x, y], dtype=np.int32),
                            angle=self.rng.uniform(0, 360),
                        )
                    )
                    break

        self._add_food(CONSTANT_FOOD_COUNT)
        observations = self._collect_observations()
        info = self._build_info()
        self.last_info = info
        if self.render_mode == "human":
            self.render()
        return observations, info

    def step(self, actions: Sequence[int]):
        if len(actions) != len(self.agents):
            raise ValueError(f"Expected {len(self.agents)} actions, got {len(actions)}")

        self.step_count += 1
        rewards = []
        movement_costs = []
        collisions = []
        waits = []

        for agent, action in zip(self.agents, actions):
            action = int(action)
            prev_energy = agent.energy
            prev_pos = agent.pos.copy()
            agent.last_action = action

            observation = self._compute_observation(agent)
            agent.perception = observation["perception_cone"]
            agent.raw_energy_gradient = observation["raw_energy_gradient"]
            agent.received_signal = observation["shared_signal"]
            agent.reservoir = self._compute_reservoir(agent)

            energy_cost, collision = self._apply_action(agent, action)
            reward = -energy_cost

            cell = tuple(agent.pos.tolist())
            if cell in self.food_positions:
                agent.energy = min(MAX_ENERGY, agent.energy + ENERGY_GAIN_FOOD)
                del self.food_positions[cell]
                agent.food_eaten += 1
                reward += ENERGY_GAIN_FOOD

            agent.energy = max(0.0, agent.energy - energy_cost)
            reward += 0.2 * (agent.energy - prev_energy)

            agent.last_movement = (agent.pos - prev_pos).astype(np.float32)
            agent.last_reward = float(reward)
            agent.movement_cost = float(energy_cost)
            if collision:
                agent.collision_count += 1
            if action == ACTION_DIM - 1:
                agent.wait_count += 1

            agent.reservoir_next = self._compute_reservoir(agent)
            agent.reservoir_star = self._compute_reservoir_star(agent, observation, collision)
            agent.leakage = self._compute_leakage(agent.reservoir_next, agent.reservoir_star)

            rewards.append(float(reward))
            movement_costs.append(float(energy_cost))
            collisions.append(bool(collision))
            waits.append(bool(action == ACTION_DIM - 1))

        self._add_food(CONSTANT_FOOD_COUNT - len(self.food_positions))

        observations = self._collect_observations()
        terminated = any(agent.energy <= 0 for agent in self.agents)
        truncated = self.step_count >= self.max_steps
        info = self._build_info(
            rewards=rewards,
            movement_costs=movement_costs,
            collisions=collisions,
            waits=waits,
        )
        self.last_info = info

        if self.render_mode == "human":
            self.render()

        return observations, np.asarray(rewards, dtype=np.float32), terminated, truncated, info

    def render(self):
        if self.viewer is not None:
            self.viewer.render()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _collect_observations(self):
        return [self._compute_observation(agent) for agent in self.agents]

    def _compute_observation(self, agent: AgentState):
        perception = np.zeros((PERCEPTION_RANGE, PERCEPTION_WINDOW), dtype=np.float32)
        shared_signal = np.zeros((PERCEPTION_RANGE, PERCEPTION_WINDOW), dtype=np.float32)

        for angle_idx in range(PERCEPTION_WINDOW):
            for distance in range(1, PERCEPTION_RANGE + 1):
                ray_angle = agent.angle + angle_idx - PERCEPTION_WINDOW / 2
                x = int(agent.pos[0] + distance * math.cos(math.radians(ray_angle))) % GRID_SIZE
                y = int(agent.pos[1] + distance * math.sin(math.radians(ray_angle))) % GRID_SIZE

                if MAZE[y, x] == 1:
                    perception[distance - 1, angle_idx] = -1.0
                    break

                if (x, y) in self.food_positions:
                    perception[distance - 1, angle_idx] = self.food_positions[(x, y)] / float(distance)

                for other in self.agents:
                    if other.agent_id != agent.agent_id and np.array_equal(other.pos, np.array([x, y])):
                        perception[distance - 1, angle_idx] = -other.energy / float(distance)
                        shared_signal[distance - 1, angle_idx] = other.energy / MAX_ENERGY

        raw_energy_gradient = self._calculate_energy_gradient(agent)
        return {
            "perception_cone": perception,
            "raw_energy_gradient": raw_energy_gradient.astype(np.float32),
            "energy": np.asarray([agent.energy / MAX_ENERGY], dtype=np.float32),
            "angle": np.asarray([agent.angle / 360.0], dtype=np.float32),
            "last_movement": agent.last_movement.astype(np.float32),
            "last_action": np.asarray([agent.last_action / float(ACTION_DIM - 1)], dtype=np.float32),
            "shared_signal": shared_signal,
        }

    def _calculate_energy_gradient(self, agent: AgentState):
        energy_gradient = np.zeros(360, dtype=np.float32)
        for angle in range(360):
            energy = 0.0
            for distance in range(1, PERCEPTION_RANGE + 1):
                x = int(agent.pos[0] + distance * math.cos(math.radians(angle))) % GRID_SIZE
                y = int(agent.pos[1] + distance * math.sin(math.radians(angle))) % GRID_SIZE
                if MAZE[y, x] == 1:
                    break
                if (x, y) in self.food_positions:
                    energy += self.food_positions[(x, y)] / float(distance)
            energy_gradient[angle] = energy
        return energy_gradient

    def _apply_action(self, agent: AgentState, action: int):
        collision = False
        if action == ACTION_DIM - 1:
            agent.observation_steps += 1
            if agent.observation_steps >= MIN_OBSERVATION_STEPS:
                agent.locomotion_steps = 0
            target_angle = int(np.argmax(agent.raw_energy_gradient))
            angle_diff = (target_angle - agent.angle + 180.0) % 360.0 - 180.0
            adjustment = min(abs(angle_diff), ROTATION_SPEED) * np.sign(angle_diff)
            agent.angle = (agent.angle + adjustment) % 360.0
            return ENERGY_LOSS_STATIONARY, collision

        if agent.locomotion_steps >= MAX_LOCOMOTION_STEPS:
            agent.observation_steps += 1
            if agent.observation_steps >= MIN_OBSERVATION_STEPS:
                agent.locomotion_steps = 0
            return ENERGY_LOSS_STATIONARY, collision

        agent.locomotion_steps += 1
        agent.observation_steps = 0

        target_angle = (action / 20.0) * 360.0
        angle_diff = (target_angle - agent.angle + 180.0) % 360.0 - 180.0
        adjustment = min(abs(angle_diff), ROTATION_SPEED) * np.sign(angle_diff)
        agent.angle = (agent.angle + adjustment) % 360.0

        move_x = int(round(math.cos(math.radians(agent.angle))))
        move_y = int(round(math.sin(math.radians(agent.angle))))
        new_x = int((agent.pos[0] + move_x) % GRID_SIZE)
        new_y = int((agent.pos[1] + move_y) % GRID_SIZE)
        if MAZE[new_y, new_x] == 0:
            agent.pos = np.array([new_x, new_y], dtype=np.int32)
            return ENERGY_LOSS_MOVE, collision

        collision = True
        return ENERGY_LOSS_STATIONARY + COLLISION_PENALTY, collision

    def _compute_reservoir(self, agent: AgentState):
        visible_food_value = float(np.clip(agent.perception[agent.perception > 0].sum() / 500.0, 0.0, 1.0))
        reachable_food_value = float(np.clip(agent.raw_energy_gradient.max() / 300.0, 0.0, 1.0))
        local_free = 0
        for angle in range(0, 360, 45):
            x = int(agent.pos[0] + math.cos(math.radians(angle))) % GRID_SIZE
            y = int(agent.pos[1] + math.sin(math.radians(angle))) % GRID_SIZE
            local_free += 1 if MAZE[y, x] == 0 else 0
        collision_safety = local_free / 8.0
        attention_budget = 1.0 if agent.last_action != ACTION_DIM - 1 else 0.5
        return {
            "self_energy": float(agent.energy / MAX_ENERGY),
            "visible_food_value": visible_food_value,
            "reachable_food_value": reachable_food_value,
            "collision_safety": float(collision_safety),
            "time_budget": float(max(0.0, 1.0 - (self.step_count / float(self.max_steps)))),
            "attention_budget": float(attention_budget),
        }

    def _compute_reservoir_star(self, agent: AgentState, observation: Dict[str, np.ndarray], collision: bool):
        # This is intentionally heuristic in v1. Later it should become learned.
        visible_food_value = float(np.clip(observation["perception_cone"][observation["perception_cone"] > 0].sum() / 500.0, 0.0, 1.0))
        reachable_food_value = float(np.clip(observation["raw_energy_gradient"].max() / 300.0, 0.0, 1.0))
        return {
            "self_energy": float(np.clip((agent.energy - ENERGY_LOSS_STATIONARY) / MAX_ENERGY, 0.0, 1.0)),
            "visible_food_value": visible_food_value,
            "reachable_food_value": reachable_food_value,
            "collision_safety": 1.0 if not collision else 0.75,
            "time_budget": float(max(0.0, 1.0 - (self.step_count / float(self.max_steps)))),
            "attention_budget": 1.0,
        }

    def _compute_leakage(self, reservoir_next: Dict[str, float], reservoir_star: Dict[str, float]):
        leakage = {}
        for key, star_value in reservoir_star.items():
            leakage[key] = float(max(0.0, star_value - reservoir_next[key]))
        return leakage

    def _build_info(
        self,
        rewards: Optional[List[float]] = None,
        movement_costs: Optional[List[float]] = None,
        collisions: Optional[List[bool]] = None,
        waits: Optional[List[bool]] = None,
    ):
        return {
            "step": self.step_count,
            "agents": [
                {
                    "agent_id": agent.agent_id,
                    "color": agent.color,
                    "position": tuple(int(v) for v in agent.pos.tolist()),
                    "energy": float(agent.energy),
                    "angle_deg": float(agent.angle),
                    "reservoir": agent.reservoir,
                    "reservoir_next": agent.reservoir_next,
                    "reservoir_star": agent.reservoir_star,
                    "leakage": agent.leakage,
                    "food_eaten": agent.food_eaten,
                    "collision_count": agent.collision_count,
                    "wait_count": agent.wait_count,
                }
                for agent in self.agents
            ],
            "food_count": len(self.food_positions),
            "rewards": rewards or [agent.last_reward for agent in self.agents],
            "movement_costs": movement_costs or [agent.movement_cost for agent in self.agents],
            "collisions": collisions or [False] * len(self.agents),
            "waits": waits or [False] * len(self.agents),
        }

    def _add_food(self, count: int):
        for _ in range(max(0, count)):
            for _attempt in range(1_000):
                x = self.rng.randint(0, GRID_SIZE - 1)
                y = self.rng.randint(0, GRID_SIZE - 1)
                if MAZE[y, x] == 0 and (x, y) not in self.food_positions:
                    occupied = any(np.array_equal(agent.pos, np.array([x, y])) for agent in self.agents)
                    if not occupied:
                        self.food_positions[(x, y)] = self.rng.randint(*FOOD_ENERGY_RANGE)
                        break
