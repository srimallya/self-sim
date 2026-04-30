from collections import deque
from typing import Any, Deque, Dict, List

import random


class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self.capacity = capacity
        self.buffer: Deque[Dict[str, Any]] = deque(maxlen=capacity)

    def push(
        self,
        obs,
        action,
        next_obs,
        reservoir,
        next_reservoir,
        reservoir_star,
        leakage,
        movement_cost,
        collision,
        done,
        info,
    ):
        self.buffer.append(
            {
                "obs": obs,
                "action": action,
                "next_obs": next_obs,
                "reservoir": reservoir,
                "next_reservoir": next_reservoir,
                "reservoir_star": reservoir_star,
                "leakage": leakage,
                "movement_cost": movement_cost,
                "collision": collision,
                "done": done,
                "info": info,
            }
        )

    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)
