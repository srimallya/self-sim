from collections import deque
from typing import Any, Deque, Dict, List

import random


class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self.capacity = capacity
        self.buffer: Deque[Dict[str, Any]] = deque(maxlen=capacity)

    def push(self, **transition):
        self.buffer.append(dict(transition))

    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)
