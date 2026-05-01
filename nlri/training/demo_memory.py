from __future__ import annotations

from collections import deque
from typing import Deque, Dict

import numpy as np


class DemoMemory:
    """Keeps high-quality recent rollout summaries for self-distillation context."""

    def __init__(self, capacity: int = 128, min_score: float = 0.0):
        self.capacity = int(capacity)
        self.min_score = float(min_score)
        self.items: Deque[Dict[str, np.ndarray | float]] = deque(maxlen=self.capacity)

    def maybe_add(self, feedback_vector, score: float):
        score = float(np.nan_to_num(score, nan=0.0))
        if score < self.min_score:
            return False
        self.items.append({"feedback_vector": np.asarray(feedback_vector, dtype=np.float32), "score": score})
        return True

    def context(self):
        if not self.items:
            return None
        scores = np.asarray([item["score"] for item in self.items], dtype=np.float32)
        weights = scores - scores.min() + 1e-3
        weights = weights / weights.sum()
        vectors = np.stack([item["feedback_vector"] for item in self.items], axis=0)
        return np.nan_to_num((vectors * weights[:, None]).sum(axis=0), nan=0.0).astype(np.float32)

    def score_mean(self):
        if not self.items:
            return 0.0
        return float(np.mean([item["score"] for item in self.items]))

    def __len__(self):
        return len(self.items)

