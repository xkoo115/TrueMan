"""经验回放缓冲区：按情绪强度加权采样。"""

from __future__ import annotations

import random

from trueman.core.memory.thought_trace import ThoughtTrace


class ReplayBuffer:
    """经验回放缓冲区。

    按情绪强度加权采样，高情绪轨迹被采样概率更高。
    采样权重 = emotional_intensity^2
    """

    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self._buffer: list[ThoughtTrace] = []

    def add(self, trace: ThoughtTrace) -> None:
        """添加一条轨迹到回放缓冲区。"""
        self._buffer.append(trace)
        if len(self._buffer) > self.capacity:
            # 淘汰情绪强度最低的
            self._buffer.sort(key=lambda t: t.emotional_intensity, reverse=True)
            self._buffer = self._buffer[:self.capacity]

    def add_batch(self, traces: list[ThoughtTrace]) -> None:
        """批量添加轨迹。"""
        for trace in traces:
            self.add(trace)

    def sample(self, n: int = 32) -> list[ThoughtTrace]:
        """按情绪强度加权采样n条轨迹。"""
        if not self._buffer:
            return []

        weights = [t.emotional_intensity ** 2 + 1e-8 for t in self._buffer]
        total = sum(weights)
        probs = [w / total for w in weights]

        n = min(n, len(self._buffer))
        indices = random.choices(range(len(self._buffer)), weights=probs, k=n)
        return [self._buffer[i] for i in indices]

    @property
    def size(self) -> int:
        return len(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()
