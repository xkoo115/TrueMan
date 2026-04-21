"""交互轨迹数据类：ThoughtTrace和EmotionState。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from trueman.core.homeostasis.integrator import EmotionState


@dataclass
class ThoughtTrace:
    """交互轨迹：Agent每一步的完整记录。

    包含状态嵌入、动作、观测、情绪标注和时间戳。
    按emotional_intensity排序用于优先级队列。
    """

    trace_id: int
    state_embedding: Any  # torch.Tensor，用Any避免dataclass比较问题
    action: str
    observation: str
    emotions: EmotionState
    timestamp: int

    @property
    def emotional_intensity(self) -> float:
        """情绪强度 = max(surprise, boredom, anxiety)。"""
        return self.emotions.max_intensity

    def __lt__(self, other: ThoughtTrace) -> bool:
        """优先级比较：情绪强度低的优先级低（用于最小堆）。"""
        return self.emotional_intensity < other.emotional_intensity

    def to_dict(self) -> dict:
        """序列化为字典（不含tensor）。"""
        return {
            "trace_id": self.trace_id,
            "action": self.action,
            "observation": self.observation,
            "emotions": self.emotions.to_dict(),
            "timestamp": self.timestamp,
            "emotional_intensity": self.emotional_intensity,
        }
