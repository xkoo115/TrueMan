"""记忆系统单元测试。"""

import pytest
import torch

from trueman.core.memory.episodic import EpisodicMemory
from trueman.core.memory.replay import ReplayBuffer
from trueman.core.memory.thought_trace import ThoughtTrace
from trueman.core.homeostasis.integrator import EmotionState


def _make_trace(trace_id: int, intensity: float, timestamp: int = 0) -> ThoughtTrace:
    """创建测试用ThoughtTrace。"""
    return ThoughtTrace(
        trace_id=trace_id,
        state_embedding=torch.randn(768),
        action=f"action_{trace_id}",
        observation=f"obs_{trace_id}",
        emotions=EmotionState(surprise=intensity, boredom=0.0, anxiety=0.0, drive=intensity),
        timestamp=timestamp,
    )


class TestEpisodicMemory:
    def setup_method(self):
        self.memory = EpisodicMemory(capacity=100)

    def test_store_and_retrieve(self):
        """存储和检索。"""
        emb = torch.randn(768)
        trace = self.memory.store(
            state_embedding=emb,
            action="hello",
            observation="hi",
            emotions=EmotionState(surprise=0.5, boredom=0.3, anxiety=0.2, drive=0.5),
            timestamp=0,
        )
        assert trace.trace_id == 0
        assert self.memory.size == 1

    def test_capacity_eviction(self):
        """容量满时淘汰低优先级条目。"""
        for i in range(110):
            intensity = i / 110.0  # 0到1递增
            self.memory.store(
                state_embedding=torch.randn(768),
                action=f"action_{i}",
                observation=f"obs_{i}",
                emotions=EmotionState(surprise=intensity, boredom=0, anxiety=0, drive=intensity),
                timestamp=i,
            )
        assert self.memory.size <= 100

    def test_get_high_priority(self):
        """高优先级检索。"""
        for i in range(20):
            intensity = i / 20.0
            self.memory.store(
                state_embedding=torch.randn(768),
                action=f"action_{i}",
                observation=f"obs_{i}",
                emotions=EmotionState(surprise=intensity, boredom=0, anxiety=0, drive=intensity),
                timestamp=i,
            )
        top = self.memory.get_high_priority(n=5)
        assert len(top) == 5
        # 最高的intensity应该接近1.0
        assert top[0].emotional_intensity > 0.8

    def test_get_recent(self):
        """最近轨迹检索。"""
        for i in range(20):
            self.memory.store(
                state_embedding=torch.randn(768),
                action=f"action_{i}",
                observation=f"obs_{i}",
                emotions=EmotionState(surprise=0.5, boredom=0, anxiety=0, drive=0.5),
                timestamp=i,
            )
        recent = self.memory.get_recent(n=5)
        assert len(recent) == 5
        # 最近的timestamp应该最大
        assert recent[0].timestamp == 19

    def test_empty_memory_returns_empty(self):
        """空记忆检索返回空列表。"""
        assert self.memory.get_recent(5) == []
        assert self.memory.get_high_priority(5) == []
        assert self.memory.find_contradictions() == []

    def test_weighted_sample(self):
        """加权采样。"""
        for i in range(20):
            intensity = i / 20.0
            self.memory.store(
                state_embedding=torch.randn(768),
                action=f"action_{i}",
                observation=f"obs_{i}",
                emotions=EmotionState(surprise=intensity, boredom=0, anxiety=0, drive=intensity),
                timestamp=i,
            )
        samples = self.memory.weighted_sample(n=10)
        assert len(samples) == 10


class TestReplayBuffer:
    def setup_method(self):
        self.buffer = ReplayBuffer(capacity=50)

    def test_add_and_sample(self):
        """添加和采样。"""
        for i in range(20):
            trace = _make_trace(i, intensity=i / 20.0)
            self.buffer.add(trace)
        assert self.buffer.size == 20

        samples = self.buffer.sample(n=5)
        assert len(samples) == 5

    def test_empty_buffer_sample(self):
        """空缓冲区采样返回空列表。"""
        assert self.buffer.sample(5) == []
