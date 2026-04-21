"""情景记忆：基于优先级队列的短期工作记忆。

存储最近的交互轨迹（Thought Traces），每条轨迹附带情绪标注，
按情绪强度排序优先级。容量满时自动淘汰低优先级条目。
"""

from __future__ import annotations

import heapq
import random
from typing import Optional

import torch

from trueman.core.memory.thought_trace import ThoughtTrace
from trueman.core.homeostasis.integrator import EmotionState


class EpisodicMemory:
    """情景记忆（海马体类比）。

    使用列表存储轨迹，支持按情绪强度排序、容量淘汰、加权采样。
    """

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self._traces: list[ThoughtTrace] = []
        self._next_id = 0

    def store(
        self,
        state_embedding: torch.Tensor,
        action: str,
        observation: str,
        emotions: EmotionState,
        timestamp: int,
    ) -> ThoughtTrace:
        """存储一条交互轨迹。

        Args:
            state_embedding: LLM状态嵌入
            action: Agent执行的动作/生成的文本
            observation: 环境返回的观测
            emotions: 情绪状态快照
            timestamp: 交互步数

        Returns:
            存储的ThoughtTrace
        """
        trace = ThoughtTrace(
            trace_id=self._next_id,
            state_embedding=state_embedding,
            action=action,
            observation=observation,
            emotions=emotions,
            timestamp=timestamp,
        )
        self._next_id += 1
        self._traces.append(trace)

        # 容量淘汰：移除情绪强度最低的
        if len(self._traces) > self.capacity:
            self._traces.sort(key=lambda t: t.emotional_intensity, reverse=True)
            self._traces = self._traces[:self.capacity]

        return trace

    def get_recent(self, n: int = 20) -> list[ThoughtTrace]:
        """按时间戳降序返回最近n条轨迹。"""
        sorted_traces = sorted(self._traces, key=lambda t: t.timestamp, reverse=True)
        return sorted_traces[:n]

    def get_high_priority(self, n: int = 100) -> list[ThoughtTrace]:
        """按情绪强度降序返回前n条轨迹。"""
        sorted_traces = sorted(self._traces, key=lambda t: t.emotional_intensity, reverse=True)
        return sorted_traces[:n]

    def find_contradictions(self, recent_n: int = 20) -> list[tuple[ThoughtTrace, ThoughtTrace]]:
        """检测最近轨迹间的逻辑矛盾。

        基于启发式规则：如果两条轨迹的action在语义上相反（包含否定词），
        且observation主题相似，则认为存在矛盾。

        Args:
            recent_n: 检索最近多少条轨迹进行检测

        Returns:
            矛盾轨迹对列表
        """
        traces = self.get_recent(recent_n)
        if len(traces) < 2:
            return []

        contradictions = []
        negation_words = {"不", "没", "无", "非", "否", "别", "未", "never", "not", "no", "don't", "can't", "won't"}

        for i in range(len(traces)):
            for j in range(i + 1, len(traces)):
                t1, t2 = traces[i], traces[j]
                if self._are_contradictory(t1, t2, negation_words):
                    contradictions.append((t1, t2))

        return contradictions

    def boost_priority(self, trace_id: int, boost_amount: float = 0.5) -> None:
        """提升指定轨迹的优先级（增加情绪强度）。

        Args:
            trace_id: 轨迹ID
            boost_amount: 优先级提升量
        """
        for trace in self._traces:
            if trace.trace_id == trace_id:
                # 通过增加情绪值来提升优先级
                new_surprise = min(1.0, trace.emotions.surprise + boost_amount)
                trace.emotions = EmotionState(
                    surprise=new_surprise,
                    boredom=trace.emotions.boredom,
                    anxiety=trace.emotions.anxiety,
                    drive=trace.emotions.drive,
                )
                break

    def weighted_sample(self, n: int = 50) -> list[ThoughtTrace]:
        """按情绪强度加权采样n条轨迹。

        采样权重 = emotional_intensity^2，使高情绪轨迹被更强烈偏好。
        """
        if not self._traces:
            return []

        weights = [t.emotional_intensity ** 2 + 1e-8 for t in self._traces]
        total = sum(weights)
        probs = [w / total for w in weights]

        n = min(n, len(self._traces))
        indices = random.choices(range(len(self._traces)), weights=probs, k=n)
        return [self._traces[i] for i in indices]

    @property
    def size(self) -> int:
        return len(self._traces)

    def clear(self) -> None:
        self._traces.clear()

    @staticmethod
    def _are_contradictory(
        t1: ThoughtTrace,
        t2: ThoughtTrace,
        negation_words: set[str],
    ) -> bool:
        """启发式矛盾检测：一条包含否定，另一条不包含，且主题相似。"""
        a1, a2 = t1.action.lower(), t2.action.lower()

        # 检查是否一条包含否定词而另一条不包含
        has_neg1 = any(neg in a1 for neg in negation_words)
        has_neg2 = any(neg in a2 for neg in negation_words)

        if has_neg1 == has_neg2:
            return False  # 两条都含或都不含否定词，不矛盾

        # 简单主题相似度：共享词比例
        words1 = set(a1.split())
        words2 = set(a2.split())
        shared = words1 & words2 - negation_words
        total = words1 | words2

        if not total:
            return False

        similarity = len(shared) / len(total)
        return similarity > 0.3  # 主题相似度超过阈值则认为矛盾
