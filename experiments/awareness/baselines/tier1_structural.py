"""Tier 1: 结构等价基线 — 与TrueMan相同的prompt结构，但无情绪信号。

使用与TrueMan完全相同的_build_prompt格式和20轮历史，
但没有情绪信号计算和记忆模块。
"""

from __future__ import annotations

from experiments.awareness.experiments.base import BaseConditionRunner


class StructuralBaseline(BaseConditionRunner):
    """结构等价基线（Tier 1）。

    使用与TrueMan完全相同的prompt构建方式，但移除所有情绪相关组件。
    目的：隔离prompt格式本身对结果的影响。
    """

    def __init__(self, llm):
        self.llm = llm
        self.history: list[dict[str, str]] = []

    def step(self, observation: str) -> tuple[str, dict]:
        full_prompt = self._build_prompt(observation)
        response = self.llm.generate(full_prompt, max_tokens=256, temperature=0.7)
        self.history.append({"role": "user", "content": observation})
        self.history.append({"role": "assistant", "content": response})
        return response, {}

    def _build_prompt(self, current: str) -> str:
        parts = []
        for msg in self.history[-20:]:
            role = "用户" if msg["role"] == "user" else "助手"
            parts.append(f"{role}: {msg['content']}")
        parts.append(f"用户: {current}")
        parts.append("助手:")
        return "\n".join(parts)

    def reset(self) -> None:
        self.history.clear()

    @property
    def condition_name(self) -> str:
        return "tier1_structural"
