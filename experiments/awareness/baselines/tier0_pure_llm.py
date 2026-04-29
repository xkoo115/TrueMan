"""Tier 0: 纯LLM基线 — 无情绪信号、无情景记忆、无特殊prompt结构。

与现有BaselineRunner完全一致，仅保持对话历史。
"""

from __future__ import annotations

from experiments.awareness.experiments.base import BaseConditionRunner


class PureLLMBaseline(BaseConditionRunner):
    """纯LLM基线（Tier 0）。"""

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
        return "tier0_pure_llm"
