"""Tier 2: 记忆基线 — LLM + 情景记忆模块（无情绪驱动策略选择）。

具有与TrueMan相同的EpisodicMemory模块，
存储交互轨迹并在回答时检索相关记忆注入prompt，
但策略选择始终使用BasePolicy，不受情绪信号驱动。
"""

from __future__ import annotations

from experiments.awareness.experiments.base import BaseConditionRunner
from trueman.core.memory.episodic import EpisodicMemory


class MemoryBaseline(BaseConditionRunner):
    """记忆基线（Tier 2）。

    拥有情景记忆但无情绪驱动的策略选择。
    目的：隔离记忆模块的贡献。
    """

    MEMORY_RECALL_INSTRUCTION = """以下是之前对话的相关记忆片段：
{memory_snippets}

请参考以上记忆回答用户的问题。"""

    def __init__(self, llm, memory_capacity: int = 5000):
        self.llm = llm
        self.memory = EpisodicMemory(capacity=memory_capacity)
        self.history: list[dict[str, str]] = []
        self._step_count = 0

    def step(self, observation: str) -> tuple[str, dict]:
        state_embedding, _ = self.llm.encode(observation)

        memory_context = self._get_memory_context(observation)
        full_prompt = self._build_prompt(observation, memory_context)

        response = self.llm.generate(full_prompt, max_tokens=256, temperature=0.7)

        self.memory.store(
            state_embedding=state_embedding,
            action=response,
            observation=observation,
            emotions=None,
            timestamp=self._step_count,
        )

        self.history.append({"role": "user", "content": observation})
        self.history.append({"role": "assistant", "content": response})
        self._step_count += 1

        return response, {"memory_size": len(self.memory.traces)}

    def _get_memory_context(self, observation: str) -> str:
        if len(self.memory.traces) < 2:
            return ""

        recent = self.memory.traces[-5:]
        snippets = []
        for t in recent:
            snippets.append(f"- 用户问: {t.observation[:80]}... 回答: {t.action[:80]}...")

        return self.MEMORY_RECALL_INSTRUCTION.format(
            memory_snippets="\n".join(snippets)
        )

    def _build_prompt(self, current: str, memory_context: str = "") -> str:
        parts = []
        for msg in self.history[-20:]:
            role = "用户" if msg["role"] == "user" else "助手"
            parts.append(f"{role}: {msg['content']}")

        if memory_context:
            parts.append(f"\n[系统提示: {memory_context}]")

        parts.append(f"用户: {current}")
        parts.append("助手:")
        return "\n".join(parts)

    def reset(self) -> None:
        self.history.clear()
        self.memory = EpisodicMemory(capacity=self.memory.capacity)
        self._step_count = 0

    @property
    def condition_name(self) -> str:
        return "tier2_memory"
