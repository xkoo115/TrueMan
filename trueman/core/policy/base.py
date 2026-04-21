"""基础策略：直接使用LLM的标准generate进行响应。"""

from __future__ import annotations

from trueman.core.llm_backend import LLMBackend


class BasePolicy:
    """基础策略：正常模式下直接使用LLM推理。"""

    def __init__(self, llm: LLMBackend):
        self.llm = llm

    def act(self, prompt: str, max_tokens: int = 512) -> str:
        """执行基础推理策略。"""
        return self.llm.generate(prompt, max_tokens=max_tokens)
