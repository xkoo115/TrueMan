"""基础策略：直接使用LLM的标准generate进行响应。"""

from __future__ import annotations

from trueman.core.llm_backend import LLMBackend


class BasePolicy:
    """基础策略：正常模式下直接使用LLM推理。"""

    def __init__(self, llm: LLMBackend):
        self.llm = llm

    def act(self, prompt: str, max_tokens: int = 160) -> str:
        """执行基础推理策略。

        默认从 512 降到 160：自定义 ``用户:/助手:`` prompt 不会触发 EOS，所以
        每次都会顶满 max_tokens；对于测量内部状态/probe 的实验，160 token 足够，
        生成开销随之线性下降。需要更长回复时显式传入更大的 max_tokens。
        """
        return self.llm.generate(prompt, max_tokens=max_tokens)
