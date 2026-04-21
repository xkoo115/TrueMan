"""内省策略：焦虑驱动的自我反思。

当焦虑信号超过阈值时，Agent检索记忆中的矛盾轨迹，
构建自我反思prompt，尝试自我纠错。
"""

from __future__ import annotations

from trueman.core.llm_backend import LLMBackend
from trueman.core.memory.episodic import EpisodicMemory
from trueman.core.homeostasis.integrator import EmotionState


class IntrospectionPolicy:
    """内省策略：焦虑驱动的自我反思和纠错。"""

    INTROSPECTION_PROMPT = """你当前感到焦虑——你的内部判断存在分歧，可能存在逻辑矛盾。
请进行自我反思：

{contradictions_text}

请分析以上矛盾，尝试：
1. 识别矛盾的核心原因
2. 判断哪个判断更可靠
3. 提出修正方案
当前焦虑程度: {anxiety:.2f}

自我反思："""

    def __init__(self, llm: LLMBackend, memory: EpisodicMemory):
        self.llm = llm
        self.memory = memory

    def act(self, prompt: str, emotions: EmotionState) -> str:
        """执行内省策略。"""
        # 检索矛盾轨迹
        contradictions = self.memory.find_contradictions(recent_n=20)

        # 提升矛盾轨迹的优先级
        for t1, t2 in contradictions:
            self.memory.boost_priority(t1.trace_id)
            self.memory.boost_priority(t2.trace_id)

        # 构建矛盾描述
        if contradictions:
            lines = []
            for i, (t1, t2) in enumerate(contradictions[:5]):  # 最多展示5对
                lines.append(f"矛盾{i+1}:")
                lines.append(f"  判断A: {t1.action[:100]}")
                lines.append(f"  判断B: {t2.action[:100]}")
            contradictions_text = "\n".join(lines)
        else:
            contradictions_text = "未发现明确的逻辑矛盾，但内部判断仍存在不确定性。"

        introspection_prompt = self.INTROSPECTION_PROMPT.format(
            contradictions_text=contradictions_text,
            anxiety=emotions.anxiety,
        )
        full_prompt = prompt + "\n\n" + introspection_prompt
        return self.llm.generate(full_prompt, max_tokens=256, temperature=0.5)
