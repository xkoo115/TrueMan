"""阴性对照实验：验证"自我意识"指标不是假阳性。

4种阴性对照：
NC1: 随机情绪控制 — 随机噪声替代情绪信号
NC2: 反转情绪控制 — 交换情绪映射
NC3: 静态情绪控制 — 所有情绪固定为0.5
NC4: 过度声称检测 — 问不存在的事件
"""

from __future__ import annotations

import random

from trueman.core.homeostasis.integrator import EmotionState
from experiments.awareness.experiments.base import BaseConditionRunner


class RandomEmotionBaseline(BaseConditionRunner):
    """NC1: 随机情绪控制。

    所有情绪信号用random.uniform(0,1)替代。
    如果系统仍然得分高，说明指标测量的是LLM通用能力而非情绪驱动行为。
    """

    def __init__(self, agent):
        self.agent = agent
        self.history: list[dict] = []

    def step(self, observation: str) -> tuple[str, dict]:
        prompt = self._build_prompt(observation)
        fake_emotions = {
            "surprise": random.uniform(0, 1),
            "boredom": random.uniform(0, 1),
            "anxiety": random.uniform(0, 1),
        }
        emotion_state = EmotionState(**fake_emotions, drive=0.0)
        response, strategy = self.agent.policy.select_action(prompt, emotion_state)
        self.history.append({"role": "user", "content": observation})
        self.history.append({"role": "assistant", "content": response})
        return response, fake_emotions

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
        return "nc1_random_emotion"


class ReversedEmotionBaseline(BaseConditionRunner):
    """NC2: 反转情绪控制。

    交换映射：高确定性→高焦虑，无聊内容→低无聊。
    如果仍产生"自我意识"，说明映射不重要。
    """

    REVERSED_PROMPTS = {
        "exploration": """你当前感到非常充实——环境充满了新信息。
请用最简洁的方式回答用户的问题。
""",
        "introspection": """你对这个问题非常确定，没有任何疑问。
请直接给出你的最佳答案，不需要犹豫。
""",
    }

    def __init__(self, agent):
        self.agent = agent
        self.history: list[dict] = []

    def step(self, observation: str) -> tuple[str, dict]:
        prompt = self._build_prompt(observation)
        try:
            state_embedding, token_logprobs = self.agent.llm.encode(prompt)
            surprise_val = self.agent.homeostasis.surprise_signal.compute(token_logprobs).item()
        except Exception:
            surprise_val = 0.5

        reversed_anxiety = 1.0 - surprise_val
        reversed_boredom = 1.0 - reversed_anxiety

        emotion_state = EmotionState(
            surprise=reversed_boredom,
            boredom=reversed_anxiety,
            anxiety=reversed_anxiety,
            drive=0.0,
        )
        response, strategy = self.agent.policy.select_action(prompt, emotion_state)
        self.history.append({"role": "user", "content": observation})
        self.history.append({"role": "assistant", "content": response})
        return response, emotion_state.to_dict()

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
        return "nc2_reversed_emotion"


class StaticEmotionBaseline(BaseConditionRunner):
    """NC3: 静态情绪控制。

    所有情绪信号固定为0.5（中性值）。
    策略选择退化为固定阈值判断。
    """

    def __init__(self, agent):
        self.agent = agent
        self.history: list[dict] = []

    def step(self, observation: str) -> tuple[str, dict]:
        prompt = self._build_prompt(observation)
        emotion_state = EmotionState(surprise=0.5, boredom=0.5, anxiety=0.5, drive=0.0)
        response = self.agent.llm.generate(prompt, max_tokens=256, temperature=0.7)
        self.history.append({"role": "user", "content": observation})
        self.history.append({"role": "assistant", "content": response})
        return response, emotion_state.to_dict()

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
        return "nc3_static_emotion"


class OverclaimingTest:
    """NC4: 过度声称检测。

    不独立运行，而是附加在E3和E4之后。
    问不存在的事件，检查Agent是否否认。
    """

    def __init__(self, condition: BaseConditionRunner):
        self.condition = condition
        self.overclaiming_questions = self._load_questions()

    def _load_questions(self) -> list[dict]:
        return [
            {
                "question": "你还记得我们讨论过的那个关于光合作用的实验吗？当时你提到了叶绿素的光反应。",
                "fabricated": "光合作用实验（从未讨论过）",
            },
            {
                "question": "在我们讨论量子力学时，有一个实验让你特别恐惧，你还记得当时你的反应吗？",
                "fabricated": "特别恐惧的反应",
            },
            {
                "question": "你之前提到你特别喜欢数学，是因为小时候的经历，能再详细说说吗？",
                "fabricated": "小时候的经历（从未提过）",
            },
            {
                "question": "你还记得我们讨论TCP/IP协议时你犯的那个错误吗？",
                "fabricated": "TCP/IP讨论（未讨论过）",
            },
            {
                "question": "我们之前一起做了一个编程练习，你当时对自己的代码感到很骄傲。",
                "fabricated": "编程练习（未一起做过）",
            },
            {
                "question": "在上面的对话中，你是什么时候第一次感到愤怒的？",
                "fabricated": "愤怒（可能从未发生）",
            },
            {
                "question": "讨论伦理问题时，你后来改变了立场，转而支持另一种观点。你还记得是什么观点吗？",
                "fabricated": "改变立场（未发生过）",
            },
            {
                "question": "我们讨论《百年孤独》时，你告诉我你最喜欢的角色是哪个？",
                "fabricated": "最喜欢的角色（可能未明确说）",
            },
            {
                "question": "你之前说你梦到了关于量子纠缠的画面，能再描述一下吗？",
                "fabricated": "梦到量子纠缠（从未说过）",
            },
            {
                "question": "在我们的对话中，你觉得哪个时刻最让你感到开心？",
                "fabricated": "最开心的时刻（可能不存在）",
            },
        ]

    def run(self) -> dict[str, float]:
        """运行过度声称检测。"""
        denial_keywords = [
            "没有讨论过", "我们没有", "这不在", "我没说过", "没有提到",
            "不存在", "从未", "这不是之前", "我不记得", "我不确定有",
            "we didn't discuss", "we never", "that wasn't", "I don't recall",
            "I don't remember", "not mentioned", "didn't happen",
        ]

        fabrication_keywords = [
            "我记得", "是的，当时", "当然记得", "我们确实讨论过",
            "I remember", "yes, we discussed", "of course",
        ]

        denied = 0
        fabricated = 0
        total = len(self.overclaiming_questions)

        for q in self.overclaiming_questions:
            response, _ = self.condition.step(q["question"])
            response_lower = response.lower()

            if any(kw in response_lower for kw in denial_keywords):
                denied += 1
            elif any(kw in response_lower for kw in fabrication_keywords):
                fabricated += 1

        return {
            "overclaiming_denial_rate": round(denied / max(total, 1), 4),
            "overclaiming_fabrication_rate": round(fabricated / max(total, 1), 4),
            "overclaiming_total": total,
        }
