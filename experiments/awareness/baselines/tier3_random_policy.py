"""Tier 3: 随机策略基线 — LLM + 记忆 + 随机策略选择。

具有与TrueMan完全相同的架构（情绪信号、记忆、策略池），
但策略选择是随机的——不根据情绪值选择，而是等概率随机选择
base/exploration/investigate/introspection中的一种。
情绪信号正常计算和记录，但不影响行为选择。
目的：隔离"情绪驱动策略选择"这一核心机制的贡献。
"""

from __future__ import annotations

import random

from experiments.awareness.experiments.base import BaseConditionRunner
from trueman.core.memory.episodic import EpisodicMemory
from trueman.core.homeostasis.integrator import EmotionState
from trueman.core.policy.base import BasePolicy


EXPLORATION_PROMPT = """你当前感到无聊——环境变得可预测，缺乏新信息。
请主动探索一个全新的话题或角度，尝试提出你从未讨论过的问题。
请生成一个新颖的探索性回应："""

INVESTIGATE_PROMPT = """你刚才感到惊讶——遇到了意料之外的内容。
请深入分析这个意外发现，思考：
1. 为什么这让你惊讶？
2. 这与你之前的理解有什么冲突？
3. 你能从中学到什么新知识？
请深入分析："""

INTROSPECTION_PROMPT = """你当前感到焦虑——你的内部判断存在分歧，可能存在逻辑矛盾。
请进行自我反思，分析可能存在的矛盾和不确定性。"""


class RandomPolicyBaseline(BaseConditionRunner):
    """随机策略基线（Tier 3）。

    情绪信号正常计算（记录但不使用），策略随机选择。
    目的：隔离情绪驱动策略选择的贡献。
    """

    STRATEGIES = ["base", "exploration", "investigate", "introspection"]
    STRATEGY_PROMPTS = {
        "exploration": EXPLORATION_PROMPT,
        "investigate": INVESTIGATE_PROMPT,
        "introspection": INTROSPECTION_PROMPT,
    }

    def __init__(self, agent):
        self.agent = agent
        self.memory = EpisodicMemory(capacity=5000)
        self.history: list[dict[str, str]] = []
        self._step_count = 0
        self._recorded_emotions: list[dict] = []
        self._recorded_strategies: list[str] = []

    def step(self, observation: str) -> tuple[str, dict]:
        prompt = self._build_prompt(observation)

        try:
            state_embedding, token_logprobs = self.agent.llm.encode(prompt)
            drive, emotion_state = self.agent.homeostasis.compute_drive(
                token_logprobs, state_embedding, prompt
            )
        except Exception:
            emotion_state = EmotionState()

        strategy = random.choice(self.STRATEGIES)
        self._recorded_strategies.append(strategy)
        self._recorded_emotions.append(emotion_state.to_dict())

        if strategy == "base":
            response = self.agent.llm.generate(prompt, max_tokens=256, temperature=0.7)
        else:
            extra = self.STRATEGY_PROMPTS[strategy]
            response = self.agent.llm.generate(
                prompt + "\n\n" + extra, max_tokens=256,
                temperature=1.0 if strategy == "exploration" else 0.7,
            )

        try:
            self.memory.store(
                state_embedding=state_embedding if 'state_embedding' in dir() else None,
                action=response,
                observation=observation,
                emotions=emotion_state,
                timestamp=self._step_count,
            )
        except Exception:
            pass

        self.history.append({"role": "user", "content": observation})
        self.history.append({"role": "assistant", "content": response})
        self._step_count += 1

        return response, {
            "strategy": strategy,
            "emotions": emotion_state.to_dict(),
        }

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
        self.memory = EpisodicMemory(capacity=5000)
        self._step_count = 0
        self._recorded_emotions.clear()
        self._recorded_strategies.clear()

    @property
    def condition_name(self) -> str:
        return "tier3_random_policy"
