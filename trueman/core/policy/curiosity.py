"""好奇心驱动策略：探索、深究、策略整合。

ExplorationPolicy: 无聊驱动的新颖内容生成
InvestigatePolicy: 惊奇驱动的深入分析
CuriosityPolicy: 按优先级整合四种策略
"""

from __future__ import annotations

from trueman.core.llm_backend import LLMBackend
from trueman.core.memory.episodic import EpisodicMemory
from trueman.core.homeostasis.integrator import EmotionState
from trueman.core.policy.base import BasePolicy
from trueman.core.config import EmotionThresholdConfig


class ExplorationPolicy:
    """探索策略：无聊驱动的新颖内容生成。

    当无聊信号超过阈值时，Agent主动生成探索性内容。
    基于近期状态嵌入的分布，生成远离已探索区域的新prompt。
    """

    EXPLORATION_PROMPT = """你当前感到无聊——环境变得可预测，缺乏新信息。
请主动探索一个全新的话题或角度，尝试提出你从未讨论过的问题。
当前无聊程度: {boredom:.2f}

请生成一个新颖的探索性回应："""

    def __init__(self, llm: LLMBackend, memory: EpisodicMemory):
        self.llm = llm
        self.memory = memory

    def act(self, prompt: str, emotions: EmotionState) -> str:
        """执行探索策略。"""
        exploration_prompt = self.EXPLORATION_PROMPT.format(boredom=emotions.boredom)
        full_prompt = prompt + "\n\n" + exploration_prompt
        return self.llm.generate(full_prompt, max_tokens=256, temperature=1.2)


class InvestigatePolicy:
    """深究策略：惊奇驱动的深入分析。

    当惊奇信号超过阈值时，Agent对意外内容进行深入分析。
    识别触发惊奇的具体内容，构建分析prompt。
    """

    INVESTIGATE_PROMPT = """你刚才感到惊讶——遇到了意料之外的内容。
请深入分析这个意外发现，思考：
1. 为什么这让你惊讶？
2. 这与你之前的理解有什么冲突？
3. 你能从中学到什么新知识？
当前惊奇程度: {surprise:.2f}

请深入分析："""

    def __init__(self, llm: LLMBackend):
        self.llm = llm

    def act(self, prompt: str, emotions: EmotionState) -> str:
        """执行深究策略。"""
        investigate_prompt = self.INVESTIGATE_PROMPT.format(surprise=emotions.surprise)
        full_prompt = prompt + "\n\n" + investigate_prompt
        return self.llm.generate(full_prompt, max_tokens=256, temperature=0.7)


class CuriosityPolicy:
    """好奇心策略整合器。

    按优先级选择策略：
    焦虑内省 > 惊奇深究 > 无聊探索 > 正常响应
    """

    def __init__(
        self,
        base_policy: BasePolicy,
        exploration_policy: ExplorationPolicy,
        investigate_policy: InvestigatePolicy,
        introspection_policy,  # IntrospectionPolicy，避免循环导入
        thresholds: EmotionThresholdConfig,
    ):
        self.base = base_policy
        self.exploration = exploration_policy
        self.investigate = investigate_policy
        self.introspection = introspection_policy
        self.thresholds = thresholds

    def select_action(
        self,
        prompt: str,
        emotions: EmotionState,
    ) -> tuple[str, str]:
        """根据情绪状态选择并执行策略。

        Returns:
            (action, strategy_name) 策略名称用于日志
        """
        # 优先级1：焦虑内省
        if emotions.anxiety > self.thresholds.anxiety_introspection_threshold:
            action = self.introspection.act(prompt, emotions)
            return action, "introspection"

        # 优先级2：惊奇深究
        if emotions.surprise > self.thresholds.surprise_update_threshold:
            action = self.investigate.act(prompt, emotions)
            return action, "investigate"

        # 优先级3：无聊探索
        if emotions.boredom > self.thresholds.boredom_explore_threshold:
            action = self.exploration.act(prompt, emotions)
            return action, "exploration"

        # 优先级4：正常响应
        action = self.base.act(prompt)
        return action, "base"
