"""内稳态内核：组合三种情绪信号和整合器，提供统一的compute_drive接口。"""

from __future__ import annotations

import torch

from trueman.core.config import AgentConfig
from trueman.core.homeostasis.signals import SurpriseSignal, BoredomSignal, AnxietySignal
from trueman.core.homeostasis.integrator import EmotionIntegrator, EmotionState
from trueman.core.llm_backend import LLMBackend


class HomeostasisCore:
    """内稳态内核。

    组合惊奇/无聊/焦虑三种信号和情绪整合器，
    提供统一的compute_drive()接口。
    """

    def __init__(self, config: AgentConfig, llm: LLMBackend):
        self.surprise_signal = SurpriseSignal(config.surprise)
        self.boredom_signal = BoredomSignal(config.boredom)
        self.anxiety_signal = AnxietySignal(config.anxiety)
        self.integrator = EmotionIntegrator(config.homeostasis)
        self.llm = llm
        self._anxiety_lightweight = config.anxiety.lightweight
        self._anxiety_n_samples = config.anxiety.n_samples

    def compute_drive(
        self,
        token_logprobs: torch.Tensor,
        state_embedding: torch.Tensor,
        current_prompt: str,
    ) -> tuple[float, EmotionState]:
        """计算内稳态驱动信号。

        Args:
            token_logprobs: LLM的token对数概率序列
            state_embedding: LLM的状态嵌入向量
            current_prompt: 当前对话prompt（用于焦虑计算）

        Returns:
            (total_drive, emotion_state)
        """
        # 惊奇信号
        surprise = self.surprise_signal.compute(token_logprobs)

        # 无聊信号
        prediction_error = -token_logprobs.mean().item() if token_logprobs.numel() > 0 else 0.0
        boredom = self.boredom_signal.compute(prediction_error, state_embedding)

        # 焦虑信号
        anxiety = self._compute_anxiety(current_prompt)

        # 整合
        total_drive, emotion_state = self.integrator.integrate(
            surprise.item(),
            boredom.item(),
            anxiety.item(),
        )

        return total_drive, emotion_state

    def _compute_anxiety(self, prompt: str) -> torch.Tensor:
        """计算焦虑信号。"""
        if self._anxiety_lightweight:
            # 轻量模式：基于多次采样文本的差异度
            try:
                texts = self.llm.generate_with_uncertainty(
                    prompt, n_samples=self._anxiety_n_samples, temperature=1.0
                )
                return self.anxiety_signal.compute_from_texts(texts)
            except Exception:
                return torch.tensor(0.0)
        else:
            # 完整模式：基于概率分布的分歧度
            try:
                predictions = []
                for _ in range(self._anxiety_n_samples):
                    probs = self.llm.get_prediction_distribution(prompt)
                    predictions.append(probs)
                return self.anxiety_signal.compute_from_predictions(predictions)
            except Exception:
                return torch.tensor(0.0)
