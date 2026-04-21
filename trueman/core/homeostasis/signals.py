"""情绪信号计算：惊奇、无聊、焦虑。

惊奇 (Surprise): 基于LLM token预测误差，运行统计量归一化到[0,1]
无聊 (Boredom): 基于时间新颖度+状态多样性+学习进度，取反
焦虑 (Anxiety): 基于多次采样预测的分歧度，支持轻量化模式
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from trueman.core.config import (
    AgentConfig,
    SurpriseConfig,
    BoredomConfig,
    AnxietyConfig,
)


class SurpriseSignal:
    """惊奇信号：基于token预测误差的归一化度量。

    计算流程：
    1. 对token_logprobs取负均值（交叉熵近似）
    2. 通过运行统计量（指数移动平均）归一化
    3. sigmoid映射到[0,1]
    """

    def __init__(self, config: SurpriseConfig):
        self.threshold = config.threshold
        self.decay = config.decay
        self.running_mean = 0.0
        self.running_var = 1.0
        self._initialized = False

    def compute(self, token_logprobs: torch.Tensor) -> torch.Tensor:
        """计算惊奇信号值。

        Args:
            token_logprobs: 每个token的log概率，形状(seq_len,)

        Returns:
            归一化到[0,1]的惊奇信号值
        """
        if token_logprobs.numel() == 0:
            return torch.tensor(0.5)

        # 负对数概率的均值 = 预测误差
        prediction_error = -token_logprobs.mean().item()

        # 首次调用初始化运行统计量
        if not self._initialized:
            self.running_mean = prediction_error
            self.running_var = 1.0
            self._initialized = True
            return torch.tensor(0.5)

        # 归一化
        std = math.sqrt(self.running_var + 1e-8)
        normalized = (prediction_error - self.running_mean) / std

        # sigmoid映射到[0,1]，threshold控制灵敏度
        surprise = torch.sigmoid(torch.tensor(normalized - self.threshold))

        # 更新运行统计量
        self._update_stats(prediction_error)

        return surprise

    def _update_stats(self, error: float) -> None:
        """指数移动平均更新运行统计量。"""
        delta = error - self.running_mean
        self.running_mean += (1 - self.decay) * delta
        self.running_var = self.decay * self.running_var + (1 - self.decay) * delta * (error - self.running_mean)
        self.running_var = max(self.running_var, 1e-8)  # 防止方差退化


class BoredomSignal:
    """无聊信号：信息熵衰减和学习进度停滞的度量。

    无聊 = 1.0 - (temporal_novelty + state_diversity + learning_progress) / 3.0
    """

    def __init__(self, config: BoredomConfig):
        self.window_size = config.window
        self.temperature = config.temperature
        self.prediction_errors: deque[float] = deque(maxlen=config.window)
        self.state_embeddings: list[torch.Tensor] = []
        self._max_embeddings = config.window

    def compute(
        self,
        prediction_error: float,
        state_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """计算无聊信号值。

        Args:
            prediction_error: 当前预测误差（惊奇信号的原始值）
            state_embedding: 当前状态嵌入向量

        Returns:
            归一化到[0,1]的无聊信号值
        """
        self.prediction_errors.append(prediction_error)
        self.state_embeddings.append(state_embedding.detach().cpu())
        if len(self.state_embeddings) > self._max_embeddings:
            self.state_embeddings = self.state_embeddings[-self._max_embeddings:]

        # 窗口不足时返回中性值
        if len(self.prediction_errors) < 10:
            return torch.tensor(0.5)

        temporal_novelty = self._temporal_novelty()
        state_diversity = self._state_diversity()
        learning_progress = self._learning_progress()

        boredom = 1.0 - (temporal_novelty + state_diversity + learning_progress) / 3.0
        return torch.tensor(max(0.0, min(1.0, boredom)))

    def _temporal_novelty(self) -> float:
        """时间新颖度：最近预测误差的方差。"""
        if len(self.prediction_errors) < 2:
            return 1.0
        errors = np.array(list(self.prediction_errors)[-20:])
        variance = float(np.var(errors))
        return min(variance / self.temperature, 1.0)

    def _state_diversity(self) -> float:
        """状态多样性：窗口内状态嵌入的pairwise平均距离。"""
        if len(self.state_embeddings) < 2:
            return 1.0
        # 采样最多20个嵌入计算距离（控制计算量）
        embeddings = self.state_embeddings[-20:]
        try:
            stacked = torch.stack(embeddings)
            pairwise_dist = torch.pdist(stacked).mean().item()
            return min(pairwise_dist / (self.temperature + 1e-8), 1.0)
        except Exception:
            return 0.5

    def _learning_progress(self) -> float:
        """学习进度：最近误差与之前误差均值之差。"""
        if len(self.prediction_errors) < 10:
            return 1.0
        errors = np.array(list(self.prediction_errors))
        recent = errors[-5:].mean()
        older = errors[-10:-5].mean()
        progress = abs(float(older - recent))
        return min(progress / (self.temperature + 1e-8), 1.0)


class AnxietySignal:
    """焦虑信号：认知失调和预测不确定性的度量。

    基于多次采样预测的分歧度计算：
    anxiety = (pairwise_KL + predictive_entropy + output_variance) / 3

    轻量模式：使用temperature扰动进行多次前向传播，开销约1.5x
    """

    def __init__(self, config: AnxietyConfig):
        self.n_samples = config.n_samples
        self.lightweight = config.lightweight
        self.temperature_scale = config.temperature_scale

    def compute_from_predictions(self, predictions: list[torch.Tensor]) -> torch.Tensor:
        """从多次预测的概率分布计算焦虑信号。

        Args:
            predictions: 多次预测的概率分布列表，每个形状(vocab_size,)

        Returns:
            归一化到[0,1]的焦虑信号值
        """
        if len(predictions) < 2:
            return torch.tensor(0.0)

        # 截断到top-k减少计算量
        k = min(1000, predictions[0].shape[0])
        truncated = [self._top_k_truncate(p, k) for p in predictions]

        disagreement = self._pairwise_disagreement(truncated)
        entropy = self._predictive_entropy(truncated)
        variance = self._output_variance(truncated)

        anxiety = (disagreement + entropy + variance) / 3.0
        anxiety_val = anxiety if isinstance(anxiety, float) else anxiety.item()
        return torch.tensor(max(0.0, min(1.0, anxiety_val)))

    def compute_from_texts(self, texts: list[str]) -> torch.Tensor:
        """从多次采样的文本计算焦虑信号（基于文本差异度）。

        当无法获取概率分布时，使用文本相似度作为分歧度的代理。
        """
        if len(texts) < 2:
            return torch.tensor(0.0)

        # 基于字符级n-gram相似度估计分歧
        similarities = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sim = self._text_similarity(texts[i], texts[j])
                similarities.append(sim)

        avg_similarity = sum(similarities) / len(similarities) if similarities else 1.0
        anxiety = 1.0 - avg_similarity  # 相似度低 → 焦虑高
        return torch.tensor(max(0.0, min(1.0, anxiety)))

    @staticmethod
    def _top_k_truncate(probs: torch.Tensor, k: int) -> torch.Tensor:
        """截断到top-k概率，剩余质量分配给一个虚拟token。"""
        topk = torch.topk(probs, k)
        truncated = torch.zeros(k + 1)
        truncated[:k] = topk.values
        truncated[k] = 1.0 - topk.values.sum()  # 剩余质量
        # 重新归一化
        truncated = truncated / (truncated.sum() + 1e-8)
        return truncated

    @staticmethod
    def _pairwise_disagreement(predictions: list[torch.Tensor]) -> float:
        """计算预测之间的pairwise KL散度。"""
        total = 0.0
        count = 0
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                p = predictions[i] + 1e-8
                q = predictions[j] + 1e-8
                kl = F.kl_div(p.log(), q, reduction='sum').item()
                total += abs(kl)
                count += 1
        return total / max(count, 1)

    @staticmethod
    def _predictive_entropy(predictions: list[torch.Tensor]) -> float:
        """计算预测概率分布的熵。"""
        avg_pred = torch.stack(predictions).mean(dim=0) + 1e-8
        entropy = -(avg_pred * avg_pred.log()).sum().item()
        # 归一化：最大熵为log(k)
        max_entropy = math.log(avg_pred.shape[0])
        return min(entropy / (max_entropy + 1e-8), 1.0) if max_entropy > 0 else 0.0

    @staticmethod
    def _output_variance(predictions: list[torch.Tensor]) -> float:
        """计算预测输出的方差。"""
        var = torch.stack(predictions).var(dim=0).mean().item()
        return min(var * 100, 1.0)  # 缩放方差到合理范围

    @staticmethod
    def _text_similarity(text1: str, text2: str) -> float:
        """基于字符3-gram的Jaccard相似度。"""
        def ngrams(text: str, n: int = 3) -> set[str]:
            return {text[i:i+n] for i in range(len(text) - n + 1)} if len(text) >= n else {text}

        ng1, ng2 = ngrams(text1), ngrams(text2)
        if not ng1 and not ng2:
            return 1.0
        intersection = len(ng1 & ng2)
        union = len(ng1 | ng2)
        return intersection / (union + 1e-8)
