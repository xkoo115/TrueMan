"""在线蒸馏：将多个LoRA专家的知识蒸馏到一个紧凑的共享适配器。

目的：当专家池中积累了多个专家后，通过蒸馏合并为一个更紧凑的适配器，
减少推理开销，同时保留各专家的核心知识。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from pathlib import Path

from trueman.core.plasticity.lora_pool import DynamicLoRAPool
from trueman.core.llm_backend import LLMBackend
from trueman.core.memory.episodic import EpisodicMemory
from trueman.utils.logging import EmotionLogger

logger = EmotionLogger("trueman.distill")


class OnlineDistill:
    """在线蒸馏：将多个LoRA专家蒸馏到一个共享适配器。

    流程：
    1. 从情景记忆采样训练数据
    2. 用各专家的输出作为软标签（teacher）
    3. 训练一个紧凑的共享适配器拟合软标签（student）
    4. 用蒸馏后的适配器替换原有专家
    """

    def __init__(
        self,
        lora_pool: DynamicLoRAPool,
        llm: LLMBackend,
        memory: EpisodicMemory,
        distill_rank: int = 8,
    ):
        self.lora_pool = lora_pool
        self.llm = llm
        self.memory = memory
        self.distill_rank = distill_rank

    def distill(
        self,
        expert_ids: list[int] | None = None,
        n_samples: int = 200,
        max_steps: int = 500,
        temperature: float = 2.0,
        learning_rate: float = 1e-4,
    ) -> str | None:
        """执行在线蒸馏。

        Args:
            expert_ids: 要蒸馏的专家ID列表，None则蒸馏所有专家
            n_samples: 采样训练数据量
            max_steps: 最大蒸馏训练步数
            temperature: 蒸馏温度（越高软标签越平滑）
            learning_rate: 学习率

        Returns:
            蒸馏后适配器的保存路径，失败返回None
        """
        if expert_ids is None:
            expert_ids = list(self.lora_pool.experts.keys())

        if len(expert_ids) < 2:
            logger.log_warning("TOO_FEW_EXPERTS", f"仅{len(expert_ids)}个专家，无需蒸馏")
            return None

        logger.log_learning_event("distill_start", {"n_experts": len(expert_ids)})

        # 采样训练数据
        traces = self.memory.weighted_sample(n=n_samples)
        if not traces:
            logger.log_warning("NO_DATA", "无训练数据用于蒸馏")
            return None

        # 收集各专家的软标签
        soft_labels = self._collect_soft_labels(traces, expert_ids, temperature)

        # 训练蒸馏适配器
        adapter_path = self._train_distill_adapter(
            traces, soft_labels, max_steps, learning_rate, temperature
        )

        if adapter_path:
            logger.log_learning_event("distill_complete", {"path": adapter_path})
        return adapter_path

    def _collect_soft_labels(
        self,
        traces,
        expert_ids: list[int],
        temperature: float,
    ) -> dict[int, list[torch.Tensor]]:
        """收集各专家在训练数据上的软标签输出。"""
        soft_labels = {}

        for expert_id in expert_ids:
            if expert_id not in self.lora_pool.experts:
                continue

            # 激活该专家
            self.lora_pool.set_active_experts([expert_id])

            expert_outputs = []
            for trace in traces[:50]:  # 限制计算量
                try:
                    probs = self.llm.get_prediction_distribution(trace.observation)
                    # 温度缩放
                    soft = F.softmax(probs / temperature, dim=-1)
                    expert_outputs.append(soft)
                except Exception:
                    continue

            soft_labels[expert_id] = expert_outputs

        return soft_labels

    def _train_distill_adapter(
        self,
        traces,
        soft_labels: dict,
        max_steps: int,
        learning_rate: float,
        temperature: float,
    ) -> str | None:
        """训练蒸馏适配器。"""
        # 简化实现：使用KL散度蒸馏
        # 完整实现需要创建student适配器并训练
        # 这里返回None表示需要完整PEFT训练流程
        logger.log_warning("DISTILL_NOT_IMPLEMENTED", "在线蒸馏完整训练流程待实现")
        return None
