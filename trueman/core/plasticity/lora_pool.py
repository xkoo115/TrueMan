"""动态LoRA专家池：管理多个LoRA专家的创建/删除/路由/推理。

关键设计：
- 通过PEFT标准接口管理适配器，不直接修改base_model权重
- 支持共享子空间（通过add_weighted_adapter合并）
- 专家元数据独立存储
- 容量满时淘汰最旧低频专家
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch

from trueman.core.config import LoRAConfig
from trueman.core.plasticity.lora_gate import NeuroLORAGate
from trueman.core.plasticity.hot_loader import HotLoader
from trueman.core.plasticity.lora_trainer import LoRATrainer
from trueman.core.memory.thought_trace import ThoughtTrace
from trueman.core.llm_backend import LLMBackend
from trueman.utils.logging import EmotionLogger

logger = EmotionLogger("trueman.lora")


@dataclass
class ExpertMetadata:
    """LoRA专家元数据。"""
    expert_id: int
    adapter_path: str
    creation_time: float
    access_count: int = 0
    rank: int = 16
    domain_tag: str = ""


class DynamicLoRAPool:
    """动态LoRA专家池。

    管理多个LoRA适配器的创建、删除、路由和推理。
    """

    def __init__(
        self,
        model,
        llm: LLMBackend,
        config: LoRAConfig,
        hidden_size: int,
    ):
        self.model = model
        self.llm = llm
        self.config = config

        # 路由器
        self.gate = NeuroLORAGate(
            hidden_size=hidden_size,
            max_experts=config.max_experts,
            orthogonality_weight=config.orthogonality_weight,
        )

        # 热加载器
        self.hot_loader = HotLoader(model)

        # 训练器
        self.trainer = LoRATrainer(model, llm, config)

        # 专家元数据
        self.experts: dict[int, ExpertMetadata] = {}
        self._next_expert_id = 0

    def add_expert(
        self,
        traces: list[ThoughtTrace],
        domain_tag: str = "",
        max_steps: int = 100,
    ) -> int | None:
        """训练并添加新专家。

        Args:
            traces: 训练数据
            domain_tag: 领域标签
            max_steps: 最大训练步数

        Returns:
            新专家ID，失败返回None
        """
        expert_id = self._next_expert_id

        # 训练
        adapter_path = self.trainer.train(
            traces, expert_id=expert_id, max_steps=max_steps
        )
        if adapter_path is None:
            return None

        # 加载到模型
        adapter_name = f"expert_{expert_id}"
        if not self.hot_loader.load(adapter_name, adapter_path):
            return None

        # 存储元数据
        metadata = ExpertMetadata(
            expert_id=expert_id,
            adapter_path=adapter_path,
            creation_time=time.time(),
            rank=self.config.rank,
            domain_tag=domain_tag,
        )
        self.experts[expert_id] = metadata
        self._next_expert_id += 1

        # 容量淘汰
        if len(self.experts) > self.config.max_experts:
            self._prune_oldest()

        logger.log_lora_event("add_expert", expert_id=expert_id, total=len(self.experts))
        return expert_id

    def route(self, context_embedding: torch.Tensor) -> list[tuple[int, float]]:
        """根据上下文嵌入路由到激活的专家。

        Args:
            context_embedding: 上下文嵌入向量

        Returns:
            [(expert_id, weight), ...] 激活的专家列表
        """
        active = self.gate.get_active_experts(context_embedding, threshold=0.1)

        # 只返回存在的专家
        result = []
        for expert_id, weight in active:
            if expert_id in self.experts:
                self.experts[expert_id].access_count += 1
                result.append((expert_id, weight))

        return result

    def set_active_experts(self, expert_ids: list[int]) -> bool:
        """设置当前活跃的专家（通过PEFT的set_adapter）。

        Args:
            expert_ids: 要激活的专家ID列表

        Returns:
            是否设置成功
        """
        if not expert_ids:
            return True

        # 使用第一个专家作为主适配器
        primary_id = expert_ids[0]
        adapter_name = f"expert_{primary_id}"
        return self.hot_loader.set_active(adapter_name)

    def remove_expert(self, expert_id: int) -> bool:
        """移除一个专家。"""
        if expert_id not in self.experts:
            return False

        adapter_name = f"expert_{expert_id}"
        success = self.hot_loader.delete(adapter_name)
        if success:
            del self.experts[expert_id]
            logger.log_lora_event("remove_expert", expert_id=expert_id)
        return success

    def export_expert(self, expert_id: int, export_path: str) -> bool:
        """导出专家到指定路径。"""
        if expert_id not in self.experts:
            return False
        metadata = self.experts[expert_id]
        try:
            import shutil
            shutil.copytree(metadata.adapter_path, export_path, dirs_exist_ok=True)
            return True
        except Exception:
            return False

    def import_expert(self, adapter_path: str, domain_tag: str = "") -> int | None:
        """导入外部专家。"""
        expert_id = self._next_expert_id
        adapter_name = f"expert_{expert_id}"

        if not self.hot_loader.load(adapter_name, adapter_path):
            return None

        metadata = ExpertMetadata(
            expert_id=expert_id,
            adapter_path=adapter_path,
            creation_time=time.time(),
            rank=self.config.rank,
            domain_tag=domain_tag,
        )
        self.experts[expert_id] = metadata
        self._next_expert_id += 1
        return expert_id

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    def _prune_oldest(self) -> None:
        """淘汰最旧且访问频率最低的专家。"""
        if not self.experts:
            return

        # 按创建时间*访问频率排序，淘汰得分最低的
        scored = [
            (eid, meta.creation_time * (meta.access_count + 1))
            for eid, meta in self.experts.items()
        ]
        scored.sort(key=lambda x: x[1])

        # 淘汰到容量以下
        while len(self.experts) > self.config.max_experts and scored:
            lowest_id, _ = scored[0]
            self.remove_expert(lowest_id)
            scored.pop(0)
