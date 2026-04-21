"""神经调制门控路由：基于上下文嵌入选择激活的LoRA专家。"""

from __future__ import annotations

import torch
import torch.nn as nn


class NeuroLORAGate(nn.Module):
    """神经调制门控路由器。

    输入上下文嵌入，输出每个专家的权重。
    权重<0.1的专家被过滤（不激活）。
    包含正交性损失，防止专家退化。
    """

    def __init__(self, hidden_size: int, max_experts: int, orthogonality_weight: float = 0.1):
        super().__init__()
        self.context_encoder = nn.Linear(hidden_size, 128)
        self.neuromod_gate = nn.Linear(128, max_experts)
        self.orthogonality_weight = orthogonality_weight
        self.max_experts = max_experts

    def forward(self, context_embedding: torch.Tensor) -> torch.Tensor:
        """计算专家路由权重。

        Args:
            context_embedding: 上下文嵌入 (hidden_size,)

        Returns:
            专家权重 (max_experts,)，经softmax归一化
        """
        if context_embedding.dim() == 1:
            context_embedding = context_embedding.unsqueeze(0)

        context = torch.relu(self.context_encoder(context_embedding))
        raw_weights = self.neuromod_gate(context)
        weights = torch.softmax(raw_weights, dim=-1)

        return weights.squeeze(0)

    def get_active_experts(
        self, context_embedding: torch.Tensor, threshold: float = 0.1
    ) -> list[tuple[int, float]]:
        """获取激活的专家列表。

        Returns:
            [(expert_id, weight), ...] 权重>threshold的专家
        """
        weights = self.forward(context_embedding)
        active = []
        for idx in range(weights.shape[0]):
            w = weights[idx].item()
            if w > threshold:
                active.append((idx, w))
        return active

    def orthogonality_loss(self) -> torch.Tensor:
        """正交性损失：||WW^T - I||^2，防止专家退化。"""
        W = self.neuromod_gate.weight  # (max_experts, 128)
        WWt = W @ W.T  # (max_experts, max_experts)
        I = torch.eye(WWt.shape[0], device=WWt.device)
        return self.orthogonality_weight * ((WWt - I) ** 2).sum()
