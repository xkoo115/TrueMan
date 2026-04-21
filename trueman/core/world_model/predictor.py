"""世界模型：状态预测器，用于惊奇计算和行为预演。

基于MLP的状态预测器，预测给定当前状态和动作后的下一状态。
支持局部在线更新（惊奇驱动）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WorldModel(nn.Module):
    """世界模型：状态预测器。

    s_{t+1} = f(s_t, a_t)
    用于：惊奇计算 | 行为预演 | 焦虑评估
    """

    def __init__(self, state_dim: int, action_dim: int = 0, hidden_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim
        # action_dim=0时仅基于状态预测
        input_dim = state_dim + action_dim

        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        # 在线更新的优化器
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, state: torch.Tensor, action: torch.Tensor | None = None) -> torch.Tensor:
        """预测下一状态。

        Args:
            state: 当前状态嵌入 (batch, state_dim) 或 (state_dim,)
            action: 动作嵌入 (batch, action_dim) 或 (action_dim,)，可选

        Returns:
            预测的下一状态
        """
        if action is not None:
            x = torch.cat([state, action], dim=-1)
        else:
            x = state
        return self.predictor(x)

    def prediction_error(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """计算预测误差。

        Returns:
            MSE预测误差（标量）
        """
        predicted = self.forward(state, action)
        return F.mse_loss(predicted, next_state)

    def update(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor | None = None,
        weight: float = 1.0,
    ) -> float:
        """局部在线更新（惊奇驱动）。

        Args:
            state: 当前状态
            next_state: 实际下一状态
            action: 动作（可选）
            weight: 更新权重（情绪强度）

        Returns:
            更新后的预测误差
        """
        self.optimizer.zero_grad()
        loss = self.prediction_error(state, next_state, action) * weight
        loss.backward()
        self.optimizer.step()
        return loss.item()
