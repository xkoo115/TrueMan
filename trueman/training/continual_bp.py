"""持续反向传播：定期重初始化低效用单元，防止塑性丧失。

依据 Dohare et al. (2023) 的持续反向传播方法。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ContinualBackprop:
    """持续反向传播：防止深度网络持续学习后丧失可塑性。

    定期重初始化低效用单元（utility低的神经元），
    保持网络的持续学习能力。
    """

    def __init__(self, model: nn.Module, replacement_rate: float = 0.001):
        self.model = model
        self.replacement_rate = replacement_rate
        self.unit_utility: dict[str, torch.Tensor] = {}

    def update_utility(self) -> None:
        """更新每个参数的单元效用估计。

        utility = |grad * param| 的指数移动平均。
        """
        for name, param in self.model.named_parameters():
            if param.grad is not None and param.dim() >= 2:
                # 计算每个输出单元的效用
                utility = (param.grad * param.data).abs().mean(dim=0)

                if name not in self.unit_utility:
                    self.unit_utility[name] = utility.detach()
                else:
                    self.unit_utility[name] = (
                        0.99 * self.unit_utility[name] + 0.01 * utility.detach()
                    )

    def maybe_reinit(self) -> int:
        """重初始化低效用单元。

        Returns:
            重初始化的单元总数
        """
        total_reinit = 0

        for name, param in self.model.named_parameters():
            if name not in self.unit_utility or param.dim() < 2:
                continue

            utility = self.unit_utility[name]
            n_replace = max(1, int(self.replacement_rate * utility.shape[0]))

            # 找到效用最低的单元
            least_useful = torch.argsort(utility)[:n_replace]

            # 重初始化
            for idx in least_useful:
                nn.init.xavier_uniform_(param.data[idx:idx+1])

            total_reinit += n_replace

        return total_reinit
