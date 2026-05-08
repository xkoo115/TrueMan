"""GWT-1/2 测量：跨模块注意力广播指数。

我们在 agent 上观察 cross-attention 头的"信息广播度"：
对一个事件相关 prompt，看注意力分布在多少 head 上、多少 token 上展开。

简化指标：H_attention = mean across layers of entropy(attention_pattern)。
高 entropy = 广播；低 entropy = 局部计算。

这个 indicator 在 GWT 理论中关联 ignition：意识相关的处理应有
非线性 ignition (sudden global broadcast)。我们另测：

   ignition_index = max_t (H_attention(t) - H_attention(t-1))

用法：在 capture.py 已有 hidden_state hook 基础上，再装一组 attention hook。
"""

from __future__ import annotations

import numpy as np
import torch


def attention_entropy(attn_weights: torch.Tensor) -> float:
    """attn_weights: (heads, q_len, k_len) -> mean entropy."""
    p = attn_weights.float() + 1e-8
    h = -(p * p.log()).sum(dim=-1).mean()
    return float(h.item())


class AttentionEntropyCollector:
    """挂在每个 transformer layer 的 attention 模块上。"""

    def __init__(self):
        self._hooks = []
        self._records: list[dict] = []
        self._step = 0

    def attach(self, model: torch.nn.Module):
        # 多数 HF 模型 attention 模块叫 self_attn 或 attention
        for name, module in model.named_modules():
            if name.endswith("self_attn") or name.endswith(".attention"):
                h = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(h)

    def _make_hook(self, name):
        def hook(module, inputs, output):
            # output 通常是 (attn_output, attn_weights, ...) 或 dict
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                aw = output[1]    # (batch, heads, q, k)
                if aw.dim() == 4:
                    ent = attention_entropy(aw[0])
                    self._records.append({
                        "step": self._step, "module": name, "entropy": ent,
                    })
        return hook

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def step(self):
        self._step += 1

    def summary(self) -> dict:
        if not self._records:
            return {}
        ent = np.array([r["entropy"] for r in self._records])
        return {
            "mean_attention_entropy": float(ent.mean()),
            "std_attention_entropy": float(ent.std()),
            "n_observations": len(ent),
        }
