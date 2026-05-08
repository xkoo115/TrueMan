"""Perturbational Complexity Index (PCI) 类比测量。

Mashour 2020 在人脑上：发送 TMS 脉冲 → 测 EEG 响应的复杂度
(Lempel-Ziv) → 区分意识状态。

LLM 类比：
1. 在 agent 残差流中注入一个 "pulse"（随机方向，幅度 σ × hidden_std）
2. 观察后续 K 步 hidden state 的演化
3. 计算演化序列的 Lempel-Ziv 复杂度（先二值化）

预期：高 ΦR / 高内稳态活跃度的 agent，PCI 显著高于 frozen baseline，
且高于 noise 注入的 baseline。

如果该差异在 ≥ 3 个底模上一致出现，是支持 H 系列的强证据。
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch


def lempel_ziv_complexity(seq: np.ndarray) -> int:
    """对 0/1 序列计算 LZ76 complexity。"""
    s = "".join(str(int(x)) for x in seq.flatten())
    n = len(s)
    i, c = 0, 1
    while i < n:
        k = 1
        while i + k <= n and s[i:i+k] in s[:i+k-1]:
            k += 1
        c += 1
        i += k
    return c


def pci_score(post_pulse_states: np.ndarray, threshold: float = 0.0) -> dict:
    """post_pulse_states: (K, hidden_dim)。

    流程：取 hidden state 时间序列 → 沿 hidden_dim PCA 降维至 1D
    → 大于均值的位置标 1，否则标 0 → LZ complexity 归一化到 [0,1]。
    """
    if post_pulse_states.shape[0] < 5:
        return {"insufficient_pulses": True}
    flat = post_pulse_states - post_pulse_states.mean(axis=0, keepdims=True)
    # 主成分
    U, S, Vt = np.linalg.svd(flat, full_matrices=False)
    pc1 = (flat @ Vt[0])
    binary = (pc1 > threshold).astype(int)

    lz = lempel_ziv_complexity(binary)
    n = len(binary)
    h_max = n / np.log2(n) if n > 1 else 1
    return {
        "lz_complexity": lz,
        "pci_normalized": float(lz / h_max),
        "n_steps": n,
    }


def administer_pci(agent, layer_idx: int, n_pulses: int = 30,
                   pulse_sigma: float = 1.0, post_steps: int = 50) -> dict:
    """在指定层注入 n_pulses 次脉冲，每次记录后 post_steps 步 hidden state。"""
    # TODO(other model): 实现脉冲注入 + 状态记录的具体逻辑。
    # 提示：可复用 pillar1_mechanistic.causal_intervention.FeatureInjector，
    # 用随机方向代替学到的特征。
    return {"todo": "implement pulse injection loop"}
