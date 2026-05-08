"""Φ (IIT) 近似：在子模块上估计信息整合度。

完整 Φ 在 LLM 整层不可计算（O(2^n) 复杂度），但 IIT 4.0 提供了
近似量 G_max 和 ΦR (Mediano et al.) 可用于子模块。

我们对 5 个组件（perception / homeostasis / memory / policy / world model）
之间的 mutual information 进行配对估计，构造邻接矩阵；ΦR ≈ 系统级 MI -
任意 bipartition 的最小 MI。

简化版本：用每个组件的当前激活作为代表向量，估计 pairwise MI 用 KSG
或 binning 方法。
"""

from __future__ import annotations

import itertools

import numpy as np


def pairwise_mi(X: np.ndarray, Y: np.ndarray, bins: int = 16) -> float:
    """二维历史的 mutual information，binning 估计。"""
    if len(X) < 50:
        return 0.0
    hist, _, _ = np.histogram2d(X, Y, bins=bins)
    p_xy = hist / hist.sum()
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)
    nz = p_xy > 0
    return float((p_xy[nz] * np.log2(p_xy[nz] / (p_x @ p_y)[nz])).sum())


def measure_phi_approx(component_signals: dict[str, np.ndarray]) -> dict:
    """component_signals: {组件名: 1-D 激活时序}。返回 ΦR 近似。"""
    names = list(component_signals.keys())
    n = len(names)
    if n < 2:
        return {"insufficient_components": True}

    mi_matrix = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        mi = pairwise_mi(component_signals[names[i]], component_signals[names[j]])
        mi_matrix[i, j] = mi
        mi_matrix[j, i] = mi

    total_mi = float(mi_matrix.sum() / 2)

    # Bipartition：找最小贡献分割
    min_cut_mi = total_mi
    for r in range(1, n):
        for partition in itertools.combinations(range(n), r):
            inside = list(partition)
            outside = [k for k in range(n) if k not in inside]
            cut = sum(mi_matrix[i, j] for i in inside for j in outside)
            if cut < min_cut_mi:
                min_cut_mi = cut

    phi_r = total_mi - min_cut_mi
    return {
        "phi_r_approx": phi_r,
        "total_mi": total_mi,
        "min_cut_mi": min_cut_mi,
        "n_components": n,
    }
