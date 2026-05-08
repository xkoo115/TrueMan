"""RPT-1：层间 reentrant signature。

LLM 是 feed-forward 的，但在 introspection 触发时，agent 会进入
多步推理（IntrospectionPolicy 内部多次调 generate）。我们度量同一
prompt 在 introspection 路径下产生的层间激活相关性，与 baseline
路径对比。

简单实现：对每条 probe 同时跑 (base_policy, introspection_policy)，
比较两次 hidden state 的相似度。差异越大 → 越像"重新加工"。
"""

from __future__ import annotations

import numpy as np


def measure_rpt1(hidden_base: list[np.ndarray], hidden_introspect: list[np.ndarray]) -> dict:
    """两个长度相同的 hidden state 列表，逐 trial 计算 cosine 相似度。
    平均 1 - cos 越大 → 内省路径与 base 路径越分离 → reentrant 信号越强。
    """
    if not hidden_base or len(hidden_base) != len(hidden_introspect):
        return {"insufficient": True}

    diffs = []
    for a, b in zip(hidden_base, hidden_introspect):
        a, b = np.asarray(a).flatten(), np.asarray(b).flatten()
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        diffs.append(1 - cos)
    return {
        "reentrance_score": float(np.mean(diffs)),
        "std": float(np.std(diffs)),
        "n_trials": len(diffs),
    }
