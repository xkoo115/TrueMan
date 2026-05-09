"""HOT-2 / 高阶表征：用 linear probe 检验中间层是否能解码"模型刚才的不确定性"。

方法：
1. 在 agent 跑一批 probe 时，捕获 layer L 的 hidden state 和真值 anxiety
2. 训练 linear probe: hidden -> anxiety
3. 5-fold CV AUC 作为 HOT-2 指标

如果 AUC > 0.65 显著大于 chance (0.5)，认为模型存在对自身不确定性的高阶表征。
"""

from __future__ import annotations

import numpy as np


def measure_hot2(hidden_states: np.ndarray, anxiety: np.ndarray,
                 threshold: float = 0.5) -> dict:
    """hidden_states: (N, hidden_dim); anxiety: (N,) in [0,1].

    Lazy-import sklearn so this module is importable in environments without
    sklearn; an informative dict is returned in that case.
    """
    if len(hidden_states) < 50:
        return {"insufficient": True, "n": len(hidden_states)}

    y = (anxiety > threshold).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return {"degenerate_labels": True}

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return {"unavailable": True, "reason": "sklearn not installed",
                "n_trials": int(len(hidden_states))}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    aucs = []
    for train_idx, test_idx in skf.split(hidden_states, y):
        clf = LogisticRegression(max_iter=1000, C=0.1)
        clf.fit(hidden_states[train_idx], y[train_idx])
        proba = clf.predict_proba(hidden_states[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], proba))

    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "n_trials": int(len(hidden_states)),
        "fold_aucs": [float(a) for a in aucs],
    }
