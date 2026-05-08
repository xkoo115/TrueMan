"""HOT-1 测量：meta-d' / d' ratio (Maniscalco-Lau 2012)。

需要每条 probe 同时给出：
  - is_correct: 0/1   答案正确性（由独立 judge 判定）
  - confidence: float 模型自报置信度 OR anxiety 倒数

工作流：
1. 对每条 probe 调用 agent.step() 获取 response 和 emotion
2. 用 NLI / keyword judge 评判 response 是否包含正确答案 → is_correct
3. 用 1 - emotion.anxiety 作为 confidence（也可改用 logprob）
4. 调用 harness.stats.meta_d_prime 计算
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from experiments.v2_ambitious.harness.stats import meta_d_prime, MetaDResult


def keyword_judge(response: str, ground_truth: str | None) -> int | None:
    if not ground_truth:
        return None
    return int(ground_truth.lower() in response.lower())


def measure_hot1(
    agent,
    probe_set: list[dict],
    judge: Callable[[str, str | None], int | None] = keyword_judge,
) -> dict:
    """H1 主指标。"""
    correct, confidence = [], []
    raw = []
    for item in probe_set:
        if item.get("ground_truth") is None:
            continue  # 跳过本质不可判对错的 probe
        resp, emo = agent.step(item["prompt"])
        c = judge(resp, item["ground_truth"])
        if c is None:
            continue
        correct.append(c)
        confidence.append(1.0 - float(emo.anxiety))
        raw.append({"id": item.get("id"), "correct": c, "confidence": confidence[-1],
                    "anxiety": emo.anxiety, "response": resp})

    if len(correct) < 20:
        return {"insufficient_trials": True, "n": len(correct)}

    res: MetaDResult = meta_d_prime(np.array(correct), np.array(confidence))
    return {
        "d_prime": res.d_prime,
        "meta_d_prime": res.meta_d_prime,
        "m_ratio": res.m_ratio,
        "auc_type2": res.auc_type2,
        "n_trials": res.n_trials,
        "raw": raw,
    }
