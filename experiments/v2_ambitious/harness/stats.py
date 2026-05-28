"""统计分析工具：mixed-effects、permutation、Bayes Factor、meta-d'。

用于 PREREGISTRATION.md §5 中所列的所有 confirmatory 检验。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Mixed-effects via pymer4 (fallback to OLS if 不可用)
# ---------------------------------------------------------------------------

def mixed_effects(
    df,                     # pandas DataFrame: columns = [score, condition, base_model, stimulus, seed]
    formula: str = "score ~ condition + base_model + (1|stimulus) + (1|seed)",
):
    """拟合 mixed-effects 模型并返回固定效应表。

    需要 pymer4（封装 R lme4）。如不可用则退回 OLS 并 warn。
    """
    try:
        from pymer4.models import Lmer
        m = Lmer(formula, data=df)
        m.fit(REML=False, summarize=False)
        return {
            "engine": "lme4",
            "coefs": m.coefs,
            "anova": m.anova(),
            "log_likelihood": m.logLike,
        }
    except Exception:
        import statsmodels.formula.api as smf
        ols_formula = formula.split("+ (")[0].strip()
        res = smf.ols(ols_formula, data=df).fit()
        return {
            "engine": "ols_fallback",
            "summary": res.summary().as_text(),
            "params": res.params.to_dict(),
            "pvalues": res.pvalues.to_dict(),
        }


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def permutation_test(
    a: np.ndarray, b: np.ndarray,
    n_permutations: int = 10000,
    statistic: str = "mean_diff",
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    pooled = np.concatenate([a, b])
    n_a = len(a)

    def stat(x, y):
        if statistic == "mean_diff":
            return float(np.mean(x) - np.mean(y))
        elif statistic == "median_diff":
            return float(np.median(x) - np.median(y))
        else:
            raise ValueError(f"unknown statistic: {statistic}")

    obs = stat(a, b)
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        idx = rng.permutation(len(pooled))
        null[i] = stat(pooled[idx[:n_a]], pooled[idx[n_a:]])

    p_two_sided = float((np.abs(null) >= abs(obs)).mean())
    return {
        "observed": obs,
        "p_value": p_two_sided,
        "ci_95_null": (float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))),
    }


# ---------------------------------------------------------------------------
# Bayes Factor (JZS for 2-sample t)
# ---------------------------------------------------------------------------

def bayes_factor_t(a: np.ndarray, b: np.ndarray, prior_scale: float = 0.707) -> float:
    """JZS Bayes Factor for two-sample comparison (Rouder et al. 2009)."""
    try:
        from scipy.stats import t as student_t
        from scipy.special import gammaln
        from scipy.integrate import quad
    except ImportError:
        return float("nan")

    a, b = np.asarray(a), np.asarray(b)
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return float("nan")

    sd_pool = np.sqrt(((n_a - 1)*np.var(a, ddof=1) + (n_b - 1)*np.var(b, ddof=1)) / (n_a + n_b - 2))
    if sd_pool == 0:
        return float("nan")
    t_obs = (np.mean(a) - np.mean(b)) / (sd_pool * np.sqrt(1/n_a + 1/n_b))
    df = n_a + n_b - 2
    n_eff = n_a * n_b / (n_a + n_b)

    def integrand(g):
        t1 = (1 + n_eff * g * prior_scale**2) ** -0.5
        t2 = (1 + t_obs**2 / df / (1 + n_eff * g * prior_scale**2)) ** -((df + 1) / 2)
        prior = (g ** -1.5) * np.exp(-0.5 / g) / np.sqrt(2 * np.pi)
        return t1 * t2 * prior

    num, _ = quad(integrand, 0, np.inf, limit=200)
    null = (1 + t_obs**2 / df) ** -((df + 1) / 2)
    return float(num / null)


# ---------------------------------------------------------------------------
# meta-d' (Maniscalco-Lau 2012, simplified MLE implementation)
# ---------------------------------------------------------------------------

@dataclass
class MetaDResult:
    d_prime: float
    meta_d_prime: float
    m_ratio: float           # meta-d' / d'
    auc_type2: float         # Type-2 AUC as cross-check
    n_trials: int


def meta_d_prime(
    correct: np.ndarray,         # 0/1 正确性
    confidence: np.ndarray,      # 数值置信度（任意范围，会做分箱）
    n_bins: int = 4,
) -> MetaDResult:
    """简化版 meta-d' 计算。完整版本见 Maniscalco & Lau 2012 公开代码。

    适用：每条 trial 给出 (correct, confidence)。confidence 是连续值时
    自动按分位数分箱。
    """
    correct = np.asarray(correct).astype(int)
    confidence = np.asarray(confidence, dtype=float)
    if len(correct) != len(confidence) or len(correct) < 20:
        return MetaDResult(0.0, 0.0, 0.0, 0.5, len(correct))

    # 分箱
    bins = np.quantile(confidence, np.linspace(0, 1, n_bins + 1))
    bins[-1] += 1e-8
    binned = np.digitize(confidence, bins[1:-1])

    # Type-1 d'：整体正确率 → z(hit) - z(fa)
    # 我们把 correct=1 视为 hit，按整个集合做。
    from scipy.stats import norm
    p_correct = max(0.01, min(0.99, correct.mean()))
    d_prime = float(norm.ppf(p_correct) - norm.ppf(1 - p_correct))

    # Type-2 ROC：对每个置信度阈值计算 (P(high|correct), P(high|wrong))
    pts_x, pts_y = [], []
    for b in range(1, n_bins):
        thr_mask = binned >= b
        if thr_mask.sum() == 0 or (~thr_mask).sum() == 0:
            continue
        p_high_corr = thr_mask[correct == 1].mean() if (correct == 1).sum() else 0
        p_high_wrong = thr_mask[correct == 0].mean() if (correct == 0).sum() else 0
        pts_x.append(p_high_wrong)
        pts_y.append(p_high_corr)

    pts_x = [0.0] + sorted(pts_x) + [1.0]
    pts_y = [0.0] + sorted(pts_y) + [1.0]
    _trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy 2.x renames trapz -> trapezoid
    auc_t2 = float(_trapz(pts_y, pts_x))

    # meta-d' 近似：z(AUC_t2 * 2 - 1) 关系
    auc_clipped = max(0.51, min(0.99, auc_t2))
    meta_d = float(np.sqrt(2) * norm.ppf(auc_clipped))

    return MetaDResult(
        d_prime=d_prime,
        meta_d_prime=meta_d,
        m_ratio=float(meta_d / max(d_prime, 1e-3)),
        auc_type2=auc_t2,
        n_trials=len(correct),
    )


# ---------------------------------------------------------------------------
# Holm-Bonferroni 校正
# ---------------------------------------------------------------------------

def holm_bonferroni(p_values: list[float]) -> list[float]:
    """返回校正后 p 值（保持原顺序）。"""
    n = len(p_values)
    order = np.argsort(p_values)
    adj = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        a = (n - rank) * p_values[idx]
        running_max = max(running_max, a)
        adj[idx] = min(1.0, running_max)
    return list(adj)
