"""统计检验模块：Bootstrap置信区间、效应量、多重比较校正。

为TrueMan实验提供严谨的统计推断支持：
- Bootstrap 95%置信区间
- 配对t检验和非参数替代
- Cohen's d效应量
- Bonferroni多重比较校正
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class StatisticalReport:
    """单次统计检验的完整报告。"""
    metric_name: str
    group_a_name: str
    group_b_name: str
    mean_a: float
    std_a: float
    mean_b: float
    std_b: float
    difference: float
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    p_value: float | None = None
    cohens_d: float = 0.0
    n_a: int = 0
    n_b: int = 0
    test_method: str = ""
    significant: bool = False

    def to_dict(self) -> dict:
        return {
            "metric": self.metric_name,
            f"{self.group_a_name}_mean": round(self.mean_a, 4),
            f"{self.group_a_name}_std": round(self.std_a, 4),
            f"{self.group_b_name}_mean": round(self.mean_b, 4),
            f"{self.group_b_name}_std": round(self.std_b, 4),
            "difference": round(self.difference, 4),
            "ci_95": [round(self.ci_lower, 4), round(self.ci_upper, 4)],
            "p_value": round(self.p_value, 4) if self.p_value is not None else None,
            "cohens_d": round(self.cohens_d, 4),
            "significant": self.significant,
            "test_method": self.test_method,
        }


def bootstrap_ci(
    data: list[float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap置信区间。

    Args:
        data: 样本数据
        n_resamples: 重采样次数
        confidence: 置信水平

    Returns:
        (lower, upper) 置信区间
    """
    arr = np.array(data)
    n = len(arr)
    if n < 2:
        return (arr[0] if n == 1 else 0.0, arr[0] if n == 1 else 0.0)

    boot_means = np.empty(n_resamples)
    rng = np.random.default_rng(42)
    for i in range(n_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = sample.mean()

    alpha = 1.0 - confidence
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (lower, upper)


def cohens_d(group_a: list[float], group_b: list[float]) -> float:
    """计算Cohen's d效应量。

    <0.2: 微小, 0.2-0.5: 小, 0.5-0.8: 中, >0.8: 大
    """
    a = np.array(group_a)
    b = np.array(group_b)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0

    var1 = a.var(ddof=1)
    var2 = b.var(ddof=1)
    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-8:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def paired_ttest(group_a: list[float], group_b: list[float]) -> tuple[float, float] | None:
    """配对t检验。

    Returns:
        (t_statistic, p_value) 或 None（数据不足时）
    """
    if not HAS_SCIPY or len(group_a) < 2 or len(group_b) < 2:
        return None
    try:
        result = scipy_stats.ttest_rel(group_a, group_b)
        return (float(result.statistic), float(result.pvalue))
    except Exception:
        return None


def welch_ttest(group_a: list[float], group_b: list[float]) -> tuple[float, float] | None:
    """Welch t检验（非等方差）。

    Returns:
        (t_statistic, p_value) 或 None
    """
    if not HAS_SCIPY or len(group_a) < 2 or len(group_b) < 2:
        return None
    try:
        result = scipy_stats.ttest_ind(group_a, group_b, equal_var=False)
        return (float(result.statistic), float(result.pvalue))
    except Exception:
        return None


def wilcoxon_test(group_a: list[float], group_b: list[float]) -> tuple[float, float] | None:
    """Wilcoxon符号秩检验（非参数替代）。

    Returns:
        (statistic, p_value) 或 None
    """
    if not HAS_SCIPY or len(group_a) < 2:
        return None
    try:
        diff = np.array(group_a) - np.array(group_b)
        if np.all(diff == 0):
            return None
        result = scipy_stats.wilcoxon(diff)
        return (float(result.statistic), float(result.pvalue))
    except Exception:
        return None


def bonferroni_correction(p_values: list[float | None], alpha: float = 0.05) -> list[bool]:
    """Bonferroni多重比较校正。

    Args:
        p_values: 原始p值列表（None视为不显著）
        alpha: 显著性水平

    Returns:
        校正后的显著性列表
    """
    valid = [p for p in p_values if p is not None]
    n_tests = max(len(valid), 1)
    corrected_alpha = alpha / n_tests
    return [p is not None and p < corrected_alpha for p in p_values]


def compare_groups(
    group_a: list[float],
    group_b: list[float],
    metric_name: str,
    group_a_name: str = "A",
    group_b_name: str = "B",
    alpha: float = 0.05,
) -> StatisticalReport:
    """完整的两组比较，自动选择检验方法。

    优先使用配对t检验，数据不足时使用Welch t检验。
    """
    a = np.array(group_a) if group_a else np.array([0.0])
    b = np.array(group_b) if group_b else np.array([0.0])

    mean_a, std_a = float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0
    mean_b, std_b = float(b.mean()), float(b.std(ddof=1)) if len(b) > 1 else 0.0
    diff = mean_a - mean_b

    ci_lower, ci_upper = bootstrap_ci(list(a))

    p_value = None
    test_method = "none"

    if len(a) >= 2 and len(b) >= 2:
        if len(a) == len(b):
            result = paired_ttest(list(a), list(b))
            test_method = "paired_ttest"
        else:
            result = welch_ttest(list(a), list(b))
            test_method = "welch_ttest"

        if result is not None:
            p_value = result[1]
        else:
            result2 = wilcoxon_test(list(a), list(b))
            if result2 is not None:
                p_value = result2[1]
                test_method = "wilcoxon"

    d = cohens_d(list(a), list(b))
    significant = p_value is not None and p_value < alpha

    return StatisticalReport(
        metric_name=metric_name,
        group_a_name=group_a_name,
        group_b_name=group_b_name,
        mean_a=mean_a,
        std_a=std_a,
        mean_b=mean_b,
        std_b=std_b,
        difference=diff,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        cohens_d=d,
        n_a=len(a),
        n_b=len(b),
        test_method=test_method,
        significant=significant,
    )


def compare_multiple_conditions(
    all_conditions: dict[str, list[float]],
    metric_name: str,
    reference_name: str = "trueman",
    alpha: float = 0.05,
) -> list[StatisticalReport]:
    """多条件对比：参考条件 vs 所有其他条件。

    Args:
        all_conditions: {condition_name: [repeat_values]}
        metric_name: 指标名
        reference_name: 参考条件名
        alpha: 显性水平

    Returns:
        每个对比的StatisticalReport列表
    """
    if reference_name not in all_conditions:
        return []

    ref_data = all_conditions[reference_name]
    reports = []
    for cond_name, cond_data in all_conditions.items():
        if cond_name == reference_name:
            continue
        report = compare_groups(
            ref_data, cond_data, metric_name,
            group_a_name=reference_name,
            group_b_name=cond_name,
            alpha=alpha,
        )
        reports.append(report)

    p_values = [r.p_value for r in reports]
    corrected = bonferroni_correction(p_values, alpha)
    for report, sig in zip(reports, corrected):
        report.significant = sig

    return reports
