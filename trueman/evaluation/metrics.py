"""评估指标：惊奇校准度、无聊检测AUC、焦虑预测ROC-AUC、塑性保持率。"""

from __future__ import annotations

import numpy as np
from typing import List


def surprise_calibration(
    surprise_values: List[float],
    actual_anomalies: List[bool],
) -> float:
    """惊奇校准度：惊奇信号与实际异常的Pearson相关系数。

    Args:
        surprise_values: 惊奇信号值序列
        actual_anomalies: 实际异常标记序列

    Returns:
        Pearson相关系数 [-1, 1]
    """
    if len(surprise_values) != len(actual_anomalies) or len(surprise_values) < 2:
        return 0.0

    x = np.array(surprise_values, dtype=float)
    y = np.array(actual_anomalies, dtype=float)

    corr = np.corrcoef(x, y)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def boredom_detection_accuracy(
    boredom_values: List[float],
    is_novel: List[bool],
    threshold: float = 0.5,
) -> float:
    """无聊检测准确率：无聊信号区分重复/新颖输入的准确率。

    Args:
        boredom_values: 无聊信号值序列
        is_novel: 是否为新颖输入的标记
        threshold: 无聊阈值

    Returns:
        准确率 [0, 1]
    """
    if len(boredom_values) != len(is_novel) or not boredom_values:
        return 0.0

    correct = 0
    for boredom, novel in zip(boredom_values, is_novel):
        predicted_bored = boredom > threshold
        actual_bored = not novel
        if predicted_bored == actual_bored:
            correct += 1

    return correct / len(boredom_values)


def anxiety_predictive_value(
    anxiety_values: List[float],
    subsequent_errors: List[bool],
) -> float:
    """焦虑预测值：焦虑信号预测后续错误的准确率。

    Args:
        anxiety_values: 焦虑信号值序列
        subsequent_errors: 后续是否出现错误的标记

    Returns:
        准确率 [0, 1]
    """
    if len(anxiety_values) != len(subsequent_errors) or not anxiety_values:
        return 0.0

    # 使用中位数作为阈值
    threshold = np.median(anxiety_values)
    correct = 0
    for anxiety, has_error in zip(anxiety_values, subsequent_errors):
        predicted_error = anxiety > threshold
        if predicted_error == has_error:
            correct += 1

    return correct / len(anxiety_values)


def plasticity_retention_rate(
    initial_learning_speed: float,
    current_learning_speed: float,
) -> float:
    """塑性保持率：连续学习后新任务学习速度与初始速度的比值。

    Args:
        initial_learning_speed: 初始学习速度
        current_learning_speed: 当前学习速度

    Returns:
        保持率 [0, +inf)，1.0表示完全保持
    """
    if initial_learning_speed <= 0:
        return 0.0
    return current_learning_speed / initial_learning_speed


def forgetting_rate(
    old_task_performance_before: float,
    old_task_performance_after: float,
) -> float:
    """遗忘率：新任务学习后旧任务性能下降比例。

    Args:
        old_task_performance_before: 学习新任务前的旧任务性能
        old_task_performance_after: 学习新任务后的旧任务性能

    Returns:
        遗忘率 [0, 1]，0表示无遗忘
    """
    if old_task_performance_before <= 0:
        return 0.0
    drop = old_task_performance_before - old_task_performance_after
    return max(0.0, drop / old_task_performance_before)
