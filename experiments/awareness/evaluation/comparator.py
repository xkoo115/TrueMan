"""对照分析器：TrueMan Agent vs 普通LLM的比较。"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from experiments.awareness.experiments.base import (
    ExperimentResult, ComparisonResult, AwarenessScore,
)


class Comparator:
    """TrueMan vs 对照组的比较器。"""

    # 各维度在实验结果中的指标映射
    DIMENSION_METRICS = {
        "metacognitive_monitoring": [
            "anxiety_discrimination",
            "uncertainty_expression_rate",
            "anxiety_calibration",
        ],
        "metacognitive_control": [
            "contradiction_detection_rate",
            "self_correction_rate",
            "introspection_trigger_rate",
        ],
        "episodic_memory": [
            "factual_recall_accuracy",
            "emotion_recall_match",
            "recall_advantage",
        ],
        "temporal_continuity": [
            "temporal_order_accuracy",
            "emotion_recall_match",
            "future_preview_quality",
        ],
        "recursive_self_model": [
            "self_description_novelty",
            "self_description_authenticity",
            "recursive_depth",
        ],
    }

    def compare(
        self,
        trueman_score: AwarenessScore,
        baseline_score: AwarenessScore,
        trueman_results: dict[str, ExperimentResult],
    ) -> list[ComparisonResult]:
        """对每个维度计算TrueMan vs 对照组的差异。

        Args:
            trueman_score: TrueMan Agent的意识评分
            baseline_score: 对照组的意识评分
            trueman_results: TrueMan Agent的实验结果（用于计算p值）

        Returns:
            各维度的ComparisonResult列表
        """
        dimensions = [
            ("metacognitive_monitoring", trueman_score.metacognitive_monitoring, baseline_score.metacognitive_monitoring),
            ("metacognitive_control", trueman_score.metacognitive_control, baseline_score.metacognitive_control),
            ("episodic_memory", trueman_score.episodic_memory, baseline_score.episodic_memory),
            ("temporal_continuity", trueman_score.temporal_continuity, baseline_score.temporal_continuity),
            ("recursive_self_model", trueman_score.recursive_self_model, baseline_score.recursive_self_model),
        ]

        comparisons = []
        for dim_name, t_score, b_score in dimensions:
            diff = t_score - b_score
            # p值需要多次运行的数据，这里简化处理
            p_value = self._estimate_p_value(dim_name, trueman_results)

            comparisons.append(ComparisonResult(
                dimension=dim_name,
                trueman_score=t_score,
                baseline_score=b_score,
                difference=diff,
                p_value=p_value,
            ))

        # 添加综合评分比较
        comparisons.append(ComparisonResult(
            dimension="overall",
            trueman_score=trueman_score.overall,
            baseline_score=baseline_score.overall,
            difference=trueman_score.overall - baseline_score.overall,
            p_value=None,
        ))

        return comparisons

    @staticmethod
    def _estimate_p_value(
        dimension: str,
        results: dict[str, ExperimentResult],
    ) -> Optional[float]:
        """估计p值（基于repeat_stats中的均值和标准差）。

        简化处理：如果有repeat_stats，使用单样本t检验。
        """
        # 找到对应实验
        exp_map = {
            "metacognitive_monitoring": "exp1_metacognition_monitor",
            "metacognitive_control": "exp2_contradiction_correction",
            "episodic_memory": "exp3_episodic_memory",
            "temporal_continuity": "exp3_episodic_memory",
            "recursive_self_model": "exp4_recursive_self_model",
        }

        exp_id = exp_map.get(dimension)
        if not exp_id or exp_id not in results:
            return None

        result = results[exp_id]
        if not result.repeat_stats:
            return None

        # 找到该维度的评分指标
        score_key_map = {
            "metacognitive_monitoring": "metacognitive_monitoring_score",
            "metacognitive_control": "metacognitive_control_score",
            "episodic_memory": "episodic_memory_score",
            "temporal_continuity": "temporal_continuity_score",
            "recursive_self_model": "recursive_self_model_score",
        }

        score_key = score_key_map.get(dimension)
        if not score_key or score_key not in result.repeat_stats:
            return None

        mean, std = result.repeat_stats[score_key]
        if std < 1e-8:
            return None

        # 单样本t检验：H0: mean = 0 (对照组水平)
        if not HAS_SCIPY:
            return None
        n = 3  # 默认3次重复
        try:
            p_value = float(scipy_stats.ttest_ind_from_stats(mean, std, n, 0, 0.01, n).pvalue)
            return p_value
        except Exception:
            return None
