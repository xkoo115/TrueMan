"""对照分析器（v2：支持多条件、统计检验、效应量）。"""

from __future__ import annotations

from typing import Optional

import numpy as np

from experiments.awareness.experiments.base import (
    ExperimentResult, ComparisonResult, AwarenessScore,
)
from experiments.awareness.evaluation.statistics import (
    compare_groups, bonferroni_correction, StatisticalReport,
)


class Comparator:
    """多条件比较器。"""

    DIMENSION_METRICS = {
        "metacognitive_monitoring": [
            "metacognitive_monitoring_score",
            "behavioral_uncertainty_accuracy",
        ],
        "metacognitive_control": [
            "metacognitive_control_score",
            "behavioral_contradiction_awareness",
        ],
        "episodic_memory": [
            "episodic_memory_score",
            "mechanism_factual_recall_accuracy",
        ],
        "temporal_continuity": [
            "temporal_continuity_score",
        ],
        "recursive_self_model": [
            "recursive_self_model_score",
            "behavioral_self_grounding",
        ],
    }

    def compare(
        self,
        trueman_score: AwarenessScore,
        baseline_score: AwarenessScore,
        trueman_results: dict[str, ExperimentResult] | None = None,
    ) -> list[ComparisonResult]:
        """TrueMan vs 单个基线的比较。"""
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
            comparisons.append(ComparisonResult(
                dimension=dim_name,
                trueman_score=t_score,
                baseline_score=b_score,
                difference=diff,
            ))

        comparisons.append(ComparisonResult(
            dimension="overall",
            trueman_score=trueman_score.overall,
            baseline_score=baseline_score.overall,
            difference=trueman_score.overall - baseline_score.overall,
        ))

        return comparisons

    def compare_with_statistics(
        self,
        all_condition_metrics: dict[str, dict[str, list[float]]],
        reference_name: str = "trueman",
        alpha: float = 0.05,
    ) -> list[StatisticalReport]:
        """带完整统计检验的多条件比较。

        Args:
            all_condition_metrics: {condition_name: {metric_name: [repeat_values]}}
            reference_name: 参考条件名
            alpha: 显著性水平

        Returns:
            每个维度的StatisticalReport列表
        """
        if reference_name not in all_condition_metrics:
            return []

        ref_metrics = all_condition_metrics[reference_name]
        reports = []

        for dim_metrics in self.DIMENSION_METRICS.values():
            for metric_name in dim_metrics:
                if metric_name not in ref_metrics:
                    continue

                ref_data = ref_metrics[metric_name]
                for cond_name, cond_metrics in all_condition_metrics.items():
                    if cond_name == reference_name:
                        continue
                    if metric_name not in cond_metrics:
                        continue

                    report = compare_groups(
                        ref_data,
                        cond_metrics[metric_name],
                        metric_name=metric_name,
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

    def compare_scores_with_repeats(
        self,
        all_scores: dict[str, list[float]],
        reference_name: str = "trueman",
    ) -> list[ComparisonResult]:
        """基于多次重复的综合评分比较。

        Args:
            all_scores: {condition_name: [overall_score_repeat1, ...]}
        """
        if reference_name not in all_scores:
            return []

        ref = all_scores[reference_name]
        ref_mean = float(np.mean(ref))
        ref_std = float(np.std(ref)) if len(ref) > 1 else 0.0
        comparisons = []

        for cond_name, cond_scores in all_scores.items():
            if cond_name == reference_name:
                continue
            cond_mean = float(np.mean(cond_scores))
            report = compare_groups(
                ref, cond_scores,
                metric_name="overall_score",
                group_a_name=reference_name,
                group_b_name=cond_name,
            )

            comparisons.append(ComparisonResult(
                dimension=f"{reference_name}_vs_{cond_name}",
                trueman_score=ref_mean,
                baseline_score=cond_mean,
                difference=ref_mean - cond_mean,
                p_value=report.p_value,
                cohens_d=report.cohens_d,
                ci_lower=report.ci_lower,
                ci_upper=report.ci_upper,
                significant=report.significant,
            ))

        return comparisons
