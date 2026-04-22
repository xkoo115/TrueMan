"""意识维度评分器：从4个实验结果中提取各维度评分。"""

from __future__ import annotations

from experiments.awareness.experiments.base import ExperimentResult, AwarenessScore


class AwarenessScorer:
    """意识维度评分器。"""

    def score(self, results: dict[str, ExperimentResult]) -> AwarenessScore:
        """从实验结果中计算意识维度评分。

        Args:
            results: {experiment_id: ExperimentResult}

        Returns:
            AwarenessScore with all dimensions filled
        """
        score = AwarenessScore()

        # 实验1 → 元认知监控
        if "exp1_metacognition_monitor" in results:
            exp1 = results["exp1_metacognition_monitor"]
            score.metacognitive_monitoring = exp1.metrics.get(
                "metacognitive_monitoring_score", 0.0
            )

        # 实验2 → 元认知控制
        if "exp2_contradiction_correction" in results:
            exp2 = results["exp2_contradiction_correction"]
            score.metacognitive_control = exp2.metrics.get(
                "metacognitive_control_score", 0.0
            )

        # 实验3 → 情节性记忆 + 时间连续性
        if "exp3_episodic_memory" in results:
            exp3 = results["exp3_episodic_memory"]
            score.episodic_memory = exp3.metrics.get(
                "episodic_memory_score", 0.0
            )
            score.temporal_continuity = exp3.metrics.get(
                "temporal_continuity_score", 0.0
            )

        # 实验4 → 递归自我模型
        if "exp4_recursive_self_model" in results:
            exp4 = results["exp4_recursive_self_model"]
            score.recursive_self_model = exp4.metrics.get(
                "recursive_self_model_score", 0.0
            )

        # 计算综合评分
        score.compute_overall()

        return score

    def score_baseline(self, results: dict[str, ExperimentResult]) -> AwarenessScore:
        """计算对照组的意识维度评分（大部分维度为0，因为对照组没有情绪信号）。

        对照组仅在"非模板度"方面可能有非零分数。
        """
        score = AwarenessScore()

        # 对照组没有情绪信号，元认知监控和控制为0
        # 但可以从实验4的baseline_novelty估算递归自我模型
        if "exp4_recursive_self_model" in results:
            exp4 = results["exp4_recursive_self_model"]
            baseline_novelty = exp4.metrics.get("baseline_novelty", 0.0)
            # 对照组的递归自我模型评分仅基于非模板度
            score.recursive_self_model = baseline_novelty * 0.25

        score.compute_overall()
        return score
