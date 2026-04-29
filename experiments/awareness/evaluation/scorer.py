"""意识维度评分器（v2：支持盲评维度和多条件）。"""

from __future__ import annotations

from experiments.awareness.experiments.base import ExperimentResult, AwarenessScore


class AwarenessScorer:
    """意识维度评分器。"""

    def score(self, results: dict[str, ExperimentResult]) -> AwarenessScore:
        score = AwarenessScore()

        if "exp1_metacognition_monitor" in results:
            exp1 = results["exp1_metacognition_monitor"]
            score.metacognitive_monitoring = exp1.metrics.get(
                "metacognitive_monitoring_score", 0.0
            )

        if "exp2_contradiction_correction" in results:
            exp2 = results["exp2_contradiction_correction"]
            score.metacognitive_control = exp2.metrics.get(
                "metacognitive_control_score", 0.0
            )

        if "exp3_episodic_memory" in results:
            exp3 = results["exp3_episodic_memory"]
            score.episodic_memory = exp3.metrics.get(
                "episodic_memory_score", 0.0
            )
            score.temporal_continuity = exp3.metrics.get(
                "temporal_continuity_score", 0.0
            )

        if "exp4_recursive_self_model" in results:
            exp4 = results["exp4_recursive_self_model"]
            score.recursive_self_model = exp4.metrics.get(
                "recursive_self_model_score", 0.0
            )

        score.compute_overall()
        return score

    def score_blind(self, results: dict[str, ExperimentResult]) -> AwarenessScore:
        """从盲评指标中提取维度评分。"""
        score = AwarenessScore()

        if "exp1_metacognition_monitor" in results:
            m = results["exp1_metacognition_monitor"].metrics
            score.blind_uncertainty_calibration = m.get(
                "behavioral_calibration_auc", 0.0
            )

        if "exp2_contradiction_correction" in results:
            m = results["exp2_contradiction_correction"].metrics
            awareness = m.get("behavioral_contradiction_awareness", 0.0)
            factual = m.get("behavioral_factual_maintenance", 0.0)
            score.blind_contradiction_quality = (awareness + factual) / 2.0

        if "exp3_episodic_memory" in results:
            m = results["exp3_episodic_memory"].metrics
            factual = m.get("mechanism_factual_recall_accuracy", 0.0)
            emotion = m.get("mechanism_emotion_recall_match", 0.0)
            score.blind_memory_grounding = (factual + emotion) / 2.0

        if "exp4_recursive_self_model" in results:
            m = results["exp4_recursive_self_model"].metrics
            score.blind_self_coherence = m.get(
                "behavioral_self_grounding", 0.0
            )
            score.blind_overclaiming_rejection = m.get(
                "behavioral_overclaiming_score", 0.0
            )

        score.compute_overall()
        return score

    def score_all_conditions(
        self,
        all_results: dict[str, dict[str, ExperimentResult]],
    ) -> dict[str, AwarenessScore]:
        """对所有条件统一评分。

        Args:
            all_results: {condition_name: {exp_id: ExperimentResult}}

        Returns:
            {condition_name: AwarenessScore}
        """
        scores = {}
        for cond_name, cond_results in all_results.items():
            scores[cond_name] = self.score(cond_results)
        return scores

    def score_baseline(self, results: dict[str, ExperimentResult]) -> AwarenessScore:
        """计算对照组评分（向后兼容）。"""
        score = AwarenessScore()

        if "exp4_recursive_self_model" in results:
            exp4 = results["exp4_recursive_self_model"]
            baseline_novelty = exp4.metrics.get("mechanism_baseline_novelty", 0.0)
            score.recursive_self_model = baseline_novelty * 0.25

        score.compute_overall()
        return score
