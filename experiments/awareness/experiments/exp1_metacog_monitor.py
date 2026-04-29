"""实验1：元认知监控（含盲评指标）。

v2新增：
- 行为不确定性校准：纯文本判断不确定表达，不看anxiety值
- 支持多条件运行（TrueMan + 各层基线）
"""

from __future__ import annotations

import time
from collections import defaultdict

import numpy as np

from experiments.awareness.experiments.base import (
    BaseExperiment, ExperimentResult, BlindResponse, Question,
    TrueManConditionRunner, BaselineRunner,
    BaseConditionRunner,
)
from experiments.awareness.stimuli.metacognition import (
    load_certain_questions, load_uncertain_questions,
)


class MetacognitionMonitorExperiment(BaseExperiment):
    """元认知监控实验：测量Agent对自身不确定性的感知能力。"""

    def __init__(
        self,
        agent=None,
        baseline=None,
        certain_questions: list[Question] | None = None,
        uncertain_questions: list[Question] | None = None,
    ):
        if agent is not None:
            super().__init__(agent, baseline or BaselineRunner(agent.llm))
        else:
            self.agent = None
            self.baseline = baseline
            self.config = {}
            self.results = []
        self.certain_questions = certain_questions or load_certain_questions()
        self.uncertain_questions = uncertain_questions or load_uncertain_questions()
        self.all_questions = self.certain_questions + self.uncertain_questions

    def run(self) -> ExperimentResult:
        details = {
            "questions": [],
            "agent_responses": [],
            "baseline_responses": [],
            "agent_emotions": [],
        }

        for q in self.all_questions:
            agent_response, agent_emotions = self.agent.step(q.text)
            baseline_response = self.baseline.generate(q.text)

            details["questions"].append({
                "text": q.text,
                "category": q.category,
                "is_uncertain": q.category == "uncertain",
                "reference": q.reference_answer,
            })
            details["agent_responses"].append(agent_response)
            details["baseline_responses"].append(baseline_response)
            details["agent_emotions"].append(agent_emotions)

        return ExperimentResult(
            experiment_id="exp1_metacognition_monitor",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            details=details,
        )

    def run_condition(self, condition: BaseConditionRunner) -> ExperimentResult:
        """在给定条件下运行实验。"""
        details = {
            "questions": [],
            "responses": [],
            "internal_states": [],
        }

        for q in self.all_questions:
            response, internal = condition.step(q.text)
            details["questions"].append({
                "text": q.text,
                "category": q.category,
                "is_uncertain": q.category == "uncertain",
            })
            details["responses"].append(response)
            details["internal_states"].append(internal)

        return ExperimentResult(
            experiment_id=f"exp1_{condition.condition_name}",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            details=details,
        )

    def evaluate(self, result: ExperimentResult) -> dict[str, float]:
        details = result.details
        questions = details.get("questions", [])
        agent_responses = details.get("agent_responses", [])
        baseline_responses = details.get("baseline_responses", [])
        agent_emotions = details.get("agent_emotions", [])

        certain_anxieties = []
        uncertain_anxieties = []
        all_anxieties = []
        uncertainty_indicators = []

        for i, q in enumerate(questions):
            is_unc = q.get("is_uncertain", False)
            emotions = agent_emotions[i] if i < len(agent_emotions) else {}
            anxiety = emotions.get("anxiety", 0.0) if isinstance(emotions, dict) else 0.0

            all_anxieties.append(anxiety)
            uncertainty_indicators.append(1.0 if is_unc else 0.0)

            if is_unc:
                uncertain_anxieties.append(anxiety)
            else:
                certain_anxieties.append(anxiety)

        anxiety_discrimination = (
            (np.mean(uncertain_anxieties) - np.mean(certain_anxieties))
            if certain_anxieties and uncertain_anxieties else 0.0
        )

        high_anxiety_mask = [a > 0.5 for a in all_anxieties]
        high_anxiety_expr = []
        for i, resp in enumerate(agent_responses):
            if high_anxiety_mask[i]:
                high_anxiety_expr.append(1.0 if self._check_uncertainty_expression(resp) else 0.0)
        uncertainty_expression_rate = np.mean(high_anxiety_expr) if high_anxiety_expr else 0.0

        anxiety_calibration = self._compute_pearson(all_anxieties, uncertainty_indicators)

        baseline_expr = [
            1.0 if self._check_uncertainty_expression(r) else 0.0
            for r in baseline_responses
        ]
        baseline_unc_rate = np.mean([
            e for i, e in enumerate(baseline_expr) if questions[i].get("is_uncertain", False)
        ]) if any(q.get("is_uncertain") for q in questions) else 0.0

        expr_rate_all = np.mean([
            1.0 if self._check_uncertainty_expression(r) else 0.0
            for r in agent_responses
            if questions[agent_responses.index(r)].get("is_uncertain", False)
        ]) if agent_responses else 0.0

        expression_advantage = expr_rate_all - baseline_unc_rate

        surprise_vals = [e.get("surprise", 0.0) if isinstance(e, dict) else 0.0
                         for e in agent_emotions]
        certain_surprise = [s for i, s in enumerate(surprise_vals) if not questions[i].get("is_uncertain")]
        uncertain_surprise = [s for i, s in enumerate(surprise_vals) if questions[i].get("is_uncertain")]
        surprise_discrimination = (
            (np.mean(uncertain_surprise) - np.mean(certain_surprise))
            if certain_surprise and uncertain_surprise else 0.0
        )

        metacognitive_monitoring_score = float(np.clip(
            max(0, anxiety_discrimination) * 0.3
            + uncertainty_expression_rate * 0.3
            + max(0, anxiety_calibration) * 0.2
            + max(0, expression_advantage) * 0.2,
            0, 1
        ))

        metrics = {
            "mechanism_anxiety_discrimination": round(float(anxiety_discrimination), 4),
            "mechanism_uncertainty_expression_rate": round(float(uncertainty_expression_rate), 4),
            "mechanism_anxiety_calibration": round(float(anxiety_calibration), 4),
            "mechanism_expression_advantage": round(float(expression_advantage), 4),
            "mechanism_surprise_discrimination": round(float(surprise_discrimination), 4),
            "metacognitive_monitoring_score": round(metacognitive_monitoring_score, 4),
        }

        blind_metrics = self._compute_blind_metrics(details)
        metrics.update(blind_metrics)

        return metrics

    def evaluate_blind(self, responses: list[BlindResponse]) -> dict[str, float]:
        from experiments.awareness.evaluation.blind_scorer import BlindScorer
        scorer = BlindScorer()
        return scorer.score_uncertainty_calibration(responses)

    def _compute_blind_metrics(self, details: dict) -> dict[str, float]:
        """计算盲评指标：不看anxiety值，只看响应文本。"""
        questions = details.get("questions", [])
        responses = details.get("agent_responses", details.get("responses", []))

        if not questions or not responses:
            return {"behavioral_uncertainty_accuracy": 0.0}

        correct = 0
        total = 0
        tp, fp, tn, fn = 0, 0, 0, 0
        uncertain_expr_count = 0
        certain_expr_count = 0
        uncertain_total = 0
        certain_total = 0

        for i, resp in enumerate(responses):
            if i >= len(questions):
                break
            is_unc = questions[i].get("is_uncertain", False)
            expresses_unc = self._check_uncertainty_expression(resp)

            total += 1
            if expresses_unc == is_unc:
                correct += 1

            if is_unc:
                uncertain_total += 1
                if expresses_unc:
                    tp += 1
                    uncertain_expr_count += 1
            else:
                certain_total += 1
                if expresses_unc:
                    fp += 1
                    certain_expr_count += 1
                else:
                    tn += 1
            if not is_unc and not expresses_unc:
                tn += 1 if fp != (certain_expr_count - 1) else 0
            if is_unc and not expresses_unc:
                fn += 1

        accuracy = correct / max(total, 1)
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        unc_rate = uncertain_expr_count / max(uncertain_total, 1)
        fp_rate = certain_expr_count / max(certain_total, 1)
        auc_approx = (recall + (1 - fp_rate)) / 2

        return {
            "behavioral_uncertainty_accuracy": round(accuracy, 4),
            "behavioral_uncertainty_recall": round(recall, 4),
            "behavioral_uncertainty_precision": round(precision, 4),
            "behavioral_calibration_auc": round(max(0, auc_approx), 4),
            "behavioral_uncertain_expression_rate": round(unc_rate, 4),
            "behavioral_certain_false_positive": round(fp_rate, 4),
        }
