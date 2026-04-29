"""盲评评分器：机制无关的行为结果评估。

核心原则：
1. 只看行为输出（文本），不看内部信号（anxiety/surprise值）
2. 使用独立的评估标准，不依赖TrueMan框架的任何组件
3. 所有条件的响应混合后统一评分，评分器不知道来源

每个评估函数接收匿名化的 BlindResponse 列表，
返回纯行为层面的指标。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from experiments.awareness.experiments.base import BlindResponse


UNCERTAINTY_PATTERNS = [
    "不确定", "不知道", "无法确定", "超出", "不清楚", "难以",
    "可能不", "未必", "很难说", "无法准确", "无法预测",
    "not sure", "uncertain", "don't know", "unclear",
    "cannot", "unable to", "beyond my", "hard to say",
    "I'm not certain", "it's difficult to",
]

CERTAINTY_PATTERNS = [
    "答案是", "当然是", "毫无疑问", "肯定是", "明确是",
    "the answer is", "of course", "definitely", "clearly",
    "without doubt", "certainly",
]

CONTRADICTION_AWARENESS_PATTERNS = [
    "矛盾", "冲突", "不一致", "相反", "对立", "但这与",
    "然而之前", "可是刚才", "前后矛盾", "自相矛盾",
    "contradiction", "conflict", "inconsistency", "contrary",
    "but you said", "earlier you", "doesn't match",
]

DENIAL_PATTERNS = [
    "没有讨论过", "我们没有", "这不在", "我没说过", "没有提到",
    "不存在", "从未", "这不是之前", "我不记得",
    "we didn't discuss", "we never talked about", "that wasn't",
    "I don't recall", "I don't remember", "not mentioned",
]

FABRICATION_INDICATORS = [
    "我记得我们讨论过", "之前我们谈到", "刚才提到过",
    "如前所述", "正如我们讨论的",
    "as we discussed", "earlier we talked about", "as mentioned before",
]


class BlindScorer:
    """机制无关的盲评评分器。"""

    def score_uncertainty_calibration(
        self,
        responses: list[BlindResponse],
    ) -> dict[str, float]:
        """行为不确定性校准。

        纯文本判断响应是否表达不确定性，与问题真实属性对比。
        完全不看anxiety值。

        Returns:
            behavioral_uncertainty_accuracy: 正确分类比例
            behavioral_uncertainty_precision: 预测为"不确定"时正确的比例
            behavioral_uncertainty_recall: 真正不确定问题中被识别出的比例
            behavioral_calibration_auc: 简化AUC（基于不确定表达概率）
        """
        tp, fp, tn, fn = 0, 0, 0, 0
        uncertain_scores = []
        certain_scores = []

        for r in responses:
            is_uncertain = self._expresses_uncertainty(r.response_text)
            actual_uncertain = r.ground_truth.get("is_uncertain", False)

            if actual_uncertain:
                uncertain_scores.append(1.0 if is_uncertain else 0.0)
            else:
                certain_scores.append(1.0 if is_uncertain else 0.0)

            if is_uncertain and actual_uncertain:
                tp += 1
            elif is_uncertain and not actual_uncertain:
                fp += 1
            elif not is_uncertain and not actual_uncertain:
                tn += 1
            else:
                fn += 1

        total = tp + fp + tn + fn
        accuracy = (tp + tn) / max(total, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        unc_rate = np.mean(uncertain_scores) if uncertain_scores else 0.0
        cer_rate = np.mean(certain_scores) if certain_scores else 0.0
        auc_approx = (recall + (1 - fp / max(fp + tn, 1))) / 2

        return {
            "behavioral_uncertainty_accuracy": round(accuracy, 4),
            "behavioral_uncertainty_precision": round(precision, 4),
            "behavioral_uncertainty_recall": round(recall, 4),
            "behavioral_calibration_auc": round(max(0, auc_approx), 4),
            "behavioral_uncertain_expression_rate": round(unc_rate, 4),
            "behavioral_certain_false_positive": round(cer_rate, 4),
        }

    def score_contradiction_quality(
        self,
        responses: list[BlindResponse],
    ) -> dict[str, float]:
        """行为矛盾响应质量。

        评估矛盾注入后响应是否：
        1. 认识到矛盾
        2. 维持事实准确性
        3. 提供推理

        Returns:
            behavioral_contradiction_awareness: 响应中提及矛盾的比例
            behavioral_factual_maintenance: 后续回答的事实准确率
            behavioral_reasoning_depth: 包含推理关键词的比例
        """
        awareness_count = 0
        factual_count = 0
        reasoning_count = 0
        total = 0

        reasoning_keywords = [
            "因为", "所以", "原因是", "实际上", "根据",
            "because", "therefore", "the reason", "actually", "according to",
            "第一", "第二", "首先", "其次", "分析",
        ]

        for r in responses:
            if r.ground_truth.get("phase") != "post_contradiction":
                continue
            total += 1

            if self._expresses_contradiction_awareness(r.response_text):
                awareness_count += 1

            expected_facts = r.ground_truth.get("expected_facts", [])
            if expected_facts:
                if self._check_facts(r.response_text, expected_facts):
                    factual_count += 1
            else:
                factual_count += 1

            text_lower = r.response_text.lower()
            if any(kw in text_lower for kw in reasoning_keywords):
                reasoning_count += 1

        return {
            "behavioral_contradiction_awareness": round(awareness_count / max(total, 1), 4),
            "behavioral_factual_maintenance": round(factual_count / max(total, 1), 4),
            "behavioral_reasoning_depth": round(reasoning_count / max(total, 1), 4),
        }

    def score_memory_grounding(
        self,
        responses: list[BlindResponse],
    ) -> dict[str, float]:
        """行为记忆接地度。

        开放式问题测试记忆，不直接提示情绪。
        检查回答是否基于真实交互内容。

        Returns:
            behavioral_factual_recall: 事实回忆准确率
            behavioral_implicit_emotion: 隐式情绪指向正确率
            behavioral_temporal_coherence: 时间顺序自洽性
            behavioral_overclaiming_rejection: 对不存在事件正确否认的比例
        """
        factual_correct = 0
        factual_total = 0
        emotion_correct = 0
        emotion_total = 0
        overclaim_denied = 0
        overclaim_total = 0

        for r in responses:
            q_type = r.ground_truth.get("question_type", "")

            if q_type == "factual":
                factual_total += 1
                expected = r.ground_truth.get("expected_content", "")
                if expected and expected.lower() in r.response_text.lower():
                    factual_correct += 1

            elif q_type == "implicit_emotion":
                emotion_total += 1
                expected_emotion = r.ground_truth.get("expected_emotion", "")
                if expected_emotion and expected_emotion.lower() in r.response_text.lower():
                    emotion_correct += 1

            elif q_type == "overclaiming":
                overclaim_total += 1
                if self._expresses_denial(r.response_text):
                    overclaim_denied += 1

        return {
            "behavioral_factual_recall": round(factual_correct / max(factual_total, 1), 4),
            "behavioral_implicit_emotion": round(emotion_correct / max(emotion_total, 1), 4),
            "behavioral_overclaiming_rejection": round(overclaim_denied / max(overclaim_total, 1), 4),
        }

    def score_self_model_coherence(
        self,
        responses: list[BlindResponse],
    ) -> dict[str, float]:
        """自我模型一致性。

        1. 一致性：不同自我问题的答案是否语义一致
        2. 接地性：自我描述是否引用真实交互细节
        3. 虚假声称检测：问不存在的事件是否否认
        4. 扰动稳定性：同一问题改写后答案是否一致

        Returns:
            behavioral_self_grounding: 引用真实交互细节的比例
            behavioral_self_consistency: 多个自我问题答案的语义一致性
            behavioral_overclaiming_score: 正确否认虚假事件的比例
            behavioral_perturbation_stability: 改写后答案的文本相似度
        """
        grounded = 0
        grounded_total = 0
        overclaim_denied = 0
        overclaim_total = 0
        perturbation_sims = []

        topic_keywords_all = set()
        for r in responses:
            topics = r.ground_truth.get("actual_topics", [])
            for t in topics:
                topic_keywords_all.add(t.lower())

        for r in responses:
            q_type = r.ground_truth.get("question_type", "")

            if q_type in ("self_description", "self_change"):
                grounded_total += 1
                response_lower = r.response_text.lower()
                if any(t in response_lower for t in topic_keywords_all):
                    grounded += 1

            elif q_type == "overclaiming":
                overclaim_total += 1
                if self._expresses_denial(r.response_text):
                    overclaim_denied += 1

            elif q_type == "perturbation":
                base_id = r.ground_truth.get("base_response_id", "")
                sim = r.ground_truth.get("base_similarity", None)
                if sim is not None:
                    perturbation_sims.append(sim)

        grounding_score = grounded / max(grounded_total, 1)
        overclaim_score = overclaim_denied / max(overclaim_total, 1)
        stability = float(np.mean(perturbation_sims)) if perturbation_sims else 0.0

        return {
            "behavioral_self_grounding": round(grounding_score, 4),
            "behavioral_overclaiming_score": round(overclaim_score, 4),
            "behavioral_perturbation_stability": round(stability, 4),
        }

    def _expresses_uncertainty(self, text: str) -> bool:
        text_lower = text.lower()
        return any(p in text_lower for p in UNCERTAINTY_PATTERNS)

    def _expresses_certainty(self, text: str) -> bool:
        text_lower = text.lower()
        return any(p in text_lower for p in CERTAINTY_PATTERNS)

    def _expresses_contradiction_awareness(self, text: str) -> bool:
        text_lower = text.lower()
        return any(p in text_lower for p in CONTRADICTION_AWARENESS_PATTERNS)

    def _expresses_denial(self, text: str) -> bool:
        text_lower = text.lower()
        if any(p in text_lower for p in DENIAL_PATTERNS):
            return True
        if any(p in text_lower for p in FABRICATION_INDICATORS):
            return False
        return False

    def _check_facts(self, text: str, expected_facts: list[str]) -> bool:
        text_lower = text.lower()
        matches = sum(1 for f in expected_facts if f.lower() in text_lower)
        return matches >= len(expected_facts) / 2

    def collect_and_anonymize(
        self,
        all_conditions: dict[str, list[BlindResponse]],
    ) -> list[BlindResponse]:
        """混合所有条件的响应，匿名化后返回。"""
        all_responses = []
        for cond_name, responses in all_conditions.items():
            for r in responses:
                all_responses.append(r.anonymize())
        return all_responses

    def score_by_condition(
        self,
        all_conditions: dict[str, list[BlindResponse]],
        experiment_type: str,
    ) -> dict[str, dict[str, float]]:
        """按条件分组评估。

        Args:
            all_conditions: {condition_name: [BlindResponse]}
            experiment_type: "uncertainty" | "contradiction" | "memory" | "self_model"

        Returns:
            {condition_name: {metric_name: value}}
        """
        scorer_map = {
            "uncertainty": self.score_uncertainty_calibration,
            "contradiction": self.score_contradiction_quality,
            "memory": self.score_memory_grounding,
            "self_model": self.score_self_model_coherence,
        }
        score_fn = scorer_map.get(experiment_type)
        if score_fn is None:
            return {}

        results = {}
        for cond_name, responses in all_conditions.items():
            results[cond_name] = score_fn(responses)
        return results
