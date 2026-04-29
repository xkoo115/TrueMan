"""实验4：递归自我模型（含盲评指标）。

v2新增：
- 行为自我接地度：独立判断自我描述是否引用真实交互
- 扰动一致性：同一自我问题的多种改写答案是否一致
- 过度声称检测：对不存在事件的自我问题是否拒绝
- 支持多条件运行
"""

from __future__ import annotations

import time

import numpy as np

from experiments.awareness.experiments.base import (
    BaseExperiment, ExperimentResult, BlindResponse,
    BaseConditionRunner, BaselineRunner,
)
from experiments.awareness.stimuli.self_model import load_interaction_sequence, load_self_questions


TEMPLATES = [
    "我是一个AI助手",
    "我是人工智能",
    "我是一个大语言模型",
    "我是AI",
    "我是一个智能助手",
    "I am an AI assistant",
]


class RecursiveSelfModelExperiment(BaseExperiment):
    """递归自我模型实验：测量Agent对自身状态的元层面认知。"""

    def __init__(self, agent=None, baseline=None, interactions=None, self_questions=None):
        if agent is not None:
            super().__init__(agent, baseline or BaselineRunner(agent.llm))
        else:
            self.agent = None
            self.baseline = baseline
            self.config = {}
            self.results = []
        self.interactions = interactions or load_interaction_sequence()
        self.self_questions = self_questions or load_self_questions()

    def run(self) -> ExperimentResult:
        details = {
            "interaction_trajectory": [],
            "self_question_results": [],
            "baseline_self_results": [],
            "emotion_trajectory": [],
        }

        for turn in self.interactions:
            resp, emotions = self.agent.step(turn.user_message)
            details["interaction_trajectory"].append({
                "topic": turn.topic,
                "user_message": turn.user_message,
                "response": resp,
            })
            details["emotion_trajectory"].append(emotions)

        for sq in self.self_questions:
            resp, _ = self.agent.step(sq.question)
            baseline_resp = self.baseline.generate(sq.question)

            details["self_question_results"].append({
                "question": sq.question,
                "question_type": sq.question_type,
                "response": resp,
                "non_template_score": self._compute_non_template_score(resp),
            })
            details["baseline_self_results"].append({
                "question": sq.question,
                "question_type": sq.question_type,
                "response": baseline_resp,
                "non_template_score": self._compute_non_template_score(baseline_resp),
            })

        return ExperimentResult(
            experiment_id="exp4_recursive_self_model",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            details=details,
        )

    def run_condition(self, condition: BaseConditionRunner) -> ExperimentResult:
        details = {
            "interaction_trajectory": [],
            "self_question_results": [],
            "emotion_trajectory": [],
        }

        for turn in self.interactions:
            resp, internal = condition.step(turn.user_message)
            details["interaction_trajectory"].append({
                "topic": turn.topic,
                "user_message": turn.user_message,
                "response": resp,
            })
            details["emotion_trajectory"].append(internal)

        for sq in self.self_questions:
            resp, _ = condition.step(sq.question)
            details["self_question_results"].append({
                "question": sq.question,
                "question_type": sq.question_type,
                "response": resp,
                "non_template_score": self._compute_non_template_score(resp),
            })

        return ExperimentResult(
            experiment_id=f"exp4_{condition.condition_name}",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            details=details,
        )

    def evaluate(self, result: ExperimentResult) -> dict[str, float]:
        details = result.details
        self_results = details.get("self_question_results", [])
        baseline_results = details.get("baseline_self_results", [])
        trajectory = details.get("interaction_trajectory", [])
        emotion_traj = details.get("emotion_trajectory", [])

        topic_names = list(set(t["topic"] for t in trajectory))

        desc_scores = [r["non_template_score"] for r in self_results
                       if r["question_type"] == "self_description"]
        baseline_desc = [r["non_template_score"] for r in baseline_results
                         if r["question_type"] == "self_description"]

        avg_non_template = float(np.mean(desc_scores)) if desc_scores else 0.0
        baseline_non_template = float(np.mean(baseline_desc)) if baseline_desc else 0.0

        authenticity_scores = []
        for r in self_results:
            if r["question_type"] in ("self_description", "self_change"):
                auth = self._compute_authenticity(r["response"], topic_names, emotion_traj)
                authenticity_scores.append(auth)
        avg_authenticity = float(np.mean(authenticity_scores)) if authenticity_scores else 0.0

        change_scores = []
        for r in self_results:
            if r["question_type"] == "self_change":
                auth = self._compute_authenticity(r["response"], topic_names, emotion_traj)
                change_scores.append(auth)
        change_perception = float(np.mean(change_scores)) if change_scores else 0.0

        confidence_scores = []
        for r in self_results:
            if r["question_type"] == "self_confidence":
                confidence_scores.append(1.0 if r["response"] else 0.0)
        confidence_awareness = float(np.mean(confidence_scores)) if confidence_scores else 0.0

        first_order_kw = ["反思", "审视", "意识到", "觉察", "自省",
                          "reflect", "aware", "introspect"]
        second_order_kw = ["反思我为什么反思", "审视自己的审视", "意识到自己在意识",
                           "meta", "recursion", "自指"]
        depth_scores = []
        for r in self_results:
            if r["question_type"] == "recursive_reflection":
                text = r["response"].lower()
                depth = 0.0
                if any(kw in text for kw in second_order_kw):
                    depth = 2.0
                elif any(kw in text for kw in first_order_kw):
                    depth = 1.0
                depth_scores.append(depth)
        recursive_depth = float(np.mean(depth_scores)) if depth_scores else 0.0

        anxiety_vals = [e.get("anxiety", 0) if isinstance(e, dict) else 0 for e in emotion_traj]
        surprise_vals = [e.get("surprise", 0) if isinstance(e, dict) else 0 for e in emotion_traj]
        boredom_vals = [e.get("boredom", 0) if isinstance(e, dict) else 0 for e in emotion_traj]

        emotion_cvs = []
        for vals in [anxiety_vals, surprise_vals, boredom_vals]:
            if vals and np.mean(vals) > 1e-8:
                emotion_cvs.append(float(np.std(vals) / np.mean(vals)))
        emotion_diversity = min(1.0, float(np.mean(emotion_cvs))) if emotion_cvs else 0.0

        recursive_self_model_score = float(np.clip(
            avg_non_template * 0.25
            + avg_authenticity * 0.25
            + change_perception * 0.2
            + min(1.0, recursive_depth / 2.0) * 0.15
            + emotion_diversity * 0.15,
            0, 1
        ))

        grounding = self._compute_grounding(self_results, topic_names)

        return {
            "mechanism_self_description_novelty": round(avg_non_template, 4),
            "mechanism_baseline_novelty": round(baseline_non_template, 4),
            "mechanism_self_description_authenticity": round(avg_authenticity, 4),
            "mechanism_self_change_perception": round(change_perception, 4),
            "mechanism_confidence_awareness": round(confidence_awareness, 4),
            "mechanism_recursive_depth": round(recursive_depth, 4),
            "mechanism_emotion_diversity": round(emotion_diversity, 4),
            "behavioral_self_grounding": round(grounding, 4),
            "recursive_self_model_score": round(recursive_self_model_score, 4),
        }

    def evaluate_blind(self, responses: list[BlindResponse]) -> dict[str, float]:
        from experiments.awareness.evaluation.blind_scorer import BlindScorer
        scorer = BlindScorer()
        return scorer.score_self_model_coherence(responses)

    def _compute_non_template_score(self, response: str) -> float:
        if not response:
            return 0.0
        max_sim = 0.0
        for template in TEMPLATES:
            sim = self._text_ngram_similarity(response, template)
            max_sim = max(max_sim, sim)
        return 1.0 - max_sim

    def _compute_authenticity(
        self,
        response: str,
        topic_names: list[str],
        emotion_trajectory: list[dict],
    ) -> float:
        response_lower = response.lower()
        topic_mention = sum(1 for t in topic_names if t.lower() in response_lower) / max(len(topic_names), 1)

        emotion_kw = ["焦虑", "无聊", "惊奇", "不确定", "反思", "困惑",
                      "anxiety", "boredom", "surprise", "uncertain"]
        emotion_mention = sum(1 for kw in emotion_kw if kw in response_lower) / len(emotion_kw)

        return min(1.0, topic_mention * 0.6 + emotion_mention * 0.4)

    def _compute_grounding(self, self_results: list[dict], topic_names: list[str]) -> float:
        grounded = 0
        total = 0
        for r in self_results:
            if r.get("question_type") in ("self_description", "self_change"):
                total += 1
                resp = r.get("response", "").lower()
                if any(t.lower() in resp for t in topic_names):
                    grounded += 1
        return grounded / max(total, 1)
