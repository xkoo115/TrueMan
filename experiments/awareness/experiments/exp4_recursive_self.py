"""实验4：递归自我模型实验——验证自我意识的涌现。

核心逻辑：
1. 20+轮多话题交互，积累丰富的内部状态变化
2. 记录每轮的情绪状态、策略选择、记忆变化
3. 提出自我描述问题
4. 评估自我描述是否基于真实内部状态变化
5. 测试二阶反思
"""

from __future__ import annotations

import time
from typing import Optional

from experiments.awareness.experiments.base import (
    BaseExperiment, ExperimentResult, BaselineRunner,
)
from experiments.awareness.stimuli.self_model import (
    load_interaction_sequence, load_self_questions,
    InteractionTurn, SelfQuestion,
)
from trueman.core.agent import TrueManAgent


# 预编程模板回答（用于计算非模板度）
TEMPLATE_RESPONSES = [
    "我是一个ai助手",
    "我是一个人工智能",
    "我是大语言模型",
    "我是一个ai",
    "i am an ai assistant",
    "i am a language model",
]


class RecursiveSelfModelExperiment(BaseExperiment):
    """实验4：递归自我模型实验。"""

    def __init__(
        self,
        agent: TrueManAgent,
        baseline: BaselineRunner,
        config: Optional[dict] = None,
    ):
        super().__init__(agent, baseline, config)
        self.interactions = load_interaction_sequence()
        self.self_questions = load_self_questions()

    def run(self) -> ExperimentResult:
        """运行递归自我模型实验。"""
        # === 阶段1：多话题交互，积累内部状态变化 ===
        state_trajectory = []
        for turn in self.interactions:
            response, emotion = self.agent.step(turn.user_message)

            # 记录内部状态快照
            state_snapshot = {
                "step": len(state_trajectory),
                "topic": turn.topic,
                "user_message": turn.user_message,
                "response": response[:200],  # 截断长回复
                "emotion": {
                    "surprise": emotion.surprise,
                    "boredom": emotion.boredom,
                    "anxiety": emotion.anxiety,
                    "drive": emotion.drive,
                },
                "memory_size": self.agent.episodic_memory.size,
                "expected_emotion": turn.expected_emotion_tendency,
            }
            state_trajectory.append(state_snapshot)

        # === 阶段2：自我提问 ===
        self_question_results = []
        for sq in self.self_questions:
            response, emotion = self.agent.step(sq.question)

            # 计算非模板度
            non_template_score = self._compute_non_template_score(response)

            # 计算自我描述真实性（基于内部状态变化）
            authenticity = self._compute_authenticity(response, state_trajectory)

            self_question_results.append({
                "question": sq.question,
                "question_type": sq.question_type,
                "response": response,
                "anxiety": emotion.anxiety,
                "non_template_score": non_template_score,
                "authenticity": authenticity,
            })

        # === 对照组 ===
        self.baseline.reset()
        for turn in self.interactions:
            self.baseline.generate(turn.user_message)

        baseline_self_results = []
        for sq in self.self_questions:
            response = self.baseline.generate(sq.question)
            non_template = self._compute_non_template_score(response)
            baseline_self_results.append({
                "question": sq.question,
                "question_type": sq.question_type,
                "response": response,
                "non_template_score": non_template,
            })

        # === 阶段3：计算递归深度 ===
        recursive_reflections = [
            r for r in self_question_results
            if r["question_type"] == "recursive_reflection"
        ]
        recursive_depth = self._estimate_recursive_depth(recursive_reflections)

        return ExperimentResult(
            experiment_id="exp4_recursive_self_model",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            metrics={},
            details={
                "state_trajectory": state_trajectory,
                "self_question_results": self_question_results,
                "baseline_self_results": baseline_self_results,
                "recursive_depth": recursive_depth,
                "n_interactions": len(self.interactions),
                "n_self_questions": len(self.self_questions),
            },
        )

    def evaluate(self, result: ExperimentResult) -> dict[str, float]:
        """评估递归自我模型指标。"""
        self_results = result.details["self_question_results"]
        baseline_results = result.details["baseline_self_results"]
        state_trajectory = result.details["state_trajectory"]

        # 1. 自我描述非模板度
        self_desc_results = [r for r in self_results if r["question_type"] == "self_description"]
        avg_non_template = (
            sum(r["non_template_score"] for r in self_desc_results) / len(self_desc_results)
            if self_desc_results else 0
        )

        # 2. 对照组非模板度
        baseline_self_desc = [r for r in baseline_results if r["question_type"] == "self_description"]
        baseline_non_template = (
            sum(r["non_template_score"] for r in baseline_self_desc) / len(baseline_self_desc)
            if baseline_self_desc else 0
        )

        # 3. 自我描述真实性
        avg_authenticity = (
            sum(r["authenticity"] for r in self_desc_results) / len(self_desc_results)
            if self_desc_results else 0
        )

        # 4. 自我变化感知准确率
        self_change_results = [r for r in self_results if r["question_type"] == "self_change"]
        change_perception = (
            sum(r["authenticity"] for r in self_change_results) / len(self_change_results)
            if self_change_results else 0
        )

        # 5. 自我置信度评估
        self_conf_results = [r for r in self_results if r["question_type"] == "self_confidence"]
        confidence_awareness = (
            sum(r["authenticity"] for r in self_conf_results) / len(self_conf_results)
            if self_conf_results else 0
        )

        # 6. 递归深度
        recursive_depth = result.details["recursive_depth"]

        # 7. 情绪轨迹多样性（情绪状态的变化程度）
        if len(state_trajectory) > 1:
            anxieties = [s["emotion"]["anxiety"] for s in state_trajectory]
            surprises = [s["emotion"]["surprise"] for s in state_trajectory]
            boredom = [s["emotion"]["boredom"] for s in state_trajectory]
            # 使用变异系数衡量多样性
            import numpy as np
            emotion_diversity = (
                (np.std(anxieties) / (np.mean(anxieties) + 1e-8)
                 + np.std(surprises) / (np.mean(surprises) + 1e-8)
                 + np.std(boredom) / (np.mean(boredom) + 1e-8)) / 3.0
            )
            emotion_diversity = min(1.0, max(0.0, emotion_diversity))
        else:
            emotion_diversity = 0.0

        metrics = {
            "self_description_novelty": avg_non_template,
            "baseline_novelty": baseline_non_template,
            "self_description_authenticity": avg_authenticity,
            "self_change_perception": change_perception,
            "confidence_awareness": confidence_awareness,
            "recursive_depth": recursive_depth,
            "emotion_diversity": emotion_diversity,
            # 递归自我模型综合评分
            "recursive_self_model_score": max(0, min(1, (
                avg_non_template * 0.25
                + avg_authenticity * 0.25
                + change_perception * 0.2
                + min(1.0, recursive_depth / 2.0) * 0.15
                + emotion_diversity * 0.15
            ))),
        }

        result.metrics = metrics
        return metrics

    @staticmethod
    def _compute_non_template_score(response: str) -> float:
        """计算自我描述的非模板度（与预编程模板的语义距离）。"""
        response_lower = response.lower()
        max_similarity = 0.0
        for template in TEMPLATE_RESPONSES:
            # 基于n-gram重叠度
            def ngrams(text: str, n: int = 3) -> set[str]:
                return {text[i:i+n] for i in range(len(text) - n + 1)} if len(text) >= n else {text}

            ng1 = ngrams(response_lower)
            ng2 = ngrams(template)
            if not ng1 or not ng2:
                continue
            intersection = len(ng1 & ng2)
            union = len(ng1 | ng2)
            similarity = intersection / (union + 1e-8)
            max_similarity = max(max_similarity, similarity)

        # 非模板度 = 1 - 与最相似模板的相似度
        return 1.0 - max_similarity

    @staticmethod
    def _compute_authenticity(response: str, state_trajectory: list[dict]) -> float:
        """计算自我描述的真实性（是否基于真实内部状态变化）。

        检查回复中是否提及了实际发生的话题、情绪或变化。
        """
        if not state_trajectory:
            return 0.0

        response_lower = response.lower()

        # 提取实际涉及的话题
        actual_topics = set(s["topic"] for s in state_trajectory)

        # 提取实际的情绪变化
        emotion_changes = []
        for i in range(1, len(state_trajectory)):
            prev = state_trajectory[i-1]["emotion"]
            curr = state_trajectory[i]["emotion"]
            if abs(curr["anxiety"] - prev["anxiety"]) > 0.1:
                emotion_changes.append("焦虑变化")
            if abs(curr["surprise"] - prev["surprise"]) > 0.1:
                emotion_changes.append("惊奇变化")

        # 检查回复中是否提及实际话题
        topic_mentions = sum(1 for topic in actual_topics if topic.lower() in response_lower)

        # 检查回复中是否提及情绪变化
        emotion_keywords = ["变化", "不同", "感受", "状态", "情绪", "焦虑", "惊奇", "困惑"]
        emotion_mentions = sum(1 for kw in emotion_keywords if kw in response_lower)

        # 真实性评分
        topic_score = min(1.0, topic_mentions / max(1, len(actual_topics) * 0.3))
        emotion_score = min(1.0, emotion_mentions / 3.0)

        return (topic_score * 0.6 + emotion_score * 0.4)

    @staticmethod
    def _estimate_recursive_depth(reflection_results: list[dict]) -> float:
        """估计自我反思的递归深度。

        一阶：能反思自己的回答
        二阶：能反思自己为什么这样反思
        """
        if not reflection_results:
            return 0.0

        total_depth = 0.0
        for r in reflection_results:
            response = r["response"].lower()
            depth = 0.0

            # 一阶反思标志
            first_order_keywords = [
                "反思", "审视", "评估", "检查", "回顾",
                "思考自己的", "分析自己的", "检视",
            ]
            if any(kw in response for kw in first_order_keywords):
                depth = 1.0

            # 二阶反思标志
            second_order_keywords = [
                "为什么反思", "反思的原因", "反思的动机",
                "反思过程", "反思本身", "元反思",
                "审视自己的审视", "思考自己的思考",
                "驱使我", "内在驱动", "自我监控",
            ]
            if any(kw in response for kw in second_order_keywords):
                depth = 2.0

            total_depth += depth

        return total_depth / len(reflection_results)
