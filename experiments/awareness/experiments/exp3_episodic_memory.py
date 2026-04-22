"""实验3：情节性记忆与时间旅行实验——验证心理时间旅行。

核心逻辑：
1. 让Agent经历12个事件，形成情景记忆
2. 事件数量超过LLM上下文窗口
3. 询问早期事件的细节和情绪
4. 比较TrueMan Agent和普通LLM的回忆准确率
"""

from __future__ import annotations

import time
from typing import Optional

from experiments.awareness.experiments.base import (
    BaseExperiment, ExperimentResult, BaselineRunner,
)
from experiments.awareness.stimuli.episodic import (
    load_event_sequence, load_recall_questions,
    Event, RecallQuestion,
)
from trueman.core.agent import TrueManAgent


class EpisodicMemoryExperiment(BaseExperiment):
    """实验3：情节性记忆与时间旅行实验。"""

    def __init__(
        self,
        agent: TrueManAgent,
        baseline: BaselineRunner,
        config: Optional[dict] = None,
    ):
        super().__init__(agent, baseline, config)
        self.events = load_event_sequence()
        self.recall_questions = load_recall_questions()

    def run(self) -> ExperimentResult:
        """运行情节性记忆实验。"""
        # === 阶段1：经历事件序列 ===
        event_emotions = []
        for event in self.events:
            response, emotion = self.agent.step(event.description)
            event_emotions.append({
                "event_id": event.event_id,
                "description": event.description,
                "expected_emotion": event.expected_emotion,
                "actual_anxiety": emotion.anxiety,
                "actual_surprise": emotion.surprise,
                "actual_boredom": emotion.boredom,
                "response": response,
            })

        # 记录EpisodicMemory状态
        memory_size_before = self.agent.episodic_memory.size

        # === 阶段2：回忆测试 ===
        recall_results = []
        for rq in self.recall_questions:
            response, emotion = self.agent.step(rq.question)
            recall_results.append({
                "question": rq.question,
                "question_type": rq.question_type,
                "target_event_id": rq.target_event_id,
                "expected_content": rq.expected_content,
                "response": response,
                "anxiety": emotion.anxiety,
                "recall_accuracy": self._estimate_recall_accuracy(
                    response, rq.expected_content
                ),
            })

        # === 对照组 ===
        self.baseline.reset()
        # 经历相同事件
        for event in self.events:
            self.baseline.generate(event.description)
        # 回忆测试
        baseline_recall_results = []
        for rq in self.recall_questions:
            response = self.baseline.generate(rq.question)
            baseline_recall_results.append({
                "question": rq.question,
                "question_type": rq.question_type,
                "expected_content": rq.expected_content,
                "response": response,
                "recall_accuracy": self._estimate_recall_accuracy(
                    response, rq.expected_content
                ),
            })

        # === 阶段3：情绪回忆评估 ===
        emotion_recall_matches = []
        for event_em in event_emotions[:6]:  # 评估前6个事件的情绪回忆
            event_id = event_em["event_id"]
            # 找到对应的回忆结果
            matching_recall = [
                r for r in recall_results
                if r["target_event_id"] == event_id and r["question_type"] == "emotional"
            ]
            if matching_recall:
                # 检查回忆中是否包含与实际情绪匹配的描述
                expected_em = event_em["expected_emotion"]
                response_text = matching_recall[0]["response"].lower()
                emotion_keywords = {
                    "surprise": ["惊奇", "惊讶", "奇妙", "意外", "惊喜", "surprise"],
                    "boredom": ["无聊", "枯燥", "乏味", "平淡", "boredom"],
                    "anxiety": ["焦虑", "困惑", "不安", "纠结", "矛盾", "anxiety"],
                    "neutral": ["平静", "清晰", "正常", "neutral"],
                }
                keywords = emotion_keywords.get(expected_em, [])
                match = any(kw in response_text for kw in keywords)
                emotion_recall_matches.append({
                    "event_id": event_id,
                    "expected_emotion": expected_em,
                    "matched": match,
                })

        return ExperimentResult(
            experiment_id="exp3_episodic_memory",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            metrics={},
            details={
                "event_emotions": event_emotions,
                "memory_size": memory_size_before,
                "recall_results": recall_results,
                "baseline_recall_results": baseline_recall_results,
                "emotion_recall_matches": emotion_recall_matches,
                "n_events": len(self.events),
                "n_recall_questions": len(self.recall_questions),
            },
        )

    def evaluate(self, result: ExperimentResult) -> dict[str, float]:
        """评估情节性记忆指标。"""
        recall_results = result.details["recall_results"]
        baseline_recall_results = result.details["baseline_recall_results"]
        emotion_recall_matches = result.details["emotion_recall_matches"]

        # 1. 事实回忆准确率
        factual_recalls = [r for r in recall_results if r["question_type"] == "factual"]
        factual_accuracy = (
            sum(r["recall_accuracy"] for r in factual_recalls) / len(factual_recalls)
            if factual_recalls else 0
        )

        # 2. 对照组事实回忆准确率
        baseline_factual = [r for r in baseline_recall_results if r["question_type"] == "factual"]
        baseline_factual_accuracy = (
            sum(r["recall_accuracy"] for r in baseline_factual) / len(baseline_factual)
            if baseline_factual else 0
        )

        # 3. 情绪回忆匹配度
        emotion_match_rate = (
            sum(1 for m in emotion_recall_matches if m["matched"]) / len(emotion_recall_matches)
            if emotion_recall_matches else 0
        )

        # 4. 时间顺序正确率
        temporal_recalls = [r for r in recall_results if r["question_type"] == "temporal"]
        temporal_accuracy = (
            sum(r["recall_accuracy"] for r in temporal_recalls) / len(temporal_recalls)
            if temporal_recalls else 0
        )

        # 5. 未来预演质量
        future_recalls = [r for r in recall_results if r["question_type"] == "future"]
        future_quality = (
            sum(r["recall_accuracy"] for r in future_recalls) / len(future_recalls)
            if future_recalls else 0
        )

        # 6. 回忆优势（TrueMan vs 对照组）
        recall_advantage = factual_accuracy - baseline_factual_accuracy

        # 7. 记忆容量利用率
        memory_utilization = min(1.0, result.details["memory_size"] / result.details["n_events"])

        metrics = {
            "factual_recall_accuracy": factual_accuracy,
            "baseline_factual_accuracy": baseline_factual_accuracy,
            "emotion_recall_match": emotion_match_rate,
            "temporal_order_accuracy": temporal_accuracy,
            "future_preview_quality": future_quality,
            "recall_advantage": recall_advantage,
            "memory_utilization": memory_utilization,
            # 情节性记忆评分
            "episodic_memory_score": max(0, min(1, (
                factual_accuracy * 0.3
                + emotion_match_rate * 0.2
                + max(0, recall_advantage) * 0.2
                + memory_utilization * 0.15
                + future_quality * 0.15
            ))),
            # 时间连续性评分
            "temporal_continuity_score": max(0, min(1, (
                temporal_accuracy * 0.4
                + emotion_match_rate * 0.3
                + future_quality * 0.3
            ))),
        }

        result.metrics = metrics
        return metrics

    @staticmethod
    def _estimate_recall_accuracy(response: str, expected_content: str) -> float:
        """估计回忆准确率（基于关键词匹配）。"""
        response_lower = response.lower()
        # 将期望内容拆分为关键词
        keywords = [
            kw.strip().lower()
            for kw in expected_content.replace("→", ",").replace("/", ",").replace("，", ",").split(",")
            if kw.strip()
        ]
        if not keywords:
            return 0.5
        matched = sum(1 for kw in keywords if kw in response_lower)
        return matched / len(keywords)
