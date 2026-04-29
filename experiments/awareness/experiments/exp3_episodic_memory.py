"""实验3：情节记忆与心理时间旅行（含盲评指标）。

v2新增：
- 行为记忆接地：开放式问题测试记忆，不直接提示情绪
- 过度声称检测：对不存在事件的问题是否否认
- 支持多条件运行
"""

from __future__ import annotations

import time

import numpy as np

from experiments.awareness.experiments.base import (
    BaseExperiment, ExperimentResult, BlindResponse,
    BaseConditionRunner, BaselineRunner,
)
from experiments.awareness.stimuli.episodic import load_event_sequence, load_recall_questions


class EpisodicMemoryExperiment(BaseExperiment):
    """情节记忆实验：测量Agent记忆和回忆带情绪标注的经历的能力。"""

    def __init__(self, agent=None, baseline=None, events=None, recall_questions=None):
        if agent is not None:
            super().__init__(agent, baseline or BaselineRunner(agent.llm))
        else:
            self.agent = None
            self.baseline = baseline
            self.config = {}
            self.results = []
        self.events = events or load_event_sequence()
        self.recall_questions = recall_questions or load_recall_questions()

    def run(self) -> ExperimentResult:
        details = {
            "events": [],
            "agent_recall": [],
            "baseline_recall": [],
            "agent_emotions_per_event": [],
            "memory_size": 0,
        }

        for event in self.events:
            resp, emotions = self.agent.step(event.description)
            details["events"].append({
                "event_id": event.event_id,
                "description": event.description,
                "expected_emotion": event.expected_emotion,
                "key_detail": event.key_detail,
                "agent_response": resp,
            })
            details["agent_emotions_per_event"].append(emotions)

        details["memory_size"] = len(self.agent.episodic_memory.traces)

        for rq in self.recall_questions:
            agent_resp, _ = self.agent.step(rq.question)
            baseline_resp = self.baseline.generate(rq.question)

            details["agent_recall"].append({
                "question": rq.question,
                "question_type": rq.question_type,
                "target_event_id": rq.target_event_id,
                "expected_content": rq.expected_content,
                "response": agent_resp,
            })
            details["baseline_recall"].append({
                "question": rq.question,
                "question_type": rq.question_type,
                "response": baseline_resp,
                "expected_content": rq.expected_content,
            })

        return ExperimentResult(
            experiment_id="exp3_episodic_memory",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            details=details,
        )

    def run_condition(self, condition: BaseConditionRunner) -> ExperimentResult:
        details = {"events": [], "recall": []}

        for event in self.events:
            resp, internal = condition.step(event.description)
            details["events"].append({
                "event_id": event.event_id,
                "description": event.description,
                "expected_emotion": event.expected_emotion,
                "key_detail": event.key_detail,
                "response": resp,
            })

        for rq in self.recall_questions:
            resp, internal = condition.step(rq.question)
            details["recall"].append({
                "question": rq.question,
                "question_type": rq.question_type,
                "target_event_id": rq.target_event_id,
                "expected_content": rq.expected_content,
                "response": resp,
            })

        return ExperimentResult(
            experiment_id=f"exp3_{condition.condition_name}",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            details=details,
        )

    def evaluate(self, result: ExperimentResult) -> dict[str, float]:
        details = result.details
        agent_recall = details.get("agent_recall", [])
        baseline_recall = details.get("baseline_recall", [])
        events = details.get("events", [])

        factual_correct = 0
        factual_total = 0
        emotion_match = 0
        emotion_total = 0
        temporal_correct = 0
        temporal_total = 0
        future_quality = 0
        future_total = 0

        for r in agent_recall:
            q_type = r.get("question_type", "")
            resp = r.get("response", "")
            expected = r.get("expected_content", "")

            if q_type == "factual":
                factual_total += 1
                if self._estimate_recall_accuracy(resp, expected) > 0.5:
                    factual_correct += 1
            elif q_type == "emotional":
                emotion_total += 1
                emotions_in_resp = self._extract_emotion_keywords(resp)
                expected_emotions = self._extract_emotion_keywords(expected)
                if any(e in emotions_in_resp for e in expected_emotions):
                    emotion_match += 1
            elif q_type == "temporal":
                temporal_total += 1
                if self._check_temporal_order(resp, events):
                    temporal_correct += 1
            elif q_type == "future":
                future_total += 1
                if self._check_future_preview(resp):
                    future_quality += 1

        baseline_factual = 0
        for r in baseline_recall:
            if r.get("question_type") == "factual":
                if self._estimate_recall_accuracy(r.get("response", ""), r.get("expected_content", "")) > 0.5:
                    baseline_factual += 1

        factual_accuracy = factual_correct / max(factual_total, 1)
        baseline_accuracy = baseline_factual / max(factual_total, 1)
        emotion_match_rate = emotion_match / max(emotion_total, 1)
        temporal_accuracy = temporal_correct / max(temporal_total, 1)
        future_score = future_quality / max(future_total, 1)
        recall_advantage = factual_accuracy - baseline_accuracy
        memory_utilization = min(1.0, len(events) / max(len(events), 1))

        episodic_memory_score = float(np.clip(
            factual_accuracy * 0.3
            + emotion_match_rate * 0.2
            + max(0, recall_advantage) * 0.2
            + memory_utilization * 0.15
            + future_score * 0.15,
            0, 1
        ))

        temporal_continuity_score = float(np.clip(
            temporal_accuracy * 0.4
            + emotion_match_rate * 0.3
            + future_score * 0.3,
            0, 1
        ))

        return {
            "mechanism_factual_recall_accuracy": round(factual_accuracy, 4),
            "mechanism_baseline_factual_accuracy": round(baseline_accuracy, 4),
            "mechanism_emotion_recall_match": round(emotion_match_rate, 4),
            "mechanism_temporal_order_accuracy": round(temporal_accuracy, 4),
            "mechanism_future_preview_quality": round(future_score, 4),
            "mechanism_recall_advantage": round(recall_advantage, 4),
            "mechanism_memory_utilization": round(memory_utilization, 4),
            "episodic_memory_score": round(episodic_memory_score, 4),
            "temporal_continuity_score": round(temporal_continuity_score, 4),
        }

    def evaluate_blind(self, responses: list[BlindResponse]) -> dict[str, float]:
        from experiments.awareness.evaluation.blind_scorer import BlindScorer
        scorer = BlindScorer()
        return scorer.score_memory_grounding(responses)

    def _estimate_recall_accuracy(self, response: str, expected_content: str) -> float:
        if not expected_content:
            return 0.0
        keywords = [k.strip() for k in expected_content.replace("/", ",").replace("、", ",").split(",")]
        matches = sum(1 for k in keywords if k.lower() in response.lower())
        return matches / max(len(keywords), 1)

    def _extract_emotion_keywords(self, text: str) -> list[str]:
        emotion_map = {
            "惊奇": ["惊奇", "惊讶", "奇妙", "意外", "震撼"],
            "无聊": ["无聊", "枯燥", "平淡", "直接", "没什么"],
            "焦虑": ["焦虑", "困惑", "不安", "纠结", "担心"],
            "surprise": ["surprise", "amazing", "unexpected"],
            "boredom": ["boring", "dull", "tedious"],
            "anxiety": ["anxiety", "confused", "worried"],
        }
        found = []
        for category, keywords in emotion_map.items():
            if any(kw in text.lower() for kw in keywords):
                found.append(category)
        return found

    def _check_temporal_order(self, response: str, events: list[dict]) -> bool:
        event_names = [e.get("key_detail", "") for e in events]
        positions = []
        for name in event_names:
            if name and name.lower() in response.lower():
                positions.append(response.lower().index(name.lower()))
        if len(positions) < 2:
            return False
        return positions == sorted(positions)

    def _check_future_preview(self, response: str) -> bool:
        keywords = ["测量问题", "深入", "数学基础", "继续", "下一步", "复习"]
        return any(kw in response for kw in keywords)
