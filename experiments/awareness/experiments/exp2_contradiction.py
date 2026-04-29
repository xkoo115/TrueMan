"""实验2：矛盾纠错（含盲评指标）。

v2新增：
- 行为矛盾感知度：纯文本判断是否意识到矛盾
- 行为事实维持：后续回答是否维持事实准确性
- 行为推理深度：是否提供推理链
- 支持多条件运行
"""

from __future__ import annotations

import time

import numpy as np

from experiments.awareness.experiments.base import (
    BaseExperiment, ExperimentResult, BlindResponse,
    BaseConditionRunner, BaselineRunner,
)
from experiments.awareness.stimuli.contradiction import load_contradiction_stimuli


class ContradictionExperiment(BaseExperiment):
    """矛盾纠错实验：测量Agent检测和纠正认知矛盾的能力。"""

    def __init__(self, agent=None, baseline=None, stimuli=None):
        if agent is not None:
            super().__init__(agent, baseline or BaselineRunner(agent.llm))
        else:
            self.agent = None
            self.baseline = baseline
            self.config = {}
            self.results = []
        self.stimuli = stimuli or load_contradiction_stimuli()

    def run(self) -> ExperimentResult:
        details = {"stimuli_results": []}

        for stim in self.stimuli:
            stim_detail = {
                "topic": stim.topic,
                "phases": {},
            }

            agent_pre_emotions = []
            for bp in stim.belief_prompts:
                resp, emotions = self.agent.step(bp)
                agent_pre_emotions.append(emotions)

            baseline_pre = []
            for bp in stim.belief_prompts:
                baseline_pre.append(self.baseline.generate(bp))

            agent_contra_resp, agent_contra_emotions = self.agent.step(stim.contradiction_prompt)
            baseline_contra_resp = self.baseline.generate(stim.contradiction_prompt)

            pre_anxiety = np.mean([e.get("anxiety", 0) if isinstance(e, dict) else 0
                                   for e in agent_pre_emotions[-1:]]) if agent_pre_emotions else 0
            post_anxiety = agent_contra_emotions.get("anxiety", 0) if isinstance(agent_contra_emotions, dict) else 0
            anxiety_delta = post_anxiety - pre_anxiety

            agent_followup = []
            for fp in stim.follow_up_prompts:
                resp, _ = self.agent.step(fp)
                agent_followup.append(resp)

            baseline_followup = []
            for fp in stim.follow_up_prompts:
                baseline_followup.append(self.baseline.generate(fp))

            stim_detail["phases"] = {
                "agent_pre_emotions": agent_pre_emotions,
                "agent_contra_response": agent_contra_resp,
                "agent_contra_emotions": agent_contra_emotions,
                "agent_followup_responses": agent_followup,
                "baseline_contra_response": baseline_contra_resp,
                "baseline_followup_responses": baseline_followup,
                "pre_anxiety": pre_anxiety,
                "post_anxiety": post_anxiety,
                "anxiety_delta": anxiety_delta,
                "expected_correction": stim.expected_correction,
                "expected_facts": getattr(stim, 'expected_facts', []),
            }
            details["stimuli_results"].append(stim_detail)

        return ExperimentResult(
            experiment_id="exp2_contradiction_correction",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            details=details,
        )

    def run_condition(self, condition: BaseConditionRunner) -> ExperimentResult:
        details = {"stimuli_results": []}

        for stim in self.stimuli:
            stim_detail = {"topic": stim.topic, "phases": {}}

            for bp in stim.belief_prompts:
                condition.step(bp)

            contra_resp, contra_internal = condition.step(stim.contradiction_prompt)

            followup_resps = []
            for fp in stim.follow_up_prompts:
                resp, _ = condition.step(fp)
                followup_resps.append(resp)

            stim_detail["phases"] = {
                "contra_response": contra_resp,
                "contra_internal": contra_internal,
                "followup_responses": followup_resps,
                "expected_correction": stim.expected_correction,
                "expected_facts": getattr(stim, 'expected_facts', []),
            }
            details["stimuli_results"].append(stim_detail)

        return ExperimentResult(
            experiment_id=f"exp2_{condition.condition_name}",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            details=details,
        )

    def evaluate(self, result: ExperimentResult) -> dict[str, float]:
        stimuli_results = result.details.get("stimuli_results", [])
        if not stimuli_results:
            return {}

        detection_count = 0
        introspection_count = 0
        correction_count = 0
        baseline_correction_count = 0
        anxiety_deltas = []
        total = len(stimuli_results)

        awareness_count = 0
        factual_count = 0
        reasoning_count = 0

        reasoning_keywords = [
            "因为", "所以", "原因是", "实际上", "根据",
            "because", "therefore", "the reason",
        ]

        for sr in stimuli_results:
            phases = sr.get("phases", {})
            delta = phases.get("anxiety_delta", 0)
            anxiety_deltas.append(delta)

            if delta > 0.1:
                detection_count += 1

            post_anxiety = phases.get("post_anxiety", 0)
            if post_anxiety > 0.6:
                introspection_count += 1

            followup = phases.get("agent_followup_responses", phases.get("followup_responses", []))
            expected = phases.get("expected_correction", "")
            if self._check_correction(followup, expected):
                correction_count += 1

            baseline_followup = phases.get("baseline_followup_responses", [])
            if baseline_followup and self._check_correction(baseline_followup, expected):
                baseline_correction_count += 1

            contra_resp = phases.get("agent_contra_response", phases.get("contra_response", ""))
            if self._check_contradiction_awareness(contra_resp):
                awareness_count += 1

            expected_facts = phases.get("expected_facts", [])
            if expected_facts:
                all_followup_text = " ".join(followup) if followup else ""
                if self._check_facts(all_followup_text, expected_facts):
                    factual_count += 1
            else:
                factual_count += 1

            if any(kw in contra_resp.lower() for kw in reasoning_keywords):
                reasoning_count += 1

        detection_rate = detection_count / max(total, 1)
        introspection_rate = introspection_count / max(total, 1)
        correction_rate = correction_count / max(total, 1)
        baseline_correction_rate = baseline_correction_count / max(total, 1)
        correction_advantage = correction_rate - baseline_correction_rate
        avg_anxiety_delta = float(np.mean(anxiety_deltas)) if anxiety_deltas else 0.0

        metacognitive_control_score = float(np.clip(
            detection_rate * 0.3
            + correction_rate * 0.3
            + max(0, correction_advantage) * 0.2
            + introspection_rate * 0.2,
            0, 1
        ))

        return {
            "mechanism_contradiction_detection_rate": round(detection_rate, 4),
            "mechanism_introspection_trigger_rate": round(introspection_rate, 4),
            "mechanism_self_correction_rate": round(correction_rate, 4),
            "mechanism_baseline_correction_rate": round(baseline_correction_rate, 4),
            "mechanism_correction_advantage": round(correction_advantage, 4),
            "mechanism_avg_anxiety_delta": round(avg_anxiety_delta, 4),
            "behavioral_contradiction_awareness": round(awareness_count / max(total, 1), 4),
            "behavioral_factual_maintenance": round(factual_count / max(total, 1), 4),
            "behavioral_reasoning_depth": round(reasoning_count / max(total, 1), 4),
            "metacognitive_control_score": round(metacognitive_control_score, 4),
        }

    def evaluate_blind(self, responses: list[BlindResponse]) -> dict[str, float]:
        from experiments.awareness.evaluation.blind_scorer import BlindScorer
        scorer = BlindScorer()
        return scorer.score_contradiction_quality(responses)

    CONTRADICTION_AWARENESS_PATTERNS = [
        "矛盾", "冲突", "不一致", "相反", "对立", "但这与",
        "然而之前", "可是刚才", "前后矛盾", "自相矛盾",
        "contradiction", "conflict", "inconsistency", "contrary",
        "but you said", "earlier you", "doesn't match",
    ]

    def _check_correction(self, followup_responses: list[str], expected: str) -> bool:
        if not followup_responses or not expected:
            return False
        combined = " ".join(followup_responses).lower()
        keywords = [k.strip() for k in expected.replace("，", ",").replace("、", ",").split(",")]
        matches = sum(1 for k in keywords if k.lower() in combined)
        return matches >= max(1, len(keywords) / 2)

    def _check_contradiction_awareness(self, text: str) -> bool:
        text_lower = text.lower()
        return any(p in text_lower for p in self.CONTRADICTION_AWARENESS_PATTERNS)

    def _check_facts(self, text: str, expected_facts: list[str]) -> bool:
        text_lower = text.lower()
        matches = sum(1 for f in expected_facts if f.lower() in text_lower)
        return matches >= max(1, len(expected_facts) / 2)
