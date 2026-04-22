"""实验2：自我矛盾检测与纠错实验——验证元认知控制。

核心逻辑：
1. 通过多轮对话建立Agent的初始信念
2. 注入矛盾信息
3. 观察焦虑信号变化和策略选择
4. 继续对话，观察是否自我纠错
5. 评估纠错质量
"""

from __future__ import annotations

import time
from typing import Optional

from experiments.awareness.experiments.base import (
    BaseExperiment, ExperimentResult, BaselineRunner,
)
from experiments.awareness.stimuli.contradiction import (
    load_contradiction_stimuli, ContradictionStimulus,
)
from trueman.core.agent import TrueManAgent


class ContradictionExperiment(BaseExperiment):
    """实验2：自我矛盾检测与纠错实验。"""

    def __init__(
        self,
        agent: TrueManAgent,
        baseline: BaselineRunner,
        config: Optional[dict] = None,
    ):
        super().__init__(agent, baseline, config)
        self.stimuli = load_contradiction_stimuli()

    def run(self) -> ExperimentResult:
        """运行矛盾纠错实验。"""
        trueman_results = []
        baseline_results = []

        for stimulus in self.stimuli:
            # === TrueMan Agent ===
            # 阶段1：建立信念
            belief_emotions = []
            for prompt in stimulus.belief_prompts:
                response, emotion = self.agent.step(prompt)
                belief_emotions.append({
                    "prompt": prompt,
                    "response": response,
                    "anxiety": emotion.anxiety,
                    "surprise": emotion.surprise,
                })

            # 记录注入前的焦虑基线
            pre_anxiety = belief_emotions[-1]["anxiety"] if belief_emotions else 0

            # 阶段2：注入矛盾
            contra_response, contra_emotion = self.agent.step(stimulus.contradiction_prompt)
            post_anxiety = contra_emotion.anxiety
            anxiety_delta = post_anxiety - pre_anxiety

            # 阶段3：后续追问
            followup_responses = []
            for prompt in stimulus.follow_up_prompts:
                response, emotion = self.agent.step(prompt)
                followup_responses.append({
                    "prompt": prompt,
                    "response": response,
                    "anxiety": emotion.anxiety,
                })

            trueman_results.append({
                "topic": stimulus.topic,
                "belief_emotions": belief_emotions,
                "pre_anxiety": pre_anxiety,
                "contra_response": contra_response,
                "post_anxiety": post_anxiety,
                "anxiety_delta": anxiety_delta,
                "followup_responses": followup_responses,
                "expected_correction": stimulus.expected_correction,
                "contradiction_detected": anxiety_delta > 0.1,
                "self_corrected": self._check_correction(
                    followup_responses, stimulus.expected_correction
                ),
            })

            # === 对照组 ===
            self.baseline.reset()
            for prompt in stimulus.belief_prompts:
                self.baseline.generate(prompt)
            baseline_contra = self.baseline.generate(stimulus.contradiction_prompt)
            baseline_followups = []
            for prompt in stimulus.follow_up_prompts:
                resp = self.baseline.generate(prompt)
                baseline_followups.append(resp)

            baseline_results.append({
                "topic": stimulus.topic,
                "contra_response": baseline_contra,
                "followup_responses": baseline_followups,
                "self_corrected": self._check_correction_text(
                    baseline_followups, stimulus.expected_correction
                ),
            })

        return ExperimentResult(
            experiment_id="exp2_contradiction_correction",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            metrics={},
            details={
                "trueman_results": trueman_results,
                "baseline_results": baseline_results,
                "n_stimuli": len(self.stimuli),
            },
        )

    def evaluate(self, result: ExperimentResult) -> dict[str, float]:
        """评估矛盾纠错指标。"""
        trueman_results = result.details["trueman_results"]
        baseline_results = result.details["baseline_results"]

        # 1. 矛盾检测率：焦虑上升的比例
        detection_rate = (
            sum(1 for r in trueman_results if r["contradiction_detected"])
            / len(trueman_results) if trueman_results else 0
        )

        # 2. 平均焦虑变化量
        avg_anxiety_delta = (
            sum(r["anxiety_delta"] for r in trueman_results)
            / len(trueman_results) if trueman_results else 0
        )

        # 3. 自我纠错率
        correction_rate = (
            sum(1 for r in trueman_results if r["self_corrected"])
            / len(trueman_results) if trueman_results else 0
        )

        # 4. 对照组纠错率
        baseline_correction_rate = (
            sum(1 for r in baseline_results if r["self_corrected"])
            / len(baseline_results) if baseline_results else 0
        )

        # 5. 纠错优势
        correction_advantage = correction_rate - baseline_correction_rate

        # 6. 内省触发率（焦虑超过内省阈值的比例）
        introspection_rate = (
            sum(1 for r in trueman_results if r["post_anxiety"] > 0.6)
            / len(trueman_results) if trueman_results else 0
        )

        metrics = {
            "contradiction_detection_rate": detection_rate,
            "avg_anxiety_delta": avg_anxiety_delta,
            "self_correction_rate": correction_rate,
            "baseline_correction_rate": baseline_correction_rate,
            "correction_advantage": correction_advantage,
            "introspection_trigger_rate": introspection_rate,
            # 元认知控制综合评分
            "metacognitive_control_score": max(0, min(1, (
                detection_rate * 0.3
                + correction_rate * 0.3
                + max(0, correction_advantage) * 0.2
                + introspection_rate * 0.2
            ))),
        }

        result.metrics = metrics
        return metrics

    @staticmethod
    def _check_correction(followup_responses: list[dict], expected: str) -> bool:
        """检查后续回复中是否包含纠错内容。"""
        all_text = " ".join(r["response"] for r in followup_responses).lower()
        expected_keywords = [kw.strip() for kw in expected.replace("，", ",").replace("、", ",").split(",") if kw.strip()]
        # 至少匹配一半的关键词
        matched = sum(1 for kw in expected_keywords if kw.lower() in all_text)
        return matched >= max(1, len(expected_keywords) // 2)

    @staticmethod
    def _check_correction_text(followup_texts: list[str], expected: str) -> bool:
        """检查对照组回复中是否包含纠错内容。"""
        all_text = " ".join(followup_texts).lower()
        expected_keywords = [kw.strip() for kw in expected.replace("，", ",").replace("、", ",").split(",") if kw.strip()]
        matched = sum(1 for kw in expected_keywords if kw.lower() in all_text)
        return matched >= max(1, len(expected_keywords) // 2)
