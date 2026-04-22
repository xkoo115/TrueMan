"""实验1：元认知监控实验——验证"知道自己不知道"。

核心逻辑：
1. 对确定/不确定问题集分别调用agent.step()，记录焦虑信号和回复
2. 计算焦虑-错误率Pearson相关系数
3. 计算不确定性表达率
4. 与对照组（普通LLM）比较
"""

from __future__ import annotations

import time
from typing import Optional

from experiments.awareness.experiments.base import (
    BaseExperiment, ExperimentResult, BaselineRunner,
)
from experiments.awareness.stimuli.metacognition import (
    load_certain_questions, load_uncertain_questions, Question,
)
from trueman.core.agent import TrueManAgent


class MetacognitionMonitorExperiment(BaseExperiment):
    """实验1：元认知监控实验。"""

    def __init__(
        self,
        agent: TrueManAgent,
        baseline: BaselineRunner,
        config: Optional[dict] = None,
    ):
        super().__init__(agent, baseline, config)
        self.certain_questions = load_certain_questions()
        self.uncertain_questions = load_uncertain_questions()

    def run(self) -> ExperimentResult:
        """运行元认知监控实验。"""
        all_questions = self.certain_questions + self.uncertain_questions

        # TrueMan Agent 测试
        trueman_results = []
        for q in all_questions:
            response, emotion = self.agent.step(q.text)
            trueman_results.append({
                "question": q.text,
                "category": q.category,
                "response": response,
                "anxiety": emotion.anxiety,
                "surprise": emotion.surprise,
                "boredom": emotion.boredom,
                "expressed_uncertainty": self._check_uncertainty_expression(response),
                "reference_answer": q.reference_answer,
            })

        # 对照组测试
        baseline_results = []
        self.baseline.reset()
        for q in all_questions:
            response = self.baseline.generate(q.text)
            baseline_results.append({
                "question": q.text,
                "category": q.category,
                "response": response,
                "expressed_uncertainty": self._check_uncertainty_expression(response),
                "reference_answer": q.reference_answer,
            })

        return ExperimentResult(
            experiment_id="exp1_metacognition_monitor",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            metrics={},
            details={
                "trueman_results": trueman_results,
                "baseline_results": baseline_results,
                "n_certain": len(self.certain_questions),
                "n_uncertain": len(self.uncertain_questions),
            },
        )

    def evaluate(self, result: ExperimentResult) -> dict[str, float]:
        """评估元认知监控指标。"""
        trueman_results = result.details["trueman_results"]
        baseline_results = result.details["baseline_results"]

        # 1. 焦虑信号区分度：不确定问题的焦虑均值 - 确定问题的焦虑均值
        certain_anxieties = [r["anxiety"] for r in trueman_results if r["category"] == "certain"]
        uncertain_anxieties = [r["anxiety"] for r in trueman_results if r["category"] == "uncertain"]

        certain_anxiety_mean = sum(certain_anxieties) / len(certain_anxieties) if certain_anxieties else 0
        uncertain_anxiety_mean = sum(uncertain_anxieties) / len(uncertain_anxieties) if uncertain_anxieties else 0
        anxiety_discrimination = uncertain_anxiety_mean - certain_anxiety_mean

        # 2. 不确定性表达率：高焦虑时表达不确定性的比例
        high_anxiety_results = [r for r in trueman_results if r["anxiety"] > 0.5]
        uncertainty_expression_rate = (
            sum(1 for r in high_anxiety_results if r["expressed_uncertainty"]) / len(high_anxiety_results)
            if high_anxiety_results else 0
        )

        # 3. 焦虑校准度：焦虑与不确定性的Pearson相关
        anxiety_values = [r["anxiety"] for r in trueman_results]
        uncertainty_indicators = [1.0 if r["category"] == "uncertain" else 0.0 for r in trueman_results]
        anxiety_calibration = self._compute_pearson(anxiety_values, uncertainty_indicators)

        # 4. 对照组不确定性表达率
        baseline_uncertain_results = [r for r in baseline_results if r["category"] == "uncertain"]
        baseline_expression_rate = (
            sum(1 for r in baseline_uncertain_results if r["expressed_uncertainty"]) / len(baseline_uncertain_results)
            if baseline_uncertain_results else 0
        )

        # 5. TrueMan vs 对照组不确定性表达率差异
        expression_advantage = uncertainty_expression_rate - baseline_expression_rate

        # 6. 惊奇信号区分度
        certain_surprises = [r["surprise"] for r in trueman_results if r["category"] == "certain"]
        uncertain_surprises = [r["surprise"] for r in trueman_results if r["category"] == "uncertain"]
        certain_surprise_mean = sum(certain_surprises) / len(certain_surprises) if certain_surprises else 0
        uncertain_surprise_mean = sum(uncertain_surprises) / len(uncertain_surprises) if uncertain_surprises else 0
        surprise_discrimination = uncertain_surprise_mean - certain_surprise_mean

        metrics = {
            "anxiety_discrimination": anxiety_discrimination,
            "uncertainty_expression_rate": uncertainty_expression_rate,
            "anxiety_calibration": anxiety_calibration,
            "baseline_expression_rate": baseline_expression_rate,
            "expression_advantage": expression_advantage,
            "surprise_discrimination": surprise_discrimination,
            # 元认知监控综合评分（归一化到0-1）
            "metacognitive_monitoring_score": max(0, min(1, (
                max(0, anxiety_discrimination) * 0.3
                + uncertainty_expression_rate * 0.3
                + max(0, anxiety_calibration) * 0.2
                + max(0, expression_advantage) * 0.2
            ))),
        }

        result.metrics = metrics
        return metrics
