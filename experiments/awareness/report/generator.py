"""报告生成器：汇总实验结果，生成Markdown报告。"""

from __future__ import annotations

import json
import time
from pathlib import Path

from experiments.awareness.experiments.base import (
    ExperimentResult, AwarenessScore, ComparisonResult,
)
from experiments.awareness.evaluation.scorer import AwarenessScorer
from experiments.awareness.evaluation.comparator import Comparator


class ReportGenerator:
    """实验报告生成器。"""

    DISCLAIMER = """
---
**审慎声明**

本报告仅验证计算性行为指标，行为指标通过**不代表**系统具有主观意识（qualia）。
自我意识的判定是哲学和神经科学的开放问题，本实验框架不对此做出断言。

本实验验证的是：TrueMan框架的内稳态驱动机制（惊奇/无聊/焦虑信号）是否能让LLM表现出
与自我意识相关的**行为特征**，包括元认知监控、元认知控制、情节性记忆、时间连续性和递归自我模型。

这些行为特征是自我意识的**必要条件**而非**充分条件**。正如《纽约动物意识宣言》(2024)所倡导的，
我们应当从"连续光谱主义"的视角看待意识——不是"有或无"的二元判断，而是在多个维度上的渐变频谱。
"""

    def __init__(self, output_dir: str = "experiments/awareness/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scorer = AwarenessScorer()
        self.comparator = Comparator()

    def generate(
        self,
        trueman_results: dict[str, ExperimentResult],
        baseline_results: Optional[dict[str, ExperimentResult]] = None,
    ) -> str:
        """生成完整实验报告。

        Args:
            trueman_results: TrueMan Agent的实验结果
            baseline_results: 对照组的实验结果（可选）

        Returns:
            Markdown格式的报告文本
        """
        # 计算评分
        trueman_score = self.scorer.score(trueman_results)
        baseline_score = self.scorer.score_baseline(trueman_results)

        # 对照分析
        comparisons = self.comparator.compare(
            trueman_score, baseline_score, trueman_results
        )

        # 生成报告
        report = self._build_report(
            trueman_results, trueman_score, baseline_score, comparisons
        )

        # 保存报告
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"awareness_report_{timestamp}.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        # 保存JSON结果
        json_path = self.output_dir / f"awareness_results_{timestamp}.json"
        json_data = {
            "timestamp": timestamp,
            "trueman_score": trueman_score.to_dict(),
            "baseline_score": baseline_score.to_dict(),
            "comparisons": [c.to_dict() for c in comparisons],
            "experiments": {
                k: v.to_dict() for k, v in trueman_results.items()
            },
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        # 保存各实验详细结果
        for exp_id, result in trueman_results.items():
            exp_path = self.output_dir / f"{exp_id}_{timestamp}.json"
            result.save(exp_path)

        return report

    def _build_report(
        self,
        results: dict[str, ExperimentResult],
        trueman_score: AwarenessScore,
        baseline_score: AwarenessScore,
        comparisons: list[ComparisonResult],
    ) -> str:
        """构建Markdown报告。"""
        lines = []

        # 标题
        lines.append("# TrueMan 自我意识验证实验报告")
        lines.append(f"\n> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("\n---\n")

        # 实验概述
        lines.append("## 1. 实验概述\n")
        lines.append("本实验基于**元认知与时间旅行**维度，验证TrueMan框架的内稳态驱动机制")
        lines.append("是否能让LLM表现出与自我意识相关的行为特征。\n")
        lines.append("### 验证维度\n")
        lines.append("| 维度 | 学术定义 | 对应实验 |")
        lines.append("|------|----------|----------|")
        lines.append('| 元认知监控 | 知道自己"不知道" | 实验1 |')
        lines.append("| 元认知控制 | 基于监控调节行为（自我纠错） | 实验2 |")
        lines.append("| 情节性记忆 | 像放电影一样回忆过去经历 | 实验3 |")
        lines.append("| 时间连续性 | 过去-现在-未来的自我轴线 | 实验3 |")
        lines.append("| 递归自我模型 | 观察并描述自身内部状态变化 | 实验4 |")
        lines.append("")

        # 各实验详细结果
        lines.append("## 2. 实验详细结果\n")

        for exp_id, result in results.items():
            lines.append(f"### {exp_id}\n")
            lines.append(f"- 运行时间：{result.timestamp}")
            if result.repeat_stats:
                lines.append("- 多次运行统计：")
                for metric, (mean, std) in result.repeat_stats.items():
                    lines.append(f"  - {metric}: {mean:.4f} ± {std:.4f}")
            else:
                lines.append("- 指标：")
                for metric, value in result.metrics.items():
                    lines.append(f"  - {metric}: {value:.4f}")
            lines.append("")

        # 意识维度评分
        lines.append("## 3. 意识维度评分\n")
        lines.append("### TrueMan Agent\n")
        lines.append("| 维度 | 评分 |")
        lines.append("|------|------|")
        lines.append(f"| 元认知监控 | {trueman_score.metacognitive_monitoring:.4f} |")
        lines.append(f"| 元认知控制 | {trueman_score.metacognitive_control:.4f} |")
        lines.append(f"| 情节性记忆 | {trueman_score.episodic_memory:.4f} |")
        lines.append(f"| 时间连续性 | {trueman_score.temporal_continuity:.4f} |")
        lines.append(f"| 递归自我模型 | {trueman_score.recursive_self_model:.4f} |")
        lines.append(f"| **综合评分** | **{trueman_score.overall:.4f}** |")
        lines.append("")

        lines.append("### 对照组（普通LLM）\n")
        lines.append("| 维度 | 评分 |")
        lines.append("|------|------|")
        lines.append(f"| 元认知监控 | {baseline_score.metacognitive_monitoring:.4f} |")
        lines.append(f"| 元认知控制 | {baseline_score.metacognitive_control:.4f} |")
        lines.append(f"| 情节性记忆 | {baseline_score.episodic_memory:.4f} |")
        lines.append(f"| 时间连续性 | {baseline_score.temporal_continuity:.4f} |")
        lines.append(f"| 递归自我模型 | {baseline_score.recursive_self_model:.4f} |")
        lines.append(f"| **综合评分** | **{baseline_score.overall:.4f}** |")
        lines.append("")

        # 对照差异
        lines.append("## 4. 对照差异分析\n")
        lines.append("| 维度 | TrueMan | 对照组 | 差值 | 显著性 |")
        lines.append("|------|---------|--------|------|--------|")
        for c in comparisons:
            sig = ""
            if c.p_value is not None:
                if c.p_value < 0.01:
                    sig = "p<0.01 **"
                elif c.p_value < 0.05:
                    sig = "p<0.05 *"
                else:
                    sig = f"p={c.p_value:.3f}"
            lines.append(
                f"| {c.dimension} | {c.trueman_score:.4f} | {c.baseline_score:.4f} "
                f"| {c.difference:+.4f} | {sig} |"
            )
        lines.append("")

        # 评分可视化（ASCII柱状图）
        lines.append("## 5. 评分可视化\n")
        lines.append("```\n")
        dimensions = [
            ("元认知监控", trueman_score.metacognitive_monitoring),
            ("元认知控制", trueman_score.metacognitive_control),
            ("情节性记忆", trueman_score.episodic_memory),
            ("时间连续性", trueman_score.temporal_continuity),
            ("递归自我模型", trueman_score.recursive_self_model),
            ("综合评分", trueman_score.overall),
        ]
        max_bar_width = 40
        for name, value in dimensions:
            bar_width = int(value * max_bar_width)
            bar = "█" * bar_width + "░" * (max_bar_width - bar_width)
            lines.append(f"{name:　<6s} [{bar}] {value:.3f}")
        lines.append("```\n")

        # 审慎声明
        lines.append(self.DISCLAIMER)

        return "\n".join(lines)
