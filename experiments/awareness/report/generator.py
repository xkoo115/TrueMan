"""实验报告生成器（v2：支持统计检验、消融、阴性对照）。"""

from __future__ import annotations

import json
import time
from pathlib import Path

from experiments.awareness.experiments.base import ExperimentResult, AwarenessScore, ComparisonResult
from experiments.awareness.evaluation.statistics import StatisticalReport


class ReportGenerator:
    """实验报告生成器。"""

    def __init__(self, output_dir: str = "experiments/awareness/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        trueman_results: dict[str, ExperimentResult],
        baseline_results: dict[str, ExperimentResult] | None = None,
        comparison_results: list[ComparisonResult] | None = None,
        statistical_reports: list[StatisticalReport] | None = None,
        ablation_results: dict[str, dict] | None = None,
        negative_control_results: dict[str, dict] | None = None,
        lora_results: dict | None = None,
    ) -> str:
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        report_parts = []
        report_parts.append("=" * 70)
        report_parts.append("TrueMan 严格实验报告")
        report_parts.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_parts.append("=" * 70)

        report_parts.append("\n## 一、主实验结果\n")
        report_parts.append(self._format_main_results(trueman_results))

        if comparison_results:
            report_parts.append("\n## 二、基线对比\n")
            report_parts.append(self._format_comparisons(comparison_results))

        if statistical_reports:
            report_parts.append("\n## 三、统计检验\n")
            report_parts.append(self._format_statistics(statistical_reports))

        if ablation_results:
            report_parts.append("\n## 四、消融实验\n")
            report_parts.append(self._format_ablation(ablation_results))

        if negative_control_results:
            report_parts.append("\n## 五、阴性对照\n")
            report_parts.append(self._format_negative_controls(negative_control_results))

        if lora_results:
            report_parts.append("\n## 六、LoRA验证\n")
            report_parts.append(self._format_lora(lora_results))

        report_parts.append("\n## 免责声明")
        report_parts.append("本实验框架测量的是行为指标，行为指标通过不意味着系统拥有主观意识。")
        report_parts.append("如论文所述，这些是自我意识的必要条件而非充分条件。")

        report_text = "\n".join(report_parts)

        report_path = self.output_dir / f"rigorous_report_{timestamp}.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        all_data = {
            "timestamp": timestamp,
            "trueman_results": {k: v.to_dict() for k, v in trueman_results.items()},
        }
        if baseline_results:
            all_data["baseline_results"] = {k: v.to_dict() for k, v in baseline_results.items()}
        if comparison_results:
            all_data["comparisons"] = [c.to_dict() for c in comparison_results]
        if statistical_reports:
            all_data["statistics"] = [r.to_dict() for r in statistical_reports]
        if ablation_results:
            all_data["ablation"] = ablation_results
        if negative_control_results:
            all_data["negative_controls"] = negative_control_results
        if lora_results:
            all_data["lora"] = lora_results

        json_path = self.output_dir / f"rigorous_results_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2, default=str)

        for exp_id, result in trueman_results.items():
            exp_path = self.output_dir / f"{exp_id}_{timestamp}.json"
            result.save(exp_path)

        print(f"  报告已保存:")
        print(f"    Markdown: {report_path}")
        print(f"    JSON:     {json_path}")

        return report_text

    def _format_main_results(self, results: dict[str, ExperimentResult]) -> str:
        lines = []
        for exp_id, result in results.items():
            lines.append(f"\n### {exp_id}")
            if result.metrics:
                lines.append("| 指标 | 值 |")
                lines.append("|------|-----|")
                for k, v in sorted(result.metrics.items()):
                    marker = "**" if "score" in k else ""
                    lines.append(f"| {marker}{k}{marker} | {v:.4f} |")
            if result.repeat_stats:
                lines.append("\n重复统计:")
                for k, (mean, std) in result.repeat_stats.items():
                    lines.append(f"  {k}: {mean:.4f} ± {std:.4f}")
        return "\n".join(lines)

    def _format_comparisons(self, comparisons: list[ComparisonResult]) -> str:
        lines = ["| 维度 | TrueMan | 基线 | 差异 | p值 | Cohen's d |", "|------|---------|------|------|-----|-----------|"]
        for c in comparisons:
            p_str = f"{c.p_value:.4f}" if c.p_value is not None else "N/A"
            d_str = f"{c.cohens_d:.4f}" if c.cohens_d else "N/A"
            sig = " *" if c.significant else ""
            lines.append(f"| {c.dimension} | {c.trueman_score:.4f} | {c.baseline_score:.4f} | {c.difference:+.4f}{sig} | {p_str} | {d_str} |")
        lines.append("\n* 表示经Bonferroni校正后显著 (p < 0.05)")
        return "\n".join(lines)

    def _format_statistics(self, reports: list[StatisticalReport]) -> str:
        lines = ["| 比较对 | 指标 | 均值A | 均值B | 95% CI | p值 | Cohen's d | 显著 |",
                 "|--------|------|-------|-------|--------|-----|-----------|------|"]
        for r in reports:
            ci = f"[{r.ci_lower:.4f}, {r.ci_upper:.4f}]"
            p = f"{r.p_value:.4f}" if r.p_value is not None else "N/A"
            lines.append(f"| {r.group_a_name} vs {r.group_b_name} | {r.metric_name} | {r.mean_a:.4f} | {r.mean_b:.4f} | {ci} | {p} | {r.cohens_d:.4f} | {'是' if r.significant else '否'} |")
        return "\n".join(lines)

    def _format_ablation(self, results: dict[str, dict]) -> str:
        lines = ["\n消融条件 vs TrueMan完整系统的差异：\n"]
        lines.append("| 消融条件 | 禁用组件 | 综合评分 | vs TrueMan |")
        lines.append("|----------|---------|---------|------------|")
        trueman_score = results.get("trueman", {}).get("overall", 0)
        for name, data in results.items():
            if name == "trueman":
                continue
            score = data.get("overall", 0)
            diff = score - trueman_score
            component = name.replace("A", "组件").replace("_", " ")
            lines.append(f"| {name} | {component} | {score:.4f} | {diff:+.4f} |")
        return "\n".join(lines)

    def _format_negative_controls(self, results: dict[str, dict]) -> str:
        lines = ["\n阴性对照结果（得分应显著低于TrueMan）：\n"]
        lines.append("| 对照条件 | 描述 | 综合评分 | vs TrueMan |")
        lines.append("|----------|------|---------|------------|")
        trueman_score = results.get("trueman", {}).get("overall", 0)
        for name, data in results.items():
            if name == "trueman":
                continue
            score = data.get("overall", 0)
            diff = score - trueman_score
            lines.append(f"| {name} | {data.get('description', '')} | {score:.4f} | {diff:+.4f} |")

        if "overclaiming" in results:
            oc = results["overclaiming"]
            lines.append(f"\n过度声称检测: 否认率={oc.get('overclaiming_denial_rate', 0):.4f}, 虚构率={oc.get('overclaiming_fabrication_rate', 0):.4f}")
        return "\n".join(lines)

    def _format_lora(self, results: dict) -> str:
        lines = ["\nLoRA可塑性验证结果：\n"]
        for key, value in results.items():
            if isinstance(value, dict):
                lines.append(f"\n### {key}")
                for k, v in value.items():
                    lines.append(f"  {k}: {v}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)
