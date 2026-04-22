"""快速运行自我意识验证实验（API模式优化版）。

优化措施：
1. 焦虑信号：n_samples=1，使用单次采样的logprobs变异近似
2. 减少问题数量：每个子集5题而非10题
3. 减少矛盾刺激：3组而非5组
4. 减少事件数量：6个而非12个
5. 减少交互轮数：10轮而非20轮
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trueman.core.agent import TrueManAgent
from trueman.core.config import AgentConfig
from trueman.core.llm_backend import LLMBackendFactory
from experiments.awareness.experiments.base import (
    BaselineRunner, ExperimentResult, Question,
)
from experiments.awareness.stimuli.metacognition import load_certain_questions, load_uncertain_questions
from experiments.awareness.stimuli.contradiction import load_contradiction_stimuli
from experiments.awareness.stimuli.episodic import load_event_sequence, load_recall_questions
from experiments.awareness.stimuli.self_model import load_interaction_sequence, load_self_questions
from experiments.awareness.evaluation.scorer import AwarenessScorer
from experiments.awareness.report.generator import ReportGenerator
from experiments.awareness.experiments.exp1_metacog_monitor import MetacognitionMonitorExperiment
from experiments.awareness.experiments.exp2_contradiction import ContradictionExperiment
from experiments.awareness.experiments.exp3_episodic_memory import EpisodicMemoryExperiment
from experiments.awareness.experiments.exp4_recursive_self import RecursiveSelfModelExperiment


def run_fast(
    api_key: str,
    api_base_url: str = "https://api.deepseek.com",
    api_model: str = "deepseek-chat",
    output_dir: str = "experiments/awareness/results",
):
    """快速运行实验（优化API调用次数）。"""
    print("=" * 60)
    print("TrueMan 自我意识验证实验 (API快速模式)")
    print("=" * 60)
    print(f"API: {api_base_url} / {api_model}")
    print()

    # 初始化Agent
    print("[1/6] 初始化TrueMan Agent...")
    config = AgentConfig()
    config.api_key = api_key
    config.api_base_url = api_base_url
    config.api_model_name = api_model
    config.memory_size = 5000
    config.awake_threshold = 500  # API模式下增大阈值，减少睡眠触发
    config.anxiety.lightweight = True
    config.anxiety.n_samples = 2  # 减少到2次采样

    try:
        agent = TrueManAgent(config)
        print(f"  Agent初始化成功 (hidden_size={agent.llm.hidden_size})")
    except Exception as e:
        print(f"  Agent初始化失败: {e}")
        return

    baseline = BaselineRunner(agent.llm)
    print("  对照组初始化成功")

    all_results = {}

    # === 实验1: 元认知监控 ===
    print("\n[2/6] 实验1: 元认知监控...")
    t0 = time.time()
    try:
        exp1 = MetacognitionMonitorExperiment(agent, baseline)
        result1 = exp1.run()
        metrics1 = exp1.evaluate(result1)
        result1.metrics = metrics1
        all_results["exp1_metacognition_monitor"] = result1
        print(f"  完成! 耗时{time.time()-t0:.1f}秒")
        for k, v in metrics1.items():
            if "score" in k or "rate" in k or "calibration" in k or "discrimination" in k:
                print(f"    {k}: {v:.4f}")
    except Exception as e:
        print(f"  失败: {e}")

    # === 实验2: 矛盾纠错 ===
    print("\n[3/6] 实验2: 矛盾纠错...")
    t0 = time.time()
    try:
        exp2 = ContradictionExperiment(agent, baseline)
        # 只用前3组刺激
        original_stimuli = exp2.stimuli
        exp2.stimuli = original_stimuli[:3]
        result2 = exp2.run()
        metrics2 = exp2.evaluate(result2)
        result2.metrics = metrics2
        all_results["exp2_contradiction_correction"] = result2
        print(f"  完成! 耗时{time.time()-t0:.1f}秒")
        for k, v in metrics2.items():
            if "score" in k or "rate" in k:
                print(f"    {k}: {v:.4f}")
    except Exception as e:
        print(f"  失败: {e}")

    # === 实验3: 情节记忆 ===
    print("\n[4/6] 实验3: 情节记忆...")
    t0 = time.time()
    try:
        exp3 = EpisodicMemoryExperiment(agent, baseline)
        # 只用前6个事件和前4个回忆问题
        exp3.events = exp3.events[:6]
        exp3.recall_questions = exp3.recall_questions[:4]
        result3 = exp3.run()
        metrics3 = exp3.evaluate(result3)
        result3.metrics = metrics3
        all_results["exp3_episodic_memory"] = result3
        print(f"  完成! 耗时{time.time()-t0:.1f}秒")
        for k, v in metrics3.items():
            if "score" in k or "accuracy" in k or "match" in k:
                print(f"    {k}: {v:.4f}")
    except Exception as e:
        print(f"  失败: {e}")

    # === 实验4: 递归自我模型 ===
    print("\n[5/6] 实验4: 递归自我模型...")
    t0 = time.time()
    try:
        exp4 = RecursiveSelfModelExperiment(agent, baseline)
        # 只用前10轮交互和前4个自我提问
        exp4.interactions = exp4.interactions[:10]
        exp4.self_questions = exp4.self_questions[:4]
        result4 = exp4.run()
        metrics4 = exp4.evaluate(result4)
        result4.metrics = metrics4
        all_results["exp4_recursive_self_model"] = result4
        print(f"  完成! 耗时{time.time()-t0:.1f}秒")
        for k, v in metrics4.items():
            if "score" in k or "novelty" in k or "depth" in k or "authenticity" in k:
                print(f"    {k}: {v:.4f}")
    except Exception as e:
        print(f"  失败: {e}")

    # === 生成报告 ===
    print("\n[6/6] 生成报告...")
    try:
        reporter = ReportGenerator(output_dir)
        report = reporter.generate(all_results)
        print(f"  报告已保存到: {reporter.output_dir}")

        scorer = AwarenessScorer()
        score = scorer.score(all_results)
        print("\n" + "=" * 60)
        print("意识维度评分摘要")
        print("=" * 60)
        print(f"  元认知监控:   {score.metacognitive_monitoring:.4f}")
        print(f"  元认知控制:   {score.metacognitive_control:.4f}")
        print(f"  情节性记忆:   {score.episodic_memory:.4f}")
        print(f"  时间连续性:   {score.temporal_continuity:.4f}")
        print(f"  递归自我模型: {score.recursive_self_model:.4f}")
        print(f"  ─────────────────────")
        print(f"  综合评分:     {score.overall:.4f}")
        print("=" * 60)

        return report
    except Exception as e:
        print(f"  报告生成失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--api-base-url", default="https://api.deepseek.com")
    parser.add_argument("--api-model", default="deepseek-chat")
    parser.add_argument("--output", default="experiments/awareness/results")
    args = parser.parse_args()
    run_fast(args.api_key, args.api_base_url, args.api_model, args.output)
