"""一键运行所有自我意识验证实验。

用法:
    # 本地模型
    python -m experiments.awareness.run_all --model Qwen/Qwen2.5-7B-Instruct --device cuda --repeats 3

    # 云端API（DeepSeek）
    python -m experiments.awareness.run_all --api --api-key YOUR_KEY --api-base-url https://api.deepseek.com --api-model deepseek-chat

    # 云端API（OpenAI）
    python -m experiments.awareness.run_all --api --api-key YOUR_KEY --api-base-url https://api.openai.com/v1 --api-model gpt-4o
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 确保项目根目录在sys.path中
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trueman.core.agent import TrueManAgent
from trueman.core.config import AgentConfig
from trueman.core.llm_backend import LLMBackendFactory
from experiments.awareness.experiments.base import BaselineRunner
from experiments.awareness.experiments.exp1_metacog_monitor import MetacognitionMonitorExperiment
from experiments.awareness.experiments.exp2_contradiction import ContradictionExperiment
from experiments.awareness.experiments.exp3_episodic_memory import EpisodicMemoryExperiment
from experiments.awareness.experiments.exp4_recursive_self import RecursiveSelfModelExperiment
from experiments.awareness.report.generator import ReportGenerator


def create_agent_config(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    device: str = "cuda",
    quantization: str | None = None,
    # API模式参数
    use_api: bool = False,
    api_key: str = "",
    api_base_url: str = "",
    api_model: str = "",
) -> AgentConfig:
    """创建Agent配置。"""
    config = AgentConfig()
    config.base_model_name = model_name
    config.device = device

    if use_api:
        config.api_key = api_key
        config.api_base_url = api_base_url
        config.api_model_name = api_model or model_name
    elif quantization:
        if quantization == "4bit":
            config.load_in_4bit = True
        elif quantization == "8bit":
            config.load_in_8bit = True

    # 减小记忆和清醒阈值以加快实验
    config.memory_size = 5000
    config.awake_threshold = 200

    # API模式下确保使用轻量焦虑计算
    if use_api:
        config.anxiety.lightweight = True

    return config


def run_experiments(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    device: str = "cuda",
    quantization: str | None = None,
    n_repeats: int = 1,
    output_dir: str = "experiments/awareness/results",
    # API模式参数
    use_api: bool = False,
    api_key: str = "",
    api_base_url: str = "",
    api_model: str = "",
) -> None:
    """运行所有自我意识验证实验。"""
    print("=" * 60)
    print("TrueMan 自我意识验证实验")
    print("=" * 60)

    if use_api:
        print(f"模式: API (云端模型)")
        print(f"API Base URL: {api_base_url}")
        print(f"API Model: {api_model or model_name}")
    else:
        print(f"模式: 本地模型")
        print(f"模型: {model_name}")
        print(f"设备: {device}")
        print(f"量化: {quantization or '无'}")

    print(f"重复次数: {n_repeats}")
    print()

    # 1. 初始化TrueMan Agent
    print("[1/6] 初始化TrueMan Agent...")
    config = create_agent_config(
        model_name, device, quantization,
        use_api, api_key, api_base_url, api_model,
    )
    try:
        agent = TrueManAgent(config)
        mode_str = "API" if use_api else "本地"
        print(f"  Agent初始化成功 ({mode_str}模式, hidden_size={agent.llm.hidden_size})")
    except Exception as e:
        print(f"  Agent初始化失败: {e}")
        if use_api:
            print("  请检查API Key和Base URL是否正确")
            print("  DeepSeek: --api-base-url https://api.deepseek.com --api-model deepseek-chat")
        else:
            print("  请检查模型名称和设备是否正确")
        return

    # 2. 初始化对照组LLM
    print("[2/6] 初始化对照组LLM...")
    baseline_runner = BaselineRunner(agent.llm)  # 共享同一个LLM实例
    print("  对照组初始化成功（共享LLM实例）")

    # 3. 运行4个实验
    all_results = {}
    experiments = [
        ("实验1: 元认知监控", MetacognitionMonitorExperiment(agent, baseline_runner)),
        ("实验2: 矛盾纠错", ContradictionExperiment(agent, baseline_runner)),
        ("实验3: 情节记忆", EpisodicMemoryExperiment(agent, baseline_runner)),
        ("实验4: 递归自我模型", RecursiveSelfModelExperiment(agent, baseline_runner)),
    ]

    for i, (name, experiment) in enumerate(experiments, start=3):
        print(f"\n[{i}/6] 运行{name}...")
        start_time = time.time()

        try:
            if n_repeats > 1:
                result = experiment.run_with_repeats(n_repeats)
            else:
                result = experiment.run()
                metrics = experiment.evaluate(result)
                result.metrics = metrics

            elapsed = time.time() - start_time
            all_results[result.experiment_id] = result

            print(f"  完成! 耗时{elapsed:.1f}秒")
            print(f"  关键指标:")
            for k, v in result.metrics.items():
                if "score" in k:
                    print(f"    {k}: {v:.4f}")

        except Exception as e:
            print(f"  实验失败: {e}")
            import traceback
            traceback.print_exc()

    # 4. 生成报告
    print(f"\n[6/6] 生成实验报告...")
    try:
        reporter = ReportGenerator(output_dir)
        report = reporter.generate(all_results)
        print("  报告生成成功!")
        print(f"  保存位置: {reporter.output_dir}")

        # 打印评分摘要
        from experiments.awareness.evaluation.scorer import AwarenessScorer
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

    except Exception as e:
        print(f"  报告生成失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="TrueMan 自我意识验证实验")

    # 模式选择
    mode_group = parser.add_argument_group("模式选择")
    mode_group.add_argument(
        "--api", action="store_true",
        help="使用云端API模式（需提供--api-key和--api-base-url）"
    )

    # 本地模型参数
    local_group = parser.add_argument_group("本地模型参数")
    local_group.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace模型名称"
    )
    local_group.add_argument(
        "--device", type=str, default="cuda",
        help="计算设备 (cuda/cpu)"
    )
    local_group.add_argument(
        "--quantization", type=str, default=None,
        choices=["4bit", "8bit"],
        help="量化方式"
    )

    # API参数
    api_group = parser.add_argument_group("API模式参数")
    api_group.add_argument(
        "--api-key", type=str, default="",
        help="API Key（如DeepSeek API Key）"
    )
    api_group.add_argument(
        "--api-base-url", type=str, default="",
        help="API Base URL（如https://api.deepseek.com）"
    )
    api_group.add_argument(
        "--api-model", type=str, default="",
        help="API中的模型名称（如deepseek-chat）"
    )

    # 通用参数
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="每个实验的重复次数"
    )
    parser.add_argument(
        "--output", type=str, default="experiments/awareness/results",
        help="结果输出目录"
    )

    args = parser.parse_args()

    # 验证API模式参数
    if args.api:
        if not args.api_key:
            parser.error("API模式需要提供 --api-key")
        if not args.api_base_url:
            parser.error("API模式需要提供 --api-base-url")

    run_experiments(
        model_name=args.model,
        device=args.device,
        quantization=args.quantization,
        n_repeats=args.repeats,
        output_dir=args.output,
        use_api=args.api,
        api_key=args.api_key,
        api_base_url=args.api_base_url,
        api_model=args.api_model,
    )


if __name__ == "__main__":
    main()
