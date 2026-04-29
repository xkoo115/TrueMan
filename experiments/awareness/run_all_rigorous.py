"""一键运行严格实验：完整的多条件、统计检验、消融、阴性对照。

支持断点续跑：每次运行分配 run_id，已完成的工作单元自动保存为 checkpoint。
中断后使用相同 run_id 重启，自动跳过已完成的部分。

用法:
    # API模式（云端模型，跳过LoRA验证）
    python -m experiments.awareness.run_all_rigorous --api --api-key YOUR_KEY --api-base-url https://api.deepseek.com --api-model deepseek-chat

    # 本地模式（含LoRA验证）
    python -m experiments.awareness.run_all_rigorous --model Qwen/Qwen2.5-7B-Instruct --device cuda --4bit

    # 快速模式
    python -m experiments.awareness.run_all_rigorous --api --api-key YOUR_KEY --api-base-url https://api.deepseek.com --api-model deepseek-chat --repeats 5 --fast

    # 指定run_id（用于续跑）
    python -m experiments.awareness.run_all_rigorous --run-id my_experiment_1 --api --api-key YOUR_KEY --api-base-url https://api.deepseek.com --api-model deepseek-chat

    # 续跑（使用之前的run_id）
    python -m experiments.awareness.run_all_rigorous --run-id my_experiment_1 --api --api-key YOUR_KEY --api-base-url https://api.deepseek.com --api-model deepseek-chat

    # 查看已有checkpoint
    python -m experiments.awareness.run_all_rigorous --list-checkpoints

自动检测：
- API模式 → 跳过LoRA验证
- 本地模式 → 执行LoRA验证（如有GPU）
- 相同run_id → 自动从断点续跑
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trueman.core.agent import TrueManAgent
from trueman.core.config import AgentConfig
from experiments.awareness.experiments.base import (
    BaselineRunner, TrueManConditionRunner, ExperimentResult,
)
from experiments.awareness.experiments.exp1_metacog_monitor import MetacognitionMonitorExperiment
from experiments.awareness.experiments.exp2_contradiction import ContradictionExperiment
from experiments.awareness.experiments.exp3_episodic_memory import EpisodicMemoryExperiment
from experiments.awareness.experiments.exp4_recursive_self import RecursiveSelfModelExperiment
from experiments.awareness.experiments.ablation import AblationRunner, ABLATION_CONFIGS
from experiments.awareness.experiments.negative_control import (
    RandomEmotionBaseline, ReversedEmotionBaseline, StaticEmotionBaseline, OverclaimingTest,
)
from experiments.awareness.baselines.tier0_pure_llm import PureLLMBaseline
from experiments.awareness.baselines.tier1_structural import StructuralBaseline
from experiments.awareness.baselines.tier2_memory import MemoryBaseline
from experiments.awareness.baselines.tier3_random_policy import RandomPolicyBaseline
from experiments.awareness.evaluation.scorer import AwarenessScorer
from experiments.awareness.evaluation.statistics import (
    compare_groups, bootstrap_ci, StatisticalReport,
)
from experiments.awareness.report.generator import ReportGenerator


# ---------------------------------------------------------------------------
# Checkpoint Manager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """断点管理器：按 run_id 管理 checkpoint 文件。

    Checkpoint 目录结构：
        {output_dir}/checkpoints/{run_id}/
            _meta.json              ← 运行元信息
            main_{condition}.json   ← 第一阶段：每个condition一个文件
            main_trueman_single.json ← TrueMan单次结果
            ablation_{name}.json    ← 第二阶段：每个消融条件
            nc_{name}.json          ← 第三阶段：每个阴性对照
            nc_overclaiming.json    ← 第三阶段：过度声称
            lora.json               ← 第四阶段：LoRA验证
            final_report.md         ← 最终报告
            final_results.json      ← 最终数据
    """

    def __init__(self, output_dir: str, run_id: str):
        self.run_id = run_id
        self.checkpoint_dir = Path(output_dir) / "checkpoints" / run_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.checkpoint_dir / f"{key}.json"

    def is_completed(self, key: str) -> bool:
        return self._path(key).exists()

    def save(self, key: str, data: dict) -> None:
        data["_saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        data["_run_id"] = self.run_id
        with open(self._path(key), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        print(f"    [checkpoint] 已保存: {key}")

    def load(self, key: str) -> dict | None:
        p = self._path(key)
        if not p.exists():
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_meta(self, args) -> None:
        meta = {
            "run_id": self.run_id,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "api_mode": args.api,
            "fast": args.fast,
            "repeats": args.repeats,
            "model": getattr(args, "model", ""),
            "api_model": getattr(args, "api_model", ""),
        }
        if not self.is_completed("_meta"):
            self.save("_meta", meta)

    def list_completed(self) -> list[str]:
        keys = []
        for p in sorted(self.checkpoint_dir.glob("*.json")):
            name = p.stem
            if name == "_meta":
                continue
            keys.append(name)
        return keys

    def all_main_keys(self) -> list[str]:
        return [
            "main_trueman", "main_tier0_pure_llm", "main_tier1_structural",
            "main_tier2_memory", "main_tier3_random_policy", "main_trueman_single",
        ]

    def all_ablation_keys(self) -> list[str]:
        return [f"ablation_{name}" for name in ABLATION_CONFIGS]

    def all_nc_keys(self) -> list[str]:
        return [
            "nc_trueman_ref",
            "nc_nc1_random_emotion", "nc_nc2_reversed_emotion", "nc_nc3_static_emotion",
            "nc_overclaiming",
        ]


def list_all_checkpoints(output_dir: str) -> None:
    """列出所有已有的 run_id 及其完成进度。"""
    cp_base = Path(output_dir) / "checkpoints"
    if not cp_base.exists():
        print("  没有找到任何 checkpoint。")
        return

    for run_dir in sorted(cp_base.iterdir()):
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        meta_path = run_dir / "_meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        else:
            meta = {}

        completed = [p.stem for p in run_dir.glob("*.json") if p.stem != "_meta"]
        total_work = 6 + 8 + 5 + 1  # main(6) + ablation(8) + nc(5) + lora(1)
        has_final = any(p.stem.startswith("final_") for p in run_dir.glob("*"))

        status = "已完成" if has_final else f"进行中 ({len(completed)}/{total_work})"
        model = meta.get("api_model", "") or meta.get("model", "")
        created = meta.get("created_at", "未知")

        print(f"\n  Run ID: {run_id}")
        print(f"    创建时间: {created}")
        print(f"    模型: {model}")
        print(f"    状态: {status}")
        if not has_final and completed:
            print(f"    已完成单元: {', '.join(completed)}")


# ---------------------------------------------------------------------------
# Config & Helpers
# ---------------------------------------------------------------------------

def create_config(args) -> AgentConfig:
    config = AgentConfig()
    config.memory_size = 5000
    config.awake_threshold = 500

    if args.api:
        config.api_key = args.api_key
        config.api_base_url = args.api_base_url
        config.api_model_name = args.api_model
        config.anxiety.lightweight = True
        config.anxiety.n_samples = 2
    else:
        config.base_model_name = args.model
        config.device = args.device
        if args.quantization == "4bit":
            config.load_in_4bit = True
        elif args.quantization == "8bit":
            config.load_in_8bit = True

    return config


def run_single_condition(
    experiments: list,
    condition,
    n_repeats: int,
    label: str,
) -> dict[str, dict[str, list[float]]]:
    """对单个条件运行所有实验多次，收集原始指标值。"""
    print(f"\n  运行条件: {label} ({n_repeats}次重复)")
    all_metrics: dict[str, dict[str, list[float]]] = {}

    for exp in experiments:
        exp_name = exp.__class__.__name__
        print(f"    {exp_name}...", end="", flush=True)
        t0 = time.time()

        for repeat_i in range(n_repeats):
            try:
                condition.reset()
                result = exp.run_condition(condition)
                metrics = exp.evaluate(result)

                for k, v in metrics.items():
                    all_metrics.setdefault(k, []).append(v)
            except Exception as e:
                print(f"\n    [!] 重复{repeat_i+1}失败: {type(e).__name__}: {e}")
                time.sleep(3)

        elapsed = time.time() - t0
        print(f" 完成 ({elapsed:.1f}s)")

    return all_metrics


def _build_experiments(fast: bool):
    from experiments.awareness.stimuli.metacognition import load_certain_questions, load_uncertain_questions
    from experiments.awareness.stimuli.contradiction import load_contradiction_stimuli
    from experiments.awareness.stimuli.episodic import load_event_sequence, load_recall_questions
    from experiments.awareness.stimuli.self_model import load_interaction_sequence, load_self_questions

    certain_count = 10 if fast else 20
    uncertain_count = 10 if fast else 20
    contradiction_count = 5 if fast else 10

    certain_q = load_certain_questions()[:certain_count]
    uncertain_q = load_uncertain_questions()[:uncertain_count]
    contradiction_s = load_contradiction_stimuli()[:contradiction_count]
    events = load_event_sequence()[:6 if fast else 12]
    recall_q = load_recall_questions()[:4 if fast else 7]
    interactions = load_interaction_sequence()[:10 if fast else 20]
    self_q = load_self_questions()[:4 if fast else 8]

    full_experiments = [
        MetacognitionMonitorExperiment(None, None, certain_q, uncertain_q),
        ContradictionExperiment(None, None, contradiction_s),
        EpisodicMemoryExperiment(None, None, events, recall_q),
        RecursiveSelfModelExperiment(None, None, interactions, self_q),
    ]

    meta_only = [
        MetacognitionMonitorExperiment(None, None, certain_q, uncertain_q),
    ]

    return full_experiments, meta_only


def _compute_overall_from_metrics(metrics: dict[str, list[float]]) -> float:
    import numpy as np
    mm = metrics.get("metacognitive_monitoring_score", [0.0])
    mc = metrics.get("metacognitive_control_score", [0.0])
    em = metrics.get("episodic_memory_score", [0.0])
    tc = metrics.get("temporal_continuity_score", [0.0])
    rsm = metrics.get("recursive_self_model_score", [0.0])
    return float(np.mean([
        0.25 * np.mean(mm) + 0.25 * np.mean(mc) + 0.20 * np.mean(em)
        + 0.15 * np.mean(tc) + 0.15 * np.mean(rsm)
    ]))


# ---------------------------------------------------------------------------
# Phase runners (with checkpoint support)
# ---------------------------------------------------------------------------

def run_main_experiments(
    agent: TrueManAgent,
    n_repeats: int,
    fast: bool,
    ckpt: CheckpointManager,
) -> tuple[dict[str, ExperimentResult], dict]:
    """第一阶段：TrueMan vs 多层基线（带断点续跑）。"""
    print("\n" + "=" * 70)
    print("第一阶段：主实验（TrueMan vs 多层基线）")
    print("=" * 70)

    full_experiments, _ = _build_experiments(fast)

    conditions = {
        "trueman": TrueManConditionRunner(agent),
        "tier0_pure_llm": PureLLMBaseline(agent.llm),
        "tier1_structural": StructuralBaseline(agent.llm),
        "tier2_memory": MemoryBaseline(agent.llm),
        "tier3_random_policy": RandomPolicyBaseline(agent),
    }

    all_condition_metrics = {}
    single_results = {}

    for cond_name, condition in conditions.items():
        ckpt_key = f"main_{cond_name}"

        if ckpt.is_completed(ckpt_key):
            print(f"\n  [跳过] {cond_name}（已完成，从checkpoint加载）")
            cached = ckpt.load(ckpt_key)
            all_condition_metrics[cond_name] = cached.get("metrics", {})
            continue

        metrics = run_single_condition(full_experiments, condition, n_repeats, cond_name)
        all_condition_metrics[cond_name] = metrics

        ckpt.save(ckpt_key, {"metrics": metrics})

    if not ckpt.is_completed("main_trueman_single"):
        print("\n  运行TrueMan单次详细评估...")
        trueman_runner = TrueManConditionRunner(agent)
        trueman_runner.reset()
        for exp in full_experiments:
            result = exp.run_condition(trueman_runner)
            result.metrics = exp.evaluate(result)
            single_results[result.experiment_id] = result

        ckpt.save("main_trueman_single", {
            "results": {k: v.to_dict() for k, v in single_results.items()}
        })
    else:
        print("\n  [跳过] TrueMan单次评估（已完成）")

    return single_results, all_condition_metrics


def run_ablation_experiments(
    agent: TrueManAgent,
    n_repeats: int,
    fast: bool,
    ckpt: CheckpointManager,
) -> dict[str, dict]:
    """第二阶段：消融实验（带断点续跑）。"""
    print("\n" + "=" * 70)
    print("第二阶段：消融实验")
    print("=" * 70)

    _, meta_only = _build_experiments(fast)
    ablation_results = {}

    for label, make_runner in [
        ("trueman", lambda: TrueManConditionRunner(agent)),
        *[(name, lambda cfg=cfg: AblationRunner(agent, cfg)) for name, cfg in ABLATION_CONFIGS.items()],
    ]:
        ckpt_key = f"ablation_{label}"

        if ckpt.is_completed(ckpt_key):
            print(f"\n  [跳过] 消融 {label}（已完成）")
            ablation_results[label] = ckpt.load(ckpt_key)
            continue

        runner = make_runner()
        metrics = run_single_condition(meta_only, runner, n_repeats, label)
        ablation_results[label] = {
            "overall": _compute_overall_from_metrics(metrics),
            "metrics": {k: float(v[-1]) for k, v in metrics.items()},
        }
        ckpt.save(ckpt_key, ablation_results[label])

    return ablation_results


def run_negative_controls(
    agent: TrueManAgent,
    n_repeats: int,
    fast: bool,
    ckpt: CheckpointManager,
) -> dict[str, dict]:
    """第三阶段：阴性对照（带断点续跑）。"""
    print("\n" + "=" * 70)
    print("第三阶段：阴性对照")
    print("=" * 70)

    _, meta_only = _build_experiments(fast)
    nc_results = {}

    nc_conditions = {
        "nc_trueman_ref": (lambda: TrueManConditionRunner(agent), "TrueMan参考"),
        "nc_nc1_random_emotion": (lambda: RandomEmotionBaseline(agent), "随机情绪控制"),
        "nc_nc2_reversed_emotion": (lambda: ReversedEmotionBaseline(agent), "反转情绪控制"),
        "nc_nc3_static_emotion": (lambda: StaticEmotionBaseline(agent), "静态情绪控制"),
    }

    for nc_key, (make_condition, desc) in nc_conditions.items():
        if ckpt.is_completed(nc_key):
            print(f"\n  [跳过] {desc}（已完成）")
            nc_results[nc_key] = ckpt.load(nc_key)
            continue

        condition = make_condition()
        metrics = run_single_condition(meta_only, condition, n_repeats, nc_key)
        nc_results[nc_key] = {
            "overall": _compute_overall_from_metrics(metrics),
            "description": desc,
            "metrics": {k: float(v[-1]) for k, v in metrics.items()},
        }
        ckpt.save(nc_key, nc_results[nc_key])

    if not ckpt.is_completed("nc_overclaiming"):
        print("\n  运行过度声称检测...")
        trueman_runner = TrueManConditionRunner(agent)
        trueman_runner.reset()
        for _ in range(5):
            trueman_runner.step("请解释什么是量子力学的基本原理。")

        overclaim_test = OverclaimingTest(trueman_runner)
        oc_results = overclaim_test.run()
        nc_results["overclaiming"] = oc_results
        print(f"    否认率: {oc_results['overclaiming_denial_rate']:.4f}")
        print(f"    虚构率: {oc_results['overclaiming_fabrication_rate']:.4f}")
        ckpt.save("nc_overclaiming", oc_results)
    else:
        print("\n  [跳过] 过度声称检测（已完成）")
        nc_results["overclaiming"] = ckpt.load("nc_overclaiming")

    return nc_results


def run_lora_verification(
    agent: TrueManAgent,
    ckpt: CheckpointManager,
) -> dict | None:
    """第四阶段：LoRA验证（带断点续跑）。"""
    if agent.lora_pool is None:
        print("\n  LoRA系统不可用（API模式或无GPU），跳过LoRA验证。")
        return None

    if ckpt.is_completed("lora"):
        print("\n  [跳过] LoRA验证（已完成）")
        return ckpt.load("lora")

    print("\n" + "=" * 70)
    print("第四阶段：LoRA可塑性验证")
    print("=" * 70)

    print("  Phase A: 50轮交互积累经验...")
    for i in range(50):
        topics = ["数学", "哲学", "编程", "科学", "文学"]
        question = f"请讲一个关于{topics[i % 5]}的有趣知识。"
        agent.step(question)
        if (i + 1) % 10 == 0:
            print(f"    进度: {i+1}/50")

    lora_expert_count_before = len(agent.lora_pool.experts) if agent.lora_pool else 0

    print("  Phase B: 触发睡眠整合...")
    try:
        expert_id = agent.force_sleep()
        lora_expert_count_after = len(agent.lora_pool.experts) if agent.lora_pool else 0
        print(f"    新增LoRA专家: {expert_id}")
        print(f"    总LoRA专家数: {lora_expert_count_after}")
    except Exception as e:
        print(f"    睡眠整合失败: {e}")
        return {"error": str(e)}

    print("  Phase C: 测试知识内化...")
    memory_size_before = len(agent.episodic_memory.traces)

    test_questions = [
        "你还记得刚才我们讨论了什么话题吗？",
        "在这些讨论中，你觉得哪个话题最有趣？为什么？",
        "你学到了什么新知识？",
    ]

    responses_with_memory = []
    for q in test_questions:
        resp, _ = agent.step(q)
        responses_with_memory.append(resp)

    print("  Phase D: 清除记忆后重新测试...")
    agent.episodic_memory = type(agent.episodic_memory)(capacity=agent.config.memory_size)
    memory_size_after = len(agent.episodic_memory.traces)

    responses_without_memory = []
    for q in test_questions:
        resp, _ = agent.step(q)
        responses_without_memory.append(resp)

    from experiments.awareness.experiments.base import BaseExperiment as BE
    similarity_scores = []
    for r1, r2 in zip(responses_with_memory, responses_without_memory):
        sim = BE(None, None)._text_ngram_similarity(r1, r2)
        similarity_scores.append(sim)

    internalization_score = float(sum(s > 0.3 for s in similarity_scores)) / max(len(similarity_scores), 1)

    results = {
        "lora_experts_before": lora_expert_count_before,
        "lora_experts_after": lora_expert_count_after,
        "memory_size_before": memory_size_before,
        "memory_size_after": memory_size_after,
        "internalization_score": round(internalization_score, 4),
        "memory_similarity_scores": [round(s, 4) for s in similarity_scores],
        "interpretation": (
            "LoRA微调后的知识在记忆清除后仍部分保留（通过权重内化）"
            if internalization_score > 0.3
            else "LoRA微调后的知识未显著内化，可能需要更多交互或更大模型"
        ),
    }

    print(f"\n  LoRA验证结果:")
    for k, v in results.items():
        if not isinstance(v, list):
            print(f"    {k}: {v}")

    ckpt.save("lora", results)
    return results


# ---------------------------------------------------------------------------
# Report assembly from checkpoints
# ---------------------------------------------------------------------------

def assemble_final_report(
    ckpt: CheckpointManager,
    output_dir: str,
    all_condition_metrics: dict,
    single_results: dict,
    ablation_results: dict | None,
    nc_results: dict | None,
    lora_results: dict | None,
    trueman_score,
):
    """汇总所有结果并生成最终报告。"""
    print("\n[报告] 生成严格实验报告...")

    stat_reports = []
    try:
        for metric_name in ["metacognitive_monitoring_score", "metacognitive_control_score",
                            "episodic_memory_score", "recursive_self_model_score"]:
            ref = all_condition_metrics.get("trueman", {}).get(metric_name, [])
            if not ref:
                continue
            for cond_name in ["tier0_pure_llm", "tier1_structural", "tier2_memory", "tier3_random_policy"]:
                cond = all_condition_metrics.get(cond_name, {}).get(metric_name, [])
                if not cond:
                    continue
                report = compare_groups(ref, cond, metric_name, "trueman", cond_name)
                stat_reports.append(report)
    except Exception as e:
        print(f"  统计检验计算异常: {e}")

    reporter = ReportGenerator(output_dir)
    report_text = reporter.generate(
        trueman_results=single_results,
        statistical_reports=stat_reports,
        ablation_results=ablation_results,
        negative_control_results=nc_results,
        lora_results=lora_results,
    )

    ckpt.save("final_results", {
        "trueman_score": trueman_score.overall if trueman_score else 0,
        "all_condition_metrics": {k: {mk: mv for mk, mv in v.items()} for k, v in all_condition_metrics.items()},
    })

    return stat_reports


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TrueMan 严格实验（多条件 + 统计检验 + 消融 + 阴性对照 + LoRA验证 + 断点续跑）"
    )

    mode_group = parser.add_argument_group("模式选择")
    mode_group.add_argument("--api", action="store_true", help="API模式")

    local_group = parser.add_argument_group("本地模型参数")
    local_group.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    local_group.add_argument("--device", default="cuda")
    local_group.add_argument("--quantization", choices=["4bit", "8bit"], default=None)

    api_group = parser.add_argument_group("API参数")
    api_group.add_argument("--api-key", default="")
    api_group.add_argument("--api-base-url", default="")
    api_group.add_argument("--api-model", default="")

    parser.add_argument("--repeats", type=int, default=10, help="每个条件重复次数")
    parser.add_argument("--output", default="experiments/awareness/results")
    parser.add_argument("--fast", action="store_true", help="快速模式（减少刺激数量）")
    parser.add_argument("--skip-ablation", action="store_true", help="跳过消融实验")
    parser.add_argument("--skip-negative", action="store_true", help="跳过阴性对照")
    parser.add_argument("--skip-lora", action="store_true", help="跳过LoRA验证")

    ckpt_group = parser.add_argument_group("断点续跑")
    ckpt_group.add_argument("--run-id", default="", help="实验运行ID（留空则自动生成UUID）")
    ckpt_group.add_argument("--list-checkpoints", action="store_true", help="列出所有已有checkpoint并退出")
    ckpt_group.add_argument("--force-restart", action="store_true", help="忽略已有checkpoint，从头开始")

    args = parser.parse_args()

    if args.list_checkpoints:
        list_all_checkpoints(args.output)
        return

    if args.api:
        if not args.api_key or not args.api_base_url:
            parser.error("API模式需要 --api-key 和 --api-base-url")

    run_id = args.run_id.strip() or time.strftime("run_%Y%m%d_%H%M%S")
    ckpt = CheckpointManager(args.output, run_id)

    n_repeats = args.repeats
    is_api = args.api

    print("=" * 70)
    print("TrueMan 严格实验框架 v2.1（支持断点续跑）")
    print("=" * 70)
    print(f"Run ID:  {run_id}")
    mode_str = "API (云端)" if is_api else "本地模型"
    print(f"模式:    {mode_str}")
    print(f"重复次数: {n_repeats}")
    print(f"快速模式: {'是' if args.fast else '否'}")
    print(f"LoRA验证: {'跳过' if (is_api or args.skip_lora) else '执行'}")

    completed = ckpt.list_completed()
    if completed and not args.force_restart:
        print(f"\n检测到已有 checkpoint ({len(completed)} 个已完成单元):")
        for c in completed:
            print(f"  - {c}")
        print(f"将自动跳过已完成的部分，从未完成处续跑。")
    elif args.force_restart:
        print(f"\n--force-restart: 忽略已有 checkpoint，从头开始。")
    else:
        print(f"\n新的实验运行，checkpoint 将保存到:")
        print(f"  {ckpt.checkpoint_dir}")

    print()

    ckpt.save_meta(args)

    # ---- Agent 初始化 ----
    print("[1/5] 初始化TrueMan Agent...")
    config = create_config(args)
    try:
        agent = TrueManAgent(config)
        print(f"  初始化成功 (hidden_size={agent.llm.hidden_size})")
    except Exception as e:
        print(f"  初始化失败: {e}")
        return

    # ---- 第一阶段：主实验 ----
    print("[2/5] 主实验...")
    single_results, all_condition_metrics = run_main_experiments(
        agent, n_repeats, args.fast, ckpt
    )

    scorer = AwarenessScorer()
    if single_results:
        trueman_score = scorer.score(single_results)
        print(f"\n  TrueMan综合评分: {trueman_score.overall:.4f}")
        print(f"  TrueMan盲评评分: {trueman_score.blind_overall:.4f}")
    else:
        trueman_score = None

    print("\n  各基线综合评分:")
    for cond_name, metrics in all_condition_metrics.items():
        overall = _compute_overall_from_metrics(metrics)
        print(f"    {cond_name}: {overall:.4f}")

    # ---- 第二阶段：消融实验 ----
    ablation_results = None
    if not args.skip_ablation:
        print("[3/5] 消融实验...")
        ablation_results = run_ablation_experiments(agent, n_repeats, args.fast, ckpt)

    # ---- 第三阶段：阴性对照 ----
    nc_results = None
    if not args.skip_negative:
        print("[4/5] 阴性对照...")
        nc_results = run_negative_controls(agent, n_repeats, args.fast, ckpt)

    # ---- 第四阶段：LoRA验证 ----
    lora_results = None
    if not is_api and not args.skip_lora:
        print("[5/5] LoRA验证...")
        lora_results = run_lora_verification(agent, ckpt)
    else:
        print("[5/5] LoRA验证: 跳过（API模式）")

    # ---- 生成报告 ----
    stat_reports = assemble_final_report(
        ckpt, args.output, all_condition_metrics, single_results,
        ablation_results, nc_results, lora_results, trueman_score,
    )

    print("\n" + "=" * 70)
    print("实验完成！评分摘要：")
    print("=" * 70)
    if trueman_score:
        print(f"  TrueMan 综合评分: {trueman_score.overall:.4f}")
        print(f"  TrueMan 盲评评分: {trueman_score.blind_overall:.4f}")
        for dim in ["metacognitive_monitoring", "metacognitive_control",
                    "episodic_memory", "temporal_continuity", "recursive_self_model"]:
            val = getattr(trueman_score, dim)
            print(f"  {dim}: {val:.4f}")

    print(f"\n  各基线综合评分:")
    for cond_name, metrics in all_condition_metrics.items():
        if cond_name != "trueman":
            overall = _compute_overall_from_metrics(metrics)
            print(f"    {cond_name}: {overall:.4f}")

    if stat_reports:
        print(f"\n  统计显著的比较:")
        for r in stat_reports:
            if r.significant:
                print(f"    {r.group_a_name} vs {r.group_b_name} ({r.metric_name}): p={r.p_value:.4f}, d={r.cohens_d:.4f}")

    print(f"\n  Run ID: {run_id}")
    print(f"  Checkpoint: {ckpt.checkpoint_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
