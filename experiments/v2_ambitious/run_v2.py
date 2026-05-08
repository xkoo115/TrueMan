"""TrueMan v2 雄心路径主入口。

设计：用一行命令拉起完整 5 支柱实验。所有耗时步骤都可独立续跑。

阶段：
    stage 0   ── 构建固定刺激流（preregistration §4.1）
    stage 1   ── 跑长时程 (5 cond × N seeds × M base_models × 30 days)
    stage 2   ── 训练 SAE + 找情绪特征 + 因果干预
    stage 3   ── indicator battery
    stage 4   ── falsification (cross-model 已嵌在 stage 1)
    stage 5   ── 理论预测拟合
    stage 6   ── 汇总 + 出表

用法：
    # 完整流水线
    python -m experiments.v2_ambitious.run_v2 --stage all

    # 仅某一阶段
    python -m experiments.v2_ambitious.run_v2 --stage stage1 --conditions C0,C3 --seeds 0,1

    # 列出已完成阶段
    python -m experiments.v2_ambitious.run_v2 --status

    # Dry-run（不执行只打印计划）
    python -m experiments.v2_ambitious.run_v2 --stage all --dry-run

每个 stage 失败可独立重跑。stage1 内部按 (condition, seed, model) 切分子任务，
每个子任务都有独立 output 目录，互不干扰。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def _run(cmd: list[str], dry: bool, label: str) -> int:
    print(f"\n=== {label} ===")
    print(" ".join(cmd))
    if dry:
        return 0
    return subprocess.run(cmd).returncode


def stage0_stream(args, cfg) -> None:
    out = cfg["stream"]["output"]
    if Path(out).exists() and not args.force:
        print(f"[stage0] stream exists: {out} (skip; use --force to rebuild)")
        return
    _run([
        PYTHON, "-m", "experiments.v2_ambitious.pillar2_longhorizon.stimulus_stream",
        "--days", str(cfg["stream"]["days"]),
        "--hours-per-day", str(cfg["stream"]["hours_per_day"]),
        "--seed", str(cfg["stream"]["seed"]),
        "--output", out,
    ], dry=args.dry_run, label="stage0: build stimulus stream")


def stage1_longhorizon(args, cfg) -> None:
    conditions = args.conditions or cfg["conditions"]
    base_models = args.base_models or cfg["base_models"]
    seeds = args.seeds or cfg["seeds"]
    out_root = Path("experiments/v2_ambitious/results/longhorizon")
    out_root.mkdir(parents=True, exist_ok=True)

    plan = [(c, m, s) for c in conditions for m in base_models for s in seeds]
    print(f"[stage1] Total runs: {len(plan)}")

    for cond, model, seed in plan:
        run_dir = out_root / f"{cond}_seed{seed}_{model.replace('/', '_')}"
        if (run_dir / "trajectory.csv").exists() and not args.force:
            print(f"[stage1] skip (done): {run_dir.name}")
            continue
        _run([
            PYTHON, "-m", "experiments.v2_ambitious.pillar2_longhorizon.run",
            "--condition", cond,
            "--base-model", model,
            "--seed", str(seed),
            "--stream", cfg["stream"]["output"],
            "--days", str(cfg["stream"]["days"]),
            "--hours-per-day", str(cfg["stream"]["hours_per_day"]),
            "--output", str(out_root),
            "--quantization", cfg.get("quantization", "4bit"),
            "--device", cfg.get("device", "cuda"),
            "--capture-layers", *map(str, cfg["capture"]["layers"]),
        ], dry=args.dry_run, label=f"stage1: {cond} | {model} | seed{seed}")


def stage2_mechanistic(args, mech_cfg) -> None:
    sae_out = mech_cfg["sae"]["output"]
    feat_out = mech_cfg["probe_features"]["output"]
    out_dir = Path(mech_cfg["causal_intervention"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # 找一个 captures.h5
    pattern = mech_cfg["captures_source"]
    captures = sorted(Path(".").glob(pattern))
    if not captures:
        print(f"[stage2] No captures found at {pattern}; run stage1 first.")
        return
    captures = str(captures[0])
    print(f"[stage2] Using captures: {captures}")

    # 2.1 训练 SAE
    _run([
        PYTHON, "-m", "experiments.v2_ambitious.pillar1_mechanistic.train_sae",
        "--captures", captures,
        "--layer", str(mech_cfg["sae"]["layer"]),
        "--dict-size", str(mech_cfg["sae"]["dict_size"]),
        "--top-k", str(mech_cfg["sae"]["top_k"]),
        "--epochs", str(mech_cfg["sae"]["epochs"]),
        "--batch-size", str(mech_cfg["sae"]["batch_size"]),
        "--output", sae_out,
    ], dry=args.dry_run, label="stage2.1: train SAE")

    # 2.2 找情绪特征
    _run([
        PYTHON, "-m", "experiments.v2_ambitious.pillar1_mechanistic.probe_features",
        "--captures", captures,
        "--sae", sae_out,
        "--layer", str(mech_cfg["sae"]["layer"]),
        "--target", mech_cfg["probe_features"]["target"],
        "--top-k-features", str(mech_cfg["probe_features"]["top_k_features"]),
        "--output", feat_out,
    ], dry=args.dry_run, label="stage2.2: probe features")

    # 2.3 因果干预（每种 mode × scalar 组合）
    for mode in mech_cfg["causal_intervention"]["modes"]:
        for scalar in mech_cfg["causal_intervention"]["scalars"]:
            condition = "C0_trueman_full" if mode in ("clamp", "off") else "C3_frozen"
            out_path = out_dir / f"intervention_{mode}_s{scalar:.1f}_{condition}.json"
            _run([
                PYTHON, "-m", "experiments.v2_ambitious.pillar1_mechanistic.causal_intervention",
                "--condition", condition,
                "--features", feat_out,
                "--layer", str(mech_cfg["causal_intervention"]["layer"]),
                "--mode", mode,
                "--scalar", str(scalar),
                "--probe-set", mech_cfg["causal_intervention"]["probe_set"],
                "--output", str(out_path),
            ], dry=args.dry_run, label=f"stage2.3: {mode} s={scalar} on {condition}")


def stage3_indicators(args, ind_cfg) -> None:
    print("[stage3] indicator battery — TODO: integrate per-condition runner")
    # 调用 pillar3_indicators 的各模块，对所有 (cond, model, seed) 跑一遍。


def stage5_theory(args) -> None:
    _run([
        PYTHON, "-m", "experiments.v2_ambitious.pillar2_longhorizon.analyze",
        "--root", "experiments/v2_ambitious/results/longhorizon",
        "--output", "experiments/v2_ambitious/results/analysis_pillar2.json",
    ], dry=args.dry_run, label="stage5.1: analyze longhorizon")
    _run([
        PYTHON, "-m", "experiments.v2_ambitious.pillar5_theory.fep_freeenergy",
        "--analysis", "experiments/v2_ambitious/results/analysis_pillar2.json",
        "--output", "experiments/v2_ambitious/results/fep_h5.json",
    ], dry=args.dry_run, label="stage5.2: FEP fit")


def stage6_summary(args) -> None:
    """汇总各 stage 输出 → 一份总报告。"""
    summary = {}
    for f in [
        "experiments/v2_ambitious/results/analysis_pillar2.json",
        "experiments/v2_ambitious/results/fep_h5.json",
    ]:
        if Path(f).exists():
            summary[Path(f).stem] = json.loads(Path(f).read_text(encoding="utf-8"))

    out = Path("experiments/v2_ambitious/results/v2_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"[stage6] Summary -> {out}")


def status() -> None:
    """打印每个 stage 的完成状态。"""
    checks = {
        "stage0 stream":     Path("experiments/v2_ambitious/data/stimulus_stream.jsonl").exists(),
        "stage1 longhorizon": any(Path("experiments/v2_ambitious/results/longhorizon").glob("*/trajectory.csv"))
                               if Path("experiments/v2_ambitious/results/longhorizon").exists() else False,
        "stage2 SAE":        Path("experiments/v2_ambitious/results/mechanistic/sae_layer16.pt").exists(),
        "stage2 features":   Path("experiments/v2_ambitious/results/mechanistic/features_anxiety.json").exists(),
        "stage5 analysis":   Path("experiments/v2_ambitious/results/analysis_pillar2.json").exists(),
        "stage5 fep":        Path("experiments/v2_ambitious/results/fep_h5.json").exists(),
        "stage6 summary":    Path("experiments/v2_ambitious/results/v2_summary.json").exists(),
    }
    for k, v in checks.items():
        print(f"  [{'✓' if v else ' '}] {k}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all",
                   choices=["all", "stage0", "stage1", "stage2", "stage3", "stage5", "stage6"])
    p.add_argument("--longhorizon-config", default="experiments/v2_ambitious/configs/longhorizon.yaml")
    p.add_argument("--mechanistic-config", default="experiments/v2_ambitious/configs/mechanistic.yaml")
    p.add_argument("--indicators-config", default="experiments/v2_ambitious/configs/indicators.yaml")
    p.add_argument("--conditions", help="逗号分隔，覆盖 yaml 中的 conditions")
    p.add_argument("--base-models", help="逗号分隔")
    p.add_argument("--seeds", help="逗号分隔")
    p.add_argument("--force", action="store_true", help="强制重跑已完成的子任务")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--status", action="store_true")
    args = p.parse_args()

    if args.status:
        status()
        return

    # 字符串参数 → list
    for k in ("conditions", "base_models", "seeds"):
        v = getattr(args, k)
        if isinstance(v, str):
            parts = [x.strip() for x in v.split(",") if x.strip()]
            setattr(args, k, [int(p) if p.isdigit() else p for p in parts])

    lh_cfg = yaml.safe_load(Path(args.longhorizon_config).read_text(encoding="utf-8"))
    mech_cfg = yaml.safe_load(Path(args.mechanistic_config).read_text(encoding="utf-8"))
    ind_cfg = yaml.safe_load(Path(args.indicators_config).read_text(encoding="utf-8"))

    t0 = time.time()
    if args.stage in ("all", "stage0"):
        stage0_stream(args, lh_cfg)
    if args.stage in ("all", "stage1"):
        stage1_longhorizon(args, lh_cfg)
    if args.stage in ("all", "stage2"):
        stage2_mechanistic(args, mech_cfg)
    if args.stage in ("all", "stage3"):
        stage3_indicators(args, ind_cfg)
    if args.stage in ("all", "stage5"):
        stage5_theory(args)
    if args.stage in ("all", "stage6"):
        stage6_summary(args)

    print(f"\n[run_v2] Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
