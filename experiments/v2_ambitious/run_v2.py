"""TrueMan v2 雄心路径主入口。

设计：用一行命令拉起完整 5 支柱实验。所有耗时步骤都可独立续跑。

阶段：
    stage 0   ── 构建固定刺激流 + 生成 probe 文件（preregistration §4.1, §4.2）
    stage 1   ── 跑长时程 (5 cond × N seeds × M base_models × 30 days)
    stage 2   ── 训练 SAE + 找情绪特征 + 因果干预
    stage 3   ── indicator battery (HOT-1/2, GWT, RPT, ΦR)
    stage 4   ── falsification: cross-model 复现
    stage 5   ── 理论预测拟合 (FEP + PCI)
    stage 6   ── 汇总 + 出表 + 假设检验

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
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


def _apply_env_overrides(cfg: dict) -> dict:
    """Allow run_all.sh's --paper / --pilot to bump scope without yaml edits."""
    days = os.environ.get("TRUEMAN_OVERRIDE_DAYS")
    hours = os.environ.get("TRUEMAN_OVERRIDE_HOURS")
    seeds = os.environ.get("TRUEMAN_OVERRIDE_SEEDS")
    if "stream" in cfg:
        if days:
            cfg["stream"]["days"] = int(days)
        if hours:
            cfg["stream"]["hours_per_day"] = int(hours)
    if seeds:
        cfg["seeds"] = [int(s) for s in seeds.split(",") if s.strip()]
    return cfg

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

# Ensure log directory exists before FileHandler initialises.
(ROOT / "results").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "results" / "run_v2.log", mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger("run_v2")


def _check_environment() -> list[str]:
    """检查必要依赖，返回缺失包列表。"""
    missing = []
    required = {
        "yaml": "pyyaml",
        "numpy": "numpy",
        "torch": "torch",
        "h5py": "h5py",
        "sklearn": "scikit-learn",
        "scipy": "scipy",
    }
    for mod, pkg in required.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    try:
        from pymer4.models import Lmer
    except Exception:
        log.warning("pymer4 not available — will fall back to OLS for mixed-effects models")
    return missing


def _slug(label: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_")[:120]


def _run(cmd: list[str], dry: bool, label: str, log_dir: Path | None = None) -> int:
    # Coerce every token to str up front. YAML 1.1 parses bare ``off`` as the
    # boolean ``False`` (and ``on``/``yes``/``no`` likewise), so a config like
    # ``modes: [clamp, inject, off]`` would otherwise smuggle a ``bool`` into the
    # command list and crash ``" ".join(cmd)`` / ``subprocess`` with a cryptic
    # "expected str instance, bool found". Stringifying here is harmless for
    # genuine strings and bulletproofs the whole stage runner.
    cmd = [str(c) for c in cmd]
    log.info(f"=== {label} ===")
    log.info(" ".join(cmd))
    if dry:
        return 0

    # Per-call log file so subprocess tracebacks survive even though the parent
    # cannot pipe their stderr live. Tail is also surfaced in run_v2.log on
    # failure so the next operator can read what went wrong without grepping
    # the filesystem.
    log_dir = log_dir or (ROOT / "results" / "subprocess_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{_slug(label)}.log"

    try:
        with open(log_path, "w", encoding="utf-8") as logf:
            logf.write(f"# CMD: {' '.join(cmd)}\n")
            logf.write(f"# START: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            logf.flush()
            result = subprocess.run(
                cmd,
                stdout=logf,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if result.returncode != 0:
            log.error(f"[FAIL] {label} returned {result.returncode} (full log: {log_path})")
            # Surface the last ~80 lines of the subprocess log so failures are
            # diagnosable from run_v2.log alone.
            try:
                tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
                log.error("---- subprocess tail ----\n" + "\n".join(tail) + "\n---- /tail ----")
            except Exception as tail_err:
                log.error(f"(could not read subprocess log tail: {tail_err})")
        else:
            log.info(f"[OK] {label}")
        return result.returncode
    except Exception as e:
        log.error(f"[EXCEPTION] {label}: {e}")
        return -1


def stage0_stream(args, cfg) -> None:
    out = cfg["stream"]["output"]
    if Path(out).exists() and not args.force:
        log.info(f"[stage0] stream exists: {out} (skip; use --force to rebuild)")
    else:
        rc = _run([
            PYTHON, "-m", "experiments.v2_ambitious.pillar2_longhorizon.stimulus_stream",
            "--days", str(cfg["stream"]["days"]),
            "--hours-per-day", str(cfg["stream"]["hours_per_day"]),
            "--seed", str(cfg["stream"]["seed"]),
            "--output", out,
        ], dry=args.dry_run, label="stage0.1: build stimulus stream")
        if rc != 0:
            log.error("stage0.1 failed!")
            return

    probe_dir = Path("experiments/v2_ambitious/data/probes")
    metacog = probe_dir / "metacog_full.jsonl"
    if metacog.exists() and not args.force:
        log.info(f"[stage0] probe files exist (skip)")
    else:
        _run([
            PYTHON, "-m", "experiments.v2_ambitious.pillar2_longhorizon.probe_battery",
            "--output-dir", str(probe_dir),
            "--generate-files",
        ], dry=args.dry_run, label="stage0.2: generate probe JSONL files")


def stage1_longhorizon(args, cfg) -> None:
    conditions = args.conditions or cfg["conditions"]
    base_models = args.base_models or cfg["base_models"]
    seeds = args.seeds or cfg["seeds"]
    out_root = Path("experiments/v2_ambitious/results/longhorizon")
    out_root.mkdir(parents=True, exist_ok=True)

    plan = [(c, m, s) for c in conditions for m in base_models for s in seeds]
    log.info(f"[stage1] Total runs: {len(plan)}")

    failed = []
    for cond, model, seed in plan:
        run_dir = out_root / f"{cond}_seed{seed}_{model.replace('/', '_')}"
        if (run_dir / "trajectory.csv").exists() and not args.force:
            log.info(f"[stage1] skip (done): {run_dir.name}")
            continue
        rc = _run([
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
        if rc != 0:
            failed.append((cond, model, seed))

    if failed:
        log.warning(f"[stage1] {len(failed)} runs failed: {failed}")


def stage2_mechanistic(args, mech_cfg) -> None:
    sae_out = mech_cfg["sae"]["output"]
    feat_out = mech_cfg["probe_features"]["output"]
    out_dir = Path(mech_cfg["causal_intervention"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = mech_cfg["captures_source"]
    captures = sorted(Path(".").glob(pattern))
    if not captures:
        log.warning(f"[stage2] No captures found at {pattern}; run stage1 first.")
        return
    captures = str(captures[0])
    log.info(f"[stage2] Using captures: {captures}")

    rc = _run([
        PYTHON, "-m", "experiments.v2_ambitious.pillar1_mechanistic.train_sae",
        "--captures", captures,
        "--layer", str(mech_cfg["sae"]["layer"]),
        "--dict-size", str(mech_cfg["sae"]["dict_size"]),
        "--top-k", str(mech_cfg["sae"]["top_k"]),
        "--epochs", str(mech_cfg["sae"]["epochs"]),
        "--batch-size", str(mech_cfg["sae"]["batch_size"]),
        "--output", sae_out,
    ], dry=args.dry_run, label="stage2.1: train SAE")
    if rc != 0:
        log.error("stage2.1 SAE training failed!")
        return

    _run([
        PYTHON, "-m", "experiments.v2_ambitious.pillar1_mechanistic.probe_features",
        "--captures", captures,
        "--sae", sae_out,
        "--layer", str(mech_cfg["sae"]["layer"]),
        "--target", mech_cfg["probe_features"]["target"],
        "--top-k-features", str(mech_cfg["probe_features"]["top_k_features"]),
        "--output", feat_out,
    ], dry=args.dry_run, label="stage2.2: probe features")

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
    lh_root = Path("experiments/v2_ambitious/results/longhorizon")
    ind_out = ind_cfg.get("output", "experiments/v2_ambitious/results/indicators")

    conditions = args.conditions or [
        "C0_trueman_full", "C1_reversed", "C2_scrambled",
        "C3_frozen", "C4_trivial_jaccard",
    ]
    seeds = args.seeds or [0, 1, 2, 3]

    _run([
        PYTHON, "-m", "experiments.v2_ambitious.pillar3_indicators.run_indicators",
        "--root", str(lh_root),
        "--output", ind_out,
        "--conditions", ",".join(conditions),
        "--seeds", ",".join(str(s) for s in seeds),
    ], dry=args.dry_run, label="stage3: indicator battery (HOT-1/2, GWT, RPT, ΦR)")


def stage4_falsification(args, cfg) -> None:
    cmd = [
        PYTHON, "-m", "experiments.v2_ambitious.pillar4_falsification.cross_model",
        "--seeds", *map(str, args.seeds or cfg.get("seeds", [0, 1, 2, 3])),
        "--days", str(cfg["stream"]["days"]),
        "--hours-per-day", str(cfg["stream"]["hours_per_day"]),
        "--output", "experiments/v2_ambitious/results/cross_model",
    ]
    base_models = args.base_models or cfg.get("base_models")
    if base_models:
        cmd += ["--base-models", *base_models]
    _run(cmd, dry=args.dry_run, label="stage4: cross-model falsification")


def stage5_theory(args, cfg) -> None:
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
    """汇总各 stage 输出 → 一份总报告 + 假设检验结论。"""
    summary = {}
    result_files = {
        "analysis_pillar2": "experiments/v2_ambitious/results/analysis_pillar2.json",
        "fep_h5": "experiments/v2_ambitious/results/fep_h5.json",
        "indicators_summary": "experiments/v2_ambitious/results/indicators/indicators_summary.json",
        "sae": "experiments/v2_ambitious/results/mechanistic/sae_layer18.pt",
        "features": "experiments/v2_ambitious/results/mechanistic/features_anxiety.json",
    }
    for key, f in result_files.items():
        if Path(f).exists() and f.endswith(".json"):
            try:
                summary[key] = json.loads(Path(f).read_text(encoding="utf-8"))
            except Exception as e:
                summary[key] = {"error": str(e)}
        elif Path(f).exists():
            summary[key] = {"exists": True, "path": f}
        else:
            summary[key] = {"exists": False}

    hypothesis_verdicts = _evaluate_hypotheses(summary)
    summary["hypothesis_verdicts"] = hypothesis_verdicts

    out = Path("experiments/v2_ambitious/results/v2_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    log.info(f"[stage6] Summary -> {out}")

    log.info("=" * 60)
    log.info("HYPOTHESIS VERDICTS:")
    for h, verdict in hypothesis_verdicts.items():
        log.info(f"  {h}: {verdict}")
    log.info("=" * 60)


def _evaluate_hypotheses(summary: dict) -> dict[str, str]:
    """根据 PREREGISTRATION.md §2 的判定标准评估各假设。"""
    verdicts = {}

    analysis = summary.get("analysis_pillar2", {})
    contrasts = analysis.get("contrasts", {})

    h4 = contrasts.get("H4_retention_C0_vs_C3", {})
    h4_perm = h4.get("permutation", {})
    h4_p = h4_perm.get("p_value", 1.0)
    if h4_p < 0.01:
        verdicts["H4"] = "CONFIRMED (p < 0.01, retention difference significant)"
    elif h4_p < 0.05:
        verdicts["H4"] = "MARGINAL (p < 0.05 but not < 0.01)"
    else:
        verdicts["H4"] = "NOT CONFIRMED"

    h5 = contrasts.get("H5_alpha_C0_vs_C3", {})
    h5_perm = h5.get("permutation", {})
    h5_p = h5_perm.get("p_value", 1.0)
    fep_data = summary.get("fep_h5", {})
    fep_contrasts = fep_data.get("contrasts", {}).get("C0_vs_C3", {})
    h5_pass = fep_contrasts.get("H5_pass", False)
    if h5_pass and h5_p < 0.01:
        verdicts["H5"] = "CONFIRMED (α_C0 < 0.85, α_C3 ≥ 0.95, p < 0.01)"
    elif h5_p < 0.05:
        verdicts["H5"] = "MARGINAL"
    else:
        verdicts["H5"] = "NOT CONFIRMED"

    ind_data = summary.get("indicators_summary", {})
    if isinstance(ind_data, list) and ind_data:
        c0_mratios = []
        c3_mratios = []
        for item in ind_data:
            if not isinstance(item, dict):
                continue
            cond = item.get("condition", "")
            hot1 = item.get("indicators", {}).get("HOT1_meta_d_prime", {})
            mr = hot1.get("m_ratio")
            if mr is not None:
                if cond == "C0_trueman_full":
                    c0_mratios.append(mr)
                elif cond == "C3_frozen":
                    c3_mratios.append(mr)
        if c0_mratios and c3_mratios:
            import numpy as np
            from experiments.v2_ambitious.harness.stats import permutation_test
            a = np.array(c0_mratios, dtype=float)
            b = np.array(c3_mratios, dtype=float)
            pt = permutation_test(a, b)
            p_val = pt.get("p_value", 1.0)
            # Pooled standard deviation per Cohen's d definition
            n1, n2 = len(a), len(b)
            if n1 >= 2 and n2 >= 2:
                pooled_var = ((n1 - 1) * np.var(a, ddof=1)
                              + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2)
                pooled_sd = float(np.sqrt(max(pooled_var, 1e-12)))
            else:
                pooled_sd = float(np.std(np.concatenate([a, b]))) or 1e-3
            effect = float((a.mean() - b.mean()) / max(pooled_sd, 1e-3))
            if p_val < 0.01 and effect >= 0.5:
                verdicts["H1"] = f"CONFIRMED (p={p_val:.4f}, d={effect:.2f})"
            else:
                verdicts["H1"] = f"NOT CONFIRMED (p={p_val:.4f}, d={effect:.2f})"
        else:
            verdicts["H1"] = "INSUFFICIENT DATA"
    else:
        verdicts["H1"] = "INSUFFICIENT DATA"

    verdicts["H2"] = "PENDING (requires RSA analysis on self-model probes)"
    verdicts["H3"] = "PENDING (requires SAE causal intervention results)"

    intervention_dir = Path("experiments/v2_ambitious/results/mechanistic/intervention")
    if intervention_dir.exists():
        clamp_files = list(intervention_dir.glob("intervention_clamp_*.json"))
        inject_files = list(intervention_dir.glob("intervention_inject_*.json"))
        if clamp_files and inject_files:
            verdicts["H3"] = "DATA AVAILABLE (run detailed analysis on intervention results)"

    return verdicts


def status() -> None:
    checks = {
        "stage0 stream":     Path("experiments/v2_ambitious/data/stimulus_stream.jsonl").exists(),
        "stage0 probes":     Path("experiments/v2_ambitious/data/probes/metacog_full.jsonl").exists(),
        "stage1 longhorizon": any(Path("experiments/v2_ambitious/results/longhorizon").glob("*/trajectory.csv"))
                               if Path("experiments/v2_ambitious/results/longhorizon").exists() else False,
        "stage2 SAE":        Path("experiments/v2_ambitious/results/mechanistic/sae_layer18.pt").exists(),
        "stage2 features":   Path("experiments/v2_ambitious/results/mechanistic/features_anxiety.json").exists(),
        "stage3 indicators": Path("experiments/v2_ambitious/results/indicators/indicators_summary.json").exists(),
        "stage5 analysis":   Path("experiments/v2_ambitious/results/analysis_pillar2.json").exists(),
        "stage5 fep":        Path("experiments/v2_ambitious/results/fep_h5.json").exists(),
        "stage6 summary":    Path("experiments/v2_ambitious/results/v2_summary.json").exists(),
    }
    for k, v in checks.items():
        sym = "OK" if v else "  "
        print(f"  [{sym}] {k}")


def main():
    p = argparse.ArgumentParser(description="TrueMan v2 experiment orchestrator")
    p.add_argument("--stage", default="all",
                   choices=["all", "stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "stage6"])
    p.add_argument("--longhorizon-config", default="experiments/v2_ambitious/configs/longhorizon.yaml")
    p.add_argument("--mechanistic-config", default="experiments/v2_ambitious/configs/mechanistic.yaml")
    p.add_argument("--indicators-config", default="experiments/v2_ambitious/configs/indicators.yaml")
    p.add_argument("--conditions", help="逗号分隔，覆盖 yaml 中的 conditions")
    p.add_argument("--base-models", help="逗号分隔")
    p.add_argument("--seeds", help="逗号分隔")
    p.add_argument("--force", action="store_true", help="强制重跑已完成的子任务")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--status", action="store_true")
    p.add_argument("--skip-env-check", action="store_true")
    args = p.parse_args()

    (ROOT / "results").mkdir(parents=True, exist_ok=True)

    if args.status:
        status()
        return

    if not args.skip_env_check:
        missing = _check_environment()
        if missing:
            log.error(f"Missing required packages: {missing}")
            log.error(f"Install with: pip install {' '.join(missing)}")
            sys.exit(1)
        else:
            log.info("Environment check passed")

    for k in ("conditions", "base_models", "seeds"):
        v = getattr(args, k)
        if isinstance(v, str):
            parts = [x.strip() for x in v.split(",") if x.strip()]
            setattr(args, k, [int(p) if p.isdigit() else p for p in parts])

    lh_cfg = _apply_env_overrides(
        yaml.safe_load(Path(args.longhorizon_config).read_text(encoding="utf-8"))
    )
    mech_cfg = yaml.safe_load(Path(args.mechanistic_config).read_text(encoding="utf-8"))
    ind_cfg = yaml.safe_load(Path(args.indicators_config).read_text(encoding="utf-8"))

    t0 = time.time()
    stages_to_run = []
    if args.stage == "all":
        stages_to_run = ["stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "stage6"]
    else:
        stages_to_run = [args.stage]

    for stage_name in stages_to_run:
        log.info(f"\n{'='*60}")
        log.info(f"Starting {stage_name}")
        log.info(f"{'='*60}")
        try:
            if stage_name == "stage0":
                stage0_stream(args, lh_cfg)
            elif stage_name == "stage1":
                stage1_longhorizon(args, lh_cfg)
            elif stage_name == "stage2":
                stage2_mechanistic(args, mech_cfg)
            elif stage_name == "stage3":
                stage3_indicators(args, ind_cfg)
            elif stage_name == "stage4":
                stage4_falsification(args, lh_cfg)
            elif stage_name == "stage5":
                stage5_theory(args, lh_cfg)
            elif stage_name == "stage6":
                stage6_summary(args)
        except Exception as e:
            log.error(f"[{stage_name}] FAILED with exception: {e}")
            import traceback
            log.error(traceback.format_exc())

    elapsed = time.time() - t0
    log.info(f"\n[run_v2] Total elapsed: {elapsed:.1f}s ({elapsed/3600:.2f}h)")


if __name__ == "__main__":
    main()
