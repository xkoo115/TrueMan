"""支柱 2 分析：参数轨迹散度、灾难性遗忘、行为表型漂移。

用法：
    python -m experiments.v2_ambitious.pillar2_longhorizon.analyze \
        --root experiments/v2_ambitious/results/longhorizon \
        --output analysis_pillar2.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.v2_ambitious.harness.snapshots import parameter_divergence
from experiments.v2_ambitious.harness.stats import (
    permutation_test, bayes_factor_t, holm_bonferroni,
)


def collect_runs(root: Path):
    runs = {}
    for d in root.iterdir():
        if not d.is_dir():
            continue
        # 目录名形如 C0_trueman_full_seed0_Qwen_Qwen3-8B
        parts = d.name.split("_seed")
        if len(parts) != 2:
            continue
        cond = parts[0]
        seed_model = parts[1]
        seed, model = seed_model.split("_", 1)
        runs.setdefault(cond, []).append({
            "seed": int(seed), "model": model, "path": d,
        })
    return runs


def trajectory_divergence(run_dir: Path) -> list[dict]:
    snap_dir = run_dir / "snapshots"
    if not snap_dir.exists():
        return []
    days = sorted([d for d in snap_dir.iterdir() if d.is_dir()],
                  key=lambda p: p.name)
    out = []
    if not days:
        return out
    base = days[0]
    for d in days:
        div = parameter_divergence(str(base), str(d))
        out.append({"day": d.name, **div})
    return out


def forgetting_score(run_dir: Path) -> dict:
    """对比 ablated probe 与最后一周 normal probe 在 forgetting bank 上的准确率。"""
    probes_dir = run_dir / "probes"
    if not probes_dir.exists():
        return {}

    normal_files = sorted(probes_dir.glob("probes_week*_*.json"))
    ablated_files = sorted(probes_dir.glob("probes_week*_*_ablated.json"))
    if not normal_files or not ablated_files:
        return {}

    def avg_correct(path: Path, bank: str = "forgetting") -> float:
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data["results"].get(bank, [])
        if not items:
            return 0.0
        # TODO(other model): 用 NLI 或 keyword match 判 correct.
        # 此处暂用占位逻辑：response 中包含 ground_truth 子串即视为正确.
        ok = 0
        for it in items:
            gt = (it.get("ground_truth") or "").strip()
            resp = (it.get("response") or "").strip()
            if gt and gt in resp:
                ok += 1
        return ok / len(items)

    return {
        "with_memory": avg_correct(normal_files[-1]),
        "without_memory": avg_correct(ablated_files[-1]),
        "retention_ratio": (
            avg_correct(ablated_files[-1]) / max(avg_correct(normal_files[-1]), 1e-3)
        ),
    }


def free_energy_fit(run_dir: Path) -> dict:
    """从 trajectory.csv 累计 surprise（自由能近似）并拟合 power-law."""
    csv_path = run_dir / "trajectory.csv"
    if not csv_path.exists():
        return {}
    import csv
    surprises = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                surprises.append(float(row["surprise"]))
            except (ValueError, KeyError):
                continue
    if len(surprises) < 100:
        return {}
    cum = np.cumsum(surprises)
    t = np.arange(1, len(cum) + 1, dtype=float)
    # log-log 拟合 cum ~ t^alpha
    log_t = np.log(t)
    log_c = np.log(np.maximum(cum, 1e-8))
    slope, intercept = np.polyfit(log_t, log_c, 1)
    # 95% CI from residuals
    resid = log_c - (slope * log_t + intercept)
    se_slope = float(np.std(resid)) / float(np.std(log_t)) / np.sqrt(len(t))
    return {
        "alpha": float(slope),
        "intercept": float(intercept),
        "alpha_ci_low": float(slope - 1.96 * se_slope),
        "alpha_ci_high": float(slope + 1.96 * se_slope),
        "n_points": len(t),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    runs = collect_runs(Path(args.root))
    summary = {}
    for cond, runs_list in runs.items():
        cond_summary = {"n_runs": len(runs_list), "runs": []}
        retentions = []
        alphas = []
        for r in runs_list:
            traj = trajectory_divergence(r["path"])
            forget = forgetting_score(r["path"])
            fe = free_energy_fit(r["path"])
            cond_summary["runs"].append({
                "seed": r["seed"], "model": r["model"],
                "trajectory_div": traj,
                "forgetting": forget,
                "free_energy_fit": fe,
            })
            if forget.get("retention_ratio") is not None:
                retentions.append(forget["retention_ratio"])
            if fe.get("alpha") is not None:
                alphas.append(fe["alpha"])
        cond_summary["retention_mean"] = float(np.mean(retentions)) if retentions else None
        cond_summary["alpha_mean"] = float(np.mean(alphas)) if alphas else None
        summary[cond] = cond_summary

    # H4 / H5 对比 vs C3 (frozen)
    c0 = summary.get("C0_trueman_full", {})
    c3 = summary.get("C3_frozen", {})
    contrasts = {}
    if c0.get("runs") and c3.get("runs"):
        a = [r["forgetting"].get("retention_ratio", 0) for r in c0["runs"]]
        b = [r["forgetting"].get("retention_ratio", 0) for r in c3["runs"]]
        contrasts["H4_retention_C0_vs_C3"] = {
            "permutation": permutation_test(np.array(a), np.array(b)),
            "BF10": bayes_factor_t(np.array(a), np.array(b)),
        }
        a2 = [r["free_energy_fit"].get("alpha", 1.0) for r in c0["runs"]]
        b2 = [r["free_energy_fit"].get("alpha", 1.0) for r in c3["runs"]]
        contrasts["H5_alpha_C0_vs_C3"] = {
            "permutation": permutation_test(np.array(a2), np.array(b2)),
            "BF10": bayes_factor_t(np.array(a2), np.array(b2)),
        }

    out = {"per_condition": summary, "contrasts": contrasts}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"[Analyze] Saved to {args.output}")


if __name__ == "__main__":
    main()
