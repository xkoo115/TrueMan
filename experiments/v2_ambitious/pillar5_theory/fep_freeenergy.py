"""FEP 自由能轨迹拟合（H5）。

实验级实现已在 pillar2_longhorizon/analyze.py::free_energy_fit 中。
此模块负责跨条件统计比较 + 论文级图表数据。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.v2_ambitious.harness.stats import (
    permutation_test, bayes_factor_t,
)


def load_alphas(analysis_json: str) -> dict[str, list[float]]:
    with open(analysis_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for cond, summary in data["per_condition"].items():
        alphas = []
        for r in summary.get("runs", []):
            a = r.get("free_energy_fit", {}).get("alpha")
            if a is not None:
                alphas.append(a)
        out[cond] = alphas
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis", required=True, help="pillar2_longhorizon/analyze.py 输出")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    alphas = load_alphas(args.analysis)
    out = {"per_condition_alphas": alphas, "contrasts": {}}

    if "C0_trueman_full" in alphas and "C3_frozen" in alphas:
        a = np.asarray(alphas["C0_trueman_full"])
        b = np.asarray(alphas["C3_frozen"])
        if len(a) >= 2 and len(b) >= 2:
            out["contrasts"]["C0_vs_C3"] = {
                "C0_mean": float(a.mean()),
                "C3_mean": float(b.mean()),
                "permutation": permutation_test(a, b),
                "BF10": bayes_factor_t(a, b),
                "H5_pass": bool(a.mean() < 0.85 and b.mean() >= 0.95),
            }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[FEP] Saved to {args.output}")


if __name__ == "__main__":
    main()
