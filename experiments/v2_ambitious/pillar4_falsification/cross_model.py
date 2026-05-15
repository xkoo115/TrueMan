"""跨底模复现：在 4 个底模上复跑核心 H1 / H2 测试。

用法：作为外层调度器，依次调用 pillar3_indicators 与 pillar2 的
analyze.py，在不同 base_model 下输出对比矩阵。

输出：cross_model_matrix.csv (rows=base_model, cols=H1/H2/H3 effects)。
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


DEFAULT_BASE_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "deepseek-ai/DeepSeek-V2-Lite",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3])
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--hours-per-day", type=int, default=24)
    p.add_argument("--base-models", nargs="*", default=DEFAULT_BASE_MODELS,
                   help="覆盖底模列表（论文级 4 模型；4080S 受限只跑 1 个）")
    p.add_argument("--conditions", nargs="*", default=None,
                   help="覆盖条件子集；默认全部 5 个")
    p.add_argument("--output", default="experiments/v2_ambitious/results/cross_model")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    BASE_MODELS = args.base_models

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    plan = []
    from experiments.v2_ambitious.harness.conditions import all_conditions
    conditions = args.conditions or all_conditions()
    for model in BASE_MODELS:
        for cond in conditions:
            for seed in args.seeds:
                plan.append({"model": model, "condition": cond, "seed": seed})

    print(f"[Cross] Total runs planned: {len(plan)}")
    plan_path = out_root / "plan.json"
    with open(plan_path, "w") as f:
        json.dump(plan, f, indent=2)
    print(f"[Cross] Plan saved to {plan_path}")

    if args.dry_run:
        return

    # 实际执行：依赖 pillar2_longhorizon.run.py
    import sys
    for job in plan:
        cmd = [
            sys.executable, "-m", "experiments.v2_ambitious.pillar2_longhorizon.run",
            "--condition", job["condition"],
            "--base-model", job["model"],
            "--seed", str(job["seed"]),
            "--stream", "experiments/v2_ambitious/data/stimulus_stream.jsonl",
            "--days", str(args.days),
            "--hours-per-day", str(args.hours_per_day),
            "--output", str(out_root / "longhorizon"),
        ]
        print(f"[Cross] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
