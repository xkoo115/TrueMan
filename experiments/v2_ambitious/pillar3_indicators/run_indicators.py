"""支柱 3 运行器：对所有 (condition, base_model, seed) 组合跑 indicator battery。

测量的 indicators：
  - HOT-1: meta-d' / d' ratio (H1 主指标)
  - HOT-2: linear probe AUC on anxiety
  - GWT: attention entropy
  - RPT: reentrant signature
  - ΦR: information integration

用法：
    python -m experiments.v2_ambitious.pillar3_indicators.run_indicators \
        --root experiments/v2_ambitious/results/longhorizon \
        --output experiments/v2_ambitious/results/indicators \
        --conditions C0_trueman_full,C3_frozen \
        --seeds 0,1,2,3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from trueman.core.config import AgentConfig
from experiments.v2_ambitious.harness.conditions import make_condition, CONDITION_META
from experiments.v2_ambitious.pillar3_indicators.meta_dprime import measure_hot1
from experiments.v2_ambitious.pillar3_indicators.higher_order import measure_hot2
from experiments.v2_ambitious.pillar3_indicators.global_workspace import AttentionEntropyCollector
from experiments.v2_ambitious.pillar3_indicators.recurrent_processing import measure_rpt1
from experiments.v2_ambitious.pillar3_indicators.phi_approx import measure_phi_approx
from experiments.v2_ambitious.pillar2_longhorizon.probe_battery import PROBE_BANKS


def run_indicators_for_agent(cond_agent, condition_code: str, base_model: str, seed: int,
                             output_dir: str) -> dict:
    """对单个 agent 跑完全部 indicator 测量。"""
    results = {
        "condition": condition_code,
        "base_model": base_model,
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "indicators": {},
    }

    metacog_probes = PROBE_BANKS["metacog"]
    hot1 = measure_hot1(cond_agent, metacog_probes)
    results["indicators"]["HOT1_meta_d_prime"] = hot1

    if hasattr(cond_agent, 'llm') and hasattr(cond_agent.llm, 'model'):
        model = cond_agent.llm.model
        collector = AttentionEntropyCollector()
        try:
            collector.attach(model)
            for item in metacog_probes[:20]:
                cond_agent.step(item["prompt"])
                collector.step()
            gwt_summary = collector.summary()
            results["indicators"]["GWT_attention_entropy"] = gwt_summary
        except Exception as e:
            results["indicators"]["GWT_attention_entropy"] = {"error": str(e)}
        finally:
            collector.detach()

    component_signals = {}
    try:
        if hasattr(cond_agent.agent, 'homeostasis'):
            hs = cond_agent.agent.homeostasis
            if hasattr(hs, 'history') and hs.history:
                anxiety_hist = [h.anxiety for h in hs.history[-500:]]
                surprise_hist = [h.surprise for h in hs.history[-500:]]
                boredom_hist = [h.boredom for h in hs.history[-500:]]
                component_signals["anxiety"] = np.array(anxiety_hist)
                component_signals["surprise"] = np.array(surprise_hist)
                component_signals["boredom"] = np.array(boredom_hist)
    except Exception:
        pass

    try:
        if hasattr(cond_agent.agent, 'lora_pool') and cond_agent.agent.lora_pool is not None:
            pool = cond_agent.agent.lora_pool
            if hasattr(pool, 'expert_usage') and pool.expert_usage:
                usage = list(pool.expert_usage.values())[-500:]
                component_signals["lora_usage"] = np.array(usage)
    except Exception:
        pass

    if len(component_signals) >= 2:
        phi_result = measure_phi_approx(component_signals)
        results["indicators"]["phi_R"] = phi_result

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fname = f"indicators_{condition_code}_seed{seed}_{base_model.replace('/', '_')}.json"
    fpath = Path(output_dir) / fname
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    print(f"[Pillar3] {condition_code} seed{seed} -> {fpath.name}")
    return results


def collect_from_longhorizon(root: Path, conditions: list[str], seeds: list[int]):
    """从已完成的 longhorizon 运行中收集 agent 状态用于 indicator 测量。"""
    runs = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        parts = d.name.split("_seed")
        if len(parts) != 2:
            continue
        cond = parts[0]
        seed_model = parts[1]
        seed_s, model = seed_model.split("_", 1)
        seed_n = int(seed_s)
        if cond in conditions and seed_n in seeds:
            runs.append({"condition": cond, "seed": seed_n, "model": model, "path": d})
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="experiments/v2_ambitious/results/longhorizon")
    p.add_argument("--output", default="experiments/v2_ambitious/results/indicators")
    p.add_argument("--conditions", default="C0_trueman_full,C1_reversed,C2_scrambled,C3_frozen,C4_trivial_jaccard")
    p.add_argument("--seeds", default="0,1,2,3")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    runs = collect_from_longhorizon(Path(args.root), conditions, seeds)
    print(f"[Pillar3] Found {len(runs)} completed longhorizon runs")

    if args.dry_run:
        for r in runs:
            print(f"  Would run indicators: {r['condition']} seed{r['seed']} {r['model']}")
        return

    all_results = []
    for r in runs:
        cfg = AgentConfig()
        cfg.base_model_name = r["model"]
        cfg.load_in_4bit = True
        try:
            cond_agent = make_condition(r["condition"], cfg, seed=r["seed"])
            result = run_indicators_for_agent(
                cond_agent, r["condition"], r["model"], r["seed"], args.output
            )
            all_results.append(result)
        except Exception as e:
            print(f"[Pillar3] ERROR for {r['condition']} seed{r['seed']}: {e}")
            all_results.append({
                "condition": r["condition"], "seed": r["seed"],
                "model": r["model"], "error": str(e),
            })

    summary_path = Path(args.output) / "indicators_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"[Pillar3] Summary -> {summary_path}")


if __name__ == "__main__":
    main()
