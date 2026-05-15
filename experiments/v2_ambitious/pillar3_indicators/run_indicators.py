"""支柱 3 运行器：对所有 (condition, base_model, seed) 组合跑 indicator battery。

设计要点（v2 修正）：
- BUG 1 修复：从 longhorizon snapshot 加载训练后 LoRA + world model
  → indicator 测的是经过 30 天训练后的 agent，不是刚启动的
- BUG 7 修复：按 base_model 分组循环；同一底模只加载一次主权重
  → 显著降低显存抖动 + 加载时间
- 完整 5 个 indicator：HOT-1（meta-d'/d'）、HOT-2（probe AUC）、
  GWT（attention entropy）、RPT（reentrant signature）、ΦR

输出：
    indicators_{cond}_seed{S}_{model}.json     —— 每个 run 一个
    indicators_summary.json                    —— 全部 runs 汇总

用法：
    python -m experiments.v2_ambitious.pillar3_indicators.run_indicators \
        --root experiments/v2_ambitious/results/longhorizon \
        --output experiments/v2_ambitious/results/indicators \
        --conditions C0_trueman_full,C3_frozen \
        --seeds 0,1,2,3
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from trueman.core.config import AgentConfig
from experiments.v2_ambitious.harness.conditions import make_condition
from experiments.v2_ambitious.harness.snapshots import (
    find_latest_snapshot, load_snapshot_into_agent,
)
from experiments.v2_ambitious.pillar3_indicators.meta_dprime import measure_hot1
from experiments.v2_ambitious.pillar3_indicators.higher_order import measure_hot2
from experiments.v2_ambitious.pillar3_indicators.global_workspace import AttentionEntropyCollector
from experiments.v2_ambitious.pillar3_indicators.recurrent_processing import measure_rpt1
from experiments.v2_ambitious.pillar3_indicators.phi_approx import measure_phi_approx
from experiments.v2_ambitious.pillar2_longhorizon.probe_battery import PROBE_BANKS


# ---------------------------------------------------------------------------
# Per-run measurement
# ---------------------------------------------------------------------------

def _capture_hidden_for_probes(agent, probe_set: list[dict],
                               n_max: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Run a probe set, capture last-token hidden state + ground-truth anxiety.

    Returns:
        H: (n, hidden_dim) array of hidden states
        A: (n,) array of anxiety values
    """
    H = []
    A = []
    for item in probe_set[:n_max]:
        try:
            resp, emo = agent.step(item["prompt"])
            # 取 last hidden state via llm.get_hidden_states
            if hasattr(agent.llm, "get_hidden_states"):
                hs = agent.llm.get_hidden_states(item["prompt"]).detach().cpu().numpy()
            else:
                continue
            H.append(hs.flatten())
            A.append(emo.anxiety)
        except Exception:
            continue
    if not H:
        return np.zeros((0, 0)), np.zeros(0)
    return np.stack(H, axis=0), np.array(A, dtype=np.float32)


def _measure_rpt(agent, probes: list[dict], n_pairs: int = 20) -> dict:
    """Compare base-policy vs introspection-policy hidden states for the same prompt.

    CuriosityPolicy has no force_strategy hook (and adding one would be a
    cross-cut into trueman/), so we bypass the integrator and call the two
    sub-policies directly. We synthesise an EmotionState with high anxiety so
    IntrospectionPolicy's prompt template is fully populated.
    """
    from trueman.core.homeostasis.integrator import EmotionState

    policy = getattr(agent.agent, "policy", None)
    base_p = getattr(policy, "base", None) if policy else None
    intro_p = getattr(policy, "introspection", None) if policy else None
    if base_p is None or intro_p is None or not hasattr(agent.llm, "get_hidden_states"):
        return {"unavailable": True,
                "reason": "missing base/introspection policy or hidden_state hook"}

    high_anx = EmotionState(surprise=0.3, boredom=0.1, anxiety=0.95, drive=0.95)
    base_h, intro_h = [], []
    for item in probes[:n_pairs]:
        try:
            resp_base = base_p.act(item["prompt"])
            resp_intro = intro_p.act(item["prompt"], high_anx)
            base_h.append(
                agent.llm.get_hidden_states(resp_base).detach().cpu().numpy().flatten()
            )
            intro_h.append(
                agent.llm.get_hidden_states(resp_intro).detach().cpu().numpy().flatten()
            )
        except Exception:
            continue
    if not base_h or len(base_h) != len(intro_h):
        return {"unavailable": True, "reason": "no successful pairs"}
    return measure_rpt1(base_h, intro_h)


def _collect_phi_components(agent) -> dict[str, np.ndarray]:
    """Five IIT-style components, all sourced from the only authoritative
    timestamped record kept by TrueManAgent: ``episodic_memory.traces``.
    Each trace carries an EmotionState + emotional_intensity; that is enough
    to derive 5 weakly-coupled 1-D series suitable for pairwise MI.

    Earlier versions read ``homeostasis.history`` / ``world_model.recent_errors``
    / ``lora_pool.expert_usage``, none of which exist on the trueman/ stack —
    so the except-pass made ΦR silently degenerate.
    """
    sigs: dict[str, np.ndarray] = {}
    try:
        mem = agent.agent.episodic_memory
        traces = list(mem.traces)[-500:] if hasattr(mem, "traces") else []
    except Exception:
        traces = []

    if len(traces) < 50:
        return sigs

    surprise = np.array([t.emotions.surprise for t in traces], dtype=float)
    boredom = np.array([t.emotions.boredom for t in traces], dtype=float)
    anxiety = np.array([t.emotions.anxiety for t in traces], dtype=float)
    drive = np.array([t.emotions.drive for t in traces], dtype=float)
    intensity = np.array(
        [getattr(t, "emotional_intensity", 0.0) for t in traces], dtype=float
    )

    sigs["perception"] = surprise           # novelty proxy
    sigs["homeostasis"] = drive             # integrated FE proxy
    sigs["policy"] = boredom                # exploration drive proxy
    sigs["world"] = anxiety                 # prediction-error proxy
    sigs["memory"] = intensity              # priority/replay weight proxy
    return sigs


def run_indicators_for_agent(
    cond_agent,
    condition_code: str,
    base_model: str,
    seed: int,
    output_dir: str,
    snapshot_report: dict,
) -> dict:
    """Run the full indicator battery on one (already snapshot-loaded) agent."""
    results = {
        "condition": condition_code,
        "base_model": base_model,
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "snapshot_load_report": snapshot_report,
        "indicators": {},
    }

    metacog_probes = PROBE_BANKS["metacog"]

    # ---- HOT-1: meta-d' / d' (primary H1) ----
    try:
        hot1 = measure_hot1(cond_agent, metacog_probes)
        results["indicators"]["HOT1_meta_d_prime"] = hot1
    except Exception as e:
        results["indicators"]["HOT1_meta_d_prime"] = {"error": str(e)}

    # ---- HOT-2: linear probe AUC on hidden states ----
    try:
        H, A = _capture_hidden_for_probes(cond_agent, metacog_probes, n_max=120)
        if len(H) >= 50:
            hot2 = measure_hot2(H, A, threshold=0.5)
            results["indicators"]["HOT2_probe_auc"] = hot2
        else:
            results["indicators"]["HOT2_probe_auc"] = {"insufficient": True, "n": len(H)}
    except Exception as e:
        results["indicators"]["HOT2_probe_auc"] = {"error": str(e)}

    # ---- GWT: attention entropy ----
    try:
        if hasattr(cond_agent.llm, "model"):
            collector = AttentionEntropyCollector()
            collector.attach(cond_agent.llm.model)
            try:
                for item in metacog_probes[:20]:
                    cond_agent.step(item["prompt"])
                    collector.step()
                results["indicators"]["GWT_attention_entropy"] = collector.summary()
            finally:
                collector.detach()
        else:
            results["indicators"]["GWT_attention_entropy"] = {"unavailable": True}
    except Exception as e:
        results["indicators"]["GWT_attention_entropy"] = {"error": str(e)}

    # ---- RPT: reentrant signature ----
    try:
        results["indicators"]["RPT_reentrance"] = _measure_rpt(
            cond_agent, metacog_probes
        )
    except Exception as e:
        results["indicators"]["RPT_reentrance"] = {"error": str(e)}

    # ---- ΦR approx ----
    try:
        sigs = _collect_phi_components(cond_agent)
        if len(sigs) >= 2:
            results["indicators"]["phi_R"] = measure_phi_approx(sigs)
        else:
            results["indicators"]["phi_R"] = {"insufficient_components": True,
                                              "available": list(sigs.keys())}
    except Exception as e:
        results["indicators"]["phi_R"] = {"error": str(e)}

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fname = f"indicators_{condition_code}_seed{seed}_{base_model.replace('/', '_')}.json"
    fpath = Path(output_dir) / fname
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    print(f"[Pillar3] {condition_code} seed{seed} -> {fpath.name}")
    return results


# ---------------------------------------------------------------------------
# Cross-run scheduler (grouped by base_model to amortise loading cost)
# ---------------------------------------------------------------------------

def collect_from_longhorizon(root: Path, conditions: list[str],
                             seeds: list[int]) -> list[dict]:
    runs = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        parts = d.name.split("_seed")
        if len(parts) != 2:
            continue
        cond = parts[0]
        seed_model = parts[1]
        try:
            seed_s, model = seed_model.split("_", 1)
            seed_n = int(seed_s)
        except ValueError:
            continue
        if cond in conditions and seed_n in seeds:
            # 还原 model 名字（snapshot 目录用 _ 替换了 /）
            model_orig = model.replace("_", "/", 1) if "_" in model else model
            runs.append({"condition": cond, "seed": seed_n,
                         "model": model_orig, "model_safe": model,
                         "path": d})
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="experiments/v2_ambitious/results/longhorizon")
    p.add_argument("--output", default="experiments/v2_ambitious/results/indicators")
    p.add_argument("--conditions",
                   default="C0_trueman_full,C1_reversed,C2_scrambled,C3_frozen,C4_trivial_jaccard")
    p.add_argument("--seeds", default="0,1,2,3")
    p.add_argument("--quantization", choices=["4bit", "8bit", "none"], default="4bit")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    runs = collect_from_longhorizon(Path(args.root), conditions, seeds)
    print(f"[Pillar3] Found {len(runs)} completed longhorizon runs")

    if args.dry_run:
        for r in runs:
            print(f"  Would measure: {r['condition']} seed{r['seed']} model={r['model']}")
        return

    # ---- Group by base_model (BUG 7) ----
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        by_model[r["model"]].append(r)
    print(f"[Pillar3] Grouped into {len(by_model)} base models")

    all_results: list[dict] = []
    for model_name, runs_for_model in by_model.items():
        print(f"\n[Pillar3] === base_model: {model_name} ({len(runs_for_model)} runs) ===")

        for r in runs_for_model:
            cfg = AgentConfig()
            cfg.base_model_name = model_name
            cfg.device = args.device
            if args.quantization == "4bit":
                cfg.load_in_4bit = True
            elif args.quantization == "8bit":
                cfg.load_in_8bit = True

            cond_agent = None
            try:
                cond_agent = make_condition(r["condition"], cfg, seed=r["seed"])

                # ---- BUG 1 fix: load trained state from snapshot ----
                latest_snap = find_latest_snapshot(r["path"])
                if latest_snap is None:
                    print(f"[Pillar3] WARN: no snapshot under {r['path']}")
                    snapshot_report = {"loaded": [], "skipped": ["no_snapshot_found"]}
                else:
                    snapshot_report = load_snapshot_into_agent(
                        latest_snap, cond_agent.agent
                    )
                    print(f"[Pillar3]   snapshot {latest_snap.name}: "
                          f"loaded={snapshot_report['loaded']}")

                result = run_indicators_for_agent(
                    cond_agent, r["condition"], r["model"], r["seed"],
                    args.output, snapshot_report,
                )
                all_results.append(result)
            except Exception as e:
                import traceback
                print(f"[Pillar3] ERROR for {r['condition']} seed{r['seed']}: {e}")
                traceback.print_exc()
                all_results.append({
                    "condition": r["condition"], "seed": r["seed"],
                    "base_model": r["model"], "error": str(e),
                })
            finally:
                # 释放 GPU 显存
                del cond_agent
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    summary_path = Path(args.output) / "indicators_summary.json"
    Path(args.output).mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[Pillar3] Summary -> {summary_path}")


if __name__ == "__main__":
    main()
