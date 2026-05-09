"""每日参数 + 状态快照工具。

用于支柱 2 (长时程) 的核心度量：参数轨迹散度、行为表型漂移。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import torch


@dataclass
class SnapshotMeta:
    day: int
    condition: str
    base_model: str
    seed: int
    cumulative_steps: int
    cumulative_lora_experts: int
    cumulative_sleep_count: int
    timestamp: str


def take_snapshot(agent, output_dir: str, meta: SnapshotMeta) -> Path:
    """保存 agent 的：
    - LoRA 权重（如有）
    - 世界模型权重
    - episodic memory dump
    - 元信息

    Returns:
        快照目录路径
    """
    out = Path(output_dir) / f"day{meta.day:03d}_{meta.condition}_seed{meta.seed}"
    out.mkdir(parents=True, exist_ok=True)

    # LoRA 权重
    if agent.lora_pool is not None:
        torch.save(
            {eid: e.state_dict() for eid, e in agent.lora_pool.experts.items()},
            out / "lora_experts.pt",
        )

    # 世界模型
    torch.save(agent.world_model.state_dict(), out / "world_model.pt")

    # 记忆 dump（仅元数据，不存原文以省空间）
    traces_meta = [
        {
            "trace_id": t.trace_id,
            "timestamp": t.timestamp,
            "emotions": t.emotions.to_dict() if hasattr(t, "emotions") else {},
            "priority": getattr(t, "priority", 0.0),
        }
        for t in list(agent.episodic_memory.traces)[:5000]
    ]
    with open(out / "memory_meta.json", "w", encoding="utf-8") as f:
        json.dump(traces_meta, f, ensure_ascii=False, indent=2, default=str)

    with open(out / "snapshot_meta.json", "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, ensure_ascii=False, indent=2, default=str)

    return out


def find_latest_snapshot(run_dir: str | Path) -> Path | None:
    """Locate the highest-day snapshot in a run directory.

    Looks under ``run_dir/snapshots/`` for sub-directories named
    ``day{D}_{condition}_seed{S}`` and returns the one with the largest day.
    """
    snap_root = Path(run_dir) / "snapshots"
    if not snap_root.exists():
        return None
    candidates = []
    for d in snap_root.iterdir():
        if not d.is_dir():
            continue
        # 形如 day029_C0_trueman_full_seed0
        try:
            day_str = d.name.split("_", 1)[0]   # "day029"
            day = int(day_str.replace("day", ""))
            candidates.append((day, d))
        except (ValueError, IndexError):
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def load_snapshot_into_agent(snapshot_dir: str | Path, agent) -> dict:
    """Load LoRA experts + world model + memory metadata back into a fresh agent.

    Returns a dict describing what was loaded; missing pieces are reported
    rather than raised, so a single-component snapshot still loads gracefully.
    """
    snap = Path(snapshot_dir)
    report = {"snapshot_dir": str(snap), "loaded": [], "skipped": []}

    if not snap.exists():
        report["skipped"].append("snapshot_dir_missing")
        return report

    # ----- LoRA experts -----
    lora_path = snap / "lora_experts.pt"
    if lora_path.exists() and getattr(agent, "lora_pool", None) is not None:
        try:
            experts_state = torch.load(lora_path, map_location="cpu")
            n_loaded = 0
            pool = agent.lora_pool
            for eid, sd in experts_state.items():
                # 优先调用 pool 自有的注册接口；若无则直接写到 pool.experts
                if hasattr(pool, "load_expert_from_state_dict"):
                    pool.load_expert_from_state_dict(eid, sd)
                elif hasattr(pool, "experts"):
                    # 简化：把 state_dict 作为 expert 表示
                    pool.experts[int(eid)] = sd
                else:
                    raise RuntimeError("lora_pool has neither load_expert_from_state_dict nor .experts")
                n_loaded += 1
            report["loaded"].append(f"lora_experts({n_loaded})")
        except Exception as e:
            report["skipped"].append(f"lora_experts:{type(e).__name__}:{e}")
    else:
        report["skipped"].append("lora_experts_unavailable")

    # ----- World model -----
    wm_path = snap / "world_model.pt"
    if wm_path.exists() and hasattr(agent, "world_model"):
        try:
            agent.world_model.load_state_dict(
                torch.load(wm_path, map_location="cpu")
            )
            report["loaded"].append("world_model")
        except Exception as e:
            report["skipped"].append(f"world_model:{type(e).__name__}:{e}")

    # ----- Episodic memory metadata -----
    # 我们只 load 优先级和情绪元数据；原始文本未存（节省空间），
    # 因此 ablation probe 的"无记忆"条件天然成立。
    mem_path = snap / "memory_meta.json"
    if mem_path.exists():
        try:
            import json
            meta = json.loads(mem_path.read_text(encoding="utf-8"))
            report["loaded"].append(f"memory_meta({len(meta)} traces)")
            report["memory_meta_count"] = len(meta)
        except Exception as e:
            report["skipped"].append(f"memory_meta:{type(e).__name__}:{e}")

    return report


def parameter_divergence(snapshot_a: str, snapshot_b: str) -> dict[str, float]:
    """计算两个快照之间的 LoRA 权重 Frobenius 距离。"""
    a_path = Path(snapshot_a) / "lora_experts.pt"
    b_path = Path(snapshot_b) / "lora_experts.pt"
    if not a_path.exists() or not b_path.exists():
        return {"frobenius": 0.0, "n_experts_a": 0, "n_experts_b": 0}

    a = torch.load(a_path, map_location="cpu")
    b = torch.load(b_path, map_location="cpu")

    common = sorted(set(a.keys()) & set(b.keys()))
    total_sq = 0.0
    n_params = 0
    for k in common:
        for name in a[k]:
            if name in b[k]:
                d = (a[k][name] - b[k][name]).flatten()
                total_sq += float((d * d).sum().item())
                n_params += d.numel()

    return {
        "frobenius": (total_sq ** 0.5),
        "n_experts_a": len(a),
        "n_experts_b": len(b),
        "common_experts": len(common),
        "n_params_compared": n_params,
    }
