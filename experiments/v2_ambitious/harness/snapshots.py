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
