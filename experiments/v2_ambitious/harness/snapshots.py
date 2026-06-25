"""每日参数 + 状态快照工具。

用于支柱 2 (长时程) 的核心度量：参数轨迹散度、行为表型漂移。

Bug 9 (致命): 旧版 ``take_snapshot`` 把 LoRA 专家序列化为
``{eid: e.state_dict()}``, 但 ``DynamicLoRAPool.experts`` 的 value 是
``ExpertMetadata`` (一个纯元数据 dataclass), 根本没有 ``.state_dict()``。
后果:
  - day0 (尚无专家) → 字典推导式得到 ``{}``, 侥幸写出一个空文件;
  - 一旦睡眠整合产出第一个专家, 推导式抛 ``AttributeError`` → 被 run.py 的
    try/except 吞掉 → 当天快照目录只剩一个空文件夹;
  - 于是 Pillar 2 轨迹散度恒为 0、Pillar 5 FEP 无数据、Pillar 3 指标电池
    因 ``lora_experts_unavailable`` 静默回退到裸 base 模型, 五个条件塌缩成同一个
    模型, 所有假设 NOT CONFIRMED。

真实的 LoRA 权重保存在磁盘上 ``ExpertMetadata.adapter_path`` 指向的 PEFT
适配器目录里。本模块现在从那里读回权重 (``peft.utils.load_peft_weights``),
序列化为 ``{eid: state_dict}`` (与下游 ``parameter_divergence`` 期望的格式一致),
并把可重建专家所需的元数据 (adapter_path / rank / domain_tag / adapter_config)
写到同目录的 ``lora_experts_meta.json`` 旁车文件。
"""

from __future__ import annotations

import json
import shutil
import tempfile
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


def _load_expert_weights(adapter_path: str | Path) -> dict[str, torch.Tensor] | None:
    """Read a single PEFT adapter's tensors back off disk.

    Returns ``None`` (rather than raising) if the directory is missing or holds
    no readable weights, so one corrupt expert never aborts a whole snapshot.
    """
    p = Path(adapter_path)
    if not p.exists():
        return None
    try:
        from peft.utils import load_peft_weights
        weights = load_peft_weights(str(p))
        return weights or None
    except Exception:
        # Fallback: raw file load, in case the peft helper rejects the dir.
        for fname in ("adapter_model.safetensors", "adapter_model.bin"):
            fp = p / fname
            if not fp.exists():
                continue
            try:
                if fname.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    return load_file(str(fp)) or None
                return torch.load(str(fp), map_location="cpu") or None
            except Exception:
                continue
        return None


def _serialize_experts(pool) -> tuple[dict[int, dict], dict[str, dict]]:
    """Pull every expert's weights + metadata out of a live LoRA pool.

    Returns ``(weights_by_eid, meta_by_eid)``.  ``weights_by_eid`` maps
    ``expert_id -> state_dict`` (the format ``parameter_divergence`` consumes);
    ``meta_by_eid`` is a JSON-serialisable sidecar keyed by ``str(expert_id)``.
    """
    weights: dict[int, dict] = {}
    meta: dict[str, dict] = {}
    experts = getattr(pool, "experts", None) or {}
    for eid, md in experts.items():
        adapter_path = getattr(md, "adapter_path", None)
        sd = _load_expert_weights(adapter_path) if adapter_path else None
        if sd is None:
            continue
        weights[int(eid)] = sd
        cfg = None
        if adapter_path:
            cfg_path = Path(adapter_path) / "adapter_config.json"
            if cfg_path.exists():
                try:
                    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                except Exception:
                    cfg = None
        meta[str(int(eid))] = {
            "adapter_path": str(adapter_path) if adapter_path else None,
            "rank": getattr(md, "rank", None),
            "domain_tag": getattr(md, "domain_tag", ""),
            "adapter_config": cfg,
        }
    return weights, meta


def take_snapshot(agent, output_dir: str, meta: SnapshotMeta) -> Path:
    """保存 agent 的：
    - LoRA 专家权重（从各专家的 adapter_path 读回真实张量）
    - 世界模型权重
    - episodic memory dump（仅元数据）
    - 快照元信息

    Returns:
        快照目录路径

    Raises:
        RuntimeError: 当 pool 报告有专家、但没有任何一个能被序列化时
        （磁盘 adapter 丢失/损坏）。宁可大声失败, 也不要再重蹈 Bug 9
        “看似完成、实则全是空” 的覆辙。
    """
    out = Path(output_dir) / f"day{meta.day:03d}_{meta.condition}_seed{meta.seed}"
    out.mkdir(parents=True, exist_ok=True)

    # ----- LoRA 专家 -----
    pool = getattr(agent, "lora_pool", None)
    n_reported = len(getattr(pool, "experts", {}) or {}) if pool is not None else 0
    if pool is not None:
        weights, expert_meta = _serialize_experts(pool)
        torch.save(weights, out / "lora_experts.pt")
        with open(out / "lora_experts_meta.json", "w", encoding="utf-8") as f:
            json.dump(expert_meta, f, ensure_ascii=False, indent=2, default=str)
        if n_reported > 0 and len(weights) == 0:
            raise RuntimeError(
                f"LoRA pool reports {n_reported} expert(s) but none could be "
                f"serialized (missing/unreadable adapter dirs). Refusing to write "
                f"a silently-empty snapshot for day {meta.day}."
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


def _materialize_adapter_dir(weights: dict, cfg: dict | None) -> str | None:
    """Write embedded expert weights + config back out as a loadable PEFT dir.

    Used when the original ``adapter_path`` is gone but the snapshot still
    carries the tensors. Returns a temp dir path, or ``None`` if we lack the
    config needed to reload (PEFT cannot attach weights without it).
    """
    if not cfg:
        return None
    tmp = Path(tempfile.mkdtemp(prefix="trueman_expert_"))
    try:
        with open(tmp / "adapter_config.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        from safetensors.torch import save_file
        save_file(weights, str(tmp / "adapter_model.safetensors"))
        return str(tmp)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        return None


def load_snapshot_into_agent(snapshot_dir: str | Path, agent) -> dict:
    """Load LoRA experts + world model + memory metadata back into a fresh agent.

    For the LoRA experts this re-attaches the adapters onto the live model via
    the pool's ``hot_loader`` so that downstream generation actually reflects
    the learned experts (not the bare base model). It prefers each expert's
    original on-disk ``adapter_path`` (from the sidecar) and falls back to
    materializing the embedded weights when that directory is gone.

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
    pool = getattr(agent, "lora_pool", None)
    if lora_path.exists() and pool is not None:
        try:
            experts_state = torch.load(lora_path, map_location="cpu")
            sidecar = {}
            meta_path = snap / "lora_experts_meta.json"
            if meta_path.exists():
                sidecar = json.loads(meta_path.read_text(encoding="utf-8"))

            n_loaded = 0
            for eid, sd in experts_state.items():
                emeta = sidecar.get(str(eid), {}) if isinstance(sidecar, dict) else {}
                adapter_dir = emeta.get("adapter_path")
                if not (adapter_dir and Path(adapter_dir).exists()):
                    adapter_dir = _materialize_adapter_dir(sd, emeta.get("adapter_config"))
                if adapter_dir and _reattach_expert(pool, int(eid), adapter_dir, emeta):
                    n_loaded += 1

            if n_loaded:
                report["loaded"].append(f"lora_experts({n_loaded})")
            if n_loaded < len(experts_state):
                report["skipped"].append(
                    f"lora_experts_partial({n_loaded}/{len(experts_state)})"
                )
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
            meta = json.loads(mem_path.read_text(encoding="utf-8"))
            report["loaded"].append(f"memory_meta({len(meta)} traces)")
            report["memory_meta_count"] = len(meta)
        except Exception as e:
            report["skipped"].append(f"memory_meta:{type(e).__name__}:{e}")

    return report


def _reattach_expert(pool, eid: int, adapter_dir: str, emeta: dict) -> bool:
    """Load one adapter dir onto the model and register its metadata.

    Returns True on success. Best-effort: any failure is swallowed and reported
    as a partial load by the caller.
    """
    adapter_name = f"expert_{eid}"
    hot_loader = getattr(pool, "hot_loader", None)
    if hot_loader is None or not hot_loader.load(adapter_name, adapter_dir):
        return False
    experts = getattr(pool, "experts", None)
    if experts is not None:
        try:
            from trueman.core.plasticity.lora_pool import ExpertMetadata
            import time as _time
            experts[eid] = ExpertMetadata(
                expert_id=eid,
                adapter_path=adapter_dir,
                creation_time=_time.time(),
                rank=emeta.get("rank") or 16,
                domain_tag=emeta.get("domain_tag", ""),
            )
        except Exception:
            # Registration is bookkeeping; the adapter is already on the model.
            pass
    return True


def parameter_divergence(snapshot_a: str, snapshot_b: str) -> dict[str, float]:
    """计算两个快照之间的 LoRA 权重 Frobenius 距离。

    ``lora_experts.pt`` 的格式是 ``{expert_id: state_dict}``。只比较两侧都存在的
    专家、且两侧都存在的张量名。
    """
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
