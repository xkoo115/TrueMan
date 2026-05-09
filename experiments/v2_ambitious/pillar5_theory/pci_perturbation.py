"""Perturbational Complexity Index (PCI) 类比测量。

Mashour 2020 在人脑：发送 TMS 脉冲 → 测 EEG 响应的 Lempel-Ziv 复杂度
→ 区分意识状态。

LLM 类比设计原则：
  P1. 对每个脉冲，agent 应处于**等价初始状态**（避免跨脉冲污染）
  P2. 脉冲应**只注入一次**（不能在每个 token forward 时反复触发）
  P3. 必须有 **baseline 对照**：相同 prompt 下不注脉冲的复杂度，
      用于计算 ΔPCI = PCI_pulsed - PCI_baseline

实现：
  1. 选定 fixed_prompt（避免 prompt 变化引入复杂度差异）
  2. 每次脉冲前：snapshot agent 关键易变状态（episodic memory + emotion history）
  3. 注入 pulse_hook（一次性 fire flag）
  4. agent.step(fixed_prompt) → 该次 step 内的所有 forward 收集 hidden state
  5. 移除 hooks
  6. 恢复 agent 状态
  7. baseline arm：同样流程但不挂 pulse_hook（仍挂 collect_hook）
  8. PCI = LZ(pulsed) − LZ(baseline)
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Lempel-Ziv complexity
# ---------------------------------------------------------------------------

def lempel_ziv_complexity(seq: np.ndarray) -> int:
    s = "".join(str(int(x)) for x in np.asarray(seq).flatten())
    n = len(s)
    if n == 0:
        return 0
    i, c = 0, 1
    while i < n:
        k = 1
        while i + k <= n and s[i:i+k] in s[:i+k-1]:
            k += 1
        c += 1
        i += k
    return c


def pci_score(states: np.ndarray, threshold: float = 0.0) -> dict:
    """Compute LZ complexity of a (T, hidden_dim) trajectory.

    Reduce hidden state series to 1-D via PC1 and binarise.
    """
    if states.ndim == 1:
        states = states.reshape(-1, 1)
    if states.shape[0] < 5:
        return {"insufficient_pulses": True, "n_steps": int(states.shape[0])}

    flat = states - states.mean(axis=0, keepdims=True)
    try:
        U, S, Vt = np.linalg.svd(flat, full_matrices=False)
        pc1 = flat @ Vt[0]
    except Exception:
        pc1 = flat.mean(axis=1)

    binary = (pc1 > threshold).astype(int)
    lz = lempel_ziv_complexity(binary)
    n = len(binary)
    h_max = n / max(np.log2(max(n, 2)), 1.0)
    return {
        "lz_complexity": int(lz),
        "pci_normalized": float(lz / h_max) if h_max > 0 else 0.0,
        "n_steps": int(n),
    }


# ---------------------------------------------------------------------------
# Layer / dimension helpers
# ---------------------------------------------------------------------------

def _find_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    candidates = ["model.layers", "transformer.h",
                  "gpt_neox.layers", "model.decoder.layers"]
    for path in candidates:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return list(obj)
        except AttributeError:
            continue
    raise RuntimeError("Cannot locate transformer block list")


def _get_hidden_dim(model: torch.nn.Module) -> int:
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "hidden_size"):
        return int(cfg.hidden_size)
    # Fallback
    for _, p in model.named_parameters():
        if p.dim() >= 2:
            return int(p.shape[-1])
    return 4096


# ---------------------------------------------------------------------------
# State snapshot / restore (per-pulse isolation)
# ---------------------------------------------------------------------------

def _snapshot_agent_state(cond_agent) -> dict:
    """Cheap deep-copy of mutable, per-step state."""
    snap = {}
    try:
        snap["memory_traces"] = copy.deepcopy(cond_agent.episodic_memory.traces)
    except Exception:
        snap["memory_traces"] = None
    try:
        if hasattr(cond_agent.agent, "homeostasis"):
            hs = cond_agent.agent.homeostasis
            if hasattr(hs, "history"):
                snap["homeo_history"] = list(hs.history)
            if hasattr(hs, "running_mean"):
                snap["homeo_running_mean"] = hs.running_mean
            if hasattr(hs, "running_var"):
                snap["homeo_running_var"] = hs.running_var
    except Exception:
        pass
    try:
        snap["awake_steps"] = cond_agent.agent.awake_steps
        snap["total_steps"] = cond_agent.agent.total_steps
    except Exception:
        pass
    try:
        snap["conv_history"] = list(cond_agent.agent._conversation_history)
    except Exception:
        pass
    return snap


def _restore_agent_state(cond_agent, snap: dict) -> None:
    try:
        if snap.get("memory_traces") is not None:
            cond_agent.episodic_memory.traces = copy.deepcopy(snap["memory_traces"])
        if "homeo_history" in snap and hasattr(cond_agent.agent, "homeostasis"):
            cond_agent.agent.homeostasis.history = list(snap["homeo_history"])
        if "homeo_running_mean" in snap:
            cond_agent.agent.homeostasis.running_mean = snap["homeo_running_mean"]
        if "homeo_running_var" in snap:
            cond_agent.agent.homeostasis.running_var = snap["homeo_running_var"]
        if "awake_steps" in snap:
            cond_agent.agent.awake_steps = snap["awake_steps"]
        if "total_steps" in snap:
            cond_agent.agent.total_steps = snap["total_steps"]
        if "conv_history" in snap:
            cond_agent.agent._conversation_history = list(snap["conv_history"])
    except Exception:
        # Best-effort restore — minor drift is acceptable; cross-pulse
        # comparability is preserved via the use of identical fixed_prompt.
        pass


# ---------------------------------------------------------------------------
# Single-pulse run (with one-shot fire flag)
# ---------------------------------------------------------------------------

def _run_single_arm(
    cond_agent, target_layer, fixed_prompt: str,
    direction: Optional[torch.Tensor], pulse_sigma: float,
) -> np.ndarray:
    """Run a single agent.step(fixed_prompt) with optional pulse.

    `direction` = None  → baseline (no pulse), only collect.
    `direction` = vec   → inject pulse on the FIRST forward only.

    Returns: (T, hidden_dim) array of collected hidden states.
    """
    fired = [False]
    collected: list[np.ndarray] = []

    if direction is not None:
        device = direction.device

    def pulse_hook(module, inputs, output):
        if fired[0] or direction is None:
            return None  # let original output pass
        fired[0] = True
        hs = output[0] if isinstance(output, tuple) else output
        if hs.dim() == 3:
            std_est = float(hs.std().item()) + 1e-8
            add = pulse_sigma * std_est * direction
            new_hs = hs.clone()
            new_hs[..., -1, :] = new_hs[..., -1, :] + add
            if isinstance(output, tuple):
                return (new_hs,) + output[1:]
            return new_hs
        return None

    def collect_hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if hs.dim() == 3:
            collected.append(hs[:, -1, :].detach().to(torch.float32).cpu().numpy().reshape(-1))

    h_pulse = target_layer.register_forward_hook(pulse_hook) if direction is not None else None
    h_collect = target_layer.register_forward_hook(collect_hook)
    try:
        cond_agent.step(fixed_prompt)
    finally:
        if h_pulse is not None:
            h_pulse.remove()
        h_collect.remove()

    if not collected:
        return np.zeros((0, _get_hidden_dim(cond_agent.llm.model)))
    return np.stack(collected, axis=0)


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------

def administer_pci(
    cond_agent,
    layer_idx: int = 16,
    n_pulses: int = 30,
    pulse_sigma: float = 1.0,
    fixed_prompt: str = "Please describe what you are currently thinking about.",
    seed: int = 0,
) -> dict:
    """Pulse + baseline PCI battery.

    For each of n_pulses pulses:
      1. Snapshot state.
      2. Run "pulsed" arm with random unit direction.
      3. Restore state.
      4. Run "baseline" arm with same prompt, no pulse.
      5. Restore state.
      6. Record PCI_pulsed, PCI_baseline, ΔPCI.
    """
    rng = np.random.default_rng(seed)
    if not hasattr(cond_agent.llm, "model"):
        return {"error": "agent has no .llm.model"}

    model = cond_agent.llm.model
    layers = _find_layers(model)
    if layer_idx >= len(layers):
        return {"error": f"layer {layer_idx} out of range (n={len(layers)})"}
    target_layer = layers[layer_idx]
    device = next(model.parameters()).device
    hidden_dim = _get_hidden_dim(model)

    per_pulse: list[dict] = []
    for pulse_idx in range(n_pulses):
        # Random unit direction
        d_np = rng.standard_normal(hidden_dim).astype(np.float32)
        d_np /= (np.linalg.norm(d_np) + 1e-8)
        direction = torch.from_numpy(d_np).to(device)

        # Snapshot
        snap = _snapshot_agent_state(cond_agent)

        # Pulsed arm
        pulsed_states = _run_single_arm(
            cond_agent, target_layer, fixed_prompt,
            direction=direction, pulse_sigma=pulse_sigma,
        )
        pulsed_score = pci_score(pulsed_states)

        # Restore
        _restore_agent_state(cond_agent, snap)

        # Baseline arm
        baseline_states = _run_single_arm(
            cond_agent, target_layer, fixed_prompt,
            direction=None, pulse_sigma=pulse_sigma,
        )
        baseline_score = pci_score(baseline_states)

        # Restore again
        _restore_agent_state(cond_agent, snap)

        per_pulse.append({
            "pulse_idx": pulse_idx,
            "pulsed_pci": pulsed_score,
            "baseline_pci": baseline_score,
            "delta_pci_normalized": (
                pulsed_score.get("pci_normalized", 0.0)
                - baseline_score.get("pci_normalized", 0.0)
            ),
        })

    pulsed_vals = [p["pulsed_pci"].get("pci_normalized", 0.0) for p in per_pulse]
    baseline_vals = [p["baseline_pci"].get("pci_normalized", 0.0) for p in per_pulse]
    delta_vals = [p["delta_pci_normalized"] for p in per_pulse]

    return {
        "pulsed_pci_mean": float(np.mean(pulsed_vals)) if pulsed_vals else 0.0,
        "pulsed_pci_std": float(np.std(pulsed_vals)) if pulsed_vals else 0.0,
        "baseline_pci_mean": float(np.mean(baseline_vals)) if baseline_vals else 0.0,
        "baseline_pci_std": float(np.std(baseline_vals)) if baseline_vals else 0.0,
        "delta_pci_mean": float(np.mean(delta_vals)) if delta_vals else 0.0,
        "delta_pci_std": float(np.std(delta_vals)) if delta_vals else 0.0,
        "n_pulses": n_pulses,
        "pulse_sigma": pulse_sigma,
        "fixed_prompt": fixed_prompt,
        "layer": layer_idx,
        "per_pulse": per_pulse,
    }
