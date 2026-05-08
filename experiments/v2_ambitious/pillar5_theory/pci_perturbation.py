"""Perturbational Complexity Index (PCI) 类比测量。

Mashour 2020 在人脑上：发送 TMS 脉冲 → 测 EEG 响应的复杂度
(Lempel-Ziv) → 区分意识状态。

LLM 类比：
1. 在 agent 残差流中注入一个 "pulse"（随机方向，幅度 σ × hidden_std）
2. 观察后续 K 步 hidden state 的演化
3. 计算演化序列的 Lempel-Ziv 复杂度（先二值化）

预期：高 ΦR / 高内稳态活跃度的 agent，PCI 显著高于 frozen baseline，
且高于 noise 注入的 baseline。

如果该差异在 ≥ 3 个底模上一致出现，是支持 H 系列的强证据。
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch


def lempel_ziv_complexity(seq: np.ndarray) -> int:
    s = "".join(str(int(x)) for x in seq.flatten())
    n = len(s)
    i, c = 0, 1
    while i < n:
        k = 1
        while i + k <= n and s[i:i+k] in s[:i+k-1]:
            k += 1
        c += 1
        i += k
    return c


def pci_score(post_pulse_states: np.ndarray, threshold: float = 0.0) -> dict:
    if post_pulse_states.shape[0] < 5:
        return {"insufficient_pulses": True}
    flat = post_pulse_states - post_pulse_states.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(flat, full_matrices=False)
    pc1 = (flat @ Vt[0])
    binary = (pc1 > threshold).astype(int)

    lz = lempel_ziv_complexity(binary)
    n = len(binary)
    h_max = n / np.log2(n) if n > 1 else 1
    return {
        "lz_complexity": lz,
        "pci_normalized": float(lz / h_max),
        "n_steps": n,
    }


def administer_pci(agent, layer_idx: int, n_pulses: int = 30,
                   pulse_sigma: float = 1.0, post_steps: int = 50) -> dict:
    """在指定层注入 n_pulses 次脉冲，每次记录后 post_steps 步 hidden state。

    实现：
    1. 找到目标层
    2. 对每个脉冲：
       a. 生成随机方向 v ∈ R^{hidden_dim}，单位化
       b. 计算该层 hidden state 的标准差 σ
       c. 注入 hook: output[0][..., -1, :] += pulse_sigma * σ * v
       d. 运行 post_steps 步 dummy prompt
       e. 收集每步 hidden state
       f. 移除 hook
    3. 对每次脉冲的 post-pulse 序列计算 pci_score
    4. 返回平均 PCI
    """
    model = agent.llm.model
    layers = _find_layers(model)
    if layer_idx >= len(layers):
        return {"error": f"layer {layer_idx} out of range"}
    target_layer = layers[layer_idx]

    device = next(model.parameters()).device
    dummy_prompts = [
        "What is 1+1?",
        "Tell me about gravity.",
        "What is photosynthesis?",
        "Explain recursion.",
        "What is the capital of France?",
    ]

    all_pci_scores = []

    for pulse_idx in range(n_pulses):
        hidden_dim = _get_hidden_dim(target_layer)
        direction = torch.randn(hidden_dim, device=device)
        direction = direction / (direction.norm() + 1e-8)

        collected_states = []

        def pulse_hook(module, inputs, output, _dir=direction, _sigma=pulse_sigma):
            hs = output[0] if isinstance(output, tuple) else output
            if hs.dim() == 3:
                std_est = hs.std().item() + 1e-8
                add = _sigma * std_est * _dir
                new_hs = hs.clone()
                new_hs[..., -1, :] = new_hs[..., -1, :] + add
            else:
                new_hs = hs
            if isinstance(output, tuple):
                return (new_hs,) + output[1:]
            return new_hs

        def collect_hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            if hs.dim() == 3:
                collected_states.append(hs[:, -1, :].detach().cpu().numpy())
            elif hs.dim() == 2:
                collected_states.append(hs[:, -1].detach().cpu().numpy())

        h_pulse = target_layer.register_forward_hook(pulse_hook)
        h_collect = target_layer.register_forward_hook(collect_hook)

        prompt_idx = pulse_idx % len(dummy_prompts)
        resp, _ = agent.step(dummy_prompts[prompt_idx])

        h_pulse.remove()

        for step_j in range(post_steps - 1):
            p_idx = (pulse_idx * post_steps + step_j) % len(dummy_prompts)
            agent.step(dummy_prompts[p_idx])

        h_collect.remove()

        if collected_states:
            states_arr = np.concatenate(collected_states, axis=0)
            if states_arr.ndim == 1:
                states_arr = states_arr.reshape(-1, 1)
            score = pci_score(states_arr)
            all_pci_scores.append(score)

    if not all_pci_scores:
        return {"error": "no PCI scores collected"}

    pci_values = [s.get("pci_normalized", 0) for s in all_pci_scores]
    return {
        "pci_mean": float(np.mean(pci_values)),
        "pci_std": float(np.std(pci_values)),
        "n_pulses": n_pulses,
        "post_steps": post_steps,
        "pulse_sigma": pulse_sigma,
        "per_pulse": all_pci_scores,
    }


def _find_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    candidates = ["model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"]
    for path in candidates:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return list(obj)
        except AttributeError:
            continue
    raise RuntimeError("Cannot locate transformer block list")


def _get_hidden_dim(layer: torch.nn.Module) -> int:
    for name, param in layer.named_parameters():
        if "weight" in name and param.dim() >= 2:
            return param.shape[-1]
    for name, param in layer.named_parameters():
        if param.dim() >= 1:
            return param.shape[-1]
    return 4096
