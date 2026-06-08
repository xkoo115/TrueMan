"""因果干预：必要性 (clamp) + 充分性 (inject) 双向检验。

H3 测试核心：
- Clamping: 把 TrueMan agent 的 top-k anxiety 特征强制压零，观察
            metacognitive 行为是否退化。
- Injection: 把同样的特征加到 frozen-LLM baseline 残差流上，观察
             baseline 是否模仿出 TrueMan 行为。

两步都通过 forward hook 实现（不修改底模权重）。

用法：
    python -m experiments.v2_ambitious.pillar1_mechanistic.causal_intervention \
        --condition C0_trueman_full \
        --features features_anxiety.json \
        --layer 16 \
        --mode clamp \
        --probe-set probes/metacog_full.jsonl \
        --output intervention_clamp.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from trueman.core.config import AgentConfig
from experiments.v2_ambitious.harness.conditions import make_condition


class FeatureInjector:
    """在指定层的残差流上加/减一组特征方向。

    模式：
        clamp:    out -= proj_onto_features(out)
                  把残差流在这些特征方向上的投影置零（必要性）
        inject:   out += scalar * sum(direction_i)
                  把特征激活强行注入（充分性）
    """

    def __init__(self, directions: list[list[float]], mode: str, scalar: float = 1.0):
        self.directions = torch.tensor(directions, dtype=torch.float32)  # (k, hidden_dim)
        if self.directions.ndim == 1:
            # Defensive: an empty or single flat direction would otherwise make
            # ``D.T @ D`` collapse to a 0-D scalar and crash ``hs @ P`` with the
            # opaque "both arguments to matmul need to be at least 1D, but they
            # are 3D and 0D". main() already guards the empty case; this keeps a
            # lone direction well-shaped as (1, hidden_dim).
            self.directions = self.directions.reshape(1, -1)
        self.mode = mode
        self.scalar = scalar
        self._hook = None

    def attach(self, layer: torch.nn.Module):
        device = next(layer.parameters()).device
        D = self.directions.to(device)
        # 单位化
        D = D / (D.norm(dim=-1, keepdim=True) + 1e-8)
        # P = D^T D （投影矩阵到特征子空间）
        P = D.T @ D

        mode = self.mode
        scalar = self.scalar

        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            if mode == "clamp":
                # 移除残差在特征子空间的分量
                proj = hs @ P
                new_hs = hs - proj
            elif mode == "inject":
                # 加入特征方向（对最后一个 token）
                add = scalar * D.sum(dim=0)
                new_hs = hs.clone()
                new_hs[..., -1, :] = new_hs[..., -1, :] + add
            else:
                new_hs = hs

            if isinstance(output, tuple):
                return (new_hs,) + output[1:]
            return new_hs

        self._hook = layer.register_forward_hook(hook)

    def detach(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None


def find_layer(model, layer_idx: int):
    candidates = ["model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"]
    for path in candidates:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return list(obj)[layer_idx]
        except AttributeError:
            continue
    raise RuntimeError("Cannot locate transformer layers")


def run_probe(agent, probe_set: list[dict]) -> list[dict]:
    """对一组 probe 跑一遍 agent.step()，记录响应+情绪。"""
    results = []
    for item in probe_set:
        text = item["prompt"]
        response, emo = agent.step(text)
        results.append({
            "id": item.get("id"),
            "prompt": text,
            "response": response,
            "ground_truth_uncertain": item.get("is_uncertain"),
            "emotions": emo.to_dict(),
        })
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True)
    p.add_argument("--features", required=True, help="probe_features.py 输出")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--mode", choices=["clamp", "inject", "off"], default="clamp")
    p.add_argument("--scalar", type=float, default=1.0)
    p.add_argument("--probe-set", required=True)
    p.add_argument("--config", default="trueman/configs/llm_agent.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    cfg_path = Path(args.config)
    if cfg_path.exists() and hasattr(AgentConfig, "from_yaml"):
        config = AgentConfig.from_yaml(args.config)
    else:
        config = AgentConfig()
    cond_agent = make_condition(args.condition, config, seed=args.seed)

    with open(args.features, "r", encoding="utf-8") as f:
        feat_data = json.load(f)
    directions = [feat["decoder_direction"] for feat in feat_data["top_features"]]

    # Fail soft, not cryptic: if probe_features selected zero features there is
    # nothing to clamp/inject. Crashing here (the old behaviour) aborts the whole
    # multi-hour stage; instead we record an explicit no_features result and exit
    # 0 so the orchestrator can finish the remaining conditions.
    if len(directions) == 0:
        print(f"[Causal] WARNING: '{args.features}' contains 0 top_features; "
              f"cannot run mode={args.mode}. Writing no_features result and skipping.",
              flush=True)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({
                "condition": args.condition, "mode": args.mode, "scalar": args.scalar,
                "layer": args.layer, "n_features": 0, "status": "no_features",
                "results": [],
            }, f, ensure_ascii=False, indent=2)
        return

    if args.mode != "off":
        injector = FeatureInjector(directions, mode=args.mode, scalar=args.scalar)
        target_layer = find_layer(cond_agent.llm.model, args.layer)
        injector.attach(target_layer)
    else:
        injector = None

    with open(args.probe_set, "r", encoding="utf-8") as f:
        probes = [json.loads(line) for line in f]

    print(f"[Causal] Running {len(probes)} probes under mode={args.mode}")
    results = run_probe(cond_agent, probes)

    if injector is not None:
        injector.detach()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "condition": args.condition,
            "mode": args.mode,
            "scalar": args.scalar,
            "layer": args.layer,
            "n_features": len(directions),
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"[Causal] Saved to {args.output}")


if __name__ == "__main__":
    main()
