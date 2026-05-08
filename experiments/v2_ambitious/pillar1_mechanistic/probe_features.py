"""寻找编码情绪信号的 SAE 特征。

方法：对每个 SAE 特征 f_i，计算其在所有 captured steps 上的激活
与同步记录的真值情绪 (surprise / boredom / anxiety) 的 Pearson 相关。
显著（Bonferroni at α=0.001）且 |r| ≥ 0.3 的特征被标记为 "emotion-encoding"。

输出：features_anxiety.json (特征 ID + 相关性 + 激活方向向量)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.v2_ambitious.pillar1_mechanistic.train_sae import TopKSAE, load_captures


def load_records(h5_path: str) -> dict:
    import h5py
    with h5py.File(h5_path, "r") as f:
        recs = f["records"][:]
    return {
        "step": recs["step"],
        "surprise": recs["surprise"],
        "boredom": recs["boredom"],
        "anxiety": recs["anxiety"],
        "drive": recs["drive"],
        "condition": [c.decode() for c in recs["condition"]],
    }


def encode_all(sae: TopKSAE, X: torch.Tensor, batch: int = 4096, device="cuda") -> np.ndarray:
    sae.eval().to(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            z = sae.encode(X[i:i+batch].to(device))
            out.append(z.cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--captures", required=True)
    p.add_argument("--sae", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--target", choices=["surprise", "boredom", "anxiety", "drive"],
                   default="anxiety")
    p.add_argument("--top-k-features", type=int, default=16)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    print(f"[Probe] Loading SAE from {args.sae}")
    ckpt = torch.load(args.sae, map_location="cpu")
    cfg = ckpt["config"]
    sae = TopKSAE(cfg["hidden_dim"], cfg["dict_size"], cfg["top_k"])
    sae.load_state_dict(ckpt["state_dict"])

    print(f"[Probe] Loading captures and records")
    X = load_captures(args.captures, args.layer)
    recs = load_records(args.captures)
    target = recs[args.target]

    if len(target) != len(X):
        cut = min(len(target), len(X))
        X = X[:cut]
        target = target[:cut]

    print(f"[Probe] Encoding {len(X)} states...")
    Z = encode_all(sae, X, device=args.device)        # (N, dict_size)
    print(f"[Probe] Z shape: {Z.shape}")

    target = np.asarray(target, dtype=np.float64)
    target_centered = target - target.mean()
    target_var = target.var() + 1e-12

    Z_centered = Z - Z.mean(axis=0, keepdims=True)
    Z_var = Z.var(axis=0) + 1e-12

    cov = (Z_centered * target_centered[:, None]).mean(axis=0)
    r = cov / np.sqrt(Z_var * target_var)

    n = len(target)
    t_stat = r * np.sqrt((n - 2) / np.maximum(1 - r**2, 1e-12))
    from scipy.stats import t as student_t
    p_vals = 2 * (1 - student_t.cdf(np.abs(t_stat), df=n - 2))

    bonf_alpha = 0.001 / cfg["dict_size"]
    significant = (np.abs(r) >= 0.3) & (p_vals < bonf_alpha)

    top_idx = np.argsort(-np.abs(r))[:args.top_k_features]
    top_idx = [i for i in top_idx if significant[i]]

    # 提取每个特征的解码方向（W_dec 的列）= 残差流上的 "推送方向"
    W_dec = sae.W_dec.weight.detach().cpu().numpy()  # (hidden_dim, dict_size)
    directions = {int(i): W_dec[:, i].tolist() for i in top_idx}

    out = {
        "target": args.target,
        "layer": args.layer,
        "n_significant": int(significant.sum()),
        "top_features": [
            {
                "feature_id": int(i),
                "r": float(r[i]),
                "p_value": float(p_vals[i]),
                "median_activation": float(np.median(Z[:, i])),
                "max_activation": float(Z[:, i].max()),
                "decoder_direction": directions[int(i)],
            }
            for i in top_idx
        ],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[Probe] {len(top_idx)} top features saved to {args.output}")


if __name__ == "__main__":
    main()
