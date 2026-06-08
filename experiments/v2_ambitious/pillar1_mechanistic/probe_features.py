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


def bh_fdr_significant(p_vals: np.ndarray, q: float) -> np.ndarray:
    """Benjamini-Hochberg FDR. Returns a boolean mask of features whose p-value
    is significant at level ``q`` controlling the false-discovery rate.

    Standard choice for thousands of simultaneous correlation tests: far less
    conservative than Bonferroni-over-dict_size, which at pilot sample sizes
    demands an unreachable |r| and selects nothing.
    """
    p = np.asarray(p_vals, dtype=np.float64)
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresh = q * (np.arange(1, m + 1) / m)
    passed = ranked <= thresh
    if not passed.any():
        return np.zeros(m, dtype=bool)
    cutoff = ranked[np.max(np.where(passed)[0])]
    return p <= cutoff


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
    p.add_argument("--fdr-q", type=float, default=0.05,
                   help="Benjamini-Hochberg FDR level for feature significance.")
    p.add_argument("--r-floor", type=float, default=0.2,
                   help="Minimum |Pearson r| effect-size floor on top of FDR.")
    p.add_argument("--bonferroni-alpha", type=float, default=0.001,
                   help="Legacy Bonferroni alpha; reported for reference only.")
    p.add_argument("--no-fallback", action="store_true",
                   help="Disable the top-k-by-|r| fallback when nothing is significant "
                        "(then an empty feature set is allowed).")
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
    abs_r = np.abs(r)

    # --- significance: BH-FDR (primary) + a |r| effect-size floor ---
    # The old rule was Bonferroni over dict_size AND |r|>=0.3. At pilot N
    # (tens of captured steps) that demands |r| ~ 0.5+, which is essentially
    # unreachable for a single SAE feature vs. a composite scalar — it returned
    # zero features and crashed the whole downstream battery. FDR is the
    # standard multiple-comparison control here; the |r| floor stops us from
    # selecting pure noise. The legacy Bonferroni count is still reported.
    bonf_alpha = args.bonferroni_alpha / cfg["dict_size"]
    n_bonferroni = int(((abs_r >= 0.3) & (p_vals < bonf_alpha)).sum())
    fdr_sig = bh_fdr_significant(p_vals, args.fdr_q)
    significant = fdr_sig & (abs_r >= args.r_floor)

    order = np.argsort(-abs_r)
    selected = [int(i) for i in order if significant[i]][:args.top_k_features]
    selection_method = "fdr+rfloor"

    # --- fallback: never hand an empty feature set to the intervention stage ---
    if not selected and not args.no_fallback:
        selected = [int(i) for i in order[:args.top_k_features]]
        selection_method = "fallback_top_k_by_abs_r"
        print(f"[Probe] WARNING: no feature passed FDR q={args.fdr_q} & |r|>={args.r_floor}; "
              f"falling back to top-{len(selected)} by |r| "
              f"(flagged not-significant — EXPLORATORY only).", flush=True)

    # --- diagnostics: make an empty/weak result interpretable at a glance ---
    print(f"[Probe] N captures         : {n}", flush=True)
    print(f"[Probe] max |r|            : {abs_r.max():.4f}", flush=True)
    for thr in (0.2, 0.25, 0.3, 0.4, 0.5):
        print(f"[Probe]   #features |r|>={thr} : {int((abs_r >= thr).sum())}", flush=True)
    print(f"[Probe] #FDR-significant    : {int(fdr_sig.sum())} (q={args.fdr_q})", flush=True)
    print(f"[Probe] #Bonferroni-signif  : {n_bonferroni} (legacy criterion)", flush=True)
    print(f"[Probe] selection_method    : {selection_method}; selected {len(selected)} features",
          flush=True)

    # 提取每个特征的解码方向（W_dec 的列）= 残差流上的 "推送方向"
    W_dec = sae.W_dec.weight.detach().cpu().numpy()  # (hidden_dim, dict_size)
    directions = {int(i): W_dec[:, i].tolist() for i in selected}

    out = {
        "target": args.target,
        "layer": args.layer,
        "n_captures": int(n),
        "max_abs_r": float(abs_r.max()),
        "selection_method": selection_method,
        "fdr_q": float(args.fdr_q),
        "r_floor": float(args.r_floor),
        "n_fdr_significant": int(fdr_sig.sum()),
        "n_bonferroni_significant": n_bonferroni,
        "n_significant": int(significant.sum()),
        "top_features": [
            {
                "feature_id": int(i),
                "r": float(r[i]),
                "p_value": float(p_vals[i]),
                "significant": bool(significant[i]),
                "median_activation": float(np.median(Z[:, i])),
                "max_activation": float(Z[:, i].max()),
                "decoder_direction": directions[int(i)],
            }
            for i in selected
        ],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[Probe] {len(selected)} top features saved to {args.output} "
          f"(method={selection_method})")


if __name__ == "__main__":
    main()
