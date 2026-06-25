"""训练 Top-K Sparse Autoencoder。

依赖：torch, h5py, tqdm
参考实现：Anthropic "Scaling Monosemanticity" (2024)，
公开实现：sae_lens (https://github.com/jbloomAus/SAELens) 可用作替代。

我们提供一个简洁的自包含 Top-K SAE 实现，避免外部依赖。

用法：
    python -m experiments.v2_ambitious.pillar1_mechanistic.train_sae \
        --captures captures.h5 \
        --layer 18 \
        --dict-size 16384 \
        --top-k 64 \
        --epochs 5 \
        --batch-size 4096 \
        --output sae_layer18.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSAE(nn.Module):
    """Top-K Sparse Autoencoder。

    forward:
        z = TopK(W_enc(x - b_dec) + b_enc)
        x_hat = W_dec @ z + b_dec
    """

    def __init__(self, hidden_dim: int, dict_size: int, top_k: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dict_size = dict_size
        self.top_k = top_k

        self.W_enc = nn.Linear(hidden_dim, dict_size, bias=True)
        self.W_dec = nn.Linear(dict_size, hidden_dim, bias=True)
        self.b_dec = nn.Parameter(torch.zeros(hidden_dim))

        # 初始化：W_dec 列单位化
        with torch.no_grad():
            self.W_dec.weight.copy_(self._unit_columns(self.W_dec.weight))
            self.W_enc.weight.copy_(self.W_dec.weight.T)

    @staticmethod
    def _unit_columns(W: torch.Tensor) -> torch.Tensor:
        return W / (W.norm(dim=0, keepdim=True) + 1e-8)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = self.W_enc(x - self.b_dec)
        # Top-K
        topk_vals, topk_idx = pre.topk(self.top_k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, topk_idx, topk_vals.relu())
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.W_dec(z) + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


def dequantize_int8(arr: np.ndarray, scale: float = None) -> np.ndarray:
    if scale is None:
        # 没存 scale 的情况下假设 1.0；如使用 capture.py 量化建议同步存 scale
        return arr.astype(np.float32)
    return arr.astype(np.float32) * scale


def load_captures(h5_path: str, layer: int, max_samples: int | None = None):
    import h5py
    with h5py.File(h5_path, "r") as f:
        ds = f[f"layer_{layer}/states"]
        n = ds.shape[0]
        if max_samples is not None:
            n = min(n, max_samples)
        data = ds[:n]
    return torch.from_numpy(dequantize_int8(data)).float()


def train(args):
    print(f"[SAE] Loading captures from {args.captures} layer {args.layer}")
    X = load_captures(args.captures, args.layer, max_samples=args.max_samples)
    n, hidden_dim = X.shape
    print(f"[SAE] Loaded {n} activations, hidden_dim={hidden_dim}")

    sae = TopKSAE(hidden_dim, args.dict_size, args.top_k)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sae = sae.to(device)
    optim = torch.optim.Adam(sae.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, args.batch_size):
            batch = X[perm[i:i+args.batch_size]].to(device)
            x_hat, z = sae(batch)
            loss = F.mse_loss(x_hat, batch)
            optim.zero_grad()
            loss.backward()
            # 解码器列单位化（每步）
            with torch.no_grad():
                sae.W_dec.weight.copy_(sae._unit_columns(sae.W_dec.weight))
            optim.step()
            total_loss += loss.item() * batch.size(0)
        print(f"[SAE] epoch {epoch}: mean MSE = {total_loss / n:.6f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": sae.state_dict(),
        "config": {
            "hidden_dim": hidden_dim,
            "dict_size": args.dict_size,
            "top_k": args.top_k,
            "layer": args.layer,
        },
    }, args.output)
    print(f"[SAE] Saved to {args.output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--captures", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--dict-size", type=int, default=16384)
    p.add_argument("--top-k", type=int, default=64)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
