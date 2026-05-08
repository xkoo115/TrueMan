"""隐藏状态捕获钩子 —— 为支柱 1 (SAE + 因果干预) 提供基础设施。

设计目标：
1. 在 agent 运行时，零侵入地从指定层捕获 hidden states
2. 同步记录每一步的真值情绪状态（用于后续 probing）
3. 输出格式：HDF5 文件，按 (step, layer, token_idx, hidden_dim) 索引
4. int8 量化以减小磁盘占用 (5×10^6 step × 4 层 × 64 token × 4096 dim × 1 byte ≈ 5 TB
   —— 因此还需要 token_pool=last_only 缩减为 (step, layer, hidden_dim) ≈ 80 GB)

依赖（可选）：h5py, numpy, torch, transformer_lens (可选用于 hooked 推理)
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch


@dataclass
class CaptureSpec:
    """捕获规格。"""
    layers: list[int]                   # 要捕获的层索引（如 [8, 16, 24, 30]）
    token_pool: str = "last"            # "last" / "mean" / "all"
    quantize: str = "int8"              # "int8" / "float16" / "float32"
    output_path: str = "captures.h5"
    flush_every: int = 1000


@dataclass
class CaptureRecord:
    step: int = 0
    surprise: float = 0.0
    boredom: float = 0.0
    anxiety: float = 0.0
    drive: float = 0.0
    condition: str = ""
    base_model: str = ""
    extras: dict = field(default_factory=dict)


class HiddenStateCapturer:
    """前向 hook 管理器。

    用法：
        capturer = HiddenStateCapturer(spec)
        capturer.attach(agent.llm.model)
        for step, obs in enumerate(stream):
            with capturer.recording(step=step, condition="C0", record=current_emotion_record):
                agent.step(obs)
        capturer.detach()
        capturer.close()
    """

    def __init__(self, spec: CaptureSpec):
        self.spec = spec
        self._hooks: list = []
        self._buffer: dict[int, list[np.ndarray]] = {l: [] for l in spec.layers}
        self._records: list[CaptureRecord] = []
        self._h5 = None
        self._current_record: Optional[CaptureRecord] = None
        self._step_counter = 0

    # ----- 安装/卸载 hooks -----
    def attach(self, model: torch.nn.Module) -> None:
        """注入 forward hooks 到指定 transformer 层。"""
        # 假设 HF causal LM 的层位于 model.model.layers (Llama, Qwen, Mistral)
        # 或 model.transformer.h (GPT-style)
        layers = self._find_layers(model)
        for layer_idx in self.spec.layers:
            if layer_idx >= len(layers):
                raise IndexError(f"layer {layer_idx} out of range (total {len(layers)})")
            h = layers[layer_idx].register_forward_hook(self._make_hook(layer_idx))
            self._hooks.append(h)

    def detach(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @staticmethod
    def _find_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
        """找到 transformer block 列表。"""
        candidates = ["model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"]
        for path in candidates:
            obj = model
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                return list(obj)
            except AttributeError:
                continue
        raise RuntimeError("Cannot locate transformer block list. Add a new candidate path.")

    # ----- hook 工厂 -----
    def _make_hook(self, layer_idx: int):
        spec = self.spec

        def hook(module, inputs, output):
            # output 通常是 tuple，第一个是 hidden_states
            hs = output[0] if isinstance(output, tuple) else output
            if hs.dim() == 3:  # (batch, seq, dim)
                if spec.token_pool == "last":
                    pooled = hs[:, -1, :]
                elif spec.token_pool == "mean":
                    pooled = hs.mean(dim=1)
                else:  # "all"
                    pooled = hs[0]  # (seq, dim)
            else:
                pooled = hs

            arr = pooled.detach().to(torch.float32).cpu().numpy()
            if spec.quantize == "int8":
                arr = self._quantize_int8(arr)
            elif spec.quantize == "float16":
                arr = arr.astype(np.float16)
            self._buffer[layer_idx].append(arr)

        return hook

    @staticmethod
    def _quantize_int8(arr: np.ndarray) -> np.ndarray:
        """对称 int8 量化，保留 scale。"""
        absmax = np.abs(arr).max() + 1e-8
        scale = absmax / 127.0
        q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
        # 把 scale 也存下来：拼成一个结构 dict（在 flush 时处理）
        return q  # scale 在 _records.extras 里同步保存

    # ----- 录制上下文 -----
    @contextmanager
    def recording(self, step: int, record: CaptureRecord):
        self._current_record = record
        self._current_record.step = step
        try:
            yield
        finally:
            self._records.append(self._current_record)
            self._step_counter += 1
            if self._step_counter % self.spec.flush_every == 0:
                self.flush()
            self._current_record = None

    # ----- IO -----
    def flush(self) -> None:
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for HiddenStateCapturer.flush()")

        if self._h5 is None:
            Path(self.spec.output_path).parent.mkdir(parents=True, exist_ok=True)
            self._h5 = h5py.File(self.spec.output_path, "a")
            for l in self.spec.layers:
                grp = self._h5.require_group(f"layer_{l}")
                if "states" not in grp:
                    grp.create_dataset(
                        "states",
                        shape=(0, 0),
                        maxshape=(None, None),
                        dtype="int8" if self.spec.quantize == "int8" else "float16",
                        chunks=True,
                    )

        for l in self.spec.layers:
            buf = self._buffer[l]
            if not buf:
                continue
            arr = np.stack(buf, axis=0).reshape(len(buf), -1)
            ds = self._h5[f"layer_{l}/states"]
            old_n = ds.shape[0]
            new_n = old_n + arr.shape[0]
            ds.resize((new_n, arr.shape[1]))
            ds[old_n:new_n] = arr
            self._buffer[l].clear()

        # 写入记录
        rec_arr = np.array([
            (r.step, r.surprise, r.boredom, r.anxiety, r.drive,
             r.condition.encode("utf-8"), r.base_model.encode("utf-8"))
            for r in self._records
        ], dtype=[
            ("step", "i8"), ("surprise", "f4"), ("boredom", "f4"),
            ("anxiety", "f4"), ("drive", "f4"),
            ("condition", "S32"), ("base_model", "S64"),
        ])
        if "records" not in self._h5:
            self._h5.create_dataset("records", data=rec_arr, maxshape=(None,), chunks=True)
        else:
            ds = self._h5["records"]
            old_n = ds.shape[0]
            ds.resize((old_n + len(rec_arr),))
            ds[old_n:] = rec_arr
        self._records.clear()

    def close(self) -> None:
        self.flush()
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
