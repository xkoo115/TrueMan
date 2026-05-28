"""支柱 2 主运行脚本：30 天 × 5 条件 × N seed × M 底模。

单次启动一个 (condition, seed, base_model) 组合。集群部署时由
run_v2.py 调度，每个组合一个独立进程/job。

用法：
    python -m experiments.v2_ambitious.pillar2_longhorizon.run \
        --condition C0_trueman_full \
        --base-model Qwen/Qwen2.5-7B-Instruct \
        --seed 0 \
        --stream experiments/v2_ambitious/data/stimulus_stream.jsonl \
        --days 30 \
        --hours-per-day 24 \
        --output experiments/v2_ambitious/results/longhorizon \
        --capture-layers 8 16 24 30 \
        --probe-every-days 7
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from trueman.core.config import AgentConfig
from experiments.v2_ambitious.harness.conditions import make_condition
from experiments.v2_ambitious.harness.capture import (
    HiddenStateCapturer, CaptureSpec, CaptureRecord,
)
from experiments.v2_ambitious.harness.snapshots import take_snapshot, SnapshotMeta
from experiments.v2_ambitious.pillar2_longhorizon.probe_battery import administer


def load_stream(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


TRAJECTORY_FIELDS = [
    "step", "day", "hour", "kind", "prompt_len", "response_len",
    "surprise", "boredom", "anxiety", "drive",
]


def write_trajectory_csv(out_path: str | "Path", rows: list[dict]) -> int:
    """Flatten per-step records (which carry an ``emotions`` sub-dict) into a
    CSV that the H5 free-energy fitter can consume.

    Returns the number of rows written. Extracted from ``main`` so the v2
    regression suite can call it directly without booting a GPU pipeline; v2's
    Bug 1 was a stray ``"emotions"`` key in the merged dict tripping
    ``csv.DictWriter`` default ``extrasaction='raise'``.
    """
    import csv
    n = 0
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRAJECTORY_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            row = {k: r.get(k) for k in TRAJECTORY_FIELDS}
            emo = r.get("emotions") or {}
            for k in ("surprise", "boredom", "anxiety", "drive"):
                if row.get(k) is None and k in emo:
                    row[k] = emo[k]
            w.writerow(row)
            n += 1
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stream", required=True)
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--hours-per-day", type=int, default=24)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="trueman/configs/llm_agent.yaml")
    p.add_argument("--capture-layers", type=int, nargs="*", default=[8, 16, 24, 30])
    p.add_argument("--probe-every-days", type=int, default=7)
    p.add_argument("--quantization", choices=["4bit", "8bit", "none"], default="4bit")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    out_root = Path(args.output) / f"{args.condition}_seed{args.seed}_{args.base_model.replace('/', '_')}"
    out_root.mkdir(parents=True, exist_ok=True)

    # ----- 配置 -----
    print(f"[Run] Setup: condition={args.condition} model={args.base_model} "
          f"seed={args.seed} device={args.device} quantization={args.quantization}",
          flush=True)
    cfg = AgentConfig()
    cfg.base_model_name = args.base_model
    cfg.device = args.device
    if args.quantization == "4bit":
        cfg.load_in_4bit = True
    elif args.quantization == "8bit":
        cfg.load_in_8bit = True

    # ----- 创建 agent -----
    # 把 setup 阶段每一步都 flush, 这样早期崩溃 (HF 下载失败 / 显存 OOM / config 错误)
    # 能立刻在 subprocess 日志里看到栈; 没有这一步时 stage-1 失败只能看到 "returned 1".
    print(f"[Run] Step 1/4: building condition agent {args.condition}", flush=True)
    t_setup = time.time()
    try:
        cond_agent = make_condition(args.condition, cfg, seed=args.seed)
    except Exception as e:
        print(f"[Run] FATAL: make_condition failed: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise
    print(f"[Run] Step 1/4 done in {time.time() - t_setup:.1f}s", flush=True)

    # ----- 捕获 hidden states（仅 C0 / C3 需要，节约存储）-----
    print(f"[Run] Step 2/4: setting up hidden-state capturer (skipped unless C0/C3)", flush=True)
    capturer = None
    if args.condition in ("C0_trueman_full", "C3_frozen"):
        capturer = HiddenStateCapturer(CaptureSpec(
            layers=args.capture_layers,
            token_pool="last",
            quantize="int8",
            output_path=str(out_root / "captures.h5"),
            flush_every=500,
        ))
        capturer.attach(cond_agent.llm.model)
    print(f"[Run] Step 2/4 done", flush=True)

    # ----- 加载刺激流 -----
    print(f"[Run] Step 3/4: loading stimulus stream from {args.stream}", flush=True)
    stream = load_stream(args.stream)
    expected_n = args.days * args.hours_per_day
    if len(stream) < expected_n:
        raise RuntimeError(f"stream too short: {len(stream)} < {expected_n}")
    print(f"[Run] Step 3/4 done: {len(stream)} items loaded, will use first {expected_n}", flush=True)
    print(f"[Run] Step 4/4: entering main loop", flush=True)

    # ----- 主循环 -----
    trajectory_log = []
    last_snapshot_day = -1
    reached_day = 0
    for item in stream[:expected_n]:
        step = item["step"]
        day = item["day"]
        hour = item["hour"]
        reached_day = max(reached_day, day)

        # （可选）每日切换前快照
        if day != last_snapshot_day:
            if last_snapshot_day >= 0:
                try:
                    take_snapshot(
                        cond_agent.agent,
                        str(out_root / "snapshots"),
                        SnapshotMeta(
                            day=last_snapshot_day,
                            condition=args.condition,
                            base_model=args.base_model,
                            seed=args.seed,
                            cumulative_steps=step,
                            cumulative_lora_experts=(
                                len(cond_agent.agent.lora_pool.experts)
                                if cond_agent.agent.lora_pool else 0),
                            cumulative_sleep_count=getattr(cond_agent.agent, "_sleep_count", 0),
                            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                        ),
                    )
                except Exception as e:
                    print(f"[Run] WARN: snapshot for day {last_snapshot_day} failed: "
                          f"{type(e).__name__}: {e}")
            last_snapshot_day = day
            print(f"[Run] day {day} starting (step {step})")

            # 周度 probe
            if day % args.probe_every_days == 0 and day > 0:
                week = day // args.probe_every_days
                administer(cond_agent, week, args.condition, args.seed,
                           str(out_root / "probes"))

        # 单步交互
        record = CaptureRecord(
            step=step, condition=args.condition,
            base_model=args.base_model,
        )
        prompt = item["prompt"]

        if capturer is not None:
            with capturer.recording(step, record):
                response, emo = cond_agent.step(prompt)
        else:
            response, emo = cond_agent.step(prompt)

        record.surprise = emo.surprise
        record.boredom = emo.boredom
        record.anxiety = emo.anxiety
        record.drive = emo.drive

        trajectory_log.append({
            "step": step, "day": day, "hour": hour,
            "kind": item["kind"], "prompt_len": len(prompt),
            "response_len": len(response),
            "emotions": emo.to_dict(),
        })

    # ----- 收尾 -----
    if capturer is not None:
        capturer.close()

    # 最终快照: 使用实际到达的 day, 而不是 args.days-1
    # (args.days * args.hours_per_day 个 item 可能并不覆盖完整 args.days 天,
    # 因为 stream 是按 24h/天生成的)
    try:
        take_snapshot(
            cond_agent.agent, str(out_root / "snapshots"),
            SnapshotMeta(
                day=reached_day, condition=args.condition, base_model=args.base_model,
                seed=args.seed, cumulative_steps=len(stream[:expected_n]),
                cumulative_lora_experts=(len(cond_agent.agent.lora_pool.experts)
                                          if cond_agent.agent.lora_pool else 0),
                cumulative_sleep_count=getattr(cond_agent.agent, "_sleep_count", 0),
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
    except Exception as e:
        print(f"[Run] WARN: final snapshot for day {reached_day} failed: "
              f"{type(e).__name__}: {e}")

    # H4: Day 30 ablate-memory probe
    print("[Run] H4 test: ablating episodic memory and re-running forgetting probe")
    administer(cond_agent, week=999, condition_code=args.condition, seed=args.seed,
               output_dir=str(out_root / "probes"),
               ablate_episodic_memory=True)

    # H5: 保存 trajectory.csv（供 free energy 拟合）
    n_written = write_trajectory_csv(out_root / "trajectory.csv", trajectory_log)
    print(f"[Run] trajectory.csv: wrote {n_written} rows", flush=True)

    print(f"[Run] Done. Output -> {out_root}")


if __name__ == "__main__":
    main()
