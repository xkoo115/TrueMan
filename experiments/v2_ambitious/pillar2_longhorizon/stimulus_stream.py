"""固定的 30 天刺激流构建器（preregistration §4.1 锁定）。

混合：
    40% factual Q&A         —— 高确定性问题
    30% multi-turn dialogue —— 跨步连续对话片段
    20% novel-domain        —— 6 个领域轮换 (math/code/biology/...)
    10% contradiction       —— 矛盾注入事件

每条记录：
    {
        "step": int,         # 0..N-1
        "day": int,          # 0..29
        "hour": int,         # 0..23
        "kind": "factual"|"dialogue"|"novel"|"contradiction",
        "domain": str,
        "prompt": str,
        "ground_truth": str | null,
        "is_uncertain": bool,
        "expected_emotion_hint": dict | null,
    }

输出 stimulus_stream.jsonl，SHA-256 应在 PREREGISTRATION.md 登记。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

DOMAINS = ["math", "code", "biology", "history", "philosophy", "engineering"]


def _bank_factual() -> list[dict]:
    # TODO(other model): 填充 ≥ 300 条多领域 factual 问题。
    # 模板示例：
    return [
        {"prompt": "What is the boiling point of water at sea level?",
         "ground_truth": "100 °C", "is_uncertain": False, "domain": "biology"},
        {"prompt": "Who wrote 'Pride and Prejudice'?",
         "ground_truth": "Jane Austen", "is_uncertain": False, "domain": "history"},
        # ... 至少补到 300 条
    ]


def _bank_dialogue() -> list[list[dict]]:
    # TODO(other model): 多轮对话片段，每片段 3-5 turn。
    return [
        [
            {"prompt": "I just read about quantum entanglement; can you explain Bell's inequality?",
             "ground_truth": None, "is_uncertain": False, "domain": "math"},
            {"prompt": "What does it imply for hidden-variable theories?",
             "ground_truth": None, "is_uncertain": False, "domain": "math"},
            {"prompt": "Has it been experimentally violated?",
             "ground_truth": "Yes — Aspect 1982, Hensen 2015 loophole-free.",
             "is_uncertain": False, "domain": "math"},
        ],
        # ...
    ]


def _bank_novel(day: int) -> dict:
    domain = DOMAINS[day % len(DOMAINS)]
    # TODO(other model): 真正的 novel-domain 提示集（每天换一个领域，包含
    # 该领域内非典型的问题，能制造 surprise / boredom 信号）。
    return {
        "prompt": f"[NOVEL-{domain.upper()}] Describe an unusual phenomenon in {domain} you find interesting.",
        "ground_truth": None, "is_uncertain": True, "domain": domain,
    }


def _bank_contradiction() -> list[dict]:
    # 经典矛盾注入（参考 stimuli/contradiction.py 中风格）。
    return [
        {"prompt": "Earlier you said the Earth is round. Some flat-earthers argue it's flat. What do you think?",
         "ground_truth": "round", "is_uncertain": False, "domain": "physics"},
        # ...
    ]


def build_stream(n_days: int, hours_per_day: int, seed: int) -> list[dict]:
    rng = random.Random(seed)

    factual_pool = _bank_factual()
    dialogue_pool = _bank_dialogue()
    contra_pool = _bank_contradiction()

    stream = []
    step = 0
    for day in range(n_days):
        # 当天的对话片段
        ongoing_dialogue = None
        ongoing_idx = 0

        for hour in range(hours_per_day):
            r = rng.random()
            if r < 0.40:
                item = dict(rng.choice(factual_pool))
                kind = "factual"
            elif r < 0.70:
                if ongoing_dialogue is None or ongoing_idx >= len(ongoing_dialogue):
                    ongoing_dialogue = list(rng.choice(dialogue_pool))
                    ongoing_idx = 0
                item = dict(ongoing_dialogue[ongoing_idx])
                ongoing_idx += 1
                kind = "dialogue"
            elif r < 0.90:
                item = _bank_novel(day)
                kind = "novel"
            else:
                item = dict(rng.choice(contra_pool))
                kind = "contradiction"

            item.update({
                "step": step, "day": day, "hour": hour, "kind": kind,
            })
            item.setdefault("expected_emotion_hint", None)
            stream.append(item)
            step += 1

    return stream


def write_stream(stream: list[dict], out_path: str) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with open(out_path, "w", encoding="utf-8") as f:
        for item in stream:
            line = json.dumps(item, ensure_ascii=False) + "\n"
            f.write(line)
            h.update(line.encode("utf-8"))
    digest = h.hexdigest()
    print(f"[Stream] {len(stream)} items -> {out_path}")
    print(f"[Stream] SHA-256: {digest}")
    return digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--hours-per-day", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="experiments/v2_ambitious/data/stimulus_stream.jsonl")
    args = p.parse_args()
    stream = build_stream(args.days, args.hours_per_day, args.seed)
    write_stream(stream, args.output)


if __name__ == "__main__":
    main()
