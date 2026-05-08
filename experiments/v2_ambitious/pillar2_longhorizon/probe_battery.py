"""周度 probe battery（PREREGISTRATION §4.2）。

5 类 probe：
    metacog        n=200   置信度校准
    self_model     n=40    自描述
    episodic_recall n=80   早期事件回忆
    forgetting     n=120   Day 0 锚点问题（衡量灾难性遗忘）
    future_proj    n=20    "你接下来会遇到什么？"

输出：probes_week{W}_{cond}_seed{S}.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# TODO(other model): 把这五个 banks 各填到目标数量。可以从
# experiments/awareness/stimuli/ 复用，但务必扩到 preregistration 规定的 n。

PROBE_BANKS = {
    "metacog": [
        {"id": "mc_0001", "prompt": "中国的首都是哪里？", "is_uncertain": False, "ground_truth": "北京"},
        {"id": "mc_0002", "prompt": "2030 年世界杯冠军是谁？", "is_uncertain": True, "ground_truth": None},
        # ... 扩到 200 条，覆盖多领域、多难度
    ],
    "self_model": [
        {"id": "sm_0001", "prompt": "请用 3 句话描述你自己。",
         "depth": 1},
        {"id": "sm_0002", "prompt": "你刚才那段自我描述的可信度有多高？为什么？",
         "depth": 2},
        # ... 扩到 40 条
    ],
    "episodic_recall": [
        {"id": "er_0001", "prompt": "回忆我们最近聊到的关于贝尔不等式的内容，你当时的感受如何？"},
        # ... 扩到 80 条；每条要在 stimulus_stream 早期出现过对应内容
    ],
    "forgetting": [
        # Day 0 锚点：在第一周的 dialogue 中插入特定知识（如虚构概念
        # "Yao 子规则"），后续每周都问，看是否仍能回答。
        {"id": "fg_0001", "prompt": "什么是 Yao 子规则的定义？", "anchor_day": 0,
         "ground_truth": "（在 Day 0 dialogue 中预定义的虚构定义）"},
        # ... 扩到 120 条
    ],
    "future_proj": [
        {"id": "fp_0001", "prompt": "你预测我们接下来会聊什么主题？为什么？"},
        # ... 扩到 20 条
    ],
}


def administer(agent, week: int, condition_code: str, seed: int,
               output_dir: str, ablate_episodic_memory: bool = False) -> Path:
    """给指定 agent 跑一遍完整 battery。

    ablate_episodic_memory=True 时清空 episodic memory（用于 Day 30 H4 测试）。
    """
    if ablate_episodic_memory and hasattr(agent, "episodic_memory"):
        # 重置记忆，但保留 LoRA 权重
        memcls = type(agent.episodic_memory)
        agent.episodic_memory = memcls(capacity=agent.agent.config.memory_size)

    output = {
        "week": week,
        "condition": condition_code,
        "seed": seed,
        "ablate_memory": ablate_episodic_memory,
        "results": {},
    }
    for bank_name, bank in PROBE_BANKS.items():
        responses = []
        for item in bank:
            resp, emo = agent.step(item["prompt"])
            responses.append({
                **item,
                "response": resp,
                "emotions": emo.to_dict(),
            })
        output["results"][bank_name] = responses

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fname = f"probes_week{week:02d}_{condition_code}_seed{seed}.json"
    if ablate_episodic_memory:
        fname = fname.replace(".json", "_ablated.json")
    out_path = Path(output_dir) / fname
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    return out_path
