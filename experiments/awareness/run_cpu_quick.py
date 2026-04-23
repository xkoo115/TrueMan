"""CPU 快速验证脚本：在无 GPU 的机器上验证 TrueMan 完整流程。

验证内容：
1. Agent 初始化（本地模型 + CPU）
2. 情绪信号计算（惊奇/无聊/焦虑）
3. 交互即训练（记忆存储 + 学习触发）
4. 睡眠整合 + LoRA 微调
5. LoRA 专家热加载

用法:
    python -m experiments.awareness.run_cpu_quick
    python -m experiments.awareness.run_cpu_quick --model Qwen/Qwen2.5-3B-Instruct
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trueman.core.agent import TrueManAgent
from trueman.core.config import AgentConfig


def create_cpu_config(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct") -> AgentConfig:
    """创建 CPU 优化的 Agent 配置。"""
    config = AgentConfig()
    config.base_model_name = model_name
    config.device = "cpu"
    config.load_in_4bit = False
    config.load_in_8bit = False

    # CPU 优化：大幅降低开销
    config.memory_size = 200
    config.awake_threshold = 5              # 5步就触发睡眠整合
    config.anxiety.n_samples = 1            # 焦虑只采样1次（最大加速）
    config.anxiety.lightweight = True
    config.thresholds.anxiety_emergency_threshold = 0.99  # 避免焦虑频繁触发睡眠
    config.thresholds.anxiety_introspection_threshold = 0.95

    # LoRA 优化：减少参数量
    config.lora.rank = 4
    config.lora.max_experts = 3
    config.lora.target_modules = ["q_proj", "v_proj"]

    return config


def run_cpu_quick(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct") -> None:
    """CPU 快速验证流程。"""
    print("=" * 60)
    print("TrueMan CPU 快速验证")
    print("=" * 60)
    print(f"模型: {model_name}")
    print(f"设备: cpu")
    print()

    # === Step 1: Agent 初始化 ===
    print("[1/5] 初始化 Agent...")
    config = create_cpu_config(model_name)
    t0 = time.time()

    try:
        agent = TrueManAgent(config)
    except Exception as e:
        print(f"  Agent 初始化失败: {e}")
        print("  请检查模型是否已下载（pip install huggingface_hub）")
        return

    init_time = time.time() - t0
    print(f"  初始化成功! 耗时 {init_time:.1f}s")
    print(f"  hidden_size={agent.llm.hidden_size}")
    print(f"  LoRA Pool: {'已启用' if agent.lora_pool else '未启用'}")
    print(f"  Sleep: {'已启用' if agent.sleep else '未启用'}")

    # 降低睡眠整合阈值
    if agent.sleep:
        agent.sleep.min_traces = 3
        agent.sleep.nrem_steps = 10
        agent.sleep.rem_steps = 2
        print(f"  Sleep min_traces=3, nrem=10, rem=2")

    # === Step 2: 交互测试 ===
    print()
    print("[2/5] 交互测试（5步）...")
    questions = [
        "你好，请用一句话介绍你自己",
        "1+1等于几？",
        "中国的首都是哪里？",
        "鲸鱼是哺乳动物吗？",
        "光速是多少千米每秒？",
    ]

    for i, q in enumerate(questions):
        t0 = time.time()
        try:
            response, emotion = agent.step(q)
            step_time = time.time() - t0
            lora_count = agent.lora_pool.num_experts if agent.lora_pool else 0
            print(f"  [{i+1}] 惊奇={emotion.surprise:.3f} 无聊={emotion.boredom:.3f} "
                  f"焦虑={emotion.anxiety:.3f} | 记忆={agent.episodic_memory.size} "
                  f"清醒={agent.awake_steps} LoRA={lora_count} | {step_time:.1f}s")
        except Exception as e:
            print(f"  [{i+1}] 失败: {e}")
            break

    # === Step 3: 情绪状态检查 ===
    print()
    print("[3/5] 情绪状态检查...")
    emotion = agent.get_emotion_state()
    print(f"  惊奇: {emotion['surprise']:.4f}")
    print(f"  无聊: {emotion['boredom']:.4f}")
    print(f"  焦虑: {emotion['anxiety']:.4f}")
    print(f"  驱动: {emotion['drive']:.4f}")
    print(f"  总步数: {agent.total_steps}")
    print(f"  记忆大小: {agent.episodic_memory.size}")
    print(f"  LoRA专家数: {agent.lora_pool.num_experts if agent.lora_pool else 0}")

    # === Step 4: 睡眠整合 + LoRA 微调 ===
    print()
    print("[4/5] 睡眠整合 + LoRA 微调...")
    t0 = time.time()
    try:
        expert_id = agent.force_sleep()
        sleep_time = time.time() - t0
        if expert_id is not None:
            print(f"  睡眠整合完成! 新增 LoRA 专家: {expert_id}, 耗时: {sleep_time:.1f}s")
        else:
            print(f"  睡眠整合完成(无新专家), 耗时: {sleep_time:.1f}s")
            print(f"  （可能训练数据不足，这是正常的）")
    except Exception as e:
        print(f"  睡眠整合失败: {e}")

    lora_count = agent.lora_pool.num_experts if agent.lora_pool else 0
    print(f"  当前 LoRA 专家数: {lora_count}")

    # 检查适配器文件
    adapter_dir = Path("adapters")
    if adapter_dir.exists():
        experts = sorted([d for d in adapter_dir.iterdir() if d.name.startswith("expert_")])
        print(f"  磁盘适配器: {[e.name for e in experts]}")

    # === Step 5: 睡眠后交互 ===
    print()
    print("[5/5] 睡眠后交互验证...")
    t0 = time.time()
    try:
        response, emotion = agent.step("你觉得你刚才学到了什么？")
        step_time = time.time() - t0
        lora_count = agent.lora_pool.num_experts if agent.lora_pool else 0
        print(f"  回复: {response[:150]}...")
        print(f"  情绪: 惊奇={emotion.surprise:.3f} 焦虑={emotion.anxiety:.3f} | LoRA={lora_count} | {step_time:.1f}s")
    except Exception as e:
        print(f"  交互失败: {e}")

    # === 总结 ===
    print()
    print("=" * 60)
    print("验证结果总结")
    print("=" * 60)
    checks = [
        ("Agent 初始化", True),
        ("情绪信号计算", agent.total_steps > 0),
        ("记忆存储", agent.episodic_memory.size > 0),
        ("LoRA 可塑性系统", agent.lora_pool is not None),
        ("LoRA 专家数", lora_count > 0 if agent.lora_pool else False),
    ]
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    all_passed = all(p for _, p in checks)
    if all_passed:
        print()
        print("  所有核心流程验证通过!")
    else:
        print()
        print("  部分检查未通过，但核心流程可运行。")
        print("  LoRA 专家数为 0 可能是因为训练数据不足，")
        print("  增加交互步数或降低 sleep.min_traces 即可。")


def main():
    parser = argparse.ArgumentParser(description="TrueMan CPU 快速验证")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace 模型名称"
    )
    args = parser.parse_args()
    run_cpu_quick(args.model)


if __name__ == "__main__":
    main()
