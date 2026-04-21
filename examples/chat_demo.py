"""TrueMan Agent 交互式对话示例。

展示Agent的情绪驱动行为：
- 惊奇驱动深究
- 无聊驱动探索
- 焦虑驱动内省
- 交互即训练（与外界交互的同时训练自身）

使用方法:
    python examples/chat_demo.py --model Qwen/Qwen2.5-7B-Instruct
    python examples/chat_demo.py --model meta-llama/Llama-3.1-8B-Instruct
    python examples/chat_demo.py --model mistralai/Mistral-7B-Instruct-v0.3
"""

import argparse
import sys

from trueman import create_agent
from trueman.core.config import AgentConfig


def main():
    parser = argparse.ArgumentParser(description="TrueMan Agent 对话示例")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace模型名称（支持Qwen/Llama/Mistral/DeepSeek等）"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="YAML配置文件路径"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="推理设备 (cuda/cpu)"
    )
    parser.add_argument(
        "--4bit", action="store_true", dest="load_4bit",
        help="使用4bit量化加载"
    )
    args = parser.parse_args()

    # 创建Agent
    print(f"正在加载模型: {args.model}")
    print("这可能需要几分钟时间...")

    if args.config:
        agent = create_agent(args.config, base_model_name=args.model, device=args.device)
    else:
        config = AgentConfig(
            base_model_name=args.model,
            device=args.device,
            load_in_4bit=args.load_4bit,
            awake_threshold=500,  # 每500步触发一次睡眠整合
        )
        from trueman.core.agent import TrueManAgent
        agent = TrueManAgent(config)

    print("\n" + "=" * 60)
    print("TrueMan Agent 已就绪！")
    print("输入消息与Agent对话，输入 'quit' 退出")
    print("输入 'emotion' 查看当前情绪状态")
    print("输入 'sleep' 强制触发睡眠整合")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("再见！")
            break

        if user_input.lower() == "emotion":
            emotion = agent.get_emotion_state()
            print(f"\n当前情绪状态:")
            print(f"  惊奇 (Surprise): {emotion['surprise']:.4f}")
            print(f"  无聊 (Boredom):  {emotion['boredom']:.4f}")
            print(f"  焦虑 (Anxiety):  {emotion['anxiety']:.4f}")
            print(f"  驱动 (Drive):    {emotion['drive']:.4f}")
            print(f"  总步数: {agent.total_steps}")
            print(f"  清醒步数: {agent.awake_steps}")
            print(f"  记忆大小: {agent.episodic_memory.size}")
            print(f"  LoRA专家数: {agent.lora_pool.num_experts}")
            print()
            continue

        if user_input.lower() == "sleep":
            print("\n触发睡眠整合...")
            expert_id = agent.force_sleep()
            if expert_id is not None:
                print(f"睡眠整合完成，新增LoRA专家: {expert_id}")
            else:
                print("睡眠整合完成（无新专家）")
            print()
            continue

        # 正常对话
        response, emotion = agent.step(user_input)

        # 显示响应和情绪状态
        print(f"\nAgent: {response}")
        print(f"  [惊奇={emotion.surprise:.2f} | 无聊={emotion.boredom:.2f} | 焦虑={emotion.anxiety:.2f} | 驱动={emotion.drive:.2f}]")
        print()


if __name__ == "__main__":
    main()
