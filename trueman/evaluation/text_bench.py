"""文本推理基准测试：评估Agent在文本推理任务上的情绪驱动学习能力。

评估指标：
- 惊奇校准度：惊奇信号与实际异常的相关系数
- 无聊检测准确率：区分重复/新颖输入的准确率
- 焦虑预测值：焦虑信号预测后续错误的准确率
- 自我纠错率：焦虑触发后逻辑一致性恢复比例
"""

from __future__ import annotations

from trueman.core.config import AgentConfig
from trueman.evaluation.metrics import (
    surprise_calibration,
    boredom_detection_accuracy,
    anxiety_predictive_value,
)


def run_text_benchmark(
    config: AgentConfig,
    test_prompts: list[str] | None = None,
    max_steps: int = 100,
) -> dict:
    """运行文本推理基准测试。

    Args:
        config: Agent配置
        test_prompts: 测试prompt列表
        max_steps: 最大测试步数

    Returns:
        评估结果字典
    """
    if test_prompts is None:
        test_prompts = [
            "What is 2+2?",
            "Explain quantum entanglement briefly.",
            "Translate 'hello world' to French.",
            "What is the capital of Japan?",
            "Solve: x^2 - 4 = 0",
        ]

    results = {
        "n_prompts": len(test_prompts),
        "surprise_values": [],
        "boredom_values": [],
        "anxiety_values": [],
        "responses": [],
    }

    try:
        from trueman.core.agent import TrueManAgent
        agent = TrueManAgent(config)

        for prompt in test_prompts[:max_steps]:
            action, emotion = agent.step(prompt)
            results["surprise_values"].append(emotion.surprise)
            results["boredom_values"].append(emotion.boredom)
            results["anxiety_values"].append(emotion.anxiety)
            results["responses"].append(action)

    except Exception as e:
        results["error"] = str(e)

    # 计算汇总统计
    if results["surprise_values"]:
        results["mean_surprise"] = sum(results["surprise_values"]) / len(results["surprise_values"])
        results["mean_boredom"] = sum(results["boredom_values"]) / len(results["boredom_values"])
        results["mean_anxiety"] = sum(results["anxiety_values"]) / len(results["anxiety_values"])

    return results
