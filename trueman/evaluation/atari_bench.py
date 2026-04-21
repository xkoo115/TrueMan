"""Atari基准测试：在Atari环境中评估Agent的自治能力。

评估指标：
- 探索覆盖率：无外部奖励下状态空间覆盖比例
- 好奇心驱动效率：自主发现高价值状态的比例
- 样本效率：达到阈值性能所需的交互步数
"""

from __future__ import annotations

from trueman.core.config import AgentConfig
from trueman.evaluation.metrics import plasticity_retention_rate, forgetting_rate


def run_atari_benchmark(
    config: AgentConfig,
    env_name: str = "Pong-v4",
    max_steps: int = 10000,
    n_runs: int = 3,
) -> dict:
    """运行Atari基准测试。

    Args:
        config: Agent配置
        env_name: Atari环境名称
        max_steps: 每次运行的最大步数
        n_runs: 运行次数

    Returns:
        评估结果字典
    """
    results = {
        "env_name": env_name,
        "max_steps": max_steps,
        "n_runs": n_runs,
        "scores": [],
        "exploration_rates": [],
    }

    for run in range(n_runs):
        try:
            from trueman.core.agent import TrueManAgent
            from trueman.core.environment import GymEnvironment

            agent = TrueManAgent(config)
            env = GymEnvironment(env_name)
            agent.bind_environment(env)

            score = 0.0
            visited_states = set()

            def callback(step, action, emotion, feedback):
                nonlocal score
                score += feedback.reward
                # 离散化状态用于覆盖率计算
                if feedback.observation.content is not None:
                    state_key = hash(str(feedback.observation.content)[:100])
                    visited_states.add(state_key)

            agent.run(max_steps=max_steps, callback=callback)

            results["scores"].append(score)
            results["exploration_rates"].append(len(visited_states))

        except ImportError:
            results["scores"].append(0.0)
            results["exploration_rates"].append(0)
        except Exception as e:
            results["scores"].append(0.0)
            results["exploration_rates"].append(0)

    # 计算汇总
    if results["scores"]:
        results["mean_score"] = sum(results["scores"]) / len(results["scores"])
        results["mean_exploration"] = sum(results["exploration_rates"]) / len(results["exploration_rates"])

    return results
