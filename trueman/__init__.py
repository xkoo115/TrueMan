"""TrueMan: Homeostasis-driven autonomous LLM Agent."""

from trueman.core.config import AgentConfig


def create_agent(config_path: str | None = None, **overrides):
    """从配置文件创建TrueManAgent实例。

    Args:
        config_path: YAML配置文件路径，None则使用默认配置
        **overrides: 覆盖配置项，如 base_model_name="Qwen/Qwen2.5-7B-Instruct"

    Returns:
        初始化完成的TrueManAgent实例
    """
    from trueman.core.agent import TrueManAgent

    config = AgentConfig.from_yaml(config_path) if config_path else AgentConfig()
    for key, value in overrides.items():
        setattr(config, key, value)
    return TrueManAgent(config)
