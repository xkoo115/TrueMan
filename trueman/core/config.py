"""Agent配置数据模型，支持YAML加载和热更新。"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any


@dataclass
class HomeostasisConfig:
    """内稳态参数配置。"""
    setpoint_surprise: float = 0.3
    setpoint_boredom: float = 0.3
    setpoint_anxiety: float = 0.2
    alpha: float = 1.0       # 惊奇权重
    beta: float = 1.0        # 无聊权重
    gamma: float = 1.5       # 焦虑权重（更高因为涉及自我纠错）


@dataclass
class EmotionThresholdConfig:
    """情绪触发阈值配置。"""
    surprise_update_threshold: float = 0.7
    boredom_explore_threshold: float = 0.8
    anxiety_introspection_threshold: float = 0.8
    anxiety_emergency_threshold: float = 0.9


@dataclass
class SurpriseConfig:
    """惊奇信号参数。"""
    decay: float = 0.99
    threshold: float = 2.0


@dataclass
class BoredomConfig:
    """无聊信号参数。"""
    window: int = 100
    temperature: float = 1.0


@dataclass
class AnxietyConfig:
    """焦虑信号参数。"""
    n_samples: int = 3
    lightweight: bool = True
    temperature_scale: float = 1.5
    # 焦虑也走运行统计归一化（同 SurpriseSignal），把"绝对分歧度"变成"相对分歧度"。
    # 否则 1-文本相似度 在高温采样下恒接近 1，信号焊死、无方差，既无法驱动行为也
    # 无法被 SAE 特征相关分析检出。decay 是 EMA 衰减系数；threshold 调灵敏度。
    decay: float = 0.9
    threshold: float = 0.0


@dataclass
class LoRAConfig:
    """LoRA可塑性系统配置。"""
    rank: int = 16
    max_experts: int = 50
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    orthogonality_weight: float = 0.1


@dataclass
class AgentConfig:
    """TrueMan Agent完整配置。"""
    base_model_name: str = "Qwen/Qwen3-8B"
    device: str = "cuda"
    memory_size: int = 10000
    # 1000→30：sleep 整合（= 训练 LoRA 专家 = 参数可塑性）按"清醒步数 >= awake_threshold
    # 或 anxiety > emergency"触发。原来 1000 在百步级 pilot run 里永不命中，而 anxiety
    # 归一化后也很少 >0.9 → 可塑性根本不会跑。降到 30 让 sleep 每约 30 步触发一次，C0
    # 才真正"可塑"。每次 consolidate 只需内存里有 ≥10 条轨迹即可训练。
    awake_threshold: int = 30
    max_inference_time: float = 30.0  # 秒

    # 量化加载
    load_in_4bit: bool = False
    load_in_8bit: bool = False

    # API模式配置（用于云端模型如DeepSeek）
    api_key: str = ""
    api_base_url: str = ""
    api_model_name: str = ""  # API中的模型名称，如"deepseek-chat"
    api_embedding_dim: int = 1024  # API模式的嵌入维度

    # 子配置
    homeostasis: HomeostasisConfig = field(default_factory=HomeostasisConfig)
    thresholds: EmotionThresholdConfig = field(default_factory=EmotionThresholdConfig)
    surprise: SurpriseConfig = field(default_factory=SurpriseConfig)
    boredom: BoredomConfig = field(default_factory=BoredomConfig)
    anxiety: AnxietyConfig = field(default_factory=AnxietyConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AgentConfig:
        """从YAML文件加载配置。"""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> AgentConfig:
        """从嵌套字典构建配置，支持子配置。"""
        sub_configs = {
            "homeostasis": HomeostasisConfig,
            "thresholds": EmotionThresholdConfig,
            "surprise": SurpriseConfig,
            "boredom": BoredomConfig,
            "anxiety": AnxietyConfig,
            "lora": LoRAConfig,
        }
        kwargs = {}
        for f in fields(cls):
            if f.name in sub_configs and f.name in data:
                sub_data = data[f.name]
                kwargs[f.name] = sub_configs[f.name](**sub_data)
            elif f.name in data:
                kwargs[f.name] = data[f.name]
        return cls(**kwargs)

    def to_yaml(self, path: str | Path) -> None:
        """保存配置到YAML文件。"""
        data = asdict(self)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    def update(self, **overrides) -> None:
        """热更新配置项，支持点号路径如 'homeostasis.alpha'。"""
        for key, value in overrides.items():
            if "." in key:
                parts = key.split(".", 1)
                sub = getattr(self, parts[0])
                setattr(sub, parts[1], value)
            else:
                setattr(self, key, value)
