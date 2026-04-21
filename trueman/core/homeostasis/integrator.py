"""情绪信号整合器：将惊奇/无聊/焦虑合成为统一内驱力。"""

from __future__ import annotations

from dataclasses import dataclass

from trueman.core.config import HomeostasisConfig


@dataclass
class EmotionState:
    """情绪状态快照。"""
    surprise: float = 0.0
    boredom: float = 0.0
    anxiety: float = 0.0
    drive: float = 0.0

    @property
    def max_intensity(self) -> float:
        """情绪强度最大值，用于优先级排序。"""
        return max(self.surprise, self.boredom, self.anxiety)

    def to_dict(self) -> dict:
        return {
            "surprise": self.surprise,
            "boredom": self.boredom,
            "anxiety": self.anxiety,
            "drive": self.drive,
        }


class EmotionIntegrator:
    """情绪信号整合器。

    计算每种信号偏离内稳态设定点的绝对值，加权求和得到总驱动信号：
    total_drive = α * |surprise - setpoint_s| + β * |boredom - setpoint_b| + γ * |anxiety - setpoint_a|
    """

    def __init__(self, config: HomeostasisConfig):
        self.alpha = config.alpha
        self.beta = config.beta
        self.gamma = config.gamma
        self.setpoint_surprise = config.setpoint_surprise
        self.setpoint_boredom = config.setpoint_boredom
        self.setpoint_anxiety = config.setpoint_anxiety

    def integrate(
        self,
        surprise: float,
        boredom: float,
        anxiety: float,
    ) -> tuple[float, EmotionState]:
        """整合三种情绪信号。

        Args:
            surprise: 惊奇信号值 [0,1]
            boredom: 无聊信号值 [0,1]
            anxiety: 焦虑信号值 [0,1]

        Returns:
            (total_drive, emotion_state)
        """
        # 约束到[0,1]
        surprise = max(0.0, min(1.0, surprise))
        boredom = max(0.0, min(1.0, boredom))
        anxiety = max(0.0, min(1.0, anxiety))

        # 计算偏离度
        dev_s = abs(surprise - self.setpoint_surprise)
        dev_b = abs(boredom - self.setpoint_boredom)
        dev_a = abs(anxiety - self.setpoint_anxiety)

        total_drive = self.alpha * dev_s + self.beta * dev_b + self.gamma * dev_a

        state = EmotionState(
            surprise=surprise,
            boredom=boredom,
            anxiety=anxiety,
            drive=total_drive,
        )

        return total_drive, state
