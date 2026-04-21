"""多尺度时间常数内稳态：不同情绪操作于不同时间尺度。

依据 MSTH (Hakim, 2026)：
- ultra_fast (5ms): 即时惊奇响应，token级预测误差
- fast (100ms): 行为策略调整，情绪驱动策略切换
- medium (1min): 探索策略切换，无聊驱动的探索方向改变
- slow (1hr): LoRA整合决策，睡眠整合触发

不同时间尺度的更新率: update_rate = 1 - exp(-1 / timescale)
"""

from __future__ import annotations

import math
import time
from enum import Enum

from trueman.core.homeostasis.integrator import EmotionState


class Timescale(Enum):
    """多尺度时间常数的层级。"""
    ULTRA_FAST = "ultra_fast"   # 5ms - 即时惊奇响应
    FAST = "fast"               # 100ms - 行为策略调整
    MEDIUM = "medium"           # 1min - 探索策略切换
    SLOW = "slow"               # 1hr - LoRA整合决策


# 默认时间常数（秒）
DEFAULT_TIMESCALES = {
    Timescale.ULTRA_FAST: 5e-3,
    Timescale.FAST: 1e-1,
    Timescale.MEDIUM: 60.0,
    Timescale.SLOW: 3600.0,
}

# 情绪信号到时间尺度的映射
EMOTION_TIMESCALE_MAP = {
    "surprise": Timescale.ULTRA_FAST,   # 惊奇：即时响应
    "boredom": Timescale.MEDIUM,         # 无聊：分钟级策略切换
    "anxiety": Timescale.FAST,           # 焦虑：百毫秒级行为调整
    "sleep": Timescale.SLOW,             # 睡眠：小时级整合决策
}


class MultiScaleHomeostasis:
    """多尺度时间常数内稳态。

    不同情绪信号操作于不同时间尺度：
    - 惊奇在ultra_fast尺度响应（5ms），快速检测意外
    - 焦虑在fast尺度响应（100ms），及时调整行为
    - 无聊在medium尺度响应（1min），切换探索方向
    - 睡眠在slow尺度触发（1hr），整合长期知识

    每个尺度维护独立的指数移动平均，更新率由时间常数决定。
    """

    def __init__(
        self,
        timescales: dict[Timescale, float] | None = None,
    ):
        self.timescales = timescales or DEFAULT_TIMESCALES

        # 每个时间尺度的运行平均值
        self._running_values: dict[Timescale, float] = {
            ts: 0.0 for ts in Timescale
        }
        # 每个时间尺度的上次更新时间
        self._last_update: dict[Timescale, float] = {
            ts: time.time() for ts in Timescale
        }

    def get_update_rate(self, timescale: Timescale) -> float:
        """获取指定时间尺度的更新率。

        update_rate = 1 - exp(-1 / timescale)

        时间常数越大，更新率越小（变化越慢）。
        """
        ts = self.timescales[timescale]
        return 1.0 - math.exp(-1.0 / ts)

    def update(self, signal_name: str, value: float) -> float:
        """更新指定情绪信号的运行平均值。

        Args:
            signal_name: 情绪信号名（"surprise"/"boredom"/"anxiety"/"sleep"）
            value: 当前信号值

        Returns:
            时间尺度平滑后的信号值
        """
        timescale = EMOTION_TIMESCALE_MAP.get(signal_name, Timescale.FAST)
        rate = self.get_update_rate(timescale)

        # 指数移动平均
        old = self._running_values[timescale]
        new = old + rate * (value - old)
        self._running_values[timescale] = new
        self._last_update[timescale] = time.time()

        return new

    def should_trigger(self, signal_name: str, current_value: float, threshold: float) -> bool:
        """判断是否应该触发某个情绪驱动的行为。

        考虑时间尺度：慢尺度的信号需要更持续的偏离才能触发。

        Args:
            signal_name: 情绪信号名
            current_value: 当前信号值
            threshold: 触发阈值

        Returns:
            是否应该触发
        """
        timescale = EMOTION_TIMESCALE_MAP.get(signal_name, Timescale.FAST)
        smoothed = self._running_values[timescale]
        return smoothed > threshold

    def get_all_smoothed(self) -> dict[str, float]:
        """获取所有情绪信号的平滑值。"""
        return {
            name: self._running_values[ts]
            for name, ts in EMOTION_TIMESCALE_MAP.items()
        }
