"""结构化日志工具，用于情绪信号和系统事件的统一记录。"""

import logging
import time
from typing import Any


class EmotionLogger:
    """情绪信号结构化日志器。"""

    def __init__(self, name: str = "trueman", level: int = logging.INFO):
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
            ))
            self._logger.addHandler(handler)

    def log_emotion(
        self,
        surprise: float,
        boredom: float,
        anxiety: float,
        drive: float,
        triggered_action: str = "none",
    ) -> None:
        """记录情绪信号状态。"""
        self._logger.info(
            "emotion|surprise=%.4f|boredom=%.4f|anxiety=%.4f|drive=%.4f|action=%s",
            surprise, boredom, anxiety, drive, triggered_action,
        )

    def log_learning_event(self, event_type: str, details: dict[str, Any] | None = None) -> None:
        """记录学习事件（惊奇更新/无聊探索/焦虑内省/睡眠整合）。"""
        detail_str = "|".join(f"{k}={v}" for k, v in (details or {}).items())
        self._logger.info("learning|type=%s|%s", event_type, detail_str)

    def log_lora_event(self, event_type: str, expert_id: int | None = None, **kwargs: Any) -> None:
        """记录LoRA专家池事件。"""
        extra = "|".join(f"{k}={v}" for k, v in kwargs.items())
        self._logger.info("lora|type=%s|expert=%s|%s", event_type, expert_id, extra)

    def log_warning(self, code: str, message: str) -> None:
        """记录警告事件。"""
        self._logger.warning("warning|code=%s|msg=%s", code, message)

    def log_error(self, code: str, message: str) -> None:
        """记录错误事件。"""
        self._logger.error("error|code=%s|msg=%s", code, message)
