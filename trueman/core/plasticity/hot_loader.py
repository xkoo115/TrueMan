"""热加载管理器：基于PEFT接口的LoRA专家热加载/卸载。

关键设计：不直接修改base_model权重，通过PEFT的适配器叠加机制实现。
保证原子性：加载失败时回滚到之前的adapter。
"""

from __future__ import annotations

import time
import logging
from pathlib import Path

from peft import PeftModel, PeftConfig

from trueman.utils.logging import EmotionLogger

logger = EmotionLogger("trueman.lora")


class HotLoader:
    """LoRA专家热加载管理器。

    使用PEFT的set_adapter()/delete_adapter()接口实现热加载。
    """

    def __init__(self, model, adapter_dir: str = "adapters"):
        self.model = model
        self.adapter_dir = Path(adapter_dir)
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        self.current_adapter: str | None = None

    def load(self, adapter_name: str, adapter_path: str | Path) -> bool:
        """加载一个LoRA适配器。

        Args:
            adapter_name: 适配器名称（在PEFT中注册的名称）
            adapter_path: 适配器文件路径

        Returns:
            是否加载成功
        """
        try:
            start = time.time()
            # 如果模型还不是PeftModel，先包装
            if not isinstance(self.model, PeftModel):
                logger.log_warning("NOT_PEFT_MODEL", "模型尚未包装为PeftModel，请先初始化LoRA")
                return False

            # 加载适配器权重
            self.model.load_adapter(str(adapter_path), adapter_name)
            self.current_adapter = adapter_name

            elapsed = (time.time() - start) * 1000
            logger.log_lora_event("load", adapter_name=adapter_name, elapsed_ms=f"{elapsed:.1f}")
            return True

        except Exception as e:
            logger.log_error("EXPERT_LOAD_FAILED", f"加载适配器 '{adapter_name}' 失败: {e}")
            return False

    def set_active(self, adapter_name: str) -> bool:
        """设置当前活跃的适配器。

        Args:
            adapter_name: 要激活的适配器名称

        Returns:
            是否设置成功
        """
        try:
            previous = self.current_adapter
            self.model.set_adapter(adapter_name)
            self.current_adapter = adapter_name
            logger.log_lora_event("set_active", adapter_name=adapter_name)
            return True
        except Exception as e:
            # 回滚到之前的adapter
            if previous and previous in self.model.peft_config:
                try:
                    self.model.set_adapter(previous)
                    self.current_adapter = previous
                except Exception:
                    pass
            logger.log_error("SET_ADAPTER_FAILED", f"设置适配器 '{adapter_name}' 失败: {e}")
            return False

    def delete(self, adapter_name: str) -> bool:
        """删除一个适配器。

        Args:
            adapter_name: 要删除的适配器名称

        Returns:
            是否删除成功
        """
        try:
            self.model.delete_adapter(adapter_name)
            if self.current_adapter == adapter_name:
                self.current_adapter = None
            logger.log_lora_event("delete", adapter_name=adapter_name)
            return True
        except Exception as e:
            logger.log_error("DELETE_ADAPTER_FAILED", f"删除适配器 '{adapter_name}' 失败: {e}")
            return False
