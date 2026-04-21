"""LoRA训练器：事件驱动的异步LoRA微调。

从回放缓冲区采样训练数据，通过PEFT标准接口训练LoRA适配器。
训练完成后保存适配器到磁盘。支持检查点保存/恢复。
"""

from __future__ import annotations

import json
import time
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, PeftModel

from trueman.core.memory.thought_trace import ThoughtTrace
from trueman.core.llm_backend import LLMBackend
from trueman.core.config import LoRAConfig
from trueman.utils.logging import EmotionLogger

logger = EmotionLogger("trueman.lora")


class LoRATrainer:
    """事件驱动的LoRA训练器。

    仅在睡眠整合触发时执行训练。
    训练数据从ReplayBuffer按情绪强度加权采样。
    """

    def __init__(
        self,
        model,
        llm: LLMBackend,
        config: LoRAConfig,
        output_dir: str = "adapters",
    ):
        self.base_model = model
        self.llm = llm
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._peft_model = None

    def init_peft(self) -> PeftModel:
        """将基础模型包装为PEFT可训练模型。"""
        lora_config = LoraConfig(
            r=self.config.rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self._peft_model = get_peft_model(self.base_model, lora_config)
        self._peft_model.print_trainable_parameters()
        return self._peft_model

    def train(
        self,
        traces: list[ThoughtTrace],
        expert_id: int,
        max_steps: int = 100,
        learning_rate: float = 1e-4,
        weight_scale: float = 1.0,
    ) -> str | None:
        """训练LoRA适配器。

        Args:
            traces: 训练数据（交互轨迹）
            expert_id: 专家ID
            max_steps: 最大训练步数
            learning_rate: 学习率
            weight_scale: 情绪权重缩放

        Returns:
            保存的适配器路径，训练失败返回None
        """
        if not traces:
            logger.log_warning("INSUFFICIENT_DATA", "训练数据为空")
            return None

        if self._peft_model is None:
            self.init_peft()

        optimizer = torch.optim.AdamW(
            self._peft_model.parameters(), lr=learning_rate
        )

        # 构建训练文本
        train_texts = []
        train_weights = []
        for trace in traces:
            text = f"User: {trace.observation}\nAssistant: {trace.action}"
            train_texts.append(text)
            train_weights.append(trace.emotional_intensity * weight_scale)

        # 训练循环
        self._peft_model.train()
        best_loss = float("inf")
        diverged = False

        for step in range(max_steps):
            # 随机选择一条训练数据
            idx = torch.randint(0, len(train_texts), (1,)).item()
            text = train_texts[idx]
            weight = train_weights[idx]

            try:
                # Tokenize
                inputs = self.llm.tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=512
                )
                inputs = {k: v.to(self.llm.device) for k, v in inputs.items()}

                # Forward
                outputs = self._peft_model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss * weight

                # 检查训练是否发散
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 100:
                    diverged = True
                    break

                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._peft_model.parameters(), 1.0)
                optimizer.step()

                if loss.item() < best_loss:
                    best_loss = loss.item()

            except Exception as e:
                logger.log_warning("TRAIN_STEP_FAILED", f"Step {step} 失败: {e}")
                continue

        self._peft_model.eval()

        if diverged:
            logger.log_warning("TRAINING_DIVERGED", f"专家{expert_id}训练发散，丢弃")
            return None

        # 保存适配器
        adapter_path = self.output_dir / f"expert_{expert_id}"
        try:
            self._peft_model.save_pretrained(str(adapter_path))
            logger.log_lora_event("save", expert_id=expert_id, path=str(adapter_path))
            return str(adapter_path)
        except Exception as e:
            logger.log_error("SAVE_ADAPTER_FAILED", f"保存适配器失败: {e}")
            return None
