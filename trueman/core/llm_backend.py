"""LLM抽象层：统一接口屏蔽不同开源LLM架构差异。

支持任意HuggingFace Transformers兼容的自回归LLM（Qwen、Llama、Mistral、DeepSeek等）。
关键设计：单次forward同时获取logits和hidden_states，避免重复计算。
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from trueman.core.config import AgentConfig


class LLMBackend(ABC):
    """LLM统一抽象接口。"""

    @abstractmethod
    def encode(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """编码文本，返回 (state_embedding, token_logprobs)。

        state_embedding: 最后一层最后一个token的hidden_state
        token_logprobs: 每个token的log概率（用于惊奇计算）
        """

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 1.0) -> str:
        """生成文本。"""

    @abstractmethod
    def generate_with_uncertainty(
        self, prompt: str, n_samples: int = 3, temperature: float = 1.0
    ) -> list[str]:
        """带不确定性的多次采样生成（用于焦虑信号计算）。"""

    @abstractmethod
    def get_hidden_states(self, text: str, layer: int = -1) -> torch.Tensor:
        """获取指定层的隐藏状态向量。"""

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """模型隐藏层维度。"""


class HuggingFaceLLM(LLMBackend):
    """HuggingFace Transformers LLM具体实现。

    封装AutoModelForCausalLM + AutoTokenizer，支持4bit/8bit量化加载。
    单次forward同时返回logits和hidden_states。
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # 量化配置
        quantization_config = None
        if config.load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        elif config.load_in_8bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # 加载模型和tokenizer
        model_kwargs = {}
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["torch_dtype"] = torch.float16

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.base_model_name, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name,
                trust_remote_code=True,
                **model_kwargs,
            )
        except Exception as e:
            raise RuntimeError(
                f"MODEL_LOAD_FAILED: 无法加载模型 '{config.base_model_name}': {e}"
            ) from e

        if quantization_config is None:
            self.model = self.model.to(self.device)

        self.model.eval()

        # 确保pad_token存在
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._hidden_size = self.model.config.hidden_size

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @torch.no_grad()
    def encode(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """编码文本，单次forward同时获取hidden_states和logits。"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(
            **inputs,
            output_hidden_states=True,
        )

        # 状态嵌入：最后一层最后一个token的hidden_state
        state_embedding = outputs.hidden_states[-1][0, -1, :]

        # Token log概率：用于惊奇信号计算
        logits = outputs.logits[0]  # (seq_len, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)

        # 获取实际token的log概率
        input_ids = inputs["input_ids"][0]  # (seq_len,)
        # logits的第i个位置预测第i+1个token，所以对齐需要偏移
        token_logprobs = log_probs[torch.arange(len(input_ids) - 1), input_ids[1:]]

        return state_embedding, token_logprobs

    @torch.no_grad()
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 1.0) -> str:
        """生成文本。"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            do_sample=temperature > 0,
            top_p=0.9,
        )

        # 只返回新生成的token
        new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def generate_with_uncertainty(
        self, prompt: str, n_samples: int = 3, temperature: float = 1.0
    ) -> list[str]:
        """带不确定性的多次采样生成。

        轻量模式：使用temperature扰动进行多次采样，开销约n_samples倍。
        """
        results = []
        for i in range(n_samples):
            # 每次使用略微不同的temperature增加多样性
            temp = temperature * (1.0 + 0.1 * i) if i > 0 else temperature
            result = self.generate(prompt, max_tokens=256, temperature=temp)
            results.append(result)
        return results

    @torch.no_grad()
    def get_hidden_states(self, text: str, layer: int = -1) -> torch.Tensor:
        """获取指定层的隐藏状态。"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[layer][0, -1, :]

    def get_prediction_distribution(self, text: str) -> torch.Tensor:
        """获取下一个token的预测概率分布（用于焦虑信号计算）。"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]  # 最后一个位置的logits
            probs = F.softmax(logits, dim=-1)
        return probs


class LLMBackendFactory:
    """LLM后端工厂，根据配置创建对应的LLMBackend实例。"""

    _registry: dict[str, type[LLMBackend]] = {
        "huggingface": HuggingFaceLLM,
    }

    @classmethod
    def create(cls, config: AgentConfig, backend_type: str = "huggingface") -> LLMBackend:
        """创建LLMBackend实例。"""
        if backend_type not in cls._registry:
            raise ValueError(
                f"不支持的LLM后端类型: '{backend_type}', "
                f"支持: {list(cls._registry.keys())}"
            )
        return cls._registry[backend_type](config)

    @classmethod
    def register(cls, name: str, backend_class: type[LLMBackend]) -> None:
        """注册自定义LLM后端。"""
        cls._registry[name] = backend_class
