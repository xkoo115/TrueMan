"""OpenAI兼容API LLM后端：支持DeepSeek、OpenAI等云端模型。

通过API调用实现LLMBackend接口，无需本地部署模型。

关键适配：
- encode(): 使用logprobs参数获取token对数概率，用embedding API或哈希投影获取状态嵌入
- generate(): 标准chat completion API
- generate_with_uncertainty(): 多次采样（不同temperature），用于焦虑信号轻量模式
- get_hidden_states(): 用文本embedding近似替代
- get_prediction_distribution(): 不支持，焦虑信号使用lightweight模式替代

支持的API提供商：
- DeepSeek (api.deepseek.com)
- OpenAI (api.openai.com)
- 任何OpenAI兼容API（提供base_url即可）
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Optional

import numpy as np
import torch

from trueman.core.config import AgentConfig
from trueman.core.llm_backend import LLMBackend

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class OpenAICompatibleLLM(LLMBackend):
    """OpenAI兼容API LLM后端。

    通过chat completion API实现LLMBackend接口。
    支持DeepSeek、OpenAI及任何OpenAI兼容的API服务。
    """

    def __init__(self, config: AgentConfig):
        if not HAS_OPENAI:
            raise RuntimeError(
                "OPENAI_PACKAGE_REQUIRED: 请安装openai包: pip install openai"
            )

        self.config = config
        self._model_name = config.api_model_name or config.base_model_name
        self._hidden_size = config.api_embedding_dim or 1024

        # 初始化OpenAI客户端
        client_kwargs = {}
        if config.api_base_url:
            client_kwargs["base_url"] = config.api_base_url
        if config.api_key:
            client_kwargs["api_key"] = config.api_key

        self.client = openai.OpenAI(**client_kwargs)

        # 验证连接
        try:
            self.client.models.list()
        except Exception as e:
            raise RuntimeError(
                f"API_CONNECTION_FAILED: 无法连接到API服务 "
                f"(base_url={config.api_base_url}): {e}"
            ) from e

        # 对话历史（用于chat completion）
        self._messages: list[dict] = []

        # 占位属性：API模式没有本地模型对象
        # LoRA系统在API模式下不可用，Agent会检测此属性为None来跳过LoRA初始化
        self.model = None
        self.is_api_mode = True

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def encode(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """编码文本，返回 (state_embedding, token_logprobs)。

        API模式下：
        - state_embedding: 基于文本内容的确定性哈希投影（保持维度一致性）
        - token_logprobs: 使用logprobs参数获取（如果API支持）
        """
        # 状态嵌入：基于文本内容的确定性投影
        state_embedding = self._text_to_embedding(text)

        # Token log概率：尝试使用logprobs参数
        token_logprobs = self._get_logprobs(text)

        return state_embedding, token_logprobs

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 1.0) -> str:
        """通过chat completion API生成文本。"""
        try:
            response = self.client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=0.9,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            raise RuntimeError(f"API_GENERATE_FAILED: {e}") from e

    def generate_with_uncertainty(
        self, prompt: str, n_samples: int = 3, temperature: float = 1.0
    ) -> list[str]:
        """带不确定性的多次采样生成（用于焦虑信号计算）。

        使用不同的temperature进行多次采样，增加多样性。
        """
        results = []
        for i in range(n_samples):
            temp = temperature * (1.0 + 0.1 * i) if i > 0 else temperature
            result = self.generate(prompt, max_tokens=256, temperature=temp)
            results.append(result)
        return results

    def get_hidden_states(self, text: str, layer: int = -1) -> torch.Tensor:
        """获取状态嵌入（API模式下用文本embedding近似）。"""
        return self._text_to_embedding(text)

    def get_prediction_distribution(self, text: str) -> torch.Tensor:
        """获取预测概率分布（API模式不支持，返回均匀分布）。"""
        # API模式无法获取完整词表概率分布
        # 焦虑信号应使用lightweight模式（基于文本差异度）
        return torch.ones(self._hidden_size) / self._hidden_size

    def _text_to_embedding(self, text: str) -> torch.Tensor:
        """将文本转换为确定性嵌入向量。

        使用文本内容的哈希值作为随机种子，生成固定维度的向量。
        同一文本始终产生同一嵌入，不同文本产生不同嵌入。
        """
        # 使用SHA256哈希作为种子
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        seed = int(text_hash[:16], 16) % (2**32)

        rng = np.random.RandomState(seed)
        embedding = rng.randn(self._hidden_size).astype(np.float32)

        # L2归一化
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm

        return torch.tensor(embedding)

    def _get_logprobs(self, text: str) -> torch.Tensor:
        """尝试通过API获取token log概率。

        DeepSeek API支持logprobs参数。
        如果不支持，返回基于文本长度的近似值。
        """
        try:
            response = self.client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": text}],
                max_tokens=1,  # 只需要logprobs，不需要生成
                logprobs=True,
                top_logprobs=5,
            )

            # 提取logprobs
            if response.choices and response.choices[0].logprobs:
                token_logprobs = []
                for token_info in response.choices[0].logprobs.content or []:
                    if token_info.token_logprob is not None:
                        token_logprobs.append(token_info.token_logprob)

                if token_logprobs:
                    return torch.tensor(token_logprobs, dtype=torch.float32)

        except Exception:
            pass

        # 降级：基于文本统计的近似log概率
        # 假设平均token概率约为1/词表大小，加入文本长度调制
        n_tokens = max(1, len(text) // 4)  # 粗估token数
        avg_logprob = -math.log(50000)  # 假设词表大小50000
        # 加入一些基于文本特征的变异
        text_complexity = min(1.0, len(set(text)) / max(1, len(text)))
        logprobs = [avg_logprob * (1.0 + 0.1 * text_complexity)] * n_tokens
        return torch.tensor(logprobs, dtype=torch.float32)
