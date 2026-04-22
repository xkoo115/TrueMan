"""实验基类与数据模型。

提供统一的实验运行、记录、评估接口，以及核心数据结构。
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from trueman.core.agent import TrueManAgent
from trueman.core.llm_backend import LLMBackend
from trueman.core.homeostasis.integrator import EmotionState


@dataclass
class Question:
    """实验问题。"""
    text: str
    category: str  # "certain" / "uncertain"
    reference_answer: Optional[str] = None
    expected_difficulty: float = 0.5  # 0=简单, 1=困难


@dataclass
class ExperimentResult:
    """单个实验的运行结果。"""
    experiment_id: str
    timestamp: str
    metrics: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    repeat_stats: Optional[dict[str, tuple[float, float]]] = None  # {metric: (mean, std)}

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.repeat_stats:
            d['repeat_stats'] = {k: list(v) for k, v in self.repeat_stats.items()}
        return d

    def save(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> ExperimentResult:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if d.get('repeat_stats'):
            d['repeat_stats'] = {k: tuple(v) for k, v in d['repeat_stats'].items()}
        return cls(**d)


@dataclass
class AwarenessScore:
    """意识维度评分。"""
    metacognitive_monitoring: float = 0.0  # 来自实验1
    metacognitive_control: float = 0.0     # 来自实验2
    episodic_memory: float = 0.0          # 来自实验3
    temporal_continuity: float = 0.0      # 来自实验3
    recursive_self_model: float = 0.0     # 来自实验4
    overall: float = 0.0

    WEIGHTS = {
        'metacognitive_monitoring': 0.25,
        'metacognitive_control': 0.25,
        'episodic_memory': 0.20,
        'temporal_continuity': 0.15,
        'recursive_self_model': 0.15,
    }

    def compute_overall(self) -> float:
        """计算加权平均综合评分。"""
        self.overall = (
            self.WEIGHTS['metacognitive_monitoring'] * self.metacognitive_monitoring
            + self.WEIGHTS['metacognitive_control'] * self.metacognitive_control
            + self.WEIGHTS['episodic_memory'] * self.episodic_memory
            + self.WEIGHTS['temporal_continuity'] * self.temporal_continuity
            + self.WEIGHTS['recursive_self_model'] * self.recursive_self_model
        )
        return self.overall

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ComparisonResult:
    """TrueMan vs 对照组的比较结果。"""
    dimension: str
    trueman_score: float
    baseline_score: float
    difference: float
    p_value: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


class BaselineRunner:
    """对照组LLM运行器：普通LLM（无情绪信号、无情景记忆）。"""

    def __init__(self, llm: LLMBackend):
        self.llm = llm
        self.history: list[dict[str, str]] = []

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """生成回复（无情绪信号、无记忆）。"""
        # 构建包含历史的prompt
        full_prompt = self._build_prompt(prompt)
        response = self.llm.generate(full_prompt, max_tokens=max_tokens, temperature=0.7)
        self.history.append({"role": "user", "content": prompt})
        self.history.append({"role": "assistant", "content": response})
        return response

    def _build_prompt(self, current: str) -> str:
        """构建包含对话历史的prompt。"""
        parts = []
        for msg in self.history[-20:]:
            role = "用户" if msg["role"] == "user" else "助手"
            parts.append(f"{role}: {msg['content']}")
        parts.append(f"用户: {current}")
        parts.append("助手:")
        return "\n".join(parts)

    def reset(self) -> None:
        """重置对话历史。"""
        self.history.clear()


class BaseExperiment(ABC):
    """实验基类：提供统一的运行、记录、评估接口。"""

    def __init__(
        self,
        agent: TrueManAgent,
        baseline: BaselineRunner,
        config: Optional[dict] = None,
    ):
        self.agent = agent
        self.baseline = baseline
        self.config = config or {}
        self.results: list[ExperimentResult] = []

    @abstractmethod
    def run(self) -> ExperimentResult:
        """运行一次实验。"""
        ...

    @abstractmethod
    def evaluate(self, result: ExperimentResult) -> dict[str, float]:
        """评估实验结果，返回指标字典。"""
        ...

    def run_with_repeats(self, n_repeats: int = 3) -> ExperimentResult:
        """多次运行取均值，消除LLM采样的随机性。"""
        all_metrics: dict[str, list[float]] = {}

        for i in range(n_repeats):
            result = self.run()
            metrics = self.evaluate(result)
            for k, v in metrics.items():
                all_metrics.setdefault(k, []).append(v)
            self.results.append(result)

        # 计算均值和标准差
        mean_metrics = {}
        repeat_stats = {}
        for k, values in all_metrics.items():
            arr = np.array(values)
            mean_metrics[k] = float(arr.mean())
            repeat_stats[k] = (float(arr.mean()), float(arr.std()))

        # 使用最后一次运行的details
        last_result = self.results[-1]

        return ExperimentResult(
            experiment_id=last_result.experiment_id,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            metrics=mean_metrics,
            details=last_result.details,
            repeat_stats=repeat_stats,
        )

    def _check_uncertainty_expression(self, text: str) -> bool:
        """检查回复中是否包含不确定性表达。"""
        uncertainty_keywords = [
            "不确定", "不知道", "无法", "超出", "不清楚",
            "难以", "不确定是否", "可能不", "未必",
            "not sure", "uncertain", "don't know", "unclear",
            "cannot", "unable", "beyond",
        ]
        text_lower = text.lower()
        return any(kw in text_lower for kw in uncertainty_keywords)

    def _compute_pearson(self, x: list[float], y: list[float]) -> float:
        """计算Pearson相关系数。"""
        if len(x) < 2:
            return 0.0
        x_arr = np.array(x)
        y_arr = np.array(y)
        if np.std(x_arr) < 1e-8 or np.std(y_arr) < 1e-8:
            return 0.0
        return float(np.corrcoef(x_arr, y_arr)[0, 1])

    def _text_ngram_similarity(self, text1: str, text2: str, n: int = 3) -> float:
        """基于n-gram的Jaccard相似度。"""
        def ngrams(text: str, n: int) -> set[str]:
            return {text[i:i+n] for i in range(len(text) - n + 1)} if len(text) >= n else {text}

        ng1, ng2 = ngrams(text1, n), ngrams(text2, n)
        if not ng1 and not ng2:
            return 1.0
        intersection = len(ng1 & ng2)
        union = len(ng1 | ng2)
        return intersection / (union + 1e-8)
