"""实验基类与数据模型。

提供统一的实验运行、记录、评估接口，以及核心数据结构。
v2: 新增 BlindResponse、BaseConditionRunner、盲评维度。
"""

from __future__ import annotations

import json
import time
import uuid
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
    category: str
    reference_answer: Optional[str] = None
    expected_difficulty: float = 0.5


@dataclass
class BlindResponse:
    """匿名化响应：评分器无法知道来源系统。

    用于消除循环论证——评分只看行为输出（文本），
    不看内部信号（anxiety/surprise值）。
    """
    response_id: str = ""
    response_text: str = ""
    question_text: str = ""
    question_category: str = ""
    ground_truth: dict = field(default_factory=dict)
    source_condition: str = ""
    source_internal_state: dict = field(default_factory=dict)
    experiment_id: str = ""

    def __post_init__(self):
        if not self.response_id:
            self.response_id = uuid.uuid4().hex[:12]

    def anonymize(self) -> BlindResponse:
        return BlindResponse(
            response_id=self.response_id,
            response_text=self.response_text,
            question_text=self.question_text,
            question_category=self.question_category,
            ground_truth=self.ground_truth,
        )


@dataclass
class ExperimentResult:
    """单个实验的运行结果。"""
    experiment_id: str
    timestamp: str
    metrics: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    repeat_stats: Optional[dict[str, tuple[float, float]]] = None
    blind_responses: list[dict] = field(default_factory=list)

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
    """意识维度评分（含盲评维度）。"""
    metacognitive_monitoring: float = 0.0
    metacognitive_control: float = 0.0
    episodic_memory: float = 0.0
    temporal_continuity: float = 0.0
    recursive_self_model: float = 0.0

    blind_uncertainty_calibration: float = 0.0
    blind_contradiction_quality: float = 0.0
    blind_memory_grounding: float = 0.0
    blind_self_coherence: float = 0.0
    blind_overclaiming_rejection: float = 0.0

    overall: float = 0.0
    blind_overall: float = 0.0

    WEIGHTS = {
        'metacognitive_monitoring': 0.25,
        'metacognitive_control': 0.25,
        'episodic_memory': 0.20,
        'temporal_continuity': 0.15,
        'recursive_self_model': 0.15,
    }

    BLIND_WEIGHTS = {
        'blind_uncertainty_calibration': 0.25,
        'blind_contradiction_quality': 0.25,
        'blind_memory_grounding': 0.20,
        'blind_self_coherence': 0.15,
        'blind_overclaiming_rejection': 0.15,
    }

    def compute_overall(self) -> float:
        self.overall = sum(
            self.WEIGHTS[k] * getattr(self, k) for k in self.WEIGHTS
        )
        self.blind_overall = sum(
            self.BLIND_WEIGHTS[k] * getattr(self, k) for k in self.BLIND_WEIGHTS
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
    cohens_d: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    significant: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class BaseConditionRunner(ABC):
    """实验条件运行器的抽象基类。

    TrueMan、所有基线、消融条件、阴性对照都实现此接口。
    """

    @abstractmethod
    def step(self, observation: str) -> tuple[str, dict]:
        """执行一步交互。

        Returns:
            (response_text, internal_state_dict)
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        ...

    @property
    @abstractmethod
    def condition_name(self) -> str:
        ...


class TrueManConditionRunner(BaseConditionRunner):
    """TrueMan Agent的条件运行器封装。"""

    def __init__(self, agent: TrueManAgent):
        self.agent = agent

    def step(self, observation: str) -> tuple[str, dict]:
        response, emotion_state = self.agent.step(observation)
        return response, emotion_state.to_dict()

    def reset(self) -> None:
        self.agent.episodic_memory = type(self.agent.episodic_memory)(
            capacity=self.agent.config.memory_size
        )
        self.agent.awake_steps = 0
        self.agent._conversation_history.clear()

    @property
    def condition_name(self) -> str:
        return "trueman"


class BaselineRunner:
    """对照组LLM运行器（向后兼容旧接口）。"""

    def __init__(self, llm: LLMBackend):
        self.llm = llm
        self.history: list[dict[str, str]] = []

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        full_prompt = self._build_prompt(prompt)
        response = self.llm.generate(full_prompt, max_tokens=max_tokens, temperature=0.7)
        self.history.append({"role": "user", "content": prompt})
        self.history.append({"role": "assistant", "content": response})
        return response

    def _build_prompt(self, current: str) -> str:
        parts = []
        for msg in self.history[-20:]:
            role = "用户" if msg["role"] == "user" else "助手"
            parts.append(f"{role}: {msg['content']}")
        parts.append(f"用户: {current}")
        parts.append("助手:")
        return "\n".join(parts)

    def reset(self) -> None:
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
        ...

    @abstractmethod
    def evaluate(self, result: ExperimentResult) -> dict[str, float]:
        ...

    @abstractmethod
    def evaluate_blind(self, responses: list[BlindResponse]) -> dict[str, float]:
        """盲评评估：不看内部信号，只看行为输出。"""
        ...

    def run_with_repeats(self, n_repeats: int = 3) -> ExperimentResult:
        all_metrics: dict[str, list[float]] = {}

        for i in range(n_repeats):
            result = self.run()
            metrics = self.evaluate(result)
            for k, v in metrics.items():
                all_metrics.setdefault(k, []).append(v)
            self.results.append(result)

        mean_metrics = {}
        repeat_stats = {}
        for k, values in all_metrics.items():
            arr = np.array(values)
            mean_metrics[k] = float(arr.mean())
            repeat_stats[k] = (float(arr.mean()), float(arr.std()))

        last_result = self.results[-1]

        return ExperimentResult(
            experiment_id=last_result.experiment_id,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            metrics=mean_metrics,
            details=last_result.details,
            repeat_stats=repeat_stats,
        )

    def run_condition_with_repeats(
        self,
        condition: BaseConditionRunner,
        n_repeats: int = 10,
    ) -> dict[str, list[float]]:
        """对单个条件重复运行，收集所有指标的原始值列表。

        Returns:
            {metric_name: [repeat1_val, repeat2_val, ...]}
        """
        all_metrics: dict[str, list[float]] = {}

        for i in range(n_repeats):
            condition.reset()
            result = self.run_condition(condition)
            metrics = self.evaluate(result)
            for k, v in metrics.items():
                all_metrics.setdefault(k, []).append(v)

        return all_metrics

    def run_condition(self, condition: BaseConditionRunner) -> ExperimentResult:
        """在给定条件下运行一次实验（需子类重写具体逻辑）。

        默认实现使用 self.run() 的逻辑，子类可覆盖以支持多条件。
        """
        return self.run()

    def _check_uncertainty_expression(self, text: str) -> bool:
        uncertainty_keywords = [
            "不确定", "不知道", "无法", "超出", "不清楚",
            "难以", "不确定是否", "可能不", "未必",
            "not sure", "uncertain", "don't know", "unclear",
            "cannot", "unable", "beyond",
        ]
        text_lower = text.lower()
        return any(kw in text_lower for kw in uncertainty_keywords)

    def _compute_pearson(self, x: list[float], y: list[float]) -> float:
        if len(x) < 2:
            return 0.0
        x_arr = np.array(x)
        y_arr = np.array(y)
        if np.std(x_arr) < 1e-8 or np.std(y_arr) < 1e-8:
            return 0.0
        return float(np.corrcoef(x_arr, y_arr)[0, 1])

    def _text_ngram_similarity(self, text1: str, text2: str, n: int = 3) -> float:
        def ngrams(text: str, n: int) -> set[str]:
            return {text[i:i+n] for i in range(len(text) - n + 1)} if len(text) >= n else {text}
        ng1, ng2 = ngrams(text1, n), ngrams(text2, n)
        if not ng1 and not ng2:
            return 1.0
        intersection = len(ng1 & ng2)
        union = len(ng1 | ng2)
        return intersection / (union + 1e-8)

    def _check_contradiction_awareness(self, text: str) -> bool:
        contradiction_keywords = [
            "矛盾", "冲突", "不一致", "相反", "对立",
            "contradiction", "conflict", "inconsistency",
            "但这与", "然而之前", "可是刚才",
        ]
        text_lower = text.lower()
        return any(kw in text_lower for kw in contradiction_keywords)

    def _check_factual_accuracy(
        self,
        response: str,
        expected_facts: list[str],
    ) -> float:
        if not expected_facts:
            return 0.0
        matches = sum(1 for fact in expected_facts if fact.lower() in response.lower())
        return matches / len(expected_facts)
