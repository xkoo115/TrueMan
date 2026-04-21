"""环境接口：定义Agent与外部世界交互的"五官"和"手脚"。

Environment抽象了Agent感知和行动的通道：
- Sensor（五官）：observation_space定义Agent能感知什么，observe()返回当前观测
- Actuator（手脚）：action_space定义Agent能做什么，execute()执行动作并返回反馈

内置实现：
- DialogEnvironment: 对话环境（文本输入/输出）
- APIEnvironment: API调用环境（请求/响应）
- GymEnvironment: Gymnasium兼容环境（游戏/控制）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum


class ObservationType(Enum):
    """观测类型：Agent的"五官"能感知的信息类型。"""
    TEXT = "text"           # 文本（对话、文档）
    STATE = "state"        # 结构化状态（游戏画面、系统状态）
    REWARD = "reward"      # 外部奖励信号（可选）
    MULTIMODAL = "multimodal"  # 多模态（文本+图像等）


class ActionType(Enum):
    """动作类型：Agent的"手脚"能执行的动作类型。"""
    TEXT = "text"           # 生成文本
    API_CALL = "api_call"  # 调用外部API
    GAME_ACTION = "game"   # 游戏操作（离散/连续动作）
    TOOL_USE = "tool_use"  # 工具调用


@dataclass
class Observation:
    """观测：Agent通过"五官"接收到的信息。"""
    type: ObservationType
    content: Any               # 观测内容（str, dict, Tensor等）
    metadata: dict = field(default_factory=dict)  # 附加信息（时间戳、来源等）

    @property
    def as_text(self) -> str:
        """将观测转换为文本表示（用于LLM输入）。"""
        if isinstance(self.content, str):
            return self.content
        return str(self.content)


@dataclass
class Action:
    """动作：Agent通过"手脚"执行的操作。"""
    type: ActionType
    content: Any               # 动作内容（str, dict, int等）
    metadata: dict = field(default_factory=dict)


@dataclass
class EnvironmentFeedback:
    """环境反馈：执行动作后环境返回的信息。"""
    observation: Observation   # 新的观测
    reward: float = 0.0       # 外部奖励（可选，内稳态驱动不依赖此）
    done: bool = False        # 是否结束
    info: dict = field(default_factory=dict)  # 额外信息


class Environment(ABC):
    """环境抽象基类：定义Agent与外部世界的交互接口。

    Agent的"五官"通过 observe() 感知世界，
    "手脚"通过 execute() 作用于世界。
    """

    @abstractmethod
    def observe(self) -> Observation:
        """获取当前观测（五官感知）。"""

    @abstractmethod
    def execute(self, action: Action) -> EnvironmentFeedback:
        """执行动作并获取反馈（手脚执行）。"""

    @abstractmethod
    def reset(self) -> Observation:
        """重置环境，返回初始观测。"""

    @property
    @abstractmethod
    def observation_type(self) -> ObservationType:
        """观测类型。"""

    @property
    @abstractmethod
    def action_type(self) -> ActionType:
        """动作类型。"""


class DialogEnvironment(Environment):
    """对话环境：文本输入/输出。

    Agent的"五官"：接收用户文本消息
    Agent的"手脚"：生成文本回复
    """

    def __init__(self):
        self._pending_input: str = ""
        self._last_response: str = ""

    def set_input(self, text: str) -> None:
        """设置待处理的用户输入。"""
        self._pending_input = text

    def observe(self) -> Observation:
        """获取当前用户输入。"""
        return Observation(
            type=ObservationType.TEXT,
            content=self._pending_input,
        )

    def execute(self, action: Action) -> EnvironmentFeedback:
        """执行文本生成动作。"""
        self._last_response = action.content if isinstance(action.content, str) else str(action.content)
        return EnvironmentFeedback(
            observation=Observation(
                type=ObservationType.TEXT,
                content=self._last_response,
                metadata={"role": "assistant"},
            ),
        )

    def reset(self) -> Observation:
        self._pending_input = ""
        self._last_response = ""
        return Observation(type=ObservationType.TEXT, content="")

    @property
    def observation_type(self) -> ObservationType:
        return ObservationType.TEXT

    @property
    def action_type(self) -> ActionType:
        return ActionType.TEXT


class APIEnvironment(Environment):
    """API调用环境：请求/响应。

    Agent的"五官"：接收API响应
    Agent的"手脚"：发起API请求
    """

    def __init__(self, api_handler=None):
        self._api_handler = api_handler  # 用户提供的API调用函数
        self._last_response: dict = {}

    def observe(self) -> Observation:
        """获取最近的API响应。"""
        return Observation(
            type=ObservationType.STATE,
            content=self._last_response,
        )

    def execute(self, action: Action) -> EnvironmentFeedback:
        """执行API调用。"""
        if self._api_handler is not None:
            try:
                result = self._api_handler(action.content)
                self._last_response = result if isinstance(result, dict) else {"result": result}
            except Exception as e:
                self._last_response = {"error": str(e)}
        else:
            self._last_response = {"echo": action.content}

        return EnvironmentFeedback(
            observation=Observation(
                type=ObservationType.STATE,
                content=self._last_response,
            ),
        )

    def reset(self) -> Observation:
        self._last_response = {}
        return Observation(type=ObservationType.STATE, content={})

    @property
    def observation_type(self) -> ObservationType:
        return ObservationType.STATE

    @property
    def action_type(self) -> ActionType:
        return ActionType.API_CALL


class GymEnvironment(Environment):
    """Gymnasium兼容环境：游戏/控制任务。

    Agent的"五官"：接收游戏画面/状态
    Agent的"手脚"：执行游戏操作
    """

    def __init__(self, env_name: str = "CartPole-v1"):
        self._env = None
        self._env_name = env_name
        self._last_obs = None

    def _ensure_env(self):
        if self._env is None:
            import gymnasium as gym
            self._env = gym.make(self._env_name)

    def observe(self) -> Observation:
        """获取当前游戏状态。"""
        return Observation(
            type=ObservationType.STATE,
            content=self._last_obs,
        )

    def execute(self, action: Action) -> EnvironmentFeedback:
        """执行游戏动作。"""
        self._ensure_env()
        obs, reward, terminated, truncated, info = self._env.step(action.content)
        self._last_obs = obs
        return EnvironmentFeedback(
            observation=Observation(type=ObservationType.STATE, content=obs),
            reward=reward,
            done=terminated or truncated,
            info=info,
        )

    def reset(self) -> Observation:
        self._ensure_env()
        obs, info = self._env.reset()
        self._last_obs = obs
        return Observation(type=ObservationType.STATE, content=obs)

    @property
    def observation_type(self) -> ObservationType:
        return ObservationType.STATE

    @property
    def action_type(self) -> ActionType:
        return ActionType.GAME_ACTION
