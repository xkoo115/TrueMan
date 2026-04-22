"""TrueMan Agent主循环：内稳态驱动的自治LLM Agent。

核心循环：感知编码 → 情绪计算 → 策略选择 → 行动执行 → 记忆存储 → 学习触发

关键特性：
- 交互即训练：与外界交互的同时通过情绪驱动机制训练自身
- 兼容任意开源LLM：通过LLM抽象层和PEFT标准接口
- 三种情绪信号驱动：惊奇(认知修正) + 无聊(好奇心探索) + 焦虑(自我纠错)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from trueman.core.config import AgentConfig
from trueman.core.llm_backend import LLMBackend, HuggingFaceLLM, LLMBackendFactory
from trueman.core.homeostasis.core import HomeostasisCore
from trueman.core.homeostasis.integrator import EmotionState
from trueman.core.memory.episodic import EpisodicMemory
from trueman.core.memory.replay import ReplayBuffer
from trueman.core.policy.base import BasePolicy
from trueman.core.policy.curiosity import ExplorationPolicy, InvestigatePolicy, CuriosityPolicy
from trueman.core.policy.introspection import IntrospectionPolicy
from trueman.core.plasticity.lora_pool import DynamicLoRAPool
from trueman.core.world_model.predictor import WorldModel
from trueman.core.environment import (
    Environment, DialogEnvironment, APIEnvironment, GymEnvironment,
    Observation, Action, ObservationType, ActionType, EnvironmentFeedback,
)
from trueman.training.sleep_consolidation import SleepConsolidation
from trueman.utils.logging import EmotionLogger


class TrueManAgent:
    """TrueMan Agent：内稳态驱动的自治LLM Agent。

    组合所有核心组件，实现感知-情绪-行动循环。
    交互即训练：每次与外界交互都通过情绪驱动机制更新自身。
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = EmotionLogger("trueman.agent")

        # 1. LLM抽象层
        self.llm: LLMBackend = LLMBackendFactory.create(config)

        # 2. 内稳态内核
        self.homeostasis = HomeostasisCore(config, self.llm)

        # 3. 记忆系统
        self.episodic_memory = EpisodicMemory(capacity=config.memory_size)
        self.replay_buffer = ReplayBuffer(capacity=config.memory_size // 2)

        # 4. 行为策略层
        base_policy = BasePolicy(self.llm)
        exploration_policy = ExplorationPolicy(self.llm, self.episodic_memory)
        investigate_policy = InvestigatePolicy(self.llm)
        introspection_policy = IntrospectionPolicy(self.llm, self.episodic_memory)
        self.policy = CuriosityPolicy(
            base_policy=base_policy,
            exploration_policy=exploration_policy,
            investigate_policy=investigate_policy,
            introspection_policy=introspection_policy,
            thresholds=config.thresholds,
        )

        # 5. 世界模型
        self.world_model = WorldModel(
            state_dim=self.llm.hidden_size,
            hidden_dim=256,
        )

        # 6. LoRA可塑性系统（API模式下跳过，因为无法访问本地模型权重）
        self._is_api_mode = getattr(self.llm, 'is_api_mode', False)
        if not self._is_api_mode and self.llm.model is not None:
            self.lora_pool = DynamicLoRAPool(
                model=self.llm.model,
                llm=self.llm,
                config=config.lora,
                hidden_size=self.llm.hidden_size,
            )
        else:
            self.lora_pool = None

        # 7. 睡眠整合（API模式下跳过）
        if self.lora_pool is not None:
            self.sleep = SleepConsolidation(
                llm=self.llm,
                memory=self.episodic_memory,
                lora_pool=self.lora_pool,
            )
        else:
            self.sleep = None

        # Agent状态
        self.awake_steps = 0
        self.total_steps = 0
        self.current_emotion = EmotionState()
        self._conversation_history: list[dict] = []

        # 8. 环境接口（五官+手脚）
        self.environment: Environment | None = None
        self._running = False

    def step(self, observation: str) -> tuple[str, EmotionState]:
        """执行一步感知-情绪-行动循环。

        这是Agent的核心方法，实现"交互即训练"：
        1. 感知编码：LLM编码观测
        2. 情绪计算：基于预测误差计算惊奇/无聊/焦虑
        3. 策略选择：根据情绪状态选择行为策略
        4. 行动执行：生成响应
        5. 记忆存储：存储带情绪标注的交互轨迹
        6. 学习触发：情绪驱动的在线学习

        Args:
            observation: 外部观测（用户消息、环境反馈等）

        Returns:
            (action, emotion_state) Agent的响应和当前情绪状态
        """
        # 构建当前prompt
        prompt = self._build_prompt(observation)

        # 1. 感知编码
        state_embedding, token_logprobs = self.llm.encode(prompt)

        # 2. 情绪计算
        drive, emotion_state = self.homeostasis.compute_drive(
            token_logprobs, state_embedding, prompt
        )
        self.current_emotion = emotion_state

        # 3. 策略选择 + 4. 行动执行
        action, strategy_name = self.policy.select_action(prompt, emotion_state)

        # 记录情绪日志
        self.logger.log_emotion(
            surprise=emotion_state.surprise,
            boredom=emotion_state.boredom,
            anxiety=emotion_state.anxiety,
            drive=emotion_state.drive,
            triggered_action=strategy_name,
        )

        # 5. 记忆存储
        trace = self.episodic_memory.store(
            state_embedding=state_embedding,
            action=action,
            observation=observation,
            emotions=emotion_state,
            timestamp=self.total_steps,
        )
        self.replay_buffer.add(trace)

        # 更新对话历史
        self._conversation_history.append({
            "role": "user",
            "content": observation,
        })
        self._conversation_history.append({
            "role": "assistant",
            "content": action,
        })

        # 6. 学习触发（交互即训练的核心）
        self._trigger_learning(emotion_state, state_embedding, trace)

        self.awake_steps += 1
        self.total_steps += 1

        return action, emotion_state

    def chat(self, message: str) -> str:
        """便捷的对话接口。

        Args:
            message: 用户消息

        Returns:
            Agent的回复文本
        """
        action, emotion = self.step(message)
        return action

    def get_emotion_state(self) -> dict:
        """获取当前情绪状态。"""
        return self.current_emotion.to_dict()

    def force_sleep(self) -> int | None:
        """强制触发睡眠整合。"""
        if self.sleep is None:
            return None
        self.logger.log_learning_event("force_sleep")
        expert_id = self.sleep.consolidate()
        self.awake_steps = 0
        return expert_id

    # ---- 环境接口（五官+手脚） ----

    def bind_environment(self, env: Environment) -> None:
        """绑定环境，连接Agent的"五官"和"手脚"。

        Args:
            env: 环境实例（DialogEnvironment/APIEnvironment/GymEnvironment等）
        """
        self.environment = env
        self.logger.log_learning_event("bind_environment", {"type": type(env).__name__})

    def perceive(self) -> Observation:
        """通过"五官"感知当前环境状态。

        Returns:
            当前观测
        """
        if self.environment is None:
            return Observation(type=ObservationType.TEXT, content="")
        return self.environment.observe()

    def act(self, action: Action) -> EnvironmentFeedback:
        """通过"手脚"执行动作并获取环境反馈。

        Args:
            action: 要执行的动作

        Returns:
            环境反馈（新观测+奖励+是否结束）
        """
        if self.environment is None:
            return EnvironmentFeedback(
                observation=Observation(type=ObservationType.TEXT, content=action.content)
            )
        return self.environment.execute(action)

    def step_with_env(self) -> tuple[str, EmotionState, EnvironmentFeedback]:
        """通过环境接口执行一步完整的感知-情绪-行动循环。

        流程：
        1. perceive() — 五官感知环境
        2. step(observation) — 情绪计算+策略选择+行动生成+学习触发
        3. act(action) — 手脚执行动作到环境
        4. 返回响应、情绪状态、环境反馈

        Returns:
            (action_text, emotion_state, env_feedback)
        """
        # 1. 五官感知
        obs = self.perceive()
        obs_text = obs.as_text

        # 2. 核心循环（情绪计算+策略+学习）
        action_text, emotion_state = self.step(obs_text)

        # 3. 手脚执行
        action = Action(type=ActionType.TEXT, content=action_text)
        feedback = self.act(action)

        return action_text, emotion_state, feedback

    def run(self, max_steps: int = -1, callback=None) -> None:
        """Agent主动运行循环：持续感知-行动直到环境结束或达到最大步数。

        这是Agent的"自主运行"模式：Agent主动从环境感知、决策、行动，
        不需要外部逐步调用step()。

        Args:
            max_steps: 最大运行步数，-1表示无限运行直到环境done
            callback: 每步回调函数 callback(step, action, emotion, feedback)
        """
        if self.environment is None:
            self.logger.log_error("NO_ENVIRONMENT", "请先调用 bind_environment() 绑定环境")
            return

        self._running = True
        step_count = 0

        # 重置环境获取初始观测
        initial_obs = self.environment.reset()

        self.logger.log_learning_event("run_start", {"max_steps": max_steps})

        while self._running:
            # 检查步数限制
            if max_steps > 0 and step_count >= max_steps:
                break

            # 感知-情绪-行动循环
            action_text, emotion_state, feedback = self.step_with_env()

            step_count += 1

            # 回调
            if callback is not None:
                callback(step_count, action_text, emotion_state, feedback)

            # 检查环境是否结束
            if feedback.done:
                self.logger.log_learning_event("run_done", {"steps": step_count})
                break

        self._running = False
        self.logger.log_learning_event("run_end", {"steps": step_count})

    def stop(self) -> None:
        """停止run()循环。"""
        self._running = False

    def save_state(self, path: str) -> None:
        """保存Agent完整状态。"""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 保存配置
        self.config.to_yaml(save_dir / "config.yaml")

        # 保存Agent状态
        state = {
            "awake_steps": self.awake_steps,
            "total_steps": self.total_steps,
            "emotion": self.current_emotion.to_dict(),
            "conversation_history": self._conversation_history[-100:],  # 最近100轮
        }
        with open(save_dir / "agent_state.json", "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

        # 保存世界模型
        torch.save(self.world_model.state_dict(), save_dir / "world_model.pt")

    def load_state(self, path: str) -> None:
        """加载Agent完整状态。"""
        load_dir = Path(path)

        # 加载Agent状态
        state_file = load_dir / "agent_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.awake_steps = state.get("awake_steps", 0)
            self.total_steps = state.get("total_steps", 0)
            self._conversation_history = state.get("conversation_history", [])

        # 加载世界模型
        wm_file = load_dir / "world_model.pt"
        if wm_file.exists():
            self.world_model.load_state_dict(torch.load(wm_file, map_location="cpu"))

    def _build_prompt(self, observation: str) -> str:
        """构建LLM输入prompt，包含对话历史。"""
        # 保留最近10轮对话
        recent = self._conversation_history[-20:]

        parts = []
        for msg in recent:
            role = "用户" if msg["role"] == "user" else "助手"
            parts.append(f"{role}: {msg['content']}")
        parts.append(f"用户: {observation}")
        parts.append("助手:")

        return "\n".join(parts)

    def _trigger_learning(
        self,
        emotions: EmotionState,
        state_embedding: torch.Tensor,
        trace,
    ) -> None:
        """情绪驱动的在线学习（交互即训练的核心）。

        根据情绪信号触发不同类型的学习：
        - 惊奇 > 阈值 → 世界模型局部更新
        - 焦虑 > 阈值 → 内省（矛盾检测+优先级提升）
        - 清醒步数 > 阈值 或 焦虑 > 紧急阈值 → 睡眠整合
        """
        # 惊奇驱动：更新世界模型
        if emotions.surprise > self.config.thresholds.surprise_update_threshold:
            try:
                # 使用当前状态作为next_state的近似（自监督）
                self.world_model.update(
                    state=state_embedding.detach(),
                    next_state=state_embedding.detach(),  # 简化：实际应使用真正的下一状态
                    weight=emotions.surprise,
                )
                self.logger.log_learning_event("surprise_update", {
                    "surprise": f"{emotions.surprise:.4f}",
                })
            except Exception:
                pass

        # 焦虑驱动：内省
        if emotions.anxiety > self.config.thresholds.anxiety_introspection_threshold:
            contradictions = self.episodic_memory.find_contradictions(recent_n=20)
            if contradictions:
                for t1, t2 in contradictions[:5]:
                    self.episodic_memory.boost_priority(t1.trace_id)
                    self.episodic_memory.boost_priority(t2.trace_id)
                self.logger.log_learning_event("introspection", {
                    "contradictions": len(contradictions),
                })

        # 睡眠整合（API模式下跳过）
        should_sleep = (
            self.sleep is not None
            and (self.awake_steps >= self.config.awake_threshold
                 or emotions.anxiety > self.config.thresholds.anxiety_emergency_threshold)
        )
        if should_sleep:
            self.logger.log_learning_event("sleep_trigger", {
                "awake_steps": self.awake_steps,
                "anxiety": f"{emotions.anxiety:.4f}",
            })
            try:
                expert_id = self.sleep.consolidate()
                self.awake_steps = 0
                if expert_id is not None:
                    self.logger.log_learning_event("sleep_complete", {
                        "new_expert": expert_id,
                    })
            except Exception as e:
                self.logger.log_error("SLEEP_FAILED", str(e))
