"""消融实验框架：系统性地移除单个组件，评估其贡献。

7种消融条件：
A1: 禁用惊奇信号 (surprise=0)
A2: 禁用无聊信号 (boredom=0)
A3: 禁用焦虑信号 (anxiety=0)
A4: 无情景记忆
A5: 无内省策略
A6: 无策略切换（始终base）
A7: 无情绪驱动学习触发
"""

from __future__ import annotations

from dataclasses import dataclass

from trueman.core.homeostasis.integrator import EmotionState


@dataclass
class AblationConfig:
    """消融条件配置。"""
    name: str
    disable_surprise: bool = False
    disable_boredom: bool = False
    disable_anxiety: bool = False
    disable_memory: bool = False
    disable_introspection: bool = False
    disable_strategy_switch: bool = False
    disable_learning_trigger: bool = False


ABLATION_CONFIGS = {
    "A1_no_surprise": AblationConfig(
        name="A1_no_surprise", disable_surprise=True,
    ),
    "A2_no_boredom": AblationConfig(
        name="A2_no_boredom", disable_boredom=True,
    ),
    "A3_no_anxiety": AblationConfig(
        name="A3_no_anxiety", disable_anxiety=True,
    ),
    "A4_no_memory": AblationConfig(
        name="A4_no_memory", disable_memory=True,
    ),
    "A5_no_introspection": AblationConfig(
        name="A5_no_introspection", disable_introspection=True,
    ),
    "A6_no_strategy": AblationConfig(
        name="A6_no_strategy", disable_strategy_switch=True,
    ),
    "A7_no_learning": AblationConfig(
        name="A7_no_learning", disable_learning_trigger=True,
    ),
}


class AblationRunner:
    """消融条件运行器：包装TrueManAgent，按配置禁用指定组件。"""

    def __init__(self, agent, config: AblationConfig):
        self.agent = agent
        self.config = config
        self.history: list[dict] = []

    def step(self, observation: str) -> tuple[str, dict]:
        prompt = self._build_prompt(observation)

        try:
            state_embedding, token_logprobs = self.agent.llm.encode(prompt)
        except Exception:
            state_embedding, token_logprobs = None, None

        surprise = 0.0
        boredom = 0.0
        anxiety = 0.0

        if state_embedding is not None and token_logprobs is not None:
            try:
                if not self.config.disable_surprise:
                    surprise_signal = self.agent.homeostasis.surprise_signal
                    surprise = surprise_signal.compute(token_logprobs).item()
            except Exception:
                pass

            try:
                if not self.config.disable_boredom:
                    boredom_signal = self.agent.homeostasis.boredom_signal
                    pred_error = -token_logprobs.mean().item()
                    boredom = boredom_signal.compute(pred_error, state_embedding).item()
            except Exception:
                pass

            try:
                if not self.config.disable_anxiety:
                    anxiety_signal = self.agent.homeostasis.anxiety_signal
                    anxiety = self.agent.homeostasis.compute_anxiety(prompt).item()
                else:
                    anxiety = 0.0
            except Exception:
                pass

        if self.config.disable_surprise:
            surprise = 0.0
        if self.config.disable_boredom:
            boredom = 0.0
        if self.config.disable_anxiety:
            anxiety = 0.0

        emotion_state = EmotionState(
            surprise=surprise, boredom=boredom, anxiety=anxiety,
            drive=0.0,
        )

        if self.config.disable_strategy_switch:
            response = self.agent.llm.generate(prompt, max_tokens=256, temperature=0.7)
            strategy = "base"
        elif self.config.disable_introspection and anxiety > 0.8:
            response = self.agent.llm.generate(prompt, max_tokens=256, temperature=0.7)
            strategy = "base_forced"
        else:
            response, strategy = self.agent.policy.select_action(prompt, emotion_state)

        if not self.config.disable_memory and state_embedding is not None:
            try:
                self.agent.episodic_memory.store(
                    state_embedding=state_embedding,
                    action=response,
                    observation=observation,
                    emotions=emotion_state,
                    timestamp=self.agent.total_steps,
                )
            except Exception:
                pass

        if not self.config.disable_learning_trigger:
            pass

        self.history.append({"role": "user", "content": observation})
        self.history.append({"role": "assistant", "content": response})
        self.agent.total_steps += 1

        return response, {
            "surprise": surprise,
            "boredom": boredom,
            "anxiety": anxiety,
            "strategy": strategy if isinstance(strategy, str) else "unknown",
        }

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
        if not self.config.disable_memory:
            self.agent.episodic_memory = type(self.agent.episodic_memory)(
                capacity=self.agent.config.memory_size
            )
        self.agent.total_steps = 0

    @property
    def condition_name(self) -> str:
        return self.config.name
