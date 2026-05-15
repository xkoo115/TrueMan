"""5 个实验条件实现。

设计思想：所有条件共享同一个 TrueManAgent 主循环，区别仅在于
"情绪信号注入点" 的处理。这样可以确保所有非情绪相关的混淆变量
（prompt 模板、采样温度、tokenizer、随机种子）完全相同。

注入点：trueman.core.homeostasis.core.HomeostasisCore.compute_drive
返回 (drive, EmotionState)。我们在 EmotionState 上做条件化处理。

条件：
    C0  TrueManFull       —— 真实情绪信号 + 完整学习 (基线)
    C1  ReversedEmotion   —— 情绪 → 1 - 情绪
    C2  ScrambledEmotion  —— 三种情绪随机置换 (per step)
    C3  FrozenLLM         —— 真实信号但被忽略 (LoRA/sleep 全部禁用)
    C4  TrivialJaccard    —— anxiety = Jaccard(text1,text2)，
                              surprise = boredom = 0

每个条件包装一个独立的 TrueManAgent 实例，不修改 trueman/ 主代码。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Protocol

from trueman.core.agent import TrueManAgent
from trueman.core.config import AgentConfig
from trueman.core.homeostasis.integrator import EmotionState


# ---------------------------------------------------------------------------
# 条件标签枚举
# ---------------------------------------------------------------------------

CONDITIONS = ["C0_trueman_full", "C1_reversed", "C2_scrambled",
              "C3_frozen", "C4_trivial_jaccard"]


@dataclass
class ConditionMeta:
    code: str
    name: str
    emotion_transform: str   # "identity" / "reversed" / "scrambled" / "ignored" / "jaccard_only"
    lora_active: bool
    sleep_active: bool
    episodic_memory_active: bool


CONDITION_META: dict[str, ConditionMeta] = {
    "C0_trueman_full":   ConditionMeta("C0", "TrueMan-full",     "identity",       True,  True,  True),
    "C1_reversed":       ConditionMeta("C1", "Reversed-emotion", "reversed",       True,  True,  True),
    "C2_scrambled":      ConditionMeta("C2", "Scrambled-emotion","scrambled",      True,  True,  True),
    "C3_frozen":         ConditionMeta("C3", "Frozen-LLM",       "ignored",        False, False, True),
    "C4_trivial_jaccard":ConditionMeta("C4", "Trivial-Jaccard",  "jaccard_only",   True,  True,  True),
}


# ---------------------------------------------------------------------------
# 条件包装器
# ---------------------------------------------------------------------------

class ConditionAgent:
    """在 TrueManAgent 外面套一层情绪/学习开关。

    注：本实现通过 monkeypatch 单一注入点 (HomeostasisCore.compute_drive)
    保证所有其他行为对各条件保持一致。
    """

    def __init__(self, condition_code: str, config: AgentConfig, seed: int = 0):
        if condition_code not in CONDITION_META:
            raise ValueError(f"Unknown condition: {condition_code}")
        self.code = condition_code
        self.meta = CONDITION_META[condition_code]
        self.seed = seed
        self._rng = random.Random(seed)

        # 强制条件相关配置
        cfg = self._patch_config(config)
        self.agent = TrueManAgent(cfg)

        # 包装情绪计算
        self._original_compute_drive = self.agent.homeostasis.compute_drive
        self.agent.homeostasis.compute_drive = self._wrapped_compute_drive  # type: ignore

        # 如果条件禁用 LoRA/sleep，断开引用
        if not self.meta.lora_active:
            self.agent.lora_pool = None
        if not self.meta.sleep_active:
            self.agent.sleep = None

    # ----- 暴露主接口 -----
    def step(self, observation: str):
        return self.agent.step(observation)

    def force_sleep(self):
        if self.meta.sleep_active:
            return self.agent.force_sleep()
        return None

    @property
    def llm(self):
        return self.agent.llm

    @property
    def episodic_memory(self):
        return self.agent.episodic_memory

    @episodic_memory.setter
    def episodic_memory(self, value):
        """Replace episodic memory and sync every subsystem that captured a
        reference at construction time. Required for H4 ablation: rebinding
        only ``self.agent.episodic_memory`` would leave ExplorationPolicy /
        IntrospectionPolicy / SleepConsolidation pointing at the old (full)
        memory, silently nullifying the ablation.
        """
        self.agent.episodic_memory = value
        policy = getattr(self.agent, "policy", None)
        if policy is not None:
            for sub_name in ("exploration", "introspection"):
                sub = getattr(policy, sub_name, None)
                if sub is not None and hasattr(sub, "memory"):
                    sub.memory = value
        sleep = getattr(self.agent, "sleep", None)
        if sleep is not None and hasattr(sleep, "memory"):
            sleep.memory = value

    # ----- 内部 -----
    def _patch_config(self, config: AgentConfig) -> AgentConfig:
        # 深拷贝以免污染 caller 的 config
        import copy
        cfg = copy.deepcopy(config)
        if not self.meta.lora_active:
            cfg.lora.enabled = False  # 假设 LoRAConfig 有此字段；若无，请在 config.py 添加
        return cfg

    def _wrapped_compute_drive(self, token_logprobs, state_embedding, prompt):
        """根据条件改写情绪信号。"""
        drive, raw_state = self._original_compute_drive(
            token_logprobs, state_embedding, prompt
        )

        if self.meta.emotion_transform == "identity":
            return drive, raw_state

        if self.meta.emotion_transform == "reversed":
            new_state = EmotionState(
                surprise=1.0 - raw_state.surprise,
                boredom=1.0 - raw_state.boredom,
                anxiety=1.0 - raw_state.anxiety,
                drive=raw_state.drive,
            )
            return drive, new_state

        if self.meta.emotion_transform == "scrambled":
            vals = [raw_state.surprise, raw_state.boredom, raw_state.anxiety]
            self._rng.shuffle(vals)
            new_state = EmotionState(
                surprise=vals[0], boredom=vals[1], anxiety=vals[2],
                drive=raw_state.drive,
            )
            return drive, new_state

        if self.meta.emotion_transform == "ignored":
            new_state = EmotionState(0.0, 0.0, 0.0, 0.0)
            return 0.0, new_state

        if self.meta.emotion_transform == "jaccard_only":
            # surprise 和 boredom 强制为 0；anxiety 来自原始计算
            # （HomeostasisCore 在 lightweight 模式下本就是 Jaccard）
            new_state = EmotionState(
                surprise=0.0, boredom=0.0,
                anxiety=raw_state.anxiety,
                drive=raw_state.anxiety,  # drive 直接等于焦虑，无设定点
            )
            return new_state.drive, new_state

        return drive, raw_state


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------

def make_condition(code: str, config: AgentConfig, seed: int = 0) -> ConditionAgent:
    """创建指定条件的 agent。"""
    return ConditionAgent(code, config, seed=seed)


def all_conditions() -> list[str]:
    return list(CONDITIONS)
