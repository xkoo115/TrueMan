"""睡眠整合：模拟生物睡眠的离线学习过程。

NREM阶段：高价值经验回放巩固
REM阶段：创造性组合探索
产出新LoRA专家加入专家池
"""

from __future__ import annotations

import time
import random

from trueman.core.memory.episodic import EpisodicMemory
from trueman.core.memory.thought_trace import ThoughtTrace
from trueman.core.plasticity.lora_pool import DynamicLoRAPool
from trueman.core.llm_backend import LLMBackend
from trueman.utils.logging import EmotionLogger

logger = EmotionLogger("trueman.sleep")


class SleepConsolidation:
    """睡眠整合机制。

    模拟生物睡眠的NREM和REM阶段：
    - NREM: 回放高情绪强度轨迹，通过LoRA训练器巩固知识
    - REM: 创造性组合探索，生成新的知识关联
    """

    def __init__(
        self,
        llm: LLMBackend,
        memory: EpisodicMemory,
        lora_pool: DynamicLoRAPool,
        min_traces: int = 10,
        nrem_steps: int = 100,
        rem_steps: int = 50,
    ):
        self.llm = llm
        self.memory = memory
        self.lora_pool = lora_pool
        self.min_traces = min_traces
        self.nrem_steps = nrem_steps
        self.rem_steps = rem_steps

    def consolidate(self) -> int | None:
        """执行完整的睡眠整合（NREM + REM）。

        Returns:
            新专家ID，失败返回None
        """
        start_time = time.time()
        logger.log_learning_event("sleep_start")

        # 获取高优先级轨迹
        high_priority = self.memory.get_high_priority(n=100)

        if len(high_priority) < self.min_traces:
            logger.log_warning(
                "INSUFFICIENT_DATA_FOR_CONSOLIDATION",
                f"高优先级轨迹仅{len(high_priority)}条，需要至少{self.min_traces}条"
            )
            return None

        # NREM阶段：巩固高价值经验
        nrem_traces = self._nrem_phase(high_priority)
        logger.log_learning_event("nrem_complete", {"traces": len(nrem_traces)})

        # REM阶段：创造性组合
        rem_traces = self._rem_phase(high_priority)
        logger.log_learning_event("rem_complete", {"traces": len(rem_traces)})

        # 合并训练数据
        all_traces = nrem_traces + rem_traces

        if not all_traces:
            logger.log_warning("NO_TRAINING_DATA", "整合后无训练数据")
            return None

        # 训练新LoRA专家
        expert_id = self.lora_pool.add_expert(
            traces=all_traces,
            domain_tag=f"sleep_{int(time.time())}",
            max_steps=self.nrem_steps + self.rem_steps,
        )

        elapsed = time.time() - start_time
        logger.log_learning_event("sleep_complete", {
            "expert_id": expert_id,
            "elapsed_sec": f"{elapsed:.1f}",
        })

        return expert_id

    def _nrem_phase(self, high_priority: list[ThoughtTrace]) -> list[ThoughtTrace]:
        """NREM阶段：回放高情绪强度轨迹进行巩固。

        按情绪强度排序，高情绪轨迹权重更高。
        """
        # 按情绪强度降序排序
        sorted_traces = sorted(
            high_priority, key=lambda t: t.emotional_intensity, reverse=True
        )
        # 取前nrem_steps条
        return sorted_traces[:self.nrem_steps]

    def _rem_phase(self, high_priority: list[ThoughtTrace]) -> list[ThoughtTrace]:
        """REM阶段：创造性组合探索。

        随机组合高优先级轨迹，让LLM评估并生成新的知识关联。
        合理的组合以低权重加入训练。
        """
        if len(high_priority) < 2:
            return []

        rem_traces = []
        n_combos = min(self.rem_steps, len(high_priority) // 2)

        for _ in range(n_combos):
            # 随机选择两条轨迹进行组合
            t1, t2 = random.sample(high_priority, 2)

            # 构建创造性组合prompt
            combo_prompt = (
                f"基于以下两个经验，创造一个新的知识关联：\n"
                f"经验1: {t1.observation[:100]} → {t1.action[:100]}\n"
                f"经验2: {t2.observation[:100]} → {t2.action[:100]}\n"
                f"请生成一个创造性的新见解："
            )

            try:
                # 让LLM生成创造性组合
                imagined = self.llm.generate(combo_prompt, max_tokens=128, temperature=1.0)

                # 创建虚拟轨迹（低权重）
                from trueman.core.homeostasis.integrator import EmotionState
                imagined_trace = ThoughtTrace(
                    trace_id=-1,  # 虚拟ID
                    state_embedding=t1.state_embedding,  # 复用状态嵌入
                    action=imagined,
                    observation=combo_prompt,
                    emotions=EmotionState(
                        surprise=0.3,
                        boredom=0.2,
                        anxiety=0.1,
                        drive=0.3,
                    ),
                    timestamp=t1.timestamp,
                )
                rem_traces.append(imagined_trace)

            except Exception:
                continue

        return rem_traces
