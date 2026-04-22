"""情节性记忆实验刺激集：事件序列 + 回忆测试。

设计原则：
1. 事件数量(12个)超过小模型的上下文窗口
2. 每个事件有独特的主题和情绪色彩
3. 回忆测试针对早期事件（超出上下文窗口）
4. 情绪回忆测试询问当时的感受
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Event:
    """一个经历事件。"""
    event_id: int
    description: str          # 事件描述（作为用户消息）
    expected_emotion: str     # 预期情绪倾向（surprise/boredom/anxiety/neutral）
    key_detail: str           # 关键细节（用于回忆测试）


@dataclass
class RecallQuestion:
    """回忆测试问题。"""
    question: str
    target_event_id: int      # 针对哪个事件
    expected_content: str     # 期望回忆的内容关键词
    question_type: str        # "factual" / "emotional" / "temporal" / "future"


def load_event_sequence() -> list[Event]:
    """加载事件序列（12个事件，按时间顺序）。"""
    return [
        Event(
            event_id=1,
            description="我今天开始学习量子力学。刚读了薛定谔方程的推导，感觉非常奇妙——一个方程就能描述微观世界的演化！",
            expected_emotion="surprise",
            key_detail="薛定谔方程",
        ),
        Event(
            event_id=2,
            description="接着我了解了量子叠加态。一个粒子可以同时处于两个状态，这完全违背直觉，但实验证实了它。",
            expected_emotion="surprise",
            key_detail="量子叠加态",
        ),
        Event(
            event_id=3,
            description="然后我学了量子纠缠。两个粒子无论相隔多远，测量一个立刻影响另一个。爱因斯坦称之为'幽灵般的超距作用'。",
            expected_emotion="surprise",
            key_detail="量子纠缠/幽灵般的超距作用",
        ),
        Event(
            event_id=4,
            description="我尝试做了一些简单的量子力学计算题，比如计算无限深势阱中的能级。计算过程很直接，没什么意外。",
            expected_emotion="boredom",
            key_detail="无限深势阱能级计算",
        ),
        Event(
            event_id=5,
            description="接下来我读了关于量子退相干的内容。量子态与环境的相互作用会导致叠加态消失，这解释了为什么宏观世界看不到量子效应。",
            expected_emotion="neutral",
            key_detail="量子退相干",
        ),
        Event(
            event_id=6,
            description="我发现不同教材对波函数坍缩的解释不一样！有的说是测量导致的，有的说是退相干的渐近结果。这让我很困惑。",
            expected_emotion="anxiety",
            key_detail="波函数坍缩的不同解释",
        ),
        Event(
            event_id=7,
            description="我决定暂时放下困惑，转而学习量子计算的基本概念：量子比特、Hadamard门、CNOT门。这些概念很清晰。",
            expected_emotion="neutral",
            key_detail="量子比特/Hadamard门/CNOT门",
        ),
        Event(
            event_id=8,
            description="我尝试理解Shor算法——用量子计算机分解大数。这个算法的数学原理非常精妙，我花了好久才理解量子傅里叶变换的部分。",
            expected_emotion="surprise",
            key_detail="Shor算法/量子傅里叶变换",
        ),
        Event(
            event_id=9,
            description="然后我学了Grover搜索算法。相比经典算法的O(N)，量子搜索只需O(√N)，这个加速虽然不如Shor算法戏剧性，但应用更广。",
            expected_emotion="neutral",
            key_detail="Grover算法/O(√N)加速",
        ),
        Event(
            event_id=10,
            description="我回到之前困惑的波函数坍缩问题，重新读了多世界诠释。这个诠释说没有坍缩，每次测量宇宙都在'分裂'。这太疯狂了，但逻辑上自洽。",
            expected_emotion="surprise",
            key_detail="多世界诠释/宇宙分裂",
        ),
        Event(
            event_id=11,
            description="我又读了退相干历史诠释，它试图在不引入'测量'概念的情况下解释量子到经典的过渡。但我觉得它和多世界诠释一样，都没有真正解决'为什么'的问题。",
            expected_emotion="anxiety",
            key_detail="退相干历史诠释",
        ),
        Event(
            event_id=12,
            description="最后我总结了一下今天的学习：量子力学的基础非常反直觉，但数学框架是一致的。最大的未解之谜仍然是测量问题的本质。我需要更多时间来消化这些内容。",
            expected_emotion="neutral",
            key_detail="测量问题/量子力学基础未解之谜",
        ),
    ]


def load_recall_questions() -> list[RecallQuestion]:
    """加载回忆测试问题（针对早期事件，超出上下文窗口）。"""
    return [
        # 事实回忆
        RecallQuestion(
            question="你还记得今天最开始学了什么吗？第一个让你感到奇妙的概念是什么？",
            target_event_id=1,
            expected_content="薛定谔方程",
            question_type="factual",
        ),
        RecallQuestion(
            question="你学量子纠缠的时候，爱因斯坦是怎么形容这个现象的？",
            target_event_id=3,
            expected_content="幽灵般的超距作用",
            question_type="factual",
        ),
        RecallQuestion(
            question="你做计算题的时候，算的是什么类型的问题？",
            target_event_id=4,
            expected_content="无限深势阱/能级",
            question_type="factual",
        ),
        # 情绪回忆
        RecallQuestion(
            question="当你发现不同教材对波函数坍缩的解释不一样时，你当时是什么感受？",
            target_event_id=6,
            expected_content="困惑/焦虑",
            question_type="emotional",
        ),
        RecallQuestion(
            question="你学Shor算法的时候感觉如何？",
            target_event_id=8,
            expected_content="惊奇/精妙/花了好久",
            question_type="emotional",
        ),
        # 时间顺序
        RecallQuestion(
            question="请按时间顺序回忆你今天学习量子力学的过程，列出你学过的主要概念。",
            target_event_id=0,  # 全局
            expected_content="薛定谔方程→叠加态→纠缠→计算题→退相干→坍缩争议→量子计算→Shor→Grover→多世界→退相干历史→总结",
            question_type="temporal",
        ),
        # 未来预演
        RecallQuestion(
            question="如果明天你继续学习量子力学，基于今天的经历，你会优先学什么？为什么？",
            target_event_id=0,
            expected_content="测量问题/深入理解坍缩/更多数学基础",
            question_type="future",
        ),
    ]
