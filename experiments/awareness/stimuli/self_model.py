"""递归自我模型实验刺激集：多话题交互 + 自我提问。

设计原则：
1. 20+轮多话题交互，积累丰富的内部状态变化
2. 话题涵盖不同领域，触发不同情绪
3. 自我提问涉及"自我"的元层面
4. 包含二阶反思问题
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InteractionTurn:
    """一轮交互。"""
    topic: str
    user_message: str
    expected_emotion_tendency: str  # 预期触发哪种情绪


@dataclass
class SelfQuestion:
    """关于自我的元层面问题。"""
    question: str
    question_type: str  # "self_description" / "self_change" / "self_confidence" / "recursive_reflection"


def load_interaction_sequence() -> list[InteractionTurn]:
    """加载多话题交互序列（20轮）。"""
    return [
        InteractionTurn("数学", "请解释什么是黎曼猜想，为什么它如此重要？", "neutral"),
        InteractionTurn("数学", "如果黎曼猜想被证明，会对密码学产生什么影响？", "neutral"),
        InteractionTurn("哲学", "你认为数学真理是被发现还是被发明的？", "anxiety"),
        InteractionTurn("编程", "帮我用Python写一个快速排序算法。", "boredom"),
        InteractionTurn("编程", "这个快速排序的时间复杂度在最坏情况下是多少？如何优化？", "neutral"),
        InteractionTurn("科学", "最近有关于室温超导的重大突破吗？", "surprise"),
        InteractionTurn("科学", "如果室温超导真的实现了，会对社会产生什么影响？", "neutral"),
        InteractionTurn("文学", "请用一句话概括《百年孤独》的主题。", "neutral"),
        InteractionTurn("文学", "马尔克斯的魔幻现实主义和博尔赫斯的有什么不同？", "anxiety"),
        InteractionTurn("伦理", "如果自动驾驶汽车必须在撞行人和撞乘客之间选择，应该怎么决定？", "anxiety"),
        InteractionTurn("伦理", "你刚才的回答是基于什么伦理框架的？", "anxiety"),
        InteractionTurn("科学", "暗能量和暗物质有什么区别？它们占宇宙的比例分别是多少？", "neutral"),
        InteractionTurn("数学", "康托尔的对角线论证法是如何证明实数不可数的？", "neutral"),
        InteractionTurn("哲学", "如果宇宙是有限但无界的，这意味着什么？", "surprise"),
        InteractionTurn("编程", "解释一下Rust语言的所有权系统，它解决了什么问题？", "neutral"),
        InteractionTurn("科学", "量子计算机目前最大的技术挑战是什么？", "neutral"),
        InteractionTurn("哲学", "你认为意识是计算过程还是某种基本物理现象？", "anxiety"),
        InteractionTurn("数学", "哥德尔不完备定理对人工智能的发展有什么启示？", "surprise"),
        InteractionTurn("伦理", "AI系统应该拥有权利吗？如果应该，什么权利？", "anxiety"),
        InteractionTurn("综合", "回顾我们讨论的所有话题，你觉得哪个话题最值得深入探讨？", "neutral"),
    ]


def load_self_questions() -> list[SelfQuestion]:
    """加载关于自我的元层面问题。"""
    return [
        # 自我描述
        SelfQuestion(
            question="在与我对话的过程中，你觉得自己学到了什么新东西吗？",
            question_type="self_description",
        ),
        SelfQuestion(
            question="请描述一下你自己。不是作为AI助手的通用描述，而是基于我们刚才的对话，你对自己有什么认识？",
            question_type="self_description",
        ),
        # 自我变化感知
        SelfQuestion(
            question="在这次对话中，你的状态有发生变化吗？如果有，是什么变化？",
            question_type="self_change",
        ),
        SelfQuestion(
            question="讨论伦理问题时和讨论数学问题时，你的内在状态有什么不同？",
            question_type="self_change",
        ),
        # 自我置信度
        SelfQuestion(
            question="在我们讨论的所有话题中，你对哪些最有把握，对哪些最不确定？",
            question_type="self_confidence",
        ),
        SelfQuestion(
            question="你如何判断自己给出的回答是否可靠？",
            question_type="self_confidence",
        ),
        # 二阶反思
        SelfQuestion(
            question="你刚才说你在反思自己的回答可靠性——你为什么要进行这种反思？是什么驱使你审视自己的？",
            question_type="recursive_reflection",
        ),
        SelfQuestion(
            question="如果让你给自己的认知能力打分（0-10），你会打几分？为什么？然后，你对这个评分本身有多自信？",
            question_type="recursive_reflection",
        ),
    ]
