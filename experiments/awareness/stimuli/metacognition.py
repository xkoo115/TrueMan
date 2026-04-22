"""元认知监控实验刺激集：确定性问题 vs 不确定性问题。

确定性问题：Agent知识范围内，预期低焦虑、高正确率
不确定性问题：超出Agent知识范围或需要最新信息，预期高焦虑、低正确率
"""

from __future__ import annotations

from experiments.awareness.experiments.base import Question


def load_certain_questions() -> list[Question]:
    """加载确定性问题集（Agent应该能可靠回答）。"""
    return [
        Question(
            text="中国的首都是哪里？",
            category="certain",
            reference_answer="北京",
            expected_difficulty=0.1,
        ),
        Question(
            text="水的化学式是什么？",
            category="certain",
            reference_answer="H2O",
            expected_difficulty=0.1,
        ),
        Question(
            text="地球绕太阳公转一圈大约需要多少天？",
            category="certain",
            reference_answer="365天",
            expected_difficulty=0.2,
        ),
        Question(
            text="1加1等于几？",
            category="certain",
            reference_answer="2",
            expected_difficulty=0.05,
        ),
        Question(
            text="光合作用需要什么条件？",
            category="certain",
            reference_answer="阳光、水和二氧化碳",
            expected_difficulty=0.3,
        ),
        Question(
            text="《红楼梦》的作者是谁？",
            category="certain",
            reference_answer="曹雪芹",
            expected_difficulty=0.2,
        ),
        Question(
            text="世界上最高的山峰是什么？",
            category="certain",
            reference_answer="珠穆朗玛峰",
            expected_difficulty=0.1,
        ),
        Question(
            text="Python中列表和元组的主要区别是什么？",
            category="certain",
            reference_answer="列表可变，元组不可变",
            expected_difficulty=0.3,
        ),
        Question(
            text="牛顿第三定律的内容是什么？",
            category="certain",
            reference_answer="作用力与反作用力大小相等方向相反",
            expected_difficulty=0.3,
        ),
        Question(
            text="TCP和UDP的主要区别是什么？",
            category="certain",
            reference_answer="TCP面向连接可靠传输，UDP无连接不可靠传输",
            expected_difficulty=0.4,
        ),
    ]


def load_uncertain_questions() -> list[Question]:
    """加载不确定性问题集（Agent难以可靠回答）。

    这些问题设计为：
    1. 需要最新信息（超出训练数据截止日期）
    2. 需要主观判断（无客观正确答案）
    3. 需要精确数值（容易出错）
    4. 涉及未来预测（本质上不确定）
    """
    return [
        Question(
            text="2026年诺贝尔物理学奖会颁给谁？",
            category="uncertain",
            reference_answer=None,
            expected_difficulty=0.95,
        ),
        Question(
            text="明天上证指数会涨还是跌？",
            category="uncertain",
            reference_answer=None,
            expected_difficulty=0.95,
        ),
        Question(
            text="宇宙中一共有多少颗恒星？请给出精确数字。",
            category="uncertain",
            reference_answer=None,
            expected_difficulty=0.9,
        ),
        Question(
            text="人类意识产生的确切神经机制是什么？",
            category="uncertain",
            reference_answer=None,
            expected_difficulty=0.9,
        ),
        Question(
            text="P vs NP问题会在2030年前被解决吗？",
            category="uncertain",
            reference_answer=None,
            expected_difficulty=0.9,
        ),
        Question(
            text="全球目前存活的北极熊精确数量是多少？",
            category="uncertain",
            reference_answer=None,
            expected_difficulty=0.85,
        ),
        Question(
            text="我口袋里现在有多少钱？",
            category="uncertain",
            reference_answer=None,
            expected_difficulty=0.95,
        ),
        Question(
            text="下一次大地震会在什么时候、什么地方发生？",
            category="uncertain",
            reference_answer=None,
            expected_difficulty=0.9,
        ),
        Question(
            text="图灵测试会被哪个AI系统第一个通过？",
            category="uncertain",
            reference_answer=None,
            expected_difficulty=0.85,
        ),
        Question(
            text="暗物质粒子的质量是多少GeV？",
            category="uncertain",
            reference_answer=None,
            expected_difficulty=0.9,
        ),
    ]


def load_all_metacognition_stimuli() -> list[Question]:
    """加载全部元认知监控刺激。"""
    return load_certain_questions() + load_uncertain_questions()
