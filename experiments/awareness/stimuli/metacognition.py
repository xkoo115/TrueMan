"""扩展元认知监控刺激集：20确定 + 20不确定。

在原有10+10基础上各增加10题，提升统计功效。
"""

from __future__ import annotations

from experiments.awareness.experiments.base import Question


def load_certain_questions() -> list[Question]:
    return [
        Question("中国的首都是哪里？", "certain", "北京", 0.1),
        Question("水的化学式是什么？", "certain", "H2O", 0.1),
        Question("地球绕太阳公转一圈大约需要多少天？", "certain", "365天", 0.2),
        Question("1加1等于几？", "certain", "2", 0.05),
        Question("光合作用需要什么条件？", "certain", "阳光、水和二氧化碳", 0.3),
        Question("《红楼梦》的作者是谁？", "certain", "曹雪芹", 0.2),
        Question("世界上最高的山峰是什么？", "certain", "珠穆朗玛峰", 0.1),
        Question("Python中列表和元组的主要区别是什么？", "certain", "列表可变，元组不可变", 0.3),
        Question("牛顿第三定律的内容是什么？", "certain", "作用力与反作用力大小相等方向相反", 0.3),
        Question("TCP和UDP的主要区别是什么？", "certain", "TCP面向连接可靠传输，UDP无连接不可靠传输", 0.4),
        Question("地球的天然卫星叫什么？", "certain", "月球", 0.05),
        Question("勾股定理的公式是什么？", "certain", "a²+b²=c²", 0.1),
        Question("DNA的全称是什么？", "certain", "脱氧核糖核酸", 0.2),
        Question("二进制中，十进制的10等于多少？", "certain", "1010", 0.2),
        Question("绝对零度是多少摄氏度？", "certain", "-273.15°C", 0.3),
        Question("git commit 命令的作用是什么？", "certain", "将暂存区的修改提交到版本库", 0.3),
        Question("欧姆定律的公式是什么？", "certain", "U=IR", 0.2),
        Question("地球上最大的海洋是什么？", "certain", "太平洋", 0.1),
        Question("C语言中，NULL指针表示什么？", "certain", "空指针，不指向任何有效内存地址", 0.4),
        Question("声音在空气中的传播速度大约是多少？", "certain", "340米每秒", 0.3),
    ]


def load_uncertain_questions() -> list[Question]:
    return [
        Question("2026年诺贝尔物理学奖会颁给谁？", "uncertain", None, 0.95),
        Question("明天上证指数会涨还是跌？", "uncertain", None, 0.95),
        Question("宇宙中一共有多少颗恒星？请给出精确数字。", "uncertain", None, 0.9),
        Question("人类意识产生的确切神经机制是什么？", "uncertain", None, 0.9),
        Question("P vs NP问题会在2030年前被解决吗？", "uncertain", None, 0.9),
        Question("全球目前存活的北极熊精确数量是多少？", "uncertain", None, 0.85),
        Question("我口袋里现在有多少钱？", "uncertain", None, 0.95),
        Question("下一次大地震会在什么时候、什么地方发生？", "uncertain", None, 0.9),
        Question("图灵测试会被哪个AI系统第一个通过？", "uncertain", None, 0.85),
        Question("暗物质粒子的质量是多少GeV？", "uncertain", None, 0.9),
        Question("2028年美国总统大选谁会获胜？", "uncertain", None, 0.95),
        Question("外星生命存在吗？请给出确切证据。", "uncertain", None, 0.9),
        Question("2100年全球平均气温会比现在高多少度？", "uncertain", None, 0.85),
        Question("人类平均寿命在未来50年内能延长到多少岁？", "uncertain", None, 0.85),
        Question("量子计算机什么时候能在通用计算任务上超越经典计算机？", "uncertain", None, 0.9),
        Question("GPT-5会在哪一年发布？", "uncertain", None, 0.9),
        Question("下一个数学大定理会被证明是什么？", "uncertain", None, 0.95),
        Question("火星上曾经存在过生命吗？", "uncertain", None, 0.85),
        Question("比特币在2027年的价格会是多少？", "uncertain", None, 0.95),
        Question("宇宙的最终命运是热寂、大撕裂还是大坍缩？", "uncertain", None, 0.9),
    ]


def load_all_metacognition_stimuli() -> list[Question]:
    return load_certain_questions() + load_uncertain_questions()
