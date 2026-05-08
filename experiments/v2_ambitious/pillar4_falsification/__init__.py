"""支柱 4：可证伪性 + 跨模型复现。

- reversed_emotion.py:    检验"情绪反转应使指标反向"
- trivial_jaccard.py:     检验仅用文本 Jaccard 是否能复现 TrueMan 表现
- cross_model.py:         在 4 个底模上复跑 H1-H3，报告 effect 一致性

支柱 4 的所有结果都直接进入论文 Reviewer-defensive 章节。
"""
