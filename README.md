# TrueMan

**实时可塑性与内稳态驱动的自治 AI Agent**

## 核心理念

放弃为 AI Agent 设定硬编码的外部 Loss，转而植入基于**内稳态（Homeostasis）**的底层生存指标。通过三种"情绪"信号驱动自发学习与演化：

- **惊奇 (Surprise)**: 预测误差驱动认知修正
- **无聊 (Boredom)**: 信息熵衰减驱动好奇心探索
- **焦虑 (Anxiety)**: 认知失调驱动自我纠错与睡眠整合

## 架构概览

```
L0: 内稳态内核 — 情绪信号生成与整合
L1: 感知执行层 — 快速权重驱动的实时响应 (System 1)
L2: 反思缓存层 — Thought Traces 短期记忆与反思
L3: 后台整合层 — 事件驱动的异步 LoRA 微调
L4: 塑性存储层 — 动态 LoRA 专家池热加载
```

## 文档

| 文档 | 说明 |
|------|------|
| [docs/references_report.md](docs/references_report.md) | 前沿研究参考文献报告（60+ 篇论文，2019-2026） |
| [docs/implementation_plan.md](docs/implementation_plan.md) | 分阶段系统实施规划（6 个 Phase，24 周） |

## 实施路线

| Phase | 内容 | 周期 |
|-------|------|------|
| 0 | 基础设施搭建 | Week 1-2 |
| 1 | 惊奇信号驱动学习 | Week 3-6 |
| 2 | 无聊信号与好奇心探索 | Week 7-9 |
| 3 | 焦虑信号与自我纠错 | Week 10-12 |
| 4 | 动态 LoRA 可塑性系统 | Week 13-16 |
| 5 | 情绪整合与 Agent 自治 | Week 17-20 |
| 6 | 高级机制与优化 | Week 21-24 |

## 关键理论基石

- **自由能原理** (Karl Friston) — 惊奇的数学基础
- **躯体标记假说** (Antonio Damasio) — 情绪作为内稳态调节器
- **快速权重编程** (Schmidhuber / Irie) — 实时可塑性机制
- **互补学习系统** — 海马-新皮层双记忆模型
- **持续反向传播** (Sutton) — 塑性保持

## 技术栈

- PyTorch + HuggingFace Transformers
- PEFT / 自研动态稀疏 LoRA
- Gymnasium (Atari) → 自定义环境
- Hydra + OmegaConf 配置管理
