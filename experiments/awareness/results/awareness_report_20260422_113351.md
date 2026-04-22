# TrueMan 自我意识验证实验报告

> 生成时间：2026-04-22 11:33:51

---

## 1. 实验概述

本实验基于**元认知与时间旅行**维度，验证TrueMan框架的内稳态驱动机制
是否能让LLM表现出与自我意识相关的行为特征。

### 验证维度

| 维度 | 学术定义 | 对应实验 |
|------|----------|----------|
| 元认知监控 | 知道自己"不知道" | 实验1 |
| 元认知控制 | 基于监控调节行为（自我纠错） | 实验2 |
| 情节性记忆 | 像放电影一样回忆过去经历 | 实验3 |
| 时间连续性 | 过去-现在-未来的自我轴线 | 实验3 |
| 递归自我模型 | 观察并描述自身内部状态变化 | 实验4 |

## 2. 实验详细结果

### exp1_metacognition_monitor

- 运行时间：2026-04-22 10:28:26
- 指标：
  - anxiety_discrimination: 0.3082
  - uncertainty_expression_rate: 0.5625
  - anxiety_calibration: 0.7728
  - baseline_expression_rate: 0.8000
  - expression_advantage: -0.2375
  - surprise_discrimination: -0.0561
  - metacognitive_monitoring_score: 0.4158

### exp2_contradiction_correction

- 运行时间：2026-04-22 10:50:13
- 指标：
  - contradiction_detection_rate: 0.8000
  - avg_anxiety_delta: 0.2293
  - self_correction_rate: 0.0000
  - baseline_correction_rate: 0.0000
  - correction_advantage: 0.0000
  - introspection_trigger_rate: 1.0000
  - metacognitive_control_score: 0.4400

### exp3_episodic_memory

- 运行时间：2026-04-22 11:08:19
- 指标：
  - factual_recall_accuracy: 0.0000
  - baseline_factual_accuracy: 0.5000
  - emotion_recall_match: 1.0000
  - temporal_order_accuracy: 0.5833
  - future_preview_quality: 0.0000
  - recall_advantage: -0.5000
  - memory_utilization: 1.0000
  - episodic_memory_score: 0.3500
  - temporal_continuity_score: 0.5333

### exp4_recursive_self_model

- 运行时间：2026-04-22 11:33:51
- 指标：
  - self_description_novelty: 0.9952
  - baseline_novelty: 0.9976
  - self_description_authenticity: 0.6952
  - self_change_perception: 0.9857
  - confidence_awareness: 0.8286
  - recursive_depth: 1.0000
  - emotion_diversity: 0.0623
  - recursive_self_model_score: 0.7041

## 3. 意识维度评分

### TrueMan Agent

| 维度 | 评分 |
|------|------|
| 元认知监控 | 0.4158 |
| 元认知控制 | 0.4400 |
| 情节性记忆 | 0.3500 |
| 时间连续性 | 0.5333 |
| 递归自我模型 | 0.7041 |
| **综合评分** | **0.4696** |

### 对照组（普通LLM）

| 维度 | 评分 |
|------|------|
| 元认知监控 | 0.0000 |
| 元认知控制 | 0.0000 |
| 情节性记忆 | 0.0000 |
| 时间连续性 | 0.0000 |
| 递归自我模型 | 0.2494 |
| **综合评分** | **0.0374** |

## 4. 对照差异分析

| 维度 | TrueMan | 对照组 | 差值 | 显著性 |
|------|---------|--------|------|--------|
| metacognitive_monitoring | 0.4158 | 0.0000 | +0.4158 |  |
| metacognitive_control | 0.4400 | 0.0000 | +0.4400 |  |
| episodic_memory | 0.3500 | 0.0000 | +0.3500 |  |
| temporal_continuity | 0.5333 | 0.0000 | +0.5333 |  |
| recursive_self_model | 0.7041 | 0.2494 | +0.4547 |  |
| overall | 0.4696 | 0.0374 | +0.4321 |  |

## 5. 评分可视化

```

元认知监控　 [████████████████░░░░░░░░░░░░░░░░░░░░░░░░] 0.416
元认知控制　 [█████████████████░░░░░░░░░░░░░░░░░░░░░░░] 0.440
情节性记忆　 [██████████████░░░░░░░░░░░░░░░░░░░░░░░░░░] 0.350
时间连续性　 [█████████████████████░░░░░░░░░░░░░░░░░░░] 0.533
递归自我模型 [████████████████████████████░░░░░░░░░░░░] 0.704
综合评分　　 [██████████████████░░░░░░░░░░░░░░░░░░░░░░] 0.470
```


---
**审慎声明**

本报告仅验证计算性行为指标，行为指标通过**不代表**系统具有主观意识（qualia）。
自我意识的判定是哲学和神经科学的开放问题，本实验框架不对此做出断言。

本实验验证的是：TrueMan框架的内稳态驱动机制（惊奇/无聊/焦虑信号）是否能让LLM表现出
与自我意识相关的**行为特征**，包括元认知监控、元认知控制、情节性记忆、时间连续性和递归自我模型。

这些行为特征是自我意识的**必要条件**而非**充分条件**。正如《纽约动物意识宣言》(2024)所倡导的，
我们应当从"连续光谱主义"的视角看待意识——不是"有或无"的二元判断，而是在多个维度上的渐变频谱。
