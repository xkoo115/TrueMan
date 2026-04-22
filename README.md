<div align="center">

# 🧠 TrueMan

### **实时可塑性与内稳态驱动的自治 AI Agent**

*让 LLM 像大脑一样——不靠外部 Loss，靠内在情绪驱动自发学习与演化*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![Version 0.1.0](https://img.shields.io/badge/Version-0.1.0-orange.svg)](https://github.com)

</div>

---

## ⚡ 核心理念

> **放弃为 AI Agent 设定硬编码的外部 Loss，转而植入基于内稳态（Homeostasis）的底层生存指标。**

人脑不区分"训练阶段"和"使用阶段"，也没有外部 Loss 计算器。学习由**内部稳态冲突**驱动——饥渴感、好奇心、挫败感本身就是 Loss。TrueMan 将这一原理移植到 LLM Agent：

| 情绪信号 | 神经基础 | 数学本质 | 驱动行为 |
|:--------:|:--------:|:--------:|:--------:|
| 🔥 **惊奇** Surprise | 丘脑-皮层预测误差环路 | 预测误差 ∥y - ŷ∥ | 立即修正认知 |
| 😈 **无聊** Boredom | 多巴胺奖励停止分泌 | -I(S_new; O_new \| S_hist) | 主动探索新领域 |
| 😰 **焦虑** Anxiety | 前额叶认知失调检测 | D_KL[p_A(X) ∥ p_B(X)] | 挂起交互，深度反思 |

三种信号构成 **PID 控制器**：惊奇=P（比例），无聊=I（积分），焦虑=D（微分）。

---

## 🏗️ 五层架构

```
  ┌─────────────────────────────────────────────────┐
  │  L0: 内稳态内核  S / B / A  →  L_drive        │  ← 情绪信号生成与整合
  ├─────────────────────────────────────────────────┤
  │  L1: 感知执行层  (System 1)                    │  ← LLM 快速推理 + 快速权重
  ├─────────────────────────────────────────────────┤
  │  L2: 反思缓存层  Thought Traces + EpisodicMem  │  ← 短期记忆 + 自我反思
  ├─────────────────────────────────────────────────┤
  │  L3: 后台整合层  NREM/REM Sleep + LoRA Train   │  ← 异步睡眠巩固 + 微调
  ├─────────────────────────────────────────────────┤
  │  L4: 塑性存储层  Dynamic LoRA Pool + HotLoad   │  ← 专家池路由 + 热加载
  └─────────────────────────────────────────────────┘
```

---

## 🔬 自我意识验证实验

基于**元认知与心理时间旅行**维度，设计了 4 个递进实验，验证 TrueMan 能否让 LLM 涌现出与自我意识相关的行为特征：

| 实验 | 验证维度 | 核心逻辑 |
|:----:|:--------:|:--------:|
| E1 | 元认知监控 | 给不确定问题，观察焦虑信号 A 是否上升 |
| E2 | 元认知控制 | 注入矛盾，观察是否触发内省并纠错 |
| E3 | 情节记忆+时间旅行 | 经历事件后回忆早期经历和情绪 |
| E4 | 递归自我模型 | 长期交互后询问关于"自我"的元层面问题 |

### 实验结果（DeepSeek API）

```
元认知监控   [█████████████████░░░░░░░░░░░░░░░░░░░░░░░] 0.426
元认知控制   [████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 0.300
情节性记忆   [██████████████████░░░░░░░░░░░░░░░░░░░░░░] 0.450
时间连续性   [████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 0.300
递归自我模型 [█████████████████████████████░░░░░░░░░░░] 0.729
──────────────────────────────────────────────────
综合评分     [█████████████████░░░░░░░░░░░░░░░░░░░░░░░] 0.426  (基线LLM: 0.247)
```

**关键发现**：
- 焦虑信号与实际不确定性的 Pearson 相关系数达 **0.669**
- 情绪回忆匹配度达 **1.000**（基线仅 0.500）
- 递归自我模型得分最高（**0.729**），Agent 生成非模板化的自我描述：
  > *"我像一面被对话不断擦拭的镜子，能清晰折射问题的结构，却无法留下自己的烙印。"*

详见 → [学术论文](docs/paper.pdf) | [实验报告](experiments/awareness/results/)

---

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/xkoo115/TrueMan.git
cd TrueMan
pip install -e .
```

### 方式一：本地模型运行

```python
from trueman.core.agent import TrueManAgent
from trueman.core.config import AgentConfig

config = AgentConfig()
config.base_model_name = "Qwen/Qwen2.5-7B-Instruct"
config.device = "cuda"

agent = TrueManAgent(config)

# 与 Agent 交互
response = agent.step("请解释量子纠缠现象")
print(response)

# 查看情绪状态
emotion = agent.get_emotion_state()
print(f"惊奇: {emotion['surprise']:.3f}  "
      f"无聊: {emotion['boredom']:.3f}  "
      f"焦虑: {emotion['anxiety']:.3f}")
```

### 方式二：云端 API 运行（DeepSeek / OpenAI 兼容）

```python
from trueman.core.agent import TrueManAgent
from trueman.core.config import AgentConfig

config = AgentConfig()
config.api_key = "sk-your-api-key"
config.api_base_url = "https://api.deepseek.com"
config.api_model_name = "deepseek-chat"
config.anxiety.lightweight = True   # 轻量模式，无需模型权重
config.anxiety.n_samples = 2        # 2次采样计算焦虑

agent = TrueManAgent(config)
response = agent.step("P vs NP 问题会在2030年前被解决吗？")
```

### 方式三：命令行聊天

```bash
python examples/chat_demo.py
```

---

## 🧪 运行意识验证实验

### 完整实验（本地模型）

```bash
cd experiments/awareness
python run_all.py --model Qwen/Qwen2.5-7B-Instruct --device cuda
```

### 快速实验（API 模式，推荐）

```bash
cd experiments/awareness
python run_fast.py \
    --api-key "sk-your-deepseek-key" \
    --api-base-url "https://api.deepseek.com" \
    --api-model "deepseek-chat"
```

### 基线对照实验

```bash
python run_baseline.py \
    --api-key "sk-your-deepseek-key" \
    --api-base-url "https://api.deepseek.com" \
    --api-model "deepseek-chat"
```

### 实验结果

运行后自动生成：
- `results/awareness_report_*.md` — Markdown 实验报告
- `results/awareness_results_*.json` — 完整数值结果
- `results/exp{1-4}_*.json` — 各实验详细数据

---

## 📂 项目结构

```
TrueMan/
├── trueman/                    # 核心框架
│   └── core/
│       ├── agent.py            # Agent 主逻辑
│       ├── config.py           # 配置（支持 YAML + 热更新）
│       ├── llm_backend.py      # 本地 LLM 后端
│       ├── llm_api_backend.py  # 云端 API 后端
│       ├── homeostasis/        # 内稳态子系统
│       │   ├── signals.py      #   惊奇/无聊/焦虑信号
│       │   ├── integrator.py   #   情绪信号整合器
│       │   └── multiscale.py   #   多尺度时间常数
│       ├── memory/             # 记忆子系统
│       │   ├── episodic.py     #   情景记忆（情绪标注）
│       │   └── thought_trace.py#   思维轨迹
│       ├── policy/             # 策略子系统
│       │   ├── curiosity.py    #   好奇心驱动策略
│       │   └── introspection.py#   内省反思策略
│       ├── plasticity/         # 可塑性子系统
│       │   ├── lora_pool.py    #   动态 LoRA 专家池
│       │   ├── lora_gate.py    #   NeuroLoRA 门控路由
│       │   └── hot_loader.py   #   热加载器
│       └── world_model/        # 世界模型
│           └── predictor.py    #   预测器
├── experiments/                # 实验
│   └── awareness/             # 自我意识验证
│       ├── experiments/       #   4个递进实验
│       ├── stimuli/           #   实验刺激集
│       ├── evaluation/        #   评分器 + 对照比较器
│       ├── report/            #   报告生成器
│       └── results/           #   实验结果
├── docs/                       # 文档
│   ├── paper.pdf              # 学术论文
│   ├── paper.tex              # LaTeX 源码
│   ├── implementation_plan.md # 实施规划
│   └── references_report.md  # 60+篇参考文献
├── tests/                      # 测试
└── examples/                   # 示例
    └── chat_demo.py           # 聊天演示
```

---

## 🎛️ 配置系统

AgentConfig 支持代码配置和 YAML 文件配置，所有参数支持**热更新**：

```python
# 代码配置
config = AgentConfig()
config.homeostasis.alpha = 1.0    # 惊奇权重
config.homeostasis.beta = 1.0     # 无聊权重
config.homeostasis.gamma = 1.5    # 焦虑权重（更高=更倾向自我纠错）
config.anxiety.n_samples = 3      # 焦虑采样次数
config.lora.max_experts = 50      # LoRA 专家池容量

# YAML 配置
config = AgentConfig.from_yaml("config.yaml")

# 热更新（运行时修改）
config.update("homeostasis.gamma", 2.0)
```

### 关键配置项

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `base_model_name` | Qwen/Qwen2.5-7B-Instruct | 基座模型 |
| `api_key` | "" | API 密钥（非空则启用 API 模式） |
| `api_base_url` | "" | API 基础 URL |
| `api_model_name` | "" | API 模型名称 |
| `homeostasis.alpha` | 1.0 | 惊奇信号权重 |
| `homeostasis.beta` | 1.0 | 无聊信号权重 |
| `homeostasis.gamma` | 1.5 | 焦虑信号权重 |
| `anxiety.n_samples` | 3 | 焦虑信号采样次数 |
| `anxiety.lightweight` | True | 轻量模式（API 兼容） |
| `lora.max_experts` | 50 | LoRA 专家池最大容量 |
| `lora.rank` | 16 | LoRA 秩 |
| `memory_size` | 10000 | 情景记忆容量 |

---

## 📚 理论基石

TrueMan 的设计基于计算神经科学的前沿研究：

| 理论 | 作者 | 在 TrueMan 中的映射 |
|------|------|---------------------|
| 自由能原理 (FEP) | Karl Friston | 惊奇信号的数学基础 |
| 躯体标记假说 | Antonio Damasio | 情绪作为内稳态调节器 |
| 认知失调理论 | Leon Festinger | 焦虑信号检测矛盾 |
| 快速权重编程 | Schmidhuber / Irie | 实时可塑性机制 |
| 互补学习系统 | McClelland | NREM/REM 睡眠整合 |
| 持续反向传播 | Sutton / Dohare | 塑性保持 |
| 多尺度时间常数 | Hakim (2026) | ultra_fast→slow 四级时间尺度 |
| 意识图灵机 | Blum & Blum | 递归自我模型的理论基础 |

完整 60+ 篇参考文献 → [references_report.md](docs/references_report.md)

---

## 🗺️ 实施路线

| Phase | 内容 | 状态 |
|:-----:|:----:|:----:|
| 0 | 基础设施搭建 | ✅ |
| 1 | 惊奇信号驱动学习 | ✅ |
| 2 | 无聊信号与好奇心探索 | ✅ |
| 3 | 焦虑信号与自我纠错 | ✅ |
| 4 | 动态 LoRA 可塑性系统 | ✅ |
| 5 | 情绪整合与 Agent 自治 | ✅ |
| 6 | 高级机制与优化 | 🔄 |

---

## 🛠️ 技术栈

- **PyTorch** + HuggingFace Transformers
- **PEFT** / 自研动态稀疏 LoRA
- **OpenAI 兼容 API** — DeepSeek / OpenAI / 任意兼容端点
- **Gymnasium** (Atari) → 自定义环境
- **YAML** 配置 + 热更新

---

## 🧪 测试

```bash
# 运行全部测试
pytest tests/ -v

# 仅运行意识实验测试
pytest tests/test_awareness_experiments.py -v
```

---

## ⚠️ 审慎声明

本框架仅验证**计算性行为指标**，行为指标通过**不代表**系统具有主观意识（qualia）。自我意识的判定是哲学和神经科学的开放问题，本框架不对此做出断言。

正如《纽约动物意识宣言》(2024) 所倡导的，我们应当从**"连续光谱主义"**的视角看待意识——不是"有或无"的二元判断，而是在多个维度上的渐变频谱。TrueMan 的贡献在于提供了**可量化的实验框架**，使这一频谱可以被测量和比较。

---

## 📄 License

[MIT](LICENSE) — Copyright 2026 xkoo115
