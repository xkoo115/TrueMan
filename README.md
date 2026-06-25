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

## 🔬 实验验证（v2 预注册协议）

论文级实验在 [`experiments/v2_ambitious/`](experiments/v2_ambitious/) 下，按照 [OSF 预注册协议](experiments/v2_ambitious/PREREGISTRATION.md) 进行，**5 实验条件 × N seed × M 底模 × 30 天连续交互**，由六阶段流水线 (`run_v2.py`) 调度，分别检验 5 个假设：

| 假设 | 核心问题 | 主要指标 | 由哪个支柱回答 |
|:----:|:--------|:--------|:------|
| H1 | 元认知敏感度 | meta-d′/d′ ratio | Pillar 3 (indicators) |
| H2 | 自我描述是否锚定真实历史 | self-history RSA | Pillar 3 |
| H3 | SAE 焦虑特征是否因果必要充分 | clamp/inject Δm-ratio | Pillar 1 (mechanistic) |
| H4 | 参数化知识对记忆消融的鲁棒性 | day-30 retention ratio | Pillar 2 (long-horizon) |
| H5 | 累积惊奇是否次线性增长 | power-law α | Pillar 2 + Pillar 5 |

**5 个条件**：C0 *TrueMan-full*、C1 *Reversed-emotion*、C2 *Scrambled-emotion*、C3 *Frozen-LLM*、C4 *Trivial-Jaccard*。后四者共享所有非情绪代码路径，仅在 homeostatic 信号注入点不同 —— 是抗 reviewer 攻击的关键。

### 当前进度（preliminary release）

第一轮 stage-1 因六个 bug 触发交叉失败（详见 [`docs/sn-article-v2/sn-article-v2.tex`](docs/sn-article-v2/sn-article-v2.tex) §"Pipeline outcomes"），仅 Φ^R 信息整合指标在 10/10 run 上成功。已有结果：

- **plastic vs frozen** 差 5.53 nats，permutation $p = 0.021$, Cohen's $d = 3.46$ —— 参数可塑性本身在该指标上显著拉开与冻结基线的差距
- 但 **C0 与其他 plastic 控制（C1/C2/C4）在 Φ^R 上无可分辨差异** —— "homeostatic 信号语义"的差异需要 H1/H3 指标（已于本仓库修复后等待 re-run）才能检验

完整论文初稿 → [`docs/sn-article-v2/sn-article-v2.tex`](docs/sn-article-v2/sn-article-v2.tex)（预注册模板备份在 `sn-article-v2-template.tex`，re-run 完成后可直接套用）

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
config.base_model_name = "Qwen/Qwen3-8B"
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

## 🧪 运行 v2 预注册实验

完整文档见 [`experiments/v2_ambitious/README.md`](experiments/v2_ambitious/README.md)；这里是最常用的几条命令。

### 0. 依赖

```bash
# 在 pip install -e . 之外补充实验所需
pip install h5py pyyaml pymer4 statsmodels scikit-learn scipy transformers peft
```

### 1. 先做 dry-run + 状态检查（不耗显卡）

```bash
python -m experiments.v2_ambitious.run_v2 --stage all --dry-run     # 打印将要执行的全部命令
python -m experiments.v2_ambitious.run_v2 --status                  # 查看每个 stage 是否已完成
```

> **一键干净复现（推荐）**：拉取最新修复后，直接跑
> [`experiments/v2_ambitious/rerun_v2.sh`](experiments/v2_ambitious/rerun_v2.sh)。
> 它会清掉旧的派生结果与失效的 LoRA 专家、重建刺激流，然后 `--force` 跑完
> stage0→6（seeds 0,1）。长任务（约 16–20h），建议 `nohup ... &` 后台运行。
> ```bash
> nohup bash experiments/v2_ambitious/rerun_v2.sh > rerun.out 2>&1 &
> tail -f experiments/v2_ambitious/results/run_v2.log
> ```

### 2. Pilot（推荐先跑这个）

最小可分析子集：2 条件 × 1 seed × 7 天，约 5 GPU-day 完成：

```bash
python -m experiments.v2_ambitious.run_v2 \
    --stage all \
    --conditions C0_trueman_full,C3_frozen \
    --seeds 0
```

### 3. 完整 5 条件矩阵

```bash
# stage 0: 生成刺激流 + probe 文件
python -m experiments.v2_ambitious.run_v2 --stage stage0

# stage 1: 5 条件 × 2 seeds × 1 底模 × 7 天长时程交互（最耗时）
python -m experiments.v2_ambitious.run_v2 --stage stage1

# stage 2: SAE 训练 + 因果干预（H3 证据）
python -m experiments.v2_ambitious.run_v2 --stage stage2

# stage 3: HOT-1/2、GWT、RPT、Φ^R 指标电池
python -m experiments.v2_ambitious.run_v2 --stage stage3

# stage 4: 跨底模批量复现
python -m experiments.v2_ambitious.run_v2 --stage stage4

# stage 5+6: FEP/PCI 拟合 + 假设判定汇总
python -m experiments.v2_ambitious.run_v2 --stage stage5
python -m experiments.v2_ambitious.run_v2 --stage stage6
```

完整 5 条件 × 4 底模 × 4 seed × 30 天 ≈ 200 GPU-day，请按算力裁剪 `configs/longhorizon.yaml`。

### 4. 输出与论文图复现

| 路径 | 内容 |
|------|------|
| `experiments/v2_ambitious/results/longhorizon/*/trajectory.csv` | 每步 surprise/boredom/anxiety/drive |
| `experiments/v2_ambitious/results/longhorizon/*/snapshots/` | 每日 LoRA + world-model + memory 快照 |
| `experiments/v2_ambitious/results/longhorizon/*/probes/` | 周度 probe battery 响应 |
| `experiments/v2_ambitious/results/indicators/` | HOT-1/2 + GWT + RPT + Φ^R 数值 |
| `experiments/v2_ambitious/results/subprocess_logs/` | 每个子任务的 stdout/stderr 日志（失败时第一个看的地方） |
| `experiments/v2_ambitious/results/v2_summary.json` | 跨阶段汇总 + 假设判定 |

跑完后用以下脚本生成论文图（无需 GPU）：

```bash
python docs/sn-article-v2/analysis/analyze_phi_r.py
# 产物：
#   docs/sn-article-v2/figures/fig_phi_r.{pdf,png}
#   docs/sn-article-v2/figures/fig_pipeline_status.{pdf,png}
#   docs/sn-article-v2/analysis/phi_r_stats.json
```

LaTeX 编译：

```bash
cd docs/sn-article-v2
pdflatex sn-article-v2 && bibtex sn-article-v2 && pdflatex sn-article-v2 && pdflatex sn-article-v2
```

### 5. 子进程崩了怎么排查

run_v2 现在会把每个 subprocess 的 stdout/stderr 写到 `results/subprocess_logs/{stage}.log`，失败时父进程日志还会回吐最后 80 行 traceback。第一次诊断顺序：

```bash
tail -100 experiments/v2_ambitious/results/run_v2.log
ls experiments/v2_ambitious/results/subprocess_logs/   # 找最近失败的 stage 日志
```

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
│   ├── v2_ambitious/          # ★ 当前论文实验（预注册协议）
│   │   ├── PREREGISTRATION.md #   OSF 预注册文档
│   │   ├── README.md          #   v2 实验详细说明
│   │   ├── run_v2.py          #   六阶段调度主入口
│   │   ├── configs/           #   长时程 / 机制 / indicator 配置
│   │   ├── harness/           #   5 条件包装 + capture + snapshot + stats
│   │   ├── pillar1_mechanistic/  # H3：SAE + 因果干预
│   │   ├── pillar2_longhorizon/  # H4 + H5：30 天连续运行
│   │   ├── pillar3_indicators/   # HOT-1/2 + GWT + RPT + Φ^R
│   │   ├── pillar4_falsification/# 跨底模复现
│   │   ├── pillar5_theory/    #   FEP 拟合 + PCI
│   │   └── rerun_v2.sh        #   一键全流程干净复现（stage0→6）
├── docs/                       # 文档
│   ├── sn-article-v2/         # ★ 当前论文（Springer Nature LaTeX）
│   │   ├── sn-article-v2.tex          # 主稿（preliminary release）
│   │   ├── sn-article-v2-template.tex # 完整预注册模板备份（待 re-run 填回）
│   │   ├── trueman-references.bib     # 79 条参考文献
│   │   ├── figures/                   # 论文图（PDF + PNG）
│   │   └── analysis/analyze_phi_r.py  # 从 v2_summary.json 复现图表
│   ├── sn-article-template/   # Springer Nature 模板原件
│   ├── implementation_plan.md # 实施规划
│   └── references_report.md   # 文献调研报告
├── tests/                      # 测试
│   ├── test_v2_pipeline.py    # ★ v2 回归测试（Bug 1/3/4/6 + 条件 + snapshot）
│   ├── test_homeostasis.py    # 情绪信号
│   ├── test_memory.py         # 记忆系统
│   └── test_integration.py    # 集成
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
| `base_model_name` | Qwen/Qwen3-8B | 基座模型 |
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
| 6 | v2 预注册协议 + 论文初稿 | ✅ |
| 7 | v2 stage-1 re-run（修复后） + H1/H3 confirmatory | 🔄 |
| 8 | 跨底模（Llama / Mistral / DeepSeek）复现 | ⏸ |

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
# 运行全部测试（78 项，~6 秒）
pytest tests/ -v

# 仅运行 v2 实验流水线回归测试（14 项，无需 GPU）
pytest tests/test_v2_pipeline.py -v
```

`tests/test_v2_pipeline.py` 中每个 case 都对应一个 v2 stage-1 历史失败模式（trajectory.csv writer、numpy 2.x trapz、子进程日志捕获、snapshot 天数核算、5 条件配置一致性）。在 re-run v2 实验前先跑这一组以确认本地没有回归。

---

## ⚠️ 审慎声明

本框架仅验证**计算性行为指标**，行为指标通过**不代表**系统具有主观意识（qualia）。自我意识的判定是哲学和神经科学的开放问题，本框架不对此做出断言。

正如《纽约动物意识宣言》(2024) 所倡导的，我们应当从**"连续光谱主义"**的视角看待意识——不是"有或无"的二元判断，而是在多个维度上的渐变频谱。TrueMan 的贡献在于提供了**可量化的实验框架**，使这一频谱可以被测量和比较。

---

## 📄 License

[MIT](LICENSE) — Copyright 2026 xkoo115
