# 实时可塑性与内稳态驱动自治 AI Agent — 前沿研究参考文献报告

> 调研时间：2026-04-21
> 调研范围：2019–2026 前沿论文，重点关注 NCS (Nature/Cell/Science) 系期刊及顶会 (NeurIPS/ICML/ICLR)

---

## 目录

1. [自由能原理与主动推断](#1-自由能原理与主动推断)
2. [内稳态驱动 AI 架构](#2-内稳态驱动-ai-架构)
3. [快速权重与实时可塑性](#3-快速权重与实时可塑性)
4. [塑性丧失与持续学习](#4-塑性丧失与持续学习)
5. [动态 LoRA 与持续适配](#5-动态-lora-与持续适配)
6. [互补学习系统与睡眠整合](#6-互补学习系统与睡眠整合)
7. [好奇心与惊奇驱动探索](#7-好奇心与惊奇驱动探索)
8. [信息增益与不确定性驱动](#8-信息增益与不确定性驱动)
9. [意识与元认知架构](#9-意识与元认知架构)
10. [自主 LLM Agent 与内在动机](#10-自主-llm-agent-与内在动机)
11. [关键研究者图谱](#11-关键研究者图谱)
12. [与架构构想的映射关系](#12-与架构构想的映射关系)

---

## 1. 自由能原理与主动推断

### 核心理论

| # | 论文 | 作者 | 出处 | 年份 | 核心贡献 |
|---|------|------|------|------|----------|
| 1.1 | *Active Inference: The Free Energy Principle in Mind, Brain, and Behaviour* | Parr, Pezzulo, Friston | MIT Press (书籍) | 2022 | 权威教材：将感知、行动、学习统一为自由能最小化。惊奇 = 预测误差 = 偏离内稳态 |
| 1.2 | *From Pixels to Planning: Scale-Free Active Inference* | Friston, Heins, Verbelen, Da Costa et al. | arXiv | 2024 | 单一 FEP 架构从像素到多步规划的端到端自主学习 (Atari) |
| 1.3 | *Active Inference and Artificial Reasoning* | Friston, Da Costa, Tschantz, Heins, Buckley, Verbelen, Parr | arXiv | 2025 | Agent 主动选择实验消除假设歧义，实现结构学习与"顿悟时刻" |
| 1.4 | *AXIOM: Learning to Play Games in Minutes with Expanding Object-Centric Models* | Heins, Van de Maele, Tschantz, Friston, Verbelen, Buckley et al. | arXiv | 2025 | ~10,000 步掌握 Atari，无外部奖励，纯惊奇驱动，贝叶斯模型约简在线扩展生成模型 |
| 1.5 | *Meta-Representational Predictive Coding: Biomimetic Self-Supervised Learning* | Ororbia, Friston, Rao | arXiv | 2025 | 无反向传播、无标签的生物可信自监督学习，纯预测误差驱动 |
| 1.6 | *Self-Orthogonalizing Attractor Neural Networks Emerging from the Free Energy Principle* | Spisak, Friston | Neurocomputing | 2025 | 证明吸引子网络从 FEP 自然涌现，"情绪状态"（吸引子盆地）从惊奇最小化自组织形成 |
| 1.7 | *Brain in the Dark: Design Principles for Neuromimetic Inference under FEP* | Bazargani, Urbas, Friston | arXiv | 2025 | FEP 预测编码网络的 PyTorch 实践路线图 |

### 预测编码

| # | 论文 | 作者 | 出处 | 年份 | 核心贡献 |
|---|------|------|------|------|----------|
| 1.8 | *Bayesian Predictive Coding* | Tschantz, Koudahl, Linander, Da Costa, Heins, Beck, Buckley | arXiv | 2025 | 从 MAP 估计扩展到全贝叶斯后验，保留局部性（Hebbian 更新）+ 不确定性量化 |
| 1.9 | *Bridging Predictive Coding and MDL: A Two-Part Code Framework for Deep Learning* | Prada, Matsumoto, Zekri, Mali | arXiv | 2025 | 首次形式化证明：预测编码 = 对 MDL 目标做块坐标下降；惊奇最小化保证最优泛化 |
| 1.10 | *Future-Guided Learning* | Gunasekaran et al., Eshraghian | **Nature Communications** | 2025 | 预测编码反馈用于癫痫预测，AUC-ROC 提升 44.8% |

---

## 2. 内稳态驱动 AI 架构

| # | 论文 | 作者 | 出处 | 年份 | 核心贡献 |
|---|------|------|------|------|----------|
| 2.1 | *Linking Homeostasis to RL: Internal State Control of Motivated Behavior (HRRL)* | Yoshida, Sprekeler, Gutkin | arXiv | 2025 | 偏离内稳态设定点即为奖励，天然产生风险规避、预期调节、自适应探索 |
| 2.2 | *CTCS-HRRL: Continuous Time Continuous Space Homeostatic RL* | Laurençon, Bhargava et al., Gutkin | arXiv | 2024 | 内稳态 RL 扩展到连续时间/空间，内驱状态连续波动 |
| 2.3 | *Multi-Scale Temporal Homeostasis Enables Efficient and Robust NNs (MSTH)* | Hakim | arXiv | 2026 | 多时间尺度（5ms→1hr）内稳态调节，消除灾难性故障，不同"情绪"操作于不同时间尺度 |
| 2.4 | *SAPIN: Structural Plasticity as Active Inference* | Hill | arXiv | 2025 | 神经元物理迁移 + Hebbian 突触学习，纯预测误差最小化解决 CartPole |
| 2.5 | *Dynamic Weight Adaptation in SNNs Inspired by Biological Homeostasis* | Zhou, Dong et al. | arXiv | 2025 | BCM 理论在脉冲神经网络中的实时内稳态权重调节 |
| 2.6 | *SGEMAS: Self-Growing Ephemeral Multi-Agent System via Entropic Homeostasis* | Hamdi | arXiv | 2025 | 智能作为动态热力学过程，Agent 诞生/消亡耦合到自由能最小化 |
| 2.7 | *The Conditions of Physical Embodiment Enable Generalization and Care* | Christov-Moore, Juliani, Kiefer, Lehman, Safron, Polani, **Damasio** et al. | arXiv | 2025 | **Damasio 亲自署名**：论证物理脆弱性 + 内稳态驱动是 AI 泛化和"关怀"的前提，躯体标记假说的 AI 版 |
| 2.8 | *Social Allostasis: Or, How I Learned To Stop Worrying and Love The Noise* | Khan | ALIFE 2025 | 2025 | 皮质醇类似物 = "焦虑"驱动适应；催产素类似物 = 社会联结信号。前瞻性稳态超越反应性稳态 |

---

## 3. 快速权重与实时可塑性

| # | 论文 | 作者 | 出处 | 年份 | 核心贡献 |
|---|------|------|------|------|----------|
| 3.1 | *Fast Weight Programming and Linear Transformers: From ML to Neurobiology* | Irie, Gershman | **TMLR** | 2025 | 权威综述：FWP = 线性 Transformer = 短期记忆。实时可塑性的 canonical 框架 |
| 3.2 | *Test-Time Regression: A Unifying Framework for Sequence Models with Associative Memory* | Wang, Shi, Fox | arXiv | 2025 | **统一框架**：线性注意力、SSM、快速权重、在线学习器全部是关联回忆回归的特例 |
| 3.3 | *Going Beyond Linear Transformers with Recurrent Fast Weight Programmers* | Irie, Schlag, Csordas, Schmidhuber | **NeurIPS** | 2021 | 循环快速权重在语言建模和 Atari RL 上的大幅提升 |
| 3.4 | *Neural Differential Equations for Learning to Program Neural Nets Through Continuous Learning Rules* | Irie, Faccio, Schmidhuber | **NeurIPS** | 2022 | 连续时间快速权重——突触动态由学习到的微分方程控制 |
| 3.5 | *Practical Computational Power of Linear Transformers and Self-Referential Extensions* | Irie, Csordas, Schmidhuber | **EMNLP** | 2023 | 自指权重矩阵——网络修改自身权重，实时可塑性的最纯粹形式 |

---

## 4. 塑性丧失与持续学习

### 塑性丧失（Loss of Plasticity）— 独立于灾难性遗忘的基本问题

| # | 论文 | 作者 | 出处 | 年份 | 核心贡献 |
|---|------|------|------|------|----------|
| 4.1 | *Maintaining Plasticity in Deep Continual Learning* | Dohare, Hernandez-Garcia, Rahman, **Mahmood**, **Sutton** | arXiv | 2023 | **里程碑**：深度网络随时间丧失学习能力（独立于遗忘），引入持续反向传播 |
| 4.2 | *Barriers for Learning in an Evolving World: Mathematical Understanding of Loss of Plasticity* | Joudaki, Lanzillotta et al. (**Google DeepMind**) | arXiv | 2025 | **数学证明**：促进静态泛化的性质（低秩、简单性偏差）直接导致塑性丧失 |
| 4.3 | *Spectral Collapse Drives Loss of Plasticity in Deep Continual Learning* | He, Guo et al. | arXiv | 2025 | Hessian 谱坍塌先于塑性丧失，τ-可训练性框架统一现有方法 |
| 4.4 | *UPGD: Addressing Loss of Plasticity and Catastrophic Forgetting* | Elsayed, Mahmood | **ICLR** | 2024 | 有用单元小修改（防遗忘）+ 无用单元大修改（恢复塑性），生物启发 |
| 4.5 | *Plastic Learning with Deep Fourier Features* | Lewandowski, Schuurmans, Machado | arXiv | 2024 | 线性函数逼近不丧失塑性，Fourier 特征维持无限梯度流 |

### 持续学习

| # | 论文 | 作者 | 出处 | 年份 | 核心贡献 |
|---|------|------|------|------|----------|
| 4.6 | *Continual Learning via Sparse Memory Finetuning* | Lin, Zettlemoyer, Ghosh et al. (Meta) | arXiv | 2025 | 稀疏记忆微调：遗忘率从全微调 89% 降至 **11%** |
| 4.7 | *Self-Composing Policies for Scalable Continual RL* | Malagon, Ceberio, Lozano | **ICML (Oral)** | 2024 | 可生长模块化网络，参数线性增长但不牺牲塑性 |
| 4.8 | *Routing Without Forgetting* | Masano, Bellitto et al. | arXiv | 2026 | 基于 Modern Hopfield Networks 的能量关联检索路由，无需任务标识符 |
| 4.9 | *KLDA: Continual Learning Using Kernel-Based Method Over Foundation Models* | Momeni, Mazumder, **Bing Liu** | arXiv | 2024 | 核方法 + 预训练特征接近联合训练上界，无需回放数据 |
| 4.10 | *InCA: In-Context Continual Learning Assisted by External Learner* | Momeni, Mazumder, Ke, Liu | arXiv | 2024 | 冻结 LLM + 外部持续学习器，零参数更新实现持续适配 |

---

## 5. 动态 LoRA 与持续适配

| # | 论文 | 作者 | 出处 | 年份 | 核心贡献 |
|---|------|------|------|------|----------|
| 5.1 | *JumpLoRA: Sparse Adapters for Continual Learning in LLMs* | Dragomir, Pintilie et al. | arXiv | 2026 | JumpReLU 门控动态稀疏 LoRA，防止任务干扰 |
| 5.2 | *NeuroLoRA: Context-Aware Neuromodulation for Multi-Task Adaptation* | Yang, Zhang et al. | arXiv | 2026 | **受生物神经调节启发**——可学习神经调制门 + 对比正交性损失 |
| 5.3 | *Share: Shared LoRA Subspaces for Almost Strict Continual Learning* | Kaushik, Vaidya et al. (**Johns Hopkins**) | arXiv | 2026 | 单一共享低秩子空间，**100x 参数削减**，一个模型替代数百个任务 LoRA |
| 5.4 | *Rank-1 Expert Pool in a Single LoRA* | Fa, Duan et al. | arXiv | 2026 | 单个 LoRA 模块重构为可分解 Rank-1 专家池，参数减少 **96.7%** |
| 5.5 | *Machine Unlearning and CL in Hybrid Resistive Memory Neuromorphic Systems* | Lin, Yang et al. | arXiv | 2026 | 硬件-软件协同：模拟电阻阵列 + 数字动态 LoRA，训练成本降低 147.76x |
| 5.6 | *CORAL: Scalable Multi-Task Robot Learning via LoRA Experts* | Luo, Chen, Liang, **Zhenguo Li** | arXiv | 2026 | 机器人连续 LoRA 学习，零推理开销热加载专家路由 |

---

## 6. 互补学习系统与睡眠整合

### 互补学习系统 (CLS)

| # | 论文 | 作者 | 出处 | 年份 | 核心贡献 |
|---|------|------|------|------|----------|
| 6.1 | *CLS-ER: Learning Fast, Learning Slow* | Arani, Sarfraz, Zonooz | **ICLR** | 2022 | 海马-新皮层双记忆系统的标准 AI 实现 |
| 6.2 | *DualNets: Continual Learning, Fast and Slow* | Pham, Liu, Hoi | arXiv | 2022 | 快速学习（监督）+ 慢速学习（自监督）双网络 |
| 6.3 | *Personalized AGI via Neuroscience-Inspired Continuous Learning Systems* | Gupta, Gupta et al. | arXiv | 2025 | 整合突触修剪、Hebbian 可塑性、稀疏编码、双记忆系统的边缘 AGI 路线图 |

### 睡眠整合

| # | 论文 | 作者 | 出处 | 年份 | 核心贡献 |
|---|------|------|------|------|----------|
| 6.4 | *Wake-Sleep Consolidated Learning (WSCL)* | Sorrenti, Bellitto et al. | arXiv | 2024 | 最完整睡眠周期：清醒→NREM（突触巩固）→REM（做梦准备）|
| 6.5 | *Sleep Replay Consolidation (SRC)* | Delanois, Ahuja, Krishnan, **Bazhenov** | arXiv | 2026 | 训练后睡眠阶段选择性重放，无需监督即可改善校准 |
| 6.6 | *Toward Lifelong Learning in Equilibrium Propagation: Sleep-like and Awake Rehearsal* | Kubo, Delanois, Bazhenov | arXiv | 2025 | 生物可信学习规则（平衡传播）+ 睡眠整合，超越 BPTT |
| 6.7 | *Replay in Deep Learning: Current Approaches and Missing Biological Elements* | Hayes, Krishnan, Bazhenov, **Siegelmann**, **Sejnowski** | **Neural Computation (MIT Press)** | 2021 | Sejnowski 署名权威综述：生物 vs 人工重放差距分析 |
| 6.8 | *Biologically Inspired Sleep Algorithm for ANNs* | Krishnan, Tadros, Ramyaa, Bazhenov | arXiv | 2019 | STDP 睡眠整合：恢复遗忘任务 + 前向迁移 + 噪声泛化 |

### 表征漂移

| # | 论文 | 作者 | 出处 | 年份 | 核心贡献 |
|---|------|------|------|------|----------|
| 6.9 | *Learning Continually with Representational Drift* | van der Veldt, van de Ven, Moorman, Etter | arXiv | 2025 | 受控表征漂移可能是特性而非缺陷，挑战"稳定表征必要"假设 |

---

## 7. 好奇心与惊奇驱动探索

| # | 论文 | 作者 | 出处 | 年份 | 核心贡献 |
|---|------|------|------|------|----------|
| 7.1 | *Language and Culture Internalisation for Human-Like Autotelic AI* | Colas, Karch, Moulin-Frier, **Oudeyer** | **Nature Machine Intelligence** | 2022 | 自我生成目标的自发 Agent + 语言文化内化 |
| 7.2 | *Autotelic Agents with Intrinsically Motivated Goal-Conditioned RL: A Short Survey* | Colas, Karch, Sigaud, Oudeyer | **JAIR** | 2022 | 内在动机分类：能力型（学习进度）、新颖性型（惊奇）、知识型（信息增益）|
| 7.3 | *Nuclear Norm Maximization-Based Curiosity-Driven RL* | Chen, Zhai, Gao et al. | IEEE TAI | 2024 | 仅内在奖励达人类标准化分数 1.09（基准 2 倍），鲁棒惊奇信号 |
| 7.4 | *SuS: Strategy-aware Surprise for Intrinsic Exploration* | Kashirskiy, Makarov | arXiv | 2026 | 策略级惊奇操作化，LLM 推理 Pass@1 提升 17.4% |
| 7.5 | *Curiosity-Driven Development of Action and Language in Robots* | Tinker, **Doya**, **Tani** | arXiv | 2025 | 好奇心驱动主动推断复现关键发展心理学现象 |
| 7.6 | *CLS-IR: Complementary Learning System-Based Intrinsic Reward* | Gao, Xu et al. | IEEE ICASSP | 2023 | 短期/长期记忆预测差异作惊奇信号，零额外训练成本 |
| 7.7 | *Development of Few-Shot Learning Through Self-Supervised Interaction* | Clay, Pipa, Kühnberger, König | IEEE TPAMI | 2023 | 好奇心探索创造的结构化表征支持单样本概念学习 |
| 7.8 | *From Psychological Curiosity to Artificial Curiosity (Survey)* | Sun, Qian, Miao | arXiv | 2022 | 心理学好奇心类型到计算模拟的统一映射框架 |

---

## 8. 信息增益与不确定性驱动

| # | 论文 | 作者 | 出处 | 年份 | 核心贡献 |
|---|------|------|------|------|----------|
| 8.1 | *MaxInfoRL: Boosting Exploration through Information Gain Maximization* | Sukhija, Coros, **Krause**, **Abbeel**, Sferrazza | **ICLR** | 2024 | 信息增益最大化框架，自然平衡外在/内在奖励 |
| 8.2 | *STEERING: Stein Information Directed Exploration for Model-Based RL* | Chakraborty, Bedi, Koppel, Mengdi Wang, Huang, Manocha | **ICML** | 2023 | 核化 Stein 差异估计信息增益，严格的次线性贝叶斯遗憾 |
| 8.3 | *Towards Empowerment Gain through Causal Structure Learning (ECL)* | Cao, Feng et al. | arXiv | 2025 | 赋权（互信息）+ 因果理解联合驱动探索 |
| 8.4 | *Entropy-Controlled Intrinsic Motivation RL (ECIM)* | Gong et al. | arXiv | 2025 | 熵控制 + 内在动机防止过早收敛 |
| 8.5 | *Information-Theoretic Policy Pre-Training with Empowerment* | Schneider, Krug et al., Boedecker | arXiv | 2025 | 赋权作为内在动机维持可控性 |
| 8.6 | *MERCI: Count-based Intrinsic Rewards for LLM Reasoning* | Zhang et al. | arXiv | 2025 | 伪计数认知不确定性作为"无聊"信号逃离局部最优 |

---

## 9. 意识与元认知架构

| # | 论文 | 作者 | 出处 | 年份 | 核心贡献 |
|---|------|------|------|------|----------|
| 9.1 | *From Internal Models Toward Metacognitive AI* | **Kawato**, Cortese | arXiv | 2021/2023 | 意识 = 责任信号熵（生成/逆模型不匹配 + 奖励预测误差）|
| 9.2 | *The Conscious Turing Machine* | **Blum**, **Blum** | arXiv | 2023 | 意识的简单理论模型，指导 AGI 构建 |
| 9.3 | *DIME Architecture: Unified Operational Algorithm for Neural Representation* | Vladu, Bizdoaca et al. | arXiv | 2026 | Detect-Integrate-Mark-Execute，集成印迹、执行线程、标记系统（神经调质）|
| 9.4 | *Modular Theory of Subjective Consciousness* | Gillon | arXiv | 2025 | 信息丰富度密度向量量化主观强度，自评估模块为"焦虑"提供基底 |
| 9.5 | *The Nature of Intelligence* | You | arXiv | 2023/2024 | 智能数学定义为熵减过程，惊奇 = 熵减失败，焦虑 = 熵增 |
| 9.6 | *Cognitive Dark Matter: Measuring What AI Misses* | Mineault, **Griffiths**, Escola | arXiv | 2026 | 真正持续学习需要训练大脑"如何学习"而非仅"输出什么" |

---

## 10. 自主 LLM Agent 与内在动机

| # | 论文 | 作者 | 出处 | 年份 | 核心贡献 |
|---|------|------|------|------|----------|
| 10.1 | *AgentEvolver: Towards Efficient Self-Evolving Agent System* | Zhai, Tao, Chen et al. | arXiv | 2025 | 自我提问（好奇心任务生成）+ 自我导航（经验复用）+ 自我归因（差异化奖励）|
| 10.2 | *Agentic AI in Autonomous Driving: LLM-Enhanced RL with Reusable Skills* | Maaroufi et al. | ICAART | 2026 | 好奇心内在奖励 + 可复用技能库，碰撞率降低 50% |
| 10.3 | *Closed-Loop Long-Horizon Robotic Planning via Equilibrium Sequence Modeling* | Li, Sun, Mu | **ICML** | 2025 | 世界模型闭环自精炼规划，预测-现实分歧驱动修正 |
| 10.4 | *Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents* | Putta, Mills et al., **Chelsea Finn** | arXiv | 2024 | MCTS + 自我批评 + DPO 迭代微调，成功率从 18.6% 提升到 81.7% |

---

## 11. 关键研究者图谱

| 研究者 | 机构 | 核心贡献 |
|--------|------|----------|
| **Karl Friston** | UCL Wellcome Centre for Neuroimaging | 自由能原理与主动推断创始人 |
| **Pierre-Yves Oudeyer** | INRIA Bordeaux (Flowers team) | 内在动机/好奇心驱动学习先驱 (IAC, RIAC, IMGEP) |
| **Antonio Damasio** | USC Brain and Creativity Institute | 躯体标记假说；情绪作为内稳态调节器 |
| **Jürgen Schmidhuber** | KAUST / IDSIA | 快速权重、RNN、好奇心驱动探索先驱 |
| **Richard S. Sutton** | UAlberta / Amii | 持续学习、RL 理论、塑性丧失研究 |
| **Terrence Sejnowski** | Salk Institute | 计算神经科学、生物重放 vs 人工重放 |
| **Maxim Bazhenov** | UCSD | 睡眠整合、STDP、记忆巩固 |
| **Boris Gutkin** | ENS Paris | 内稳态强化学习 (HRRL) |
| **Kenji Doya** | OIST | 神经调节与元学习、好奇心探索 |
| **Cédric Colas** | INRIA Flowers | 自发 Agent、目标条件 RL + 内在动机 |
| **Bing Liu** | University of Illinois at Chicago | 持续学习理论 (LAMOL, KLDA, InCA) |
| **Anil Seth** | University of Sussex | 预测处理；意识（与 Friston 对抗合作）|
| **Kazuki Irie** | KAUST | 快速权重编程统一理论 |
| **Alexander Ororbia** | Penn State | 预测编码、生物可信学习规则 |
| **Michael Levin** | Tufts University | 多样智能、形态计算 |

---

## 12. 与架构构想的映射关系

### 架构层级 → 实现方案

| 架构层级 | 功能描述 | 对应实现方案 | 关键论文 |
|----------|----------|-------------|----------|
| **感知执行层** (System 1) | 高频实时响应 | Fast Weight Programmer / 线性 Transformer | 3.1, 3.2 |
| **反思缓存层** (短期工作记忆) | Thought Traces 暂存与反思 | CLS-ER 双记忆系统 / 稀疏记忆微调 | 6.1, 4.6 |
| **后台整合层** (异步微调) | 事件驱动极轻量训练 | Wake-Sleep 周期 / 动态 LoRA | 6.4, 5.2 |
| **塑性存储层** (内化机制) | LoRA 热加载实现拓扑改变 | Share 共享子空间 / CORAL 零开销路由 | 5.3, 5.6 |

### 情绪信号 → 计算实现

| 情绪信号 | 数学映射 | 计算实现 | 关键论文 |
|----------|----------|----------|----------|
| **惊奇/不适** | 变分自由能；预测误差；KL 散度 | ICM 前向模型误差、RND、核范数最大化 | 1.1, 7.3, 7.4 |
| **无聊** | 信息熵衰减；预测增益趋零；新颖度消退 | 学习进度监控、RND 新颖度衰减、熵控制 | 7.2, 8.4, 8.6 |
| **焦虑** | 认知失调；预测方差；责任信号高熵 | 自由能累积、Stein 差异、赋权最小化 | 1.1, 8.2, 8.3, 9.1 |

### NCS 系期刊论文汇总

| 论文 | 期刊 |
|------|------|
| Gunasekaran et al. *Future-Guided Learning* | **Nature Communications** (2025) |
| Colas, Karch, Moulin-Frier, Oudeyer. *Autotelic AI* | **Nature Machine Intelligence** (2022) |
| Hayes, Siegelmann, Sejnowski et al. *Replay in DL* | **Neural Computation** MIT Press (2021) |

---

*报告结束。本文档为后续实验提供理论基础与实施参考。*
