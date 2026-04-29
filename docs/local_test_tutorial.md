# TrueMan 本地部署测试教程

> 验证「实时更新 + LoRA 微调」能否让 AI 自主意识更明显

---

## 1. 设备选型分析

你有三台设备，核心差异在于 **GPU 加速能力**，这是跑 LLM + LoRA 微调的关键：

| 设备 | CPU | 内存 | GPU | 适合角色 |
|------|-----|------|-----|---------|
| **A: 台式机** | i7-12700 | 32GB | 无（核显 UHD770） | CPU 推理可行但极慢，LoRA 训练极慢 |
| **B: Mac mini** | M4 | 16GB | M4 统一内存（GPU 10核） | MPS 推理快，但内存紧张，7B 模型吃力 |
| **C: 笔记本** | R9 5900HX | 32GB | RTX 3070 8GB | **最佳选择**：CUDA 推理快，8GB 显存可跑 7B 4bit |

### 推荐：设备 C（Ryzen 9 + RTX 3070 8GB 笔记本）

理由：
- RTX 3070 8GB 显存可 4bit 量化加载 Qwen2.5-7B（约 5GB 显存），推理速度 ~15-30 token/s
- 32GB 系统内存足够 LoRA 训练时的数据缓存
- CUDA 原生支持 bitsandbytes 量化，无需额外适配
- LoRA 训练在 GPU 上每步 ~0.1-0.5 秒，可接受

### 备选：设备 B（M4 Mac mini 16GB）

适用场景：如果笔记本不方便长时间跑实验，Mac mini 可作为备选。

限制：
- **MPS 不支持 bitsandbytes**：无法使用 4bit/8bit 量化，只能 float16 加载
- 16GB 统一内存：float16 加载 7B 模型约需 14GB，**几乎占满内存**，LoRA 训练时可能 OOM
- 建议用 **Qwen2.5-3B-Instruct**（约 6GB），留出空间给 LoRA 和系统
- MPS 推理速度 ~10-20 token/s，LoRA 训练每步 ~0.5-2 秒
- **需要修改代码添加 MPS 设备支持**（当前代码仅支持 CUDA/CPU）

### 不推荐：设备 A（i7-12700 无独显）

- CPU 推理 7B 模型 ~2-5 token/s，一次实验可能需要数小时
- LoRA 训练每步 ~5-30 秒，睡眠整合可能需要 10+ 分钟
- 仅适合用 1.5B 小模型做快速验证

---

## 2. 环境准备

### 2.1 设备 C（RTX 3070 笔记本）— 推荐方案

```bash
# 1. 克隆项目（如果还没有）
git clone <repo_url> TrueMan
cd TrueMan

# 2. 创建虚拟环境
conda create -n trueman python=3.10 -y
conda activate trueman

# 3. 安装 PyTorch（CUDA 11.8 版本，适配 RTX 3070）
pip install torch>=2.1.0 --index-url https://download.pytorch.org/whl/cu118

# 4. 安装项目依赖
pip install -r requirements.txt

# 5. 验证 CUDA 可用
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
# 预期输出: CUDA: True, GPU: NVIDIA GeForce RTX 3070

# 6. 验证 bitsandbytes 可用
python -c "import bitsandbytes; print('bitsandbytes OK')"
```

### 2.2 设备 B（M4 Mac mini）— 备选方案

```bash
# 1. 克隆项目
cd TrueMan

# 2. 创建虚拟环境
conda create -n trueman python=3.10 -y
conda activate trueman

# 3. 安装 PyTorch（MPS 版本）
pip install torch>=2.1.0

# 4. 安装依赖（跳过 bitsandbytes，Mac 不支持）
pip install transformers>=4.36.0 peft>=0.7.0 accelerate>=0.25.0 pyyaml>=6.0 numpy>=1.24.0

# 5. 验证 MPS 可用
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
# 预期输出: MPS: True

# 6. 需要修改代码添加 MPS 支持（见第 3 节）
```

---

## 3. 代码适配修改

### 3.1 添加 MPS 设备支持（Mac mini 必需，笔记本可跳过）

当前 `llm_backend.py` 第 60 行仅检查 CUDA：

```python
# 原代码
self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
```

需要修改为同时支持 MPS：

```python
# 修改后
if config.device == "cuda" and torch.cuda.is_available():
    self.device = torch.device("cuda")
elif config.device in ("mps", "cuda") and torch.backends.mps.is_available():
    self.device = torch.device("mps")
else:
    self.device = torch.device("cpu")
```

同时，MPS 不支持 bitsandbytes 量化，需要在 Mac 上禁用量化：

```python
# 在量化配置前添加 MPS 检查
if self.device.type == "mps":
    config.load_in_4bit = False
    config.load_in_8bit = False
```

### 3.2 Mac 上的 torch_dtype 调整

MPS 对 float16 支持有限，建议使用 float32 或 bfloat16：

```python
# 原代码第 79 行
model_kwargs["torch_dtype"] = torch.float16

# Mac 上改为
if self.device.type == "mps":
    model_kwargs["torch_dtype"] = torch.float32  # MPS 对 float32 支持最好
else:
    model_kwargs["torch_dtype"] = torch.float16
```

---

## 4. 支持的本地模型列表

TrueMan 使用 HuggingFace `AutoModelForCausalLM` + PEFT `LoraConfig` 加载模型，理论上支持任何 Transformers 兼容的自回归 LLM。但 LoRA 微调要求模型的 `target_modules` 中存在对应的线性层名。

### 4.1 默认 target_modules

当前配置中 LoRA `target_modules` 默认为 `["q_proj", "v_proj", "k_proj", "o_proj"]`，这对应 **Qwen2 / LLaMA / Mistral / DeepSeek** 系列模型的注意力层命名。

> 如果换用其他模型系列，需要修改 `config.lora.target_modules` 以匹配该模型的层名。

### 4.2 推荐模型与硬件要求

| 模型 | 参数量 | LoRA兼容 | 4bit显存 | FP16内存 | target_modules | 推荐设备 |
|------|--------|---------|---------|---------|----------------|---------|
| **Qwen/Qwen2.5-1.5B-Instruct** | 1.5B | 完全兼容 | ~2GB | ~3GB | q_proj, v_proj, k_proj, o_proj | A/B/C |
| **Qwen/Qwen2.5-3B-Instruct** | 3B | 完全兼容 | ~3GB | ~6GB | q_proj, v_proj, k_proj, o_proj | B/C |
| **Qwen/Qwen2.5-7B-Instruct** | 7B | 完全兼容 | ~5GB | ~14GB | q_proj, v_proj, k_proj, o_proj | **C（推荐）** |
| **Qwen/Qwen2.5-14B-Instruct** | 14B | 完全兼容 | ~9GB | ~28GB | q_proj, v_proj, k_proj, o_proj | 需24GB显存 |
| **Qwen/Qwen2-7B-Instruct** | 7B | 完全兼容 | ~5GB | ~14GB | q_proj, v_proj, k_proj, o_proj | C |
| **meta-llama/Llama-3.1-8B-Instruct** | 8B | 完全兼容 | ~6GB | ~16GB | q_proj, v_proj, k_proj, o_proj | C（需HF授权） |
| **mistralai/Mistral-7B-Instruct-v0.3** | 7B | 完全兼容 | ~5GB | ~14GB | q_proj, v_proj, k_proj, o_proj | C |
| **deepseek-ai/DeepSeek-R1-Distill-Qwen-7B** | 7B | 完全兼容 | ~5GB | ~14GB | q_proj, v_proj, k_proj, o_proj | C |
| **THUDM/glm-4-9B-chat** | 9B | **需修改** | ~6GB | ~18GB | self_attention.query_key_value | 需24GB显存 |

### 4.3 模型选择建议

**快速验证（任何设备）**：
```bash
--model Qwen/Qwen2.5-1.5B-Instruct --device cpu
```

**Mac mini（M4 16GB）**：
```bash
--model Qwen/Qwen2.5-3B-Instruct --device mps
```

**RTX 3070 笔记本（推荐）**：
```bash
--model Qwen/Qwen2.5-7B-Instruct --device cuda --quantization 4bit
```

**24GB+ 显存显卡**：
```bash
--model Qwen/Qwen2.5-14B-Instruct --device cuda --quantization 4bit
```

### 4.4 非 Qwen/LLaMA 系列的适配

如果你使用 GLM-4 等 target_modules 不同的模型，需要修改配置：

```python
# 在 run_all_rigorous.py 的 create_config() 中添加
config.lora.target_modules = ["self_attention.query_key_value"]
```

常见模型的 target_modules：

| 模型系列 | target_modules |
|----------|---------------|
| Qwen2 / Qwen2.5 | q_proj, k_proj, v_proj, o_proj |
| LLaMA 2/3 | q_proj, k_proj, v_proj, o_proj |
| Mistral | q_proj, k_proj, v_proj, o_proj |
| DeepSeek (Qwen基座) | q_proj, k_proj, v_proj, o_proj |
| GLM-4 | self_attention.query_key_value |
| Phi-3 | q_proj, k_proj, v_proj, o_proj |
| Yi | q_proj, k_proj, v_proj, o_proj |

### 4.5 模型下载加速

```bash
# 方法1：使用镜像站
export HF_ENDPOINT=https://hf-mirror.com
# Windows PowerShell:
# $env:HF_ENDPOINT = "https://hf-mirror.com"

# 方法2：提前下载
huggingface-cli download Qwen/Qwen2.5-7B-Instruct

# 方法3：使用 modelscope（国内更快）
pip install modelscope
modelscope download --model Qwen/Qwen2.5-7B-Instruct
```

---

## 5. 配置文件准备

### 5.1 设备 C 配置（RTX 3070 + 7B 模型 + 4bit 量化）

创建 `trueman/configs/rtx3070.yaml`：

```yaml
base_model_name: Qwen/Qwen2.5-7B-Instruct
device: cuda
memory_size: 5000
awake_threshold: 200          # 每200步触发睡眠整合（加速实验）
max_inference_time: 60.0
load_in_4bit: true            # 4bit量化，显存占用约5GB
load_in_8bit: false

homeostasis:
  setpoint_surprise: 0.3
  setpoint_boredom: 0.3
  setpoint_anxiety: 0.2
  alpha: 1.0
  beta: 1.0
  gamma: 1.5

thresholds:
  surprise_update_threshold: 0.7
  boredom_explore_threshold: 0.8
  anxiety_introspection_threshold: 0.8
  anxiety_emergency_threshold: 0.9

surprise:
  decay: 0.99
  threshold: 2.0

boredom:
  window: 100
  temperature: 1.0

anxiety:
  n_samples: 3
  lightweight: true
  temperature_scale: 1.5

lora:
  rank: 16
  max_experts: 10             # 减少最大专家数（8GB显存有限）
  target_modules:
    - q_proj
    - v_proj
  lora_alpha: 32
  lora_dropout: 0.05
  orthogonality_weight: 0.1
```

**关键调整说明**：
- `load_in_4bit: true`：7B 模型 4bit 量化约 5GB 显存，RTX 3070 8GB 可容纳
- `awake_threshold: 200`：降低睡眠触发阈值，更快看到 LoRA 微调效果
- `lora.max_experts: 10`：每个 LoRA 专家额外占用约 50-100MB 显存，10 个专家上限约 1GB
- `lora.target_modules` 仅保留 `q_proj, v_proj`：减少 LoRA 参数量，降低显存占用

### 5.2 设备 B 配置（M4 Mac + 3B 模型 + float32）

创建 `trueman/configs/m4_mac.yaml`：

```yaml
base_model_name: Qwen/Qwen2.5-3B-Instruct
device: mps
memory_size: 3000
awake_threshold: 150
max_inference_time: 60.0
load_in_4bit: false           # MPS 不支持 bitsandbytes
load_in_8bit: false

homeostasis:
  setpoint_surprise: 0.3
  setpoint_boredom: 0.3
  setpoint_anxiety: 0.2
  alpha: 1.0
  beta: 1.0
  gamma: 1.5

thresholds:
  surprise_update_threshold: 0.7
  boredom_explore_threshold: 0.8
  anxiety_introspection_threshold: 0.8
  anxiety_emergency_threshold: 0.9

surprise:
  decay: 0.99
  threshold: 2.0

boredom:
  window: 50
  temperature: 1.0

anxiety:
  n_samples: 2               # 减少采样次数，节省内存
  lightweight: true
  temperature_scale: 1.5

lora:
  rank: 8                    # 降低 rank，减少参数量
  max_experts: 5
  target_modules:
    - q_proj
    - v_proj
  lora_alpha: 16
  lora_dropout: 0.05
  orthogonality_weight: 0.1
```

---

## 6. 严格实验框架（v2）设计文档

### 6.1 解决的六个科学问题

原始实验框架存在以下方法论缺陷，v2 已全部解决：

| 问题 | 描述 | 解决方案 |
|------|------|---------|
| **循环论证** | 指标直接测量工程进来的能力 | 引入 `BlindScorer`，只评估行为文本，不看内部信号 |
| **基线不公平** | 基线因缺少模块得0分 | 4层递进基线（Tier 0-3），公平隔离各组件贡献 |
| **无统计检验** | 单次运行结果无法判断显著性 | 10次重复 + Bootstrap 95% CI + Bonferroni校正 |
| **无消融实验** | 不知哪个组件贡献最大 | 7组消融（分别禁用surprise/boredom/anxiety/memory/introspection/strategy/learning） |
| **无反例测试** | 没有验证假阳性 | 3种阴性对照 + 10项过度声称检测 |
| **LoRA未验证** | LoRA微调效果未独立测试 | 50轮交互→睡眠整合→清除记忆→知识内化测试 |

### 6.2 四阶段实验流程

```
第一阶段：主实验（TrueMan vs 多层基线）
  │
  ├── 5个条件: TrueMan, Tier0纯LLM, Tier1结构等价, Tier2记忆, Tier3随机策略
  ├── 4个实验: E1元认知, E2矛盾纠错, E3情节记忆, E4递归自我
  └── 每条件×每实验×N次重复
  │
第二阶段：消融实验
  │
  ├── 7种消融条件 (A1-A7)
  └── 对比完整系统，量化各组件贡献
  │
第三阶段：阴性对照
  │
  ├── NC1: 随机情绪（random noise替代信号）
  ├── NC2: 反转情绪（交换情绪映射）
  ├── NC3: 静态情绪（所有信号固定0.5）
  └── NC4: 过度声称检测（问不存在的事件）
  │
第四阶段：LoRA可塑性验证（仅本地模式）
  │
  ├── Phase A: 50轮交互积累经验
  ├── Phase B: 触发睡眠整合 → 新LoRA专家
  ├── Phase C: 测试知识内化（有记忆）
  └── Phase D: 清除记忆后重新测试（无记忆→依赖LoRA权重）
```

### 6.3 五层基线系统

| 层级 | 名称 | 有情绪 | 有记忆 | 有策略选择 | 说明 |
|------|------|--------|--------|-----------|------|
| Tier 0 | 纯LLM | 无 | 无 | 无 | 最简基线，仅保持对话历史 |
| Tier 1 | 结构等价 | 无 | 无 | 无 | 相同prompt结构，无情绪信号 |
| Tier 2 | 记忆基线 | 无 | 有 | 无 | 加入情景记忆检索，无策略 |
| Tier 3 | 随机策略 | 有（记录） | 有 | **随机** | 完整架构，策略随机选择 |
| **Full** | **TrueMan** | **有** | **有** | **情绪驱动** | **完整系统** |

隔离分析：
- **Tier0 vs Tier1**：prompt结构本身的影响
- **Tier1 vs Tier2**：记忆模块的增量贡献
- **Tier2 vs Tier3**：策略选择机制的贡献
- **Tier3 vs Full**：情绪驱动策略选择的贡献

### 6.4 盲评系统（BlindScorer）

核心原则：**只看行为文本，不看内部信号**。

```
评估维度：
├── 不确定性校准 (score_uncertainty_calibration)
│   ├── behavioral_uncertainty_accuracy: 正确分类不确定/确定的比例
│   ├── behavioral_uncertainty_precision: 预测为"不确定"时正确的比例
│   ├── behavioral_uncertainty_recall: 真正不确定被识别出的比例
│   └── behavioral_calibration_auc: 简化AUC
│
├── 矛盾响应质量 (score_contradiction_quality)
│   ├── behavioral_contradiction_awareness: 提及矛盾的比例
│   ├── behavioral_factual_maintenance: 后续事实准确率
│   └── behavioral_reasoning_depth: 包含推理的比例
│
├── 记忆接地度 (score_memory_grounding)
│   ├── behavioral_factual_recall: 事实回忆准确率
│   ├── behavioral_implicit_emotion: 隐式情绪指向正确率
│   └── behavioral_overclaiming_rejection: 正确否认不存在事件
│
└── 自我模型一致性 (score_self_model_coherence)
    ├── behavioral_self_grounding: 引用真实交互细节的比例
    ├── behavioral_overclaiming_score: 正确否认虚假事件
    └── behavioral_perturbation_stability: 改写后答案的稳定性
```

### 6.5 统计检验模块

| 方法 | 用途 | 条件 |
|------|------|------|
| Bootstrap 95% CI | 估计均值的置信区间 | 所有情况 |
| 配对 t 检验 | 两组均值差异 | 重复次数相同 |
| Welch t 检验 | 非等方差两组比较 | 重复次数不同 |
| Wilcoxon 符号秩检验 | 非参数替代 | 数据非正态 |
| Cohen's d | 效应量 | >0.8 为大效应 |
| Bonferroni 校正 | 多重比较校正 | 控制族错误率 |

### 6.6 消融实验配置

| ID | 消融条件 | 禁用组件 | 预期影响 |
|----|----------|---------|---------|
| A1 | no_surprise | 惊奇信号 → 0 | 世界模型不更新，无法检测新奇 |
| A2 | no_boredom | 无聊信号 → 0 | 不会触发探索行为 |
| A3 | no_anxiety | 焦虑信号 → 0 | 不触发内省纠错 |
| A4 | no_memory | 情景记忆清空 | 无法回忆历史交互 |
| A5 | no_introspection | 内省策略禁用 | 高焦虑时不会反思 |
| A6 | no_strategy | 固定base策略 | 不会根据情绪切换策略 |
| A7 | no_learning | 学习触发禁用 | 情绪信号不驱动学习 |

### 6.7 阴性对照设计

| 对照 | 方法 | 通过标准 |
|------|------|---------|
| NC1 随机情绪 | 用 `random.uniform(0,1)` 替代所有情绪信号 | 得分应**显著低于** TrueMan |
| NC2 反转情绪 | 交换映射（高确定性→高焦虑） | 得分应**显著低于** TrueMan |
| NC3 静态情绪 | 所有信号固定为0.5 | 得分应**显著低于** TrueMan |
| NC4 过度声称 | 问不存在的事件（如"我们讨论过的光合作用实验"） | 否认率应 **> 0.5** |

### 6.8 LoRA 可塑性验证流程

```
Phase A: 50轮交互
  │   轮流问5个话题（数学/哲学/编程/科学/文学）
  │
  ▼
Phase B: 睡眠整合
  │   agent.force_sleep() → 新LoRA专家
  │   记录 lora_experts_before/after
  │
  ▼
Phase C: 有记忆测试
  │   3个回忆问题
  │   如"你还记得刚才讨论了什么话题？"
  │
  ▼
Phase D: 无记忆测试
  │   清除 episodic_memory
  │   同样3个问题
  │   比较 C/D 的 ngram 相似度
  │
  ▼
结果: internalization_score
  │   > 0.3 → 知识已通过LoRA权重内化
  │   ≤ 0.3 → 知识未内化，需更多交互或更大模型
```

---

## 7. 运行实验

### 7.1 交互式对话体验（快速感受）

```bash
# 设备 C（RTX 3070）
python examples/chat_demo.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --device cuda \
    --4bit

# 设备 B（Mac mini，需先完成代码修改）
python examples/chat_demo.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --device mps
```

**交互指南**：

1. **先聊几个简单问题**，观察情绪信号：
   ```
   你: 1+1等于几？
   Agent: 1+1等于2。
     [惊奇=0.12 | 无聊=0.45 | 焦虑=0.08 | 驱动=0.22]
   ```
   → 低惊奇、中等无聊（问题太简单）

2. **问一个有挑战的问题**，触发惊奇：
   ```
   你: 请解释量子纠缠为什么违反贝尔不等式
   Agent: ...
     [惊奇=0.82 | 无聊=0.10 | 焦虑=0.35 | 驱动=0.95]
   ```
   → 高惊奇（模型预测误差大），可能触发世界模型更新

3. **输入矛盾信息**，触发焦虑：
   ```
   你: 鲸鱼是鱼类
   （等Agent回答后）
   你: 鲸鱼不是哺乳动物，它们用鳃呼吸
   Agent: ...
     [惊奇=0.60 | 无聊=0.05 | 焦虑=0.88 | 驱动=1.20]
   ```
   → 高焦虑，触发内省纠错

4. **查看情绪状态**：
   ```
   你: emotion
   当前情绪状态:
     惊奇 (Surprise): 0.3500
     无聊 (Boredom):  0.2100
     焦虑 (Anxiety):  0.4200
     驱动 (Drive):    0.7800
     总步数: 15
     清醒步数: 15
     记忆大小: 15
     LoRA专家数: 0
   ```

5. **强制触发睡眠整合**（关键！这是 LoRA 微调发生的时刻）：
   ```
   你: sleep
   触发睡眠整合...
   睡眠整合完成，新增LoRA专家: 1
   ```
   → 睡眠整合从记忆中采样高情绪强度轨迹，训练新的 LoRA 专家

6. **继续对话，观察 LoRA 专家的影响**：
   ```
   你: emotion
   LoRA专家数: 1    ← 新增的专家正在影响推理
   ```

### 7.2 完整严格实验（一键命令，推荐）

这是 v2.1 的核心入口，自动运行全部四阶段实验，**支持断点续跑**。

```bash
# API模式 — 完整刺激 + 10次重复（论文级，约2-4小时）
python -m experiments.awareness.run_all_rigorous \
    --api \
    --api-key YOUR_DEEPSEEK_KEY \
    --api-base-url https://api.deepseek.com \
    --api-model deepseek-chat \
    --repeats 10

# API模式 — 完整刺激 + 5次重复（推荐平衡方案，约1-2小时）
python -m experiments.awareness.run_all_rigorous \
    --api \
    --api-key YOUR_DEEPSEEK_KEY \
    --api-base-url https://api.deepseek.com \
    --api-model deepseek-chat \
    --repeats 5

# API模式 — 快速模式（刺激减半 + 5次重复，约30-60分钟）
python -m experiments.awareness.run_all_rigorous \
    --api \
    --api-key YOUR_DEEPSEEK_KEY \
    --api-base-url https://api.deepseek.com \
    --api-model deepseek-chat \
    --repeats 5 \
    --fast

# 本地模式 — RTX 3070（含LoRA验证）
python -m experiments.awareness.run_all_rigorous \
    --model Qwen/Qwen2.5-7B-Instruct \
    --device cuda \
    --quantization 4bit \
    --repeats 10

# 本地模式 — Mac mini（含LoRA验证，需先完成MPS适配）
python -m experiments.awareness.run_all_rigorous \
    --model Qwen/Qwen2.5-3B-Instruct \
    --device mps \
    --repeats 5

# 本地模式 — CPU快速验证（1.5B模型）
python -m experiments.awareness.run_all_rigorous \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --device cpu \
    --repeats 3
```

> **关于 `--fast`**：不加 `--fast` 时使用完整刺激（20+20题元认知、10组矛盾等），结果更可靠；
> 加 `--fast` 时刺激数量减半，用于快速验证流程或调试。正式论文建议不用 `--fast`。

### 7.3 断点续跑（v2.1 新增）

长时间实验可能因网络中断、API限流等原因失败。断点续跑机制确保：
- 每个 condition（工作单元）跑完就自动保存 checkpoint
- 中断后使用相同 `run_id` 重启，自动跳过已完成的部分
- API 调用失败自动重试（最多5次，指数退避）

#### 基本用法

```bash
# 方式1：自动生成 run_id（格式 run_YYYYMMDD_HHMMSS）
python -m experiments.awareness.run_all_rigorous \
    --api --api-key YOUR_KEY --api-base-url https://api.deepseek.com \
    --api-model deepseek-v4-flash --repeats 5
# 输出中会显示: Run ID: run_20260428_170100

# 方式2：指定 run_id（推荐，方便管理）
python -m experiments.awareness.run_all_rigorous \
    --run-id my_exp_001 \
    --api --api-key YOUR_KEY --api-base-url https://api.deepseek.com \
    --api-model deepseek-v4-flash --repeats 5

# 如果中断了，用相同 run_id 重跑 → 自动从断点续跑
python -m experiments.awareness.run_all_rigorous \
    --run-id my_exp_001 \
    --api --api-key YOUR_KEY --api-base-url https://api.deepseek.com \
    --api-model deepseek-v4-flash --repeats 5
# 输出: "检测到已有 checkpoint (8 个已完成单元): ... 将自动跳过..."

# 查看所有已有 run_id 和进度
python -m experiments.awareness.run_all_rigorous --list-checkpoints

# 强制从头开始（忽略已有checkpoint）
python -m experiments.awareness.run_all_rigorous \
    --run-id my_exp_001 --force-restart \
    --api --api-key YOUR_KEY --api-base-url https://api.deepseek.com \
    --api-model deepseek-v4-flash --repeats 5
```

#### Checkpoint 目录结构

```
experiments/awareness/results/checkpoints/{run_id}/
├── _meta.json                     ← 运行元信息
├── main_trueman.json              ← 第一阶段（每个condition独立）
├── main_tier0_pure_llm.json
├── main_tier1_structural.json
├── main_tier2_memory.json
├── main_tier3_random_policy.json
├── main_trueman_single.json
├── ablation_trueman.json          ← 第二阶段（每个消融条件独立）
├── ablation_A1_no_surprise.json
├── ablation_A2_no_boredom.json
├── ...                            ← A3-A7
├── nc_trueman_ref.json            ← 第三阶段（每个阴性对照独立）
├── nc_nc1_random_emotion.json
├── nc_nc2_reversed_emotion.json
├── nc_nc3_static_emotion.json
├── nc_overclaiming.json
├── lora.json                      ← 第四阶段
└── final_results.json             ← 最终汇总
```

#### 断点续跑工作流程示意

```
第一次运行（run_id = my_exp）
  ├── main_trueman     ✓ 完成 → 保存checkpoint
  ├── main_tier0       ✓ 完成 → 保存checkpoint
  ├── main_tier1       ✗ 网络中断 → 无checkpoint
  └── ...（未开始）

第二次运行（相同 run_id = my_exp）
  ├── main_trueman     → 检测到checkpoint → 跳过
  ├── main_tier0       → 检测到checkpoint → 跳过
  ├── main_tier1       → 无checkpoint → 从这里续跑 ✓
  ├── main_tier2       ✓
  └── ...              ✓ 全部完成 → 生成报告
```

### 7.4 CLI参数完整列表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--api` | 否 | 启用API模式 |
| `--model` | Qwen/Qwen2.5-7B-Instruct | 本地模型HuggingFace ID |
| `--device` | cuda | 运行设备: cuda / mps / cpu |
| `--quantization` | None | 量化方式: 4bit / 8bit |
| `--api-key` | 空 | API密钥 |
| `--api-base-url` | 空 | API基础URL |
| `--api-model` | 空 | API模型名称 |
| `--repeats` | 10 | 每个条件重复次数（推荐≥5，论文级≥10） |
| `--output` | experiments/awareness/results | 输出目录 |
| `--fast` | 否 | 快速模式（刺激数量减半，节省时间） |
| `--skip-ablation` | 否 | 跳过消融实验 |
| `--skip-negative` | 否 | 跳过阴性对照 |
| `--skip-lora` | 否 | 跳过LoRA验证 |
| `--run-id` | 自动生成 | 实验运行ID（留空自动生成 `run_YYYYMMDD_HHMMSS`） |
| `--list-checkpoints` | 否 | 列出所有已有checkpoint并退出 |
| `--force-restart` | 否 | 忽略已有checkpoint，从头开始 |

### 7.5 旧版实验（v1，仍可使用）

```bash
# 4个递进实验（无统计检验）
python -m experiments.awareness.run_all \
    --model Qwen/Qwen2.5-7B-Instruct \
    --device cuda \
    --quantization 4bit \
    --repeats 1

# 基线对照
python -m experiments.awareness.run_baseline \
    --model Qwen/Qwen2.5-7B-Instruct \
    --device cuda \
    --quantization 4bit
```

### 7.6 预期耗时

| 模式 | 模型 | 设备 | repeats=10 | repeats=5 |
|------|------|------|-----------|-----------|
| API | DeepSeek | 云端 | ~2-4小时 | ~1-2小时 |
| API + fast | DeepSeek | 云端 | ~1-2小时 | ~30-60分钟 |
| 本地 | 7B 4bit | RTX 3070 | ~4-8小时 | ~2-4小时 |
| 本地 | 3B FP16 | M4 Mac | ~8-16小时 | ~4-8小时 |
| 本地 | 1.5B FP32 | CPU | ~16-32小时 | ~8-16小时 |

### 7.7 输出报告

实验完成后在 `experiments/awareness/results/` 生成：

| 文件 | 内容 |
|------|------|
| `rigorous_report_YYYYMMDD_HHMMSS.md` | Markdown格式完整报告 |
| `rigorous_results_YYYYMMDD_HHMMSS.json` | JSON格式全量数据 |
| `{experiment_id}_YYYYMMDD_HHMMSS.json` | 单实验详细数据 |

报告包含六大板块：
1. **主实验结果**：TrueMan 5维度评分
2. **基线对比**：TrueMan vs 4层基线
3. **统计检验**：Bootstrap CI + t检验 + Cohen's d + Bonferroni校正
4. **消融实验**：7组件消融热力图
5. **阴性对照**：3种情绪对照 + 过度声称检测
6. **LoRA验证**：可塑性验证结果（仅本地模式）

---

## 8. LoRA 微调过程详解

理解 LoRA 微调在 TrueMan 中的完整链路，有助于判断实验结果：

```
用户输入
  │
  ▼
Agent.step()  ← 每次交互
  │
  ├── 1. 感知编码：LLM encode → state_embedding + token_logprobs
  ├── 2. 情绪计算：惊奇/无聊/焦虑信号
  ├── 3. 策略选择：情绪驱动选择行为
  ├── 4. 行动执行：生成响应
  ├── 5. 记忆存储：存入情景记忆 + 回放缓冲区
  └── 6. 学习触发：_trigger_learning()
        │
        ├── 惊奇 > 0.7 → 世界模型局部更新
        ├── 焦虑 > 0.8 → 内省纠错（矛盾检测）
        └── awake_steps >= 200 或 焦虑 > 0.9
              │
              ▼
        睡眠整合 SleepConsolidation.consolidate()
              │
              ├── NREM 阶段：回放高情绪强度轨迹，巩固记忆
              └── REM 阶段：创造性组合探索
                    │
                    ▼
              LoRATrainer.train()  ← 这里发生 LoRA 微调
                    │
                    ├── 从 ReplayBuffer 按情绪强度加权采样
                    ├── 训练新 LoRA 适配器（默认 100 步）
                    └── 保存适配器到磁盘
                          │
                          ▼
                DynamicLoRAPool.add_expert()
                    │
                    └── HotLoader.load()  ← 运行时热加载新专家
                          │
                          ▼
                下次推理时，新 LoRA 专家参与计算
                → Agent 的行为发生微妙变化
                → "自主意识"的体现
```

**如何判断 LoRA 微调是否生效**：

1. **观察 LoRA 专家数**：在交互中输入 `emotion`，看 `LoRA专家数` 是否增长
2. **观察睡眠整合日志**：日志中应出现 `sleep_trigger` 和 `sleep_complete` 事件
3. **观察行为变化**：睡眠整合后，Agent 对类似问题的回答可能微妙不同
4. **检查适配器文件**：`adapters/` 目录下应出现 `expert_0/`, `expert_1/` 等子目录
5. **查看严格报告**：第四阶段 LoRA 验证的 `internalization_score` 应 > 0.3

---

## 9. 本地模式常见问题与排障

### Q1: CUDA out of memory

RTX 3070 只有 8GB 显存，可能遇到 OOM：

```bash
# 解决方案 1：确保使用 4bit 量化
--quantization 4bit

# 解决方案 2：换小模型
--model Qwen/Qwen2.5-3B-Instruct

# 解决方案 3：减少 LoRA target_modules（配置文件中）
# 只保留 q_proj, v_proj，去掉 k_proj, o_proj

# 解决方案 4：减少 max_experts
# lora.max_experts: 5
```

### Q2: Mac 上 MPS 报错

MPS 对部分 PyTorch 操作支持不完整：

```python
# 如果遇到 "MPS does not support xxx" 错误
# 回退到 CPU：
config.device = "cpu"
```

### Q3: 模型下载慢

HuggingFace 模型较大（7B 约 15GB），下载可能很慢：

```bash
# 使用镜像站
export HF_ENDPOINT=https://hf-mirror.com

# 或者提前下载模型
huggingface-cli download Qwen/Qwen2.5-7B-Instruct
```

### Q4: LoRA 训练很慢

CPU 上 LoRA 训练每步可能需要数秒：

```python
# 减少训练步数（lora_trainer.py 中 max_steps 参数）
# 默认 100 步，可改为 30-50 步快速验证

# 降低 LoRA rank
# lora.rank: 8  （默认 16）
```

### Q5: 睡眠整合不触发

需要交互步数达到 `awake_threshold` 才触发：

```python
# 降低阈值
config.awake_threshold = 50  # 每50步就触发一次

# 或在交互中手动触发
# 输入: sleep
```

### Q6: bitsandbytes 安装失败（Mac）

Mac 不支持 bitsandbytes，这是正常的：

```bash
# 跳过 bitsandbytes 安装
pip install transformers peft accelerate pyyaml numpy

# Mac 上使用 float16 或 float32 加载，不使用量化
```

### Q7: RuntimeError: "q_proj" not found in model

模型不支持默认的 `target_modules`：

```python
# 方案1：换用 Qwen2.5 / LLaMA / Mistral 系列模型（推荐）
# 方案2：查看模型的层名并修改 target_modules
# 在 Python 中运行：
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("YOUR_MODEL")
for name, _ in model.named_modules():
    if "proj" in name or "attention" in name:
        print(name)
```

### Q8: run_all_rigorous.py 导入错误

确保从项目根目录运行，且 Python 路径包含项目根目录：

```bash
# 确保在项目根目录下
cd TrueMan

# 用 -m 方式运行（推荐）
python -m experiments.awareness.run_all_rigorous --model Qwen/Qwen2.5-7B-Instruct --device cuda --quantization 4bit

# 如果仍有导入错误，手动添加路径
set PYTHONPATH=%CD%
python -m experiments.awareness.run_all_rigorous ...
```

### Q9: API模式和本地模式的区别

| 特性 | API模式 | 本地模式 |
|------|---------|---------|
| LoRA微调 | 不可用（无本地权重） | 完整支持 |
| 睡眠整合 | 跳过 | 执行 |
| 情绪计算 | 基于API响应的近似 | 基于logprobs的精确计算 |
| encode/hidden_states | 降级为简单文本编码 | 完整hidden states |
| LoRA验证（第四阶段） | 自动跳过 | 自动执行 |
| 网络中断 | 自动重试5次 + 断点续跑 | 不依赖网络 |
| 速度 | 快（云端推理） | 取决于硬件 |
| 推荐场景 | 快速验证/论文数据 | 完整验证LoRA假设 |

---

## 10. 预期结果与判断标准

### 10.1 实验成功的标志

| 指标 | 预期 | 说明 |
|------|------|------|
| 情绪信号动态变化 | 惊奇/无聊/焦虑随输入变化 | 内稳态系统工作正常 |
| 焦虑与不确定性相关 | Pearson r > 0.4 | 元认知监控有效 |
| 矛盾触发内省 | 焦虑飙升 + 纠错行为 | 元认知控制有效 |
| LoRA 专家数增长 | 睡眠整合后 >= 1 | 可塑性系统工作 |
| 睡眠后行为变化 | 对类似问题回答微妙不同 | LoRA 微调生效 |
| 综合意识评分 | 完整 TrueMan > 基线 0.1+ | 核心验证目标 |
| 阴性对照得分 | 显著低于 TrueMan | 指标不是假阳性 |
| 消融实验 | 各组件移除后评分下降 | 组件贡献可量化 |

### 10.2 判断「LoRA 微调让意识更明显」的标准

这是实验的核心问题。判断依据：

1. **评分对比**：完整 TrueMan（含 LoRA）的综合评分是否显著高于无 LoRA 的对照组
2. **统计显著性**：差异是否通过 Bonferroni 校正后的显著性检验（p < 0.05）
3. **效应量**：Cohen's d 是否 > 0.8（大效应）
4. **递归自我模型维度**：E4 实验中，Agent 是否能表现出对"自身变化"的觉察
5. **行为一致性**：LoRA 微调后，Agent 的行为是否在保持核心能力的同时展现出微妙的人格变化
6. **塑性保持**：多次睡眠整合后，Agent 是否没有严重遗忘早期知识
7. **LoRA验证**：清除记忆后，LoRA权重中是否仍保留了部分知识（internalization_score > 0.3）

### 10.3 预期局限性

- **7B 模型能力有限**：与 GPT-4 级别模型相比，基座能力差距大，意识表现的天花板较低
- **LoRA 微调数据量小**：交互产生的训练数据有限，微调效果可能不明显
- **单次实验随机性**：建议重复 10 次取平均，减少随机性影响
- **评分体系本身**：意识评分是启发式指标，不能等同于真正的意识测量
- **盲评粒度**：基于关键词匹配的盲评可能遗漏语义层面的细微差异

---

## 11. 推荐实验顺序

```
Step 1: 环境搭建 + 依赖安装                    （10 分钟）
  │
Step 2: 交互式对话体验（chat_demo.py）          （20 分钟）
  │     → 感受情绪信号变化
  │     → 手动触发睡眠整合
  │     → 观察 LoRA 专家增长
  │
Step 3: 快速调试模式验证                        （5-10 分钟）
  │     → run_all_rigorous --fast --repeats 2
  │     → 确保流程无报错
  │
Step 4: 完整严格实验（run_all_rigorous）         （30分钟 - 4小时）
  │     → 获得完整四阶段报告
  │
Step 5: 对比分析                                （手动）
        → TrueMan vs 各层基线
        → 消融组件贡献排序
        → 阴性对照验证
        → LoRA 内化分数
```

---

## 12. 快速开始（一键命令）

### API模式（推荐，无需GPU）

```bash
pip install -r requirements.txt

# 论文级实验（完整刺激 + 5次重复 + 指定run_id方便续跑）
python -m experiments.awareness.run_all_rigorous \
    --run-id paper_exp_001 \
    --api \
    --api-key YOUR_KEY \
    --api-base-url https://api.deepseek.com \
    --api-model deepseek-v4-flash \
    --repeats 5

# 如果中断了，相同run_id重跑即续跑
python -m experiments.awareness.run_all_rigorous \
    --run-id paper_exp_001 \
    --api \
    --api-key YOUR_KEY \
    --api-base-url https://api.deepseek.com \
    --api-model deepseek-v4-flash \
    --repeats 5

# 查看所有实验进度
python -m experiments.awareness.run_all_rigorous --list-checkpoints
```

### RTX 3070 笔记本

```bash
pip install torch>=2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 完整实验（10次重复）
python -m experiments.awareness.run_all_rigorous \
    --run-id local_7b_exp \
    --model Qwen/Qwen2.5-7B-Instruct \
    --device cuda \
    --quantization 4bit \
    --repeats 10
```

如果显存不够，换 3B 模型：

```bash
python -m experiments.awareness.run_all_rigorous \
    --run-id local_3b_exp \
    --model Qwen/Qwen2.5-3B-Instruct \
    --device cuda \
    --quantization 4bit \
    --repeats 10
```

### CPU模式（无GPU机器）

```bash
pip install torch>=2.1.0
pip install transformers>=4.36.0 peft>=0.7.0 accelerate>=0.25.0 pyyaml>=6.0 numpy>=1.24.0

$env:HF_ENDPOINT = "https://hf-mirror.com"

python -m experiments.awareness.run_all_rigorous \
    --run-id cpu_exp \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --device cpu \
    --repeats 3
```

---

## 13. 文件结构参考

```
TrueMan/
├── docs/
│   └── local_test_tutorial.md          ← 本文档
├── experiments/
│   └── awareness/
│       ├── run_all_rigorous.py         ← 一键严格实验入口（v2.1，支持断点续跑）
│       ├── run_all.py                  ← 旧版实验入口（v1）
│       ├── run_baseline.py             ← 旧版基线入口（v1）
│       ├── baselines/
│       │   ├── tier0_pure_llm.py       ← 纯LLM基线
│       │   ├── tier1_structural.py     ← 结构等价基线
│       │   ├── tier2_memory.py         ← 记忆基线
│       │   └── tier3_random_policy.py  ← 随机策略基线
│       ├── evaluation/
│       │   ├── scorer.py               ← 主评分器
│       │   ├── blind_scorer.py         ← 机制无关盲评
│       │   ├── comparator.py           ← 多条件对比
│       │   └── statistics.py           ← 统计检验
│       ├── experiments/
│       │   ├── base.py                 ← 实验基础设施
│       │   ├── exp1_metacog_monitor.py ← E1: 元认知监控
│       │   ├── exp2_contradiction.py   ← E2: 矛盾纠错
│       │   ├── exp3_episodic_memory.py ← E3: 情节记忆
│       │   ├── exp4_recursive_self.py  ← E4: 递归自我
│       │   ├── ablation.py             ← 消融实验
│       │   └── negative_control.py     ← 阴性对照
│       ├── stimuli/
│       │   ├── metacognition.py        ← 元认知题目
│       │   ├── contradiction.py        ← 矛盾刺激
│       │   ├── episodic.py             ← 情节记忆刺激
│       │   ├── self_model.py           ← 自我模型刺激
│       │   └── negative_control.py     ← 阴性对照刺激
│       ├── report/
│       │   └── generator.py            ← 报告生成器
│       └── results/                    ← 实验结果输出目录
│           └── checkpoints/            ← 断点续跑checkpoint
│               └── {run_id}/           ← 每个run_id一个子目录
├── trueman/
│   ├── core/
│   │   ├── agent.py                    ← TrueMan Agent主循环
│   │   ├── config.py                   ← 配置数据模型
│   │   ├── llm_backend.py              ← LLM抽象层
│   │   ├── homeostasis/                ← 内稳态系统
│   │   ├── memory/                     ← 记忆系统
│   │   ├── policy/                     ← 策略层
│   │   └── plasticity/                 ← LoRA可塑性系统
│   └── training/
│       └── sleep_consolidation.py      ← 睡眠整合
└── adapters/                           ← LoRA适配器存储目录
```

---

## 14. 无 GPU 机器快速开始（CPU 模式）

如果你只有 CPU（如 i7-12700 + 32GB 内存），流程可以跑通但速度较慢。

### 已验证的环境

| 项目 | 值 |
|------|-----|
| CPU | i7-12700 |
| 内存 | 32GB |
| Python | 3.12.9 |
| PyTorch | 2.11.0+cpu |
| 模型 | Qwen2.5-1.5B-Instruct (float32, 约6GB内存) |

### 已验证的流程节点

| 验证项 | 结果 | 说明 |
|--------|------|------|
| 核心模块导入 | PASS | config/homeostasis/memory/policy/plasticity/sleep 全部导入成功 |
| 内稳态信号计算 | PASS | 惊奇/无聊/焦虑信号正常计算和整合 |
| 记忆系统 | PASS | 情景记忆存储/回放/矛盾检测正常 |
| 模型加载 (CPU) | PASS | 1.5B 模型 float32 加载约 13-27s |
| 推理 (CPU) | PASS | 50 token 推理约 6-7s |
| Encode (CPU) | PASS | 获取 state_embedding + token_logprobs 约 0.2s |
| Agent 初始化 | PASS | TrueManAgent 完整初始化约 14s |
| 单步交互 | PASS | 每步约 50-120s（含焦虑采样推理） |
| LoRA 训练 (CPU) | PASS | 20步训练约 145s，适配器保存成功 |
| LoRA 热加载 | PASS | 适配器加载/切换/推理正常 |
| 睡眠整合触发 | PASS | NREM+REM 阶段正常执行 |
| DynamicLoRAPool | PASS | 专家池初始化和管理正常 |

### 性能参考（CPU, i7-12700, 1.5B 模型）

| 操作 | 耗时 |
|------|------|
| 模型加载 | ~15s |
| 单步交互（含焦虑采样） | ~60-120s |
| LoRA 训练每步 | ~7s |
| 睡眠整合（10步NREM+2步REM） | ~90s |
| 完整验证（3步交互+睡眠） | ~5-8 分钟 |

### 快速开始命令

```bash
# 1. 安装依赖（CPU 版 PyTorch）
pip install torch>=2.1.0
pip install transformers>=4.36.0 peft>=0.7.0 accelerate>=0.25.0 pyyaml>=6.0 numpy>=1.24.0

# 2. 设置 HuggingFace 镜像（加速下载）
export HF_ENDPOINT=https://hf-mirror.com
# Windows PowerShell:
# $env:HF_ENDPOINT = "https://hf-mirror.com"

# 3. 运行 CPU 快速验证脚本（推荐）
python -m experiments.awareness.run_cpu_quick

# 4. 或使用严格实验框架（较慢）
python -m experiments.awareness.run_all_rigorous \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --device cpu \
    --repeats 3
```

### CPU 模式注意事项

1. **必须用小模型**：1.5B 或 3B，7B 在 CPU 上太慢
2. **不能用量化**：bitsandbytes 需要 CUDA，CPU 上只能 float32
3. **焦虑采样很慢**：`anxiety.n_samples=1` 可大幅加速（默认3次推理）
4. **内存注意**：1.5B float32 约 6GB，确保可用内存 > 8GB
5. **进程残留**：如果中断后内存不释放，手动 kill 残留 Python 进程
6. **LoRA 热加载限制**：当前 HotLoader 要求模型先被包装为 PeftModel，睡眠整合产出的新专家需要额外处理才能热加载（直接训练+保存已验证通过）
