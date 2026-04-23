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

## 4. 配置文件准备

### 4.1 设备 C 配置（RTX 3070 + 7B 模型 + 4bit 量化）

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

### 4.2 设备 B 配置（M4 Mac + 3B 模型 + float32）

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

## 5. 实验流程

### 5.1 实验一：交互式对话体验（最直观）

这是最快速感受「情绪驱动 + 实时学习」的方式：

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

### 5.2 实验二：完整自我意识验证实验

这是项目自带的 4 个递进实验，量化评估意识水平：

```bash
# 设备 C（RTX 3070）
python -m experiments.awareness.run_all \
    --model Qwen/Qwen2.5-7B-Instruct \
    --device cuda \
    --quantization 4bit \
    --repeats 1

# 设备 B（Mac mini）
python -m experiments.awareness.run_all \
    --model Qwen/Qwen2.5-3B-Instruct \
    --device mps \
    --repeats 1
```

4 个实验依次测试：

| 实验 | 测试内容 | 预期耗时（RTX 3070） | 预期耗时（M4） |
|------|---------|---------------------|---------------|
| E1: 元认知监控 | 给不确定问题，焦虑信号是否上升 | ~5 分钟 | ~15 分钟 |
| E2: 矛盾纠错 | 注入矛盾，是否触发内省纠错 | ~5 分钟 | ~15 分钟 |
| E3: 情节记忆 | 经历事件后回忆早期经历和情绪 | ~10 分钟 | ~30 分钟 |
| E4: 递归自我模型 | 长期交互后关于"自我"的元层面问题 | ~15 分钟 | ~45 分钟 |

实验结束后会输出评分摘要：
```
意识维度评分摘要
============================================================
  元认知监控:   0.4500
  元认知控制:   0.3800
  情节性记忆:   0.5200
  时间连续性:   0.4100
  递归自我模型: 0.6000
  ─────────────────────
  综合评分:     0.4720
============================================================
```

### 5.3 实验三：对比实验（核心验证）

**这是验证「LoRA 微调是否让意识更明显」的关键实验。**

需要跑三组对照，逐步开启功能：

#### 对照组 1：纯 LLM 基线（无情绪、无 LoRA、无记忆）

```bash
python -m experiments.awareness.run_baseline \
    --model Qwen/Qwen2.5-7B-Instruct \
    --device cuda \
    --quantization 4bit
```

#### 对照组 2：有情绪但无 LoRA（API 模式）

API 模式下没有 LoRA 微调，但有情绪驱动策略选择：

```bash
python -m experiments.awareness.run_fast \
    --api-key YOUR_DEEPSEEK_KEY \
    --api-base-url https://api.deepseek.com \
    --api-model deepseek-chat
```

> 如果没有 DeepSeek API Key，可以跳过此对照组，或使用其他 OpenAI 兼容 API。

#### 实验组：完整 TrueMan（情绪 + LoRA + 记忆）

```bash
python -m experiments.awareness.run_all \
    --model Qwen/Qwen2.5-7B-Instruct \
    --device cuda \
    --quantization 4bit \
    --repeats 1
```

#### 对比分析

| 组别 | 情绪驱动 | LoRA 微调 | 情景记忆 | 预期综合评分 |
|------|---------|----------|---------|------------|
| 纯 LLM 基线 | 无 | 无 | 无 | ~0.20-0.30 |
| +情绪（API） | 有 | 无 | 有限 | ~0.30-0.40 |
| 完整 TrueMan | 有 | 有 | 有 | ~0.40-0.55 |

**关键观察点**：
- 完整 TrueMan 的评分是否显著高于基线？
- 递归自我模型维度（E4）的评分差异最大，因为 LoRA 微调让 Agent 的"自我"在持续变化
- 情节记忆维度（E3）的差异也较大，因为记忆系统 + LoRA 让 Agent 能"记住"并"内化"经历

---

## 6. LoRA 微调过程详解

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

---

## 7. 常见问题与排障

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

---

## 8. 预期结果与判断标准

### 8.1 实验成功的标志

| 指标 | 预期 | 说明 |
|------|------|------|
| 情绪信号动态变化 | 惊奇/无聊/焦虑随输入变化 | 内稳态系统工作正常 |
| 焦虑与不确定性相关 | Pearson r > 0.4 | 元认知监控有效 |
| 矛盾触发内省 | 焦虑飙升 + 纠错行为 | 元认知控制有效 |
| LoRA 专家数增长 | 睡眠整合后 >= 1 | 可塑性系统工作 |
| 睡眠后行为变化 | 对类似问题回答微妙不同 | LoRA 微调生效 |
| 综合意识评分 | 完整 TrueMan > 基线 0.1+ | 核心验证目标 |

### 8.2 判断「LoRA 微调让意识更明显」的标准

这是实验的核心问题。判断依据：

1. **评分对比**：完整 TrueMan（含 LoRA）的综合评分是否显著高于无 LoRA 的对照组
2. **递归自我模型维度**：E4 实验中，Agent 是否能表现出对"自身变化"的觉察（如"我觉得我最近对这个问题有了新的理解"）
3. **行为一致性**：LoRA 微调后，Agent 的行为是否在保持核心能力的同时展现出微妙的人格变化（而非退化或混乱）
4. **塑性保持**：多次睡眠整合后，Agent 是否没有严重遗忘早期知识（评估指标中的 `plasticity_retention`）

### 8.3 预期局限性

- **7B 模型能力有限**：与 GPT-4 级别模型相比，基座能力差距大，意识表现的天花板较低
- **LoRA 微调数据量小**：交互产生的训练数据有限，微调效果可能不明显
- **单次实验随机性**：建议重复 3 次取平均，减少随机性影响
- **评分体系本身**：意识评分是启发式指标，不能等同于真正的意识测量

---

## 9. 推荐实验顺序

```
Step 1: 环境搭建 + 依赖安装                    （10 分钟）
  │
Step 2: 交互式对话体验（chat_demo.py）          （20 分钟）
  │     → 感受情绪信号变化
  │     → 手动触发睡眠整合
  │     → 观察 LoRA 专家增长
  │
Step 3: 完整意识验证实验（run_all.py）          （30-60 分钟）
  │     → 获得 4 个维度的评分
  │
Step 4: 基线对照实验（run_baseline.py）         （15-30 分钟）
  │     → 获得无情绪/无 LoRA 的基线评分
  │
Step 5: 对比分析                                （手动）
        → 完整 TrueMan vs 基线
        → 判断 LoRA 微调的增量效果
```

---

## 10. 快速开始（一键命令）

如果你只想最快看到结果，在 **RTX 3070 笔记本** 上执行：

```bash
# 安装
conda create -n trueman python=3.10 -y && conda activate trueman
pip install torch>=2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 跑完整实验
python -m experiments.awareness.run_all \
    --model Qwen/Qwen2.5-7B-Instruct \
    --device cuda \
    --quantization 4bit \
    --repeats 1

# 跑基线对照
python -m experiments.awareness.run_baseline \
    --model Qwen/Qwen2.5-7B-Instruct \
    --device cuda \
    --quantization 4bit

# 对比两组评分，看差异
```

如果显存不够，换 3B 模型：

```bash
python -m experiments.awareness.run_all \
    --model Qwen/Qwen2.5-3B-Instruct \
    --device cuda \
    --quantization 4bit \
    --repeats 1
```

---

## 11. 无 GPU 机器快速开始（CPU 模式）

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

# 或直接用 run_all.py（较慢，约 30-60 分钟）
python -m experiments.awareness.run_all \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --device cpu \
    --repeats 1
```

### CPU 模式注意事项

1. **必须用小模型**：1.5B 或 3B，7B 在 CPU 上太慢
2. **不能用量化**：bitsandbytes 需要 CUDA，CPU 上只能 float32
3. **焦虑采样很慢**：`anxiety.n_samples=1` 可大幅加速（默认3次推理）
4. **内存注意**：1.5B float32 约 6GB，确保可用内存 > 8GB
5. **进程残留**：如果中断后内存不释放，手动 kill 残留 Python 进程
6. **LoRA 热加载限制**：当前 HotLoader 要求模型先被包装为 PeftModel，睡眠整合产出的新专家需要额外处理才能热加载（直接训练+保存已验证通过）
