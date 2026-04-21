# TrueMan — 系统实施规划

> 基于"实时可塑性与内稳态驱动的自治 AI Agent"架构构想

---

## 1. 项目愿景

构建一个能够**实时长脑子**的自治 AI Agent：不依赖外部标注的 Loss 函数，而是通过内稳态驱动的"情绪信号"（惊奇、无聊、焦虑）实现自发学习、自我组织与持续演化。

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    TrueMan Agent                         │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │           L0: 内稳态内核 (Homeostasis Core)        │  │
│  │                                                   │  │
│  │   ┌─────────┐  ┌─────────┐  ┌─────────────┐     │  │
│  │   │ 惊奇    │  │ 无聊    │  │ 焦虑        │     │  │
│  │   │Surprise │  │Boredom  │  │ Anxiety     │     │  │
│  │   │ δ = y-ŷ │  │ H(s)≈0  │  │ σ²(s) ↑↑   │     │  │
│  │   └────┬────┘  └────┬────┘  └──────┬──────┘     │  │
│  │        │            │               │             │  │
│  │        └─────────┬──┴───────────────┘             │  │
│  │                  ▼                                 │  │
│  │         ┌──────────────────┐                      │  │
│  │         │  情绪信号整合器   │                      │  │
│  │         │  L_total = αS   │                      │  │
│  │         │         + βB    │                      │  │
│  │         │         + γA    │                      │  │
│  │         └────────┬─────────┘                      │  │
│  └──────────────────┼───────────────────────────────┘  │
│                     │                                    │
│  ┌──────────────────┼───────────────────────────────┐  │
│  │           L1: 感知执行层 (System 1)               │  │
│  │                                                   │  │
│  │   外部输入 → [感知编码器] → 快速权重记忆 → 行动输出  │  │
│  │              (ViT/LLM)    (FWP)         (动作空间)  │  │
│  │                                                   │  │
│  │   特征: 毫秒级响应, 无梯度更新                      │  │
│  └──────────────────┬───────────────────────────────┘  │
│                     │ 情绪信号触发                      │
│  ┌──────────────────┼───────────────────────────────┐  │
│  │           L2: 反思缓存层 (Epistemic Buffer)       │  │
│  │                                                   │  │
│  │   ┌──────────────┐  ┌──────────────────────┐     │  │
│  │   │ 短期记忆     │  │ Thought Traces 缓存  │     │  │
│  │   │ (海马体)     │  │ (交互轨迹+情绪标注)  │     │  │
│  │   │ 容量: N_recent│  │ 优先级: 情绪强度排序 │     │  │
│  │   └──────┬───────┘  └──────────┬───────────┘     │  │
│  │          │ 互补学习             │                  │  │
│  │          ▼                     ▼                  │  │
│  │   ┌──────────────────────────────────────────┐   │  │
│  │   │         反思推理引擎 (LLM Self-Reflect)  │   │  │
│  │   │    "为什么这件事让我惊讶?"                 │   │  │
│  │   │    "我是否陷入了重复模式?"                 │   │  │
│  │   └──────────────────┬───────────────────────┘   │  │
│  └──────────────────────┼───────────────────────────┘  │
│                     │ 高价值事件                       │
│  ┌──────────────────┼───────────────────────────────┐  │
│  │           L3: 后台整合层 (Async Trainer)          │  │
│  │                                                   │  │
│  │   ┌────────────┐  ┌────────────┐  ┌───────────┐ │  │
│  │   │ 经验回放    │  │ 睡眠整合    │  │ 在线蒸馏   │ │  │
│  │   │ (Replay)   │  │ (NREM/REM) │  │ (Distill) │ │  │
│  │   └──────┬─────┘  └──────┬─────┘  └─────┬─────┘ │  │
│  │          └───────────┬───┘              │         │  │
│  │                      ▼                  │         │  │
│  │              ┌──────────────┐           │         │  │
│  │              │ LoRA 训练器  │◄──────────┘         │  │
│  │              │ (事件驱动)   │                      │  │
│  │              │ 稀疏 + 正交  │                      │  │
│  │              └──────┬───────┘                      │  │
│  └─────────────────────┼────────────────────────────┘  │
│                        │ 训练完成                       │
│  ┌─────────────────────┼────────────────────────────┐  │
│  │           L4: 塑性存储层 (Plastic Memory)         │  │
│  │                                                   │  │
│  │   ┌──────────────────────────────────────────┐   │  │
│  │   │          动态 LoRA 专家池                  │   │  │
│  │   │                                          │   │  │
│  │   │  [LoRA_v1] [LoRA_v2] ... [LoRA_vN]       │   │  │
│  │   │      │        │              │            │   │  │
│  │   │      └────────┼──────────────┘            │   │  │
│  │   │               ▼                           │   │  │
│  │   │     共享低秩子空间 (Share)                 │   │  │
│  │   │     + 热加载路由器 (NeuroLoRA Gate)        │   │  │
│  │   └──────────────────────────────────────────┘   │  │
│  │                                                   │  │
│  │   特征: 物理拓扑改变, 从"查阅"到"内化"             │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │           世界模型 (Internal Simulator)            │  │
│  │                                                   │  │
│  │   状态预测器 s_{t+1} = f(s_t, a_t)               │  │
│  │   用于: 惊奇计算 | 行为预演 | 焦虑评估             │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 3. 分阶段实施路线图

### Phase 0: 基础设施搭建 (2 周)

**目标**: 建立项目骨架与实验框架

```
trueman/
├── core/
│   ├── homeostasis/          # 内稳态内核
│   │   ├── signals.py        # 情绪信号计算 (惊奇/无聊/焦虑)
│   │   ├── integrator.py     # 信号整合器
│   │   └── drives.py         # 内驱状态管理
│   ├── memory/
│   │   ├── episodic.py       # 短期情景记忆 (海马体)
│   │   ├── semantic.py       # 长期语义记忆 (新皮层)
│   │   └── replay.py         # 经验回放缓冲区
│   ├── plasticity/
│   │   ├── fast_weights.py   # 快速权重模块
│   │   ├── lora_pool.py      # 动态 LoRA 专家池
│   │   ├── trainer.py        # 异步 LoRA 训练器
│   │   └── hot_loader.py     # 热加载路由器
│   ├── world_model/
│   │   ├── predictor.py      # 状态预测器
│   │   └── simulator.py      # 内部模拟器
│   └── agent.py              # Agent 主循环
├── training/
│   ├── sleep_consolidation.py # 睡眠整合模块
│   ├── online_distill.py      # 在线蒸馏
│   └── continual_bp.py        # 持续反向传播
├── evaluation/
│   ├── atari_bench.py         # Atari 基准测试
│   ├── text_bench.py          # 文本推理基准
│   └── metrics.py             # 评估指标
├── configs/
│   ├── base.yaml
│   ├── atari.yaml
│   └── llm_agent.yaml
├── docs/
│   ├── references_report.md
│   └── implementation_plan.md
└── experiments/
    ├── phase1_surprise/
    ├── phase2_boredom/
    └── phase3_anxiety/
```

**技术选型**:
- 框架: PyTorch + HuggingFace Transformers
- LoRA: PEFT 库 / 自研稀疏动态 LoRA
- 强化学习: Gymnasium (Atari) → 自定义环境
- 分布式训练: PyTorch DDP / Ray
- 配置管理: Hydra + OmegaConf

---

### Phase 1: 惊奇信号驱动学习 (4 周)

**理论依据**: 自由能原理、预测编码 (参考文献 1.1–1.10)

**目标**: Agent 能通过预测误差自动识别"意外"并触发学习

#### 1.1 世界模型构建

```python
class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, state, action):
        predicted_next_state = self.predictor(
            torch.cat([state, action], dim=-1)
        )
        return predicted_next_state

    def prediction_error(self, state, action, next_state):
        predicted = self.forward(state, action)
        return F.mse_loss(predicted, next_state, reduction='none')
```

#### 1.2 惊奇信号计算

```python
class SurpriseSignal:
    def __init__(self, threshold=2.0, decay=0.99):
        self.threshold = threshold
        self.decay = decay
        self.running_mean = 0.0
        self.running_var = 1.0

    def compute(self, prediction_error):
        normalized = (prediction_error - self.running_mean) / (self.running_var + 1e-8)
        surprise = torch.relu(normalized - self.threshold)
        self._update_stats(prediction_error)
        return surprise

    def _update_stats(self, error):
        self.running_mean = self.decay * self.running_mean + (1 - self.decay) * error.item()
        self.running_var = self.decay * self.running_var + (1 - self.decay) * (error.item() - self.running_mean) ** 2
```

#### 1.3 验证实验

| 实验 | 环境 | 成功指标 |
|------|------|----------|
| E1.1 | Atari (Pong/Breakout) | 仅惊奇驱动探索 vs 随机探索的得分曲线 |
| E1.2 | CartPole | 预测误差收敛速度 |
| E1.3 | 文本推理 (GSM8K 子集) | 错误类型预测准确率 |

---

### Phase 2: 无聊信号与好奇心探索 (3 周)

**理论依据**: 内在动机、信息熵、学习进度 (参考文献 7.1–7.8, 8.1–8.6)

**目标**: 当环境高度可预测时，Agent 主动寻求新信息

#### 2.1 无聊信号计算

```python
class BoredomSignal:
    def __init__(self, window_size=100, temperature=1.0):
        self.window_size = window_size
        self.temperature = temperature
        self.prediction_errors = deque(maxlen=window_size)
        self.state_embeddings = deque(maxlen=window_size)

    def compute(self, prediction_error, state_embedding):
        self.prediction_errors.append(prediction_error.item())
        self.state_embeddings.append(state_embedding.detach())

        temporal_novelty = self._temporal_novelty()
        state_diversity = self._state_diversity()
        learning_progress = self._learning_progress()

        boredom = 1.0 - (temporal_novelty + state_diversity + learning_progress) / 3.0
        return torch.tensor(boredom)

    def _temporal_novelty(self):
        if len(self.prediction_errors) < 2:
            return 1.0
        errors = np.array(self.prediction_errors)
        variance = np.var(errors[-20:])
        return min(variance / self.temperature, 1.0)

    def _state_diversity(self):
        if len(self.state_embeddings) < 2:
            return 1.0
        embeddings = torch.stack(list(self.state_embeddings))
        pairwise_dist = torch.pdist(embeddings).mean()
        return min(pairwise_dist.item() / self.temperature, 1.0)

    def _learning_progress(self):
        if len(self.prediction_errors) < 10:
            return 1.0
        errors = np.array(self.prediction_errors)
        recent = errors[-5:].mean()
        older = errors[-10:-5].mean()
        progress = abs(older - recent)
        return min(progress / self.temperature, 1.0)
```

#### 2.2 好奇心驱动的行为策略

```python
class CuriosityPolicy:
    def __init__(self, base_policy, surprise_signal, boredom_signal):
        self.base_policy = base_policy
        self.surprise = surprise_signal
        self.boredom = boredom_signal

    def select_action(self, state, world_model, action_space):
        surprise = self.surprise.compute(
            world_model.prediction_error(state, ...)
        )
        boredom = self.boredom.compute(...)

        if boredom > 0.8:
            action = self._exploratory_action(state, action_space)
        elif surprise > 0.5:
            action = self._investigate_action(state, action_space)
        else:
            action = self.base_policy.act(state)

        return action

    def _exploratory_action(self, state, action_space):
        embeddings = [world_model.encode(state, a) for a in action_space]
        distances = [torch.pdist(torch.stack([state_emb, e])).item() for e in embeddings]
        return action_space[np.argmax(distances)]
```

#### 2.3 验证实验

| 实验 | 环境 | 成功指标 |
|------|------|----------|
| E2.1 | Atari (Montezuma's Revenge) | 无聊驱动 vs ε-greedy 的房间探索数 |
| E2.2 | 文本对话 | 自动切换话题的连贯性 |
| E2.3 | 迷宫导航 | 探索覆盖率曲线 |

---

### Phase 3: 焦虑信号与自我纠错 (3 周)

**理论依据**: 认知失调、不确定性、睡眠整合 (参考文献 6.1–6.9, 9.1–9.6)

**目标**: Agent 在内部逻辑冲突时进入内省模式

#### 3.1 焦虑信号计算

```python
class AnxietySignal:
    def __init__(self, n_models=3, threshold=0.3):
        self.n_models = n_models
        self.threshold = threshold

    def compute(self, predictions: list[torch.Tensor]):
        pairwise_disagreement = self._disagreement(predictions)
        entropy = self._predictive_entropy(predictions)
        variance = self._output_variance(predictions)

        anxiety = (pairwise_disagreement + entropy + variance) / 3.0
        return anxiety

    def _disagreement(self, predictions):
        total = 0.0
        count = 0
        for i, j in itertools.combinations(range(len(predictions)), 2):
            total += F.kl_div(
                predictions[i].log(), predictions[j], reduction='batchmean'
            )
            count += 1
        return total / max(count, 1)

    def _predictive_entropy(self, predictions):
        avg_pred = torch.stack(predictions).mean(dim=0)
        return -(avg_pred * (avg_pred + 1e-8).log()).sum(-1).mean()

    def _output_variance(self, predictions):
        return torch.stack(predictions).var(dim=0).mean()
```

#### 3.2 睡眠/内省机制

```python
class SleepConsolidation:
    def __init__(self, model, episodic_memory, lora_trainer):
        self.model = model
        self.memory = episodic_memory
        self.trainer = lora_trainer

    def nrem_phase(self, n_steps=100):
        high_value_traces = self.memory.get_by_priority(top_k=n_steps)
        for trace in high_value_traces:
            self.trainer.update(
                input=trace.state,
                target=trace.corrected_target,
                weight=trace.emotional_intensity
            )

    def rem_phase(self, n_steps=50):
        novel_combinations = self._generate_novel_combinations()
        for combo in novel_combinations:
            self.trainer.update(
                input=combo.state,
                target=combo.imagined_target,
                weight=combo.confidence * 0.5
            )

    def consolidate(self):
        self.nrem_phase()
        self.rem_phase()
        new_lora = self.trainer.compile()
        return new_lora
```

#### 3.3 验证实验

| 实验 | 环境 | 成功指标 |
|------|------|----------|
| E3.1 | 矛盾信息注入 | 焦虑信号上升 → 内省 → 逻辑一致性恢复 |
| E3.2 | 灾难性遗忘测试 | 睡眠整合 vs 无整合的任务保持率 |
| E3.3 | 长对话推理 | 自我纠错率 |

---

### Phase 4: 动态 LoRA 可塑性系统 (4 周)

**理论依据**: 快速权重、动态 LoRA、互补学习系统 (参考文献 3.1–3.5, 5.1–5.6, 6.1–6.9)

**目标**: 实现从"短期缓存"到"长期参数化记忆"的物理拓扑改变

#### 4.1 动态 LoRA 专家池

```python
class DynamicLoRAPool:
    def __init__(self, base_model, rank=16, max_experts=50):
        self.base_model = base_model
        self.rank = rank
        self.max_experts = max_experts
        self.experts = {}
        self.shared_subspace = self._init_shared_subspace()
        self.router = NeuroLORAGate(base_model.config.hidden_size, max_experts)

    def _init_shared_subspace(self):
        return {
            name: SharedLowRankTensor(
                weight.shape, rank=self.rank
            )
            for name, weight in self.base_model.named_parameters()
            if 'attention' in name or 'mlp' in name
        }

    def add_expert(self, expert_id, trace_data):
        lora_adapter = self._train_expert(trace_data)
        self.experts[expert_id] = lora_adapter
        self._merge_to_shared(lora_adapter)

        if len(self.experts) > self.max_experts:
            self._prune_oldest()

    def forward(self, input_ids, context_embedding):
        expert_weights = self.router(context_embedding)
        active_experts = [
            (self.experts[eid], w)
            for eid, w in enumerate(expert_weights)
            if w > 0.1 and eid in self.experts
        ]

        base_output = self.base_model(input_ids)
        lora_outputs = [
            self._apply_lora(lora, base_output, w)
            for lora, w in active_experts
        ]
        return base_output + sum(lora_outputs)

    def _prune_oldest(self):
        oldest_id = min(self.experts.keys())
        self._unmerge_from_shared(self.experts[oldest_id])
        del self.experts[oldest_id]
```

#### 4.2 神经调制门控路由

```python
class NeuroLORAGate(nn.Module):
    def __init__(self, hidden_size, max_experts):
        super().__init__()
        self.context_encoder = nn.Linear(hidden_size, 128)
        self.neuromod_gate = nn.Linear(128, max_experts)
        self.orthogonality_loss_weight = 0.1

    def forward(self, context_embedding):
        context = torch.relu(self.context_encoder(context_embedding))
        raw_weights = self.neuromod_gate(context)
        weights = torch.softmax(raw_weights, dim=-1)
        return weights

    def orthogonality_loss(self):
        W = self.neuromod_gate.weight
        WWt = W @ W.T
        I = torch.eye(WWt.shape[0], device=WWt.device)
        return self.orthogonality_loss_weight * ((WWt - I) ** 2).sum()
```

#### 4.3 热加载管理器

```python
class HotLoader:
    def __init__(self, base_model, lora_pool):
        self.base_model = base_model
        self.lora_pool = lora_pool
        self.current_active = set()

    def swap_expert(self, expert_id):
        for eid in self.current_active:
            self._unload(eid)
        self._load(expert_id)
        self.current_active = {expert_id}

    def _load(self, expert_id):
        lora = self.lora_pool.experts[expert_id]
        for name, delta_W in lora.items():
            orig_weight = dict(self.base_model.named_parameters())[name]
            orig_weight.data += delta_W.to(orig_weight.device)

    def _unload(self, expert_id):
        lora = self.lora_pool.experts[expert_id]
        for name, delta_W in lora.items():
            orig_weight = dict(self.base_model.named_parameters())[name]
            orig_weight.data -= delta_W.to(orig_weight.device)
```

#### 4.4 验证实验

| 实验 | 环境 | 成功指标 |
|------|------|----------|
| E4.1 | 多任务 Atari | LoRA 专家数增长 vs 任务保持率 |
| E4.2 | 多领域文本 | 热加载延迟 < 100ms |
| E4.3 | 持续对话 | 新知识内化率（参数改变 vs RAG 检索对比）|

---

### Phase 5: 情绪整合与 Agent 自治 (4 周)

**目标**: 三种情绪信号协同工作，Agent 实现完全自治

#### 5.1 情绪整合器

```python
class HomeostasisCore:
    def __init__(self, config):
        self.surprise = SurpriseSignal(config.surprise_threshold, config.surprise_decay)
        self.boredom = BoredomSignal(config.boredom_window, config.boredom_temp)
        self.anxiety = AnxietySignal(config.anxiety_models, config.anxiety_threshold)

        self.alpha = config.surprise_weight
        self.beta = config.boredom_weight
        self.gamma = config.anxiety_weight

        self.homeostatic_setpoints = {
            'surprise': config.setpoint_surprise,
            'boredom': config.setpoint_boredom,
            'anxiety': config.setpoint_anxiety,
        }

    def compute_drive(self, state, world_model, action):
        prediction_error = world_model.prediction_error(state, action)
        surprise = self.surprise.compute(prediction_error)
        boredom = self.boredom.compute(prediction_error, state)

        predictions = [world_model.predict_with_uncertainty(state) for _ in range(3)]
        anxiety = self.anxiety.compute(predictions)

        deviations = {
            'surprise': abs(surprise - self.homeostatic_setpoints['surprise']),
            'boredom': abs(boredom - self.homeostatic_setpoints['boredom']),
            'anxiety': abs(anxiety - self.homeostatic_setpoints['anxiety']),
        }

        total_drive = (
            self.alpha * deviations['surprise']
            + self.beta * deviations['boredom']
            + self.gamma * deviations['anxiety']
        )

        return total_drive, {
            'surprise': surprise.item(),
            'boredom': boredom.item(),
            'anxiety': anxiety.item(),
            'drive': total_drive.item(),
        }
```

#### 5.2 Agent 主循环

```python
class TrueManAgent:
    def __init__(self, config):
        self.perception = PerceptionLayer(config.base_model)
        self.world_model = WorldModel(config.state_dim, config.action_dim)
        self.homeostasis = HomeostasisCore(config)
        self.episodic_memory = EpisodicMemory(config.memory_size)
        self.lora_pool = DynamicLoRAPool(config.base_model, config.lora_rank)
        self.sleep = SleepConsolidation(config.base_model, self.episodic_memory, self.lora_pool)
        self.policy = CuriosityPolicy(config.base_policy, self.homeostasis.surprise, self.homeostasis.boredom)

        self.awake_steps = 0
        self.awake_threshold = config.awake_threshold

    def step(self, observation):
        state = self.perception.encode(observation)
        action = self.policy.select_action(state, self.world_model, self.action_space)
        next_observation, reward, done, info = self.env.step(action)

        drive, emotions = self.homeostasis.compute_drive(state, self.world_model, action)

        trace = ThoughtTrace(
            state=state,
            action=action,
            observation=next_observation,
            emotions=emotions,
            drive=drive,
            timestamp=self.awake_steps,
        )
        self.episodic_memory.store(trace)

        if emotions['surprise'] > 0.7:
            self._trigger_local_update(trace)

        if emotions['anxiety'] > 0.8:
            self._enter_introspection()

        if self.awake_steps >= self.awake_threshold or emotions['anxiety'] > 0.9:
            self._enter_sleep()

        self.awake_steps += 1
        return action

    def _trigger_local_update(self, trace):
        self.world_model.update(trace.state, trace.action, trace.observation)

    def _enter_introspection(self):
        traces = self.episodic_memory.get_recent(n=20)
        contradictions = self._find_contradictions(traces)
        for c in contradictions:
            self.episodic_memory.boost_priority(c)

    def _enter_sleep(self):
        new_lora = self.sleep.consolidate()
        self.lora_pool.add_expert(
            expert_id=len(self.lora_pool.experts),
            trace_data=self.episodic_memory.get_high_priority(n=100),
        )
        self.awake_steps = 0

    def _find_contradictions(self, traces):
        contradictions = []
        for i, t1 in enumerate(traces):
            for t2 in traces[i+1:]:
                if self._are_contradictory(t1, t2):
                    contradictions.extend([t1, t2])
        return contradictions
```

#### 5.3 验证实验

| 实验 | 环境 | 成功指标 |
|------|------|----------|
| E5.1 | 开放世界探索 | 自主学习曲线（无外部奖励） |
| E5.2 | 持续对话系统 | 知识内化深度（LoRA 权重变化分析） |
| E5.3 | 多任务迁移 | 单一 Agent 自动切换任务能力 |
| E5.4 | 情绪可解释性 | 情绪轨迹 vs 人类标注的一致性 |

---

### Phase 6: 高级机制与优化 (4 周)

**目标**: 优化系统效率，添加高级机制

#### 6.1 持续反向传播（防止塑性丧失）

依据 Dohare et al. (2023) 的持续反向传播方法，定期重初始化低效用单元：

```python
class ContinualBackprop:
    def __init__(self, model, replacement_rate=0.001):
        self.model = model
        self.replacement_rate = replacement_rate
        self.unit_utility = {}

    def update_utility(self):
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                utility = (param.grad * param.data).abs().mean(dim=0)
                if name not in self.unit_utility:
                    self.unit_utility[name] = utility
                else:
                    self.unit_utility[name] = 0.99 * self.unit_utility[name] + 0.01 * utility

    def maybe_reinit(self):
        for name, param in self.model.named_parameters():
            if name in self.unit_utility:
                utility = self.unit_utility[name]
                n_replace = int(self.replacement_rate * utility.shape[0])
                least_useful = torch.argsort(utility)[:n_replace]
                init = torch.nn.init.xavier_uniform_
                for idx in least_useful:
                    init(param.data[idx:idx+1])
```

#### 6.2 多尺度时间常数

依据 MSTH (Hakim, 2026)，不同情绪操作于不同时间尺度：

```python
class MultiScaleHomeostasis:
    def __init__(self):
        self.timescales = {
            'ultra_fast': 5e-3,     # 5ms - 即时惊奇响应
            'fast': 1e-1,           # 100ms - 行为策略调整
            'medium': 60.0,         # 1min - 探索策略切换
            'slow': 3600.0,         # 1hr - LoRA 整合决策
        }

    def get_update_rate(self, signal_type, timescale):
        return 1.0 - math.exp(-1.0 / self.timescales[timescale])
```

---

## 4. 评估指标体系

### 4.1 学习能力指标

| 指标 | 定义 | 目标 |
|------|------|------|
| 样本效率 | 达到阈值性能所需的交互步数 | 优于纯 RL 基线 10x |
| 塑性保持率 | 连续学习 N 个任务后的新任务学习速度 | > 初始学习速度的 80% |
| 遗忘率 | 新任务学习后旧任务性能下降比例 | < 15% |
| LoRA 内化率 | 参数改变量 vs 等效 RAG 检索的性能比 | > 1.0 |

### 4.2 自治能力指标

| 指标 | 定义 | 目标 |
|------|------|------|
| 探索覆盖率 | 无外部奖励下状态空间覆盖比例 | Atari > 80% |
| 好奇心驱动效率 | 自主发现高价值状态的比例 | > 50% |
| 自我纠错率 | 焦虑触发后逻辑一致性恢复比例 | > 70% |
| 睡眠整合增益 | 睡眠前后任务性能提升 | > 5% |

### 4.3 情绪信号质量指标

| 指标 | 定义 | 目标 |
|------|------|------|
| 惊奇校准度 | 惊奇信号与实际异常的相关系数 | > 0.7 |
| 无聊检测准确率 | 无聊信号区分重复/新颖输入的 AUC | > 0.85 |
| 焦虑预测值 | 焦虑信号预测后续错误的 ROC-AUC | > 0.8 |

---

## 5. 技术风险与缓解策略

| 风险 | 描述 | 缓解策略 |
|------|------|----------|
| 塑性丧失 | 深度网络持续学习后丧失可塑性 | 持续反向传播 + Deep Fourier Features |
| 灾难性遗忘 | 新知识覆盖旧知识 | 稀疏 LoRA + 睡眠整合回放 |
| 情绪信号失衡 | 三种信号互相干扰或退化 | 内稳态设定点自适应调节 |
| LoRA 爆炸 | 专家数量无限增长 | Share 共享子空间 + 修剪策略 |
| 计算瓶颈 | 异步训练与推理的资源竞争 | 事件驱动触发 + 优先级队列 |
| 世界模型偏差 | 错误的世界模型导致错误惊奇 | 集成多模型 + 不确定性量化 |

---

## 6. 时间线总览

```
Week  1-2:  Phase 0 — 基础设施搭建
Week  3-6:  Phase 1 — 惊奇信号 (世界模型 + 预测误差)
Week  7-9:  Phase 2 — 无聊信号 (好奇心探索 + 信息熵)
Week 10-12: Phase 3 — 焦虑信号 (认知失调 + 睡眠整合)
Week 13-16: Phase 4 — 动态 LoRA 可塑性系统
Week 17-20: Phase 5 — 情绪整合与 Agent 自治
Week 21-24: Phase 6 — 高级机制优化与论文撰写
```

---

## 7. 论文发表策略

| 目标期刊/会议 | 内容 | 投稿时间 |
|--------------|------|----------|
| NeurIPS / ICML Workshop | Phase 1-2 单篇：内稳态情绪信号驱动学习 | Week 12 |
| ICLR | Phase 3-4 单篇：动态 LoRA + 睡眠整合的持续学习 | Week 20 |
| Nature Machine Intelligence | Phase 5-6 完整架构：内稳态驱动的自治 Agent | Week 28 |
| arXiv (随时) | 技术报告与中间结果 | 每个 Phase 结束 |

---

*本规划为迭代文档，将随实验进展持续更新。*
