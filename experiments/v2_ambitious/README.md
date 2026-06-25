# TrueMan v2 — 雄心路径实验框架

这是论文使用的实验框架（v1 已从仓库移除）。v2 严格遵循
`PREREGISTRATION.md` 中登记的 5 项假设 (H1–H5)，自包含、不依赖任何
旧实验资产。

---

## 一、目录结构

```
experiments/v2_ambitious/
├── PREREGISTRATION.md          # OSF 预注册（A 交付物）
├── README.md                   # 本文件
├── run_v2.py                   # 一行命令主入口
├── configs/
│   ├── longhorizon.yaml        # 支柱 2 配置
│   ├── mechanistic.yaml        # 支柱 1 配置
│   ├── indicators.yaml         # 支柱 3 配置
│   └── ablations.yaml          # 5 条件矩阵
├── harness/                    # 共享基础设施
│   ├── conditions.py           # 5 实验条件包装器
│   ├── capture.py              # 隐藏状态 hook
│   ├── snapshots.py            # 每日参数/记忆快照
│   └── stats.py                # mixed-effects, permutation, BF, meta-d'
├── pillar1_mechanistic/        # H3：SAE + 因果干预
│   ├── train_sae.py
│   ├── probe_features.py
│   └── causal_intervention.py
├── pillar2_longhorizon/        # H4 + H5：30 天连续运行
│   ├── stimulus_stream.py
│   ├── probe_battery.py
│   ├── run.py                  # 单次 (cond, seed, model) 运行
│   └── analyze.py              # 跨运行汇总
├── pillar3_indicators/         # Butlin 14 indicator 子集
│   ├── meta_dprime.py          # HOT-1
│   ├── higher_order.py         # HOT-2
│   ├── global_workspace.py     # GWT-1/2
│   ├── recurrent_processing.py # RPT-1
│   └── phi_approx.py           # ΦR
├── pillar4_falsification/
│   └── cross_model.py          # 跨底模批量调度
└── pillar5_theory/
    ├── fep_freeenergy.py       # H5
    └── pci_perturbation.py     # PCI 类比
```

---

## 二、快速开始（最小可运行 demo）

```bash
# 1. 检查依赖
pip install h5py pyyaml pymer4 statsmodels scikit-learn scipy torch transformers peft

# 2. Dry-run 全流程，确认计划无误
python -m experiments.v2_ambitious.run_v2 --stage all --dry-run

# 3. 看进度
python -m experiments.v2_ambitious.run_v2 --status

# 4. 真正跑一个最小条件子集（Qwen2.5-7B + C0 + 1 seed）
python -m experiments.v2_ambitious.run_v2 \
    --stage all \
    --conditions C0_trueman_full,C3_frozen \
    --seeds 0
```

完整 5 条件 × 4 底模 × 4 seed × 30 天 ≈ 200 GPU-day。Pilot
建议先用 1 model × 2 cond × 2 seed × 7 天 (~5 GPU-day) 验证管线。

---

## 三、5 个支柱 vs 5 个假设

| 支柱 | 文件 | 测试假设 | Nature 角度 |
|------|------|----------|-------------|
| 1 机制 | `pillar1_*` | H3 | 因果证据，最强 |
| 2 长时程 | `pillar2_*` | H4 + H5 | 主图，参数演化 |
| 3 indicator | `pillar3_*` | H1 + H2 | 跨概念覆盖 |
| 4 证伪 | `pillar4_*` + 5 conditions | 全部 | 抗 reviewer 攻击 |
| 5 理论 | `pillar5_*` | H5 + 补充 | FEP/IIT 预测验证 |

---

## 四、数据流

```
stimulus_stream.jsonl                 ← stage 0
            │
            ▼
   pillar2/run.py × N runs            ← stage 1
            │
            ├── snapshots/             (lora, world_model, memory_meta)
            ├── captures.h5            (hidden states + emotion records)
            ├── probes/                (weekly battery)
            └── trajectory.csv         (per-step emotions)
            │
   ┌────────┼────────┐
   ▼        ▼        ▼
pillar1  pillar3   pillar5
SAE +    indicator  FEP fit
causal   battery    PCI
   │        │        │
   └────────┼────────┘
            ▼
     stage 6 → v2_summary.json
```

---

## 五、给"其他模型"的接力提示

代码中 `TODO(other model)` 标注了需要扩充的位置。优先级：

1. **`pillar2_longhorizon/stimulus_stream.py`** — 把 `_bank_factual()` /
   `_bank_dialogue()` / `_bank_contradiction()` 各扩到 ≥ 300 / 30 / 30 条。
2. **`pillar2_longhorizon/probe_battery.py`** — 把 5 个 `PROBE_BANKS`
   扩到 preregistration 规定的 n。
3. **`pillar5_theory/pci_perturbation.py`** — 实现 `administer_pci` 中
   的脉冲注入循环。
4. **`pillar3_indicators/global_workspace.py`** — 验证 `attach()` 在
   你目标底模 (Qwen/Llama/Mistral) 上确实捕获到了 attention weights，
   不同底模 attention 输出 schema 略有差异。
5. **`pillar2_longhorizon/analyze.py::forgetting_score`** — 把占位
   keyword-match judge 替换为 NLI 判断（HuggingFace
   `cross-encoder/nli-deberta-v3-base` 或类似）。

---

## 六、参考资料

- Butlin et al. 2023 (arXiv:2308.08708) — 14 consciousness indicators
- Maniscalco & Lau 2012 — meta-d' / d'
- Tononi et al. 2023 — IIT 4.0 (PLOS Comp Biol)
- Mashour 2020 — PCI in human consciousness
- Cunningham et al. 2024 — Sparse Autoencoders for transformers
- Templeton et al. 2024 — Scaling Monosemanticity (Anthropic)
