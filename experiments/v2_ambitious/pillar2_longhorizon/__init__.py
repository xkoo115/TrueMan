"""支柱 2：长时程参数演化与塑性 (H4 + H5)。

核心实验：5 个条件 × 4 个底模 × ≥4 seed × 30 天连续运行
- 每天 24 步交互（720 步/agent）
- 每 24h 快照参数
- 每周 + Day 0/30 administer probe battery
- Day 30 时清空 episodic memory，测试参数化记忆保留率

输出：
- snapshots/day{D}_{cond}_seed{S}/  —— LoRA + 世界模型 + 记忆 dump
- probes/week{W}_{cond}_seed{S}.json
- captures.h5  (供支柱 1 使用)
- trajectory.csv (用于 H5 free energy 拟合)
"""
