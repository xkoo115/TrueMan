"""支柱 1：机制因果证据（SAE + probing + causal intervention）。

工作流：
1. capture_states.py     —— 用 harness/capture.py 捕获 hidden states
2. train_sae.py          —— 训练稀疏自编码器
3. probe_features.py     —— 找出与情绪信号高度相关的特征
4. causal_intervention.py —— 必要性 (clamp) + 充分性 (inject) 双向检验

测试 H3。
"""
