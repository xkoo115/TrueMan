"""支柱 3：Butlin et al. (2023) 14 indicators 的实证测量。

我们挑选 5 项最可计算、最与本框架相关的 indicator：

| Indicator   | 测量方法                                                |
|-------------|--------------------------------------------------------|
| HOT-1       | meta-d' / d' ratio (probe-based)                       |
| HOT-2       | RSA between confidence dimension and behavior          |
| GWT-1/2     | Cross-module attention 整合度                          |
| RPT-1       | Reentrant signature: layer-wise activation correlation |
| AE          | Goal-directed persistence under perturbation           |

每个 indicator 接收 (agent, probe_set) → 返回 dict[str, float] 指标。

测试 H1（meta-d' 通过 HOT-1 实现），其余作为 multi-indicator coverage。

运行器：run_indicators.py 对所有 (cond, model, seed) 调度完整 battery。
"""
