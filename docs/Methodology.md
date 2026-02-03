# Methodology

本节基于 019/020 报告与脚本中的描述进行摘要。

## 1) Purged + Embargo
- 019 脚本固定 Purge/Embargo：40/40 bars（M5），用于防止时间泄漏。

## 2) Walk-Forward 阈值
- 使用 walk-forward 的阈值文件（见 `configs/thresholds_walkforward.json`），用于冻结非 TP2 部分的门限。

## 3) TP2 Posterior
- 019 TP2 policy（节选）：{"name": "tp2_deep_019", "H2": 240, "thresh_prob": 0.55, "q_target": 0.4, "regime_weighting": "vol_only", "scale_base": 0.8, "trail_no_tp2_mult": 0.6, "trail_tp2_mult": 1000000.0, "posterior_P(p>=0.60)": 0.9999996656475405, "posterior_gate": {"p0": 0.6, "require_P_ge": 0.8}}
- 020 中要求 posterior gate 满足 p0 / require_P_ge（见 020 报告）。

## 4) Stability Audit (020)
- 置信区间/抽样：- method=daily_block_bootstrap | seed=10
- 多随机种子：- N=20 多随机种子复核：seed 仅影响 bootstrap/统计抽样；交易信号/执行不允许受 seed 影响。
- 包含 DD bootstrap、年度/Regime 分段、成本压力与一致性核验。
