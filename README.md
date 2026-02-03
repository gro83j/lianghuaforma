# lianghuaforma — Mode4 (019/020) 可展示可复现仓库

## 项目简介
Mode4 策略（验收口径：epd、hit@TP1、ev_r、maxDD，以及 p80/p95 DD）。本仓库整理了 019 TP2-only 优化与 020 稳定性审计的完整交付物与复现脚本。

## 关键结论（来自 019/020 报告）
- - preOS: epd=1.0103, hit@TP1=0.8262, hit@TP2=0.3604, PF=1.8653, ev_r=0.1731, maxDD_usd=23.18
- - OS: epd=0.4995, hit@TP1=0.8183, hit@TP2=0.3596, PF=1.9198, ev_r=0.1875, maxDD_usd=14.71
- - ALL: epd=0.8714, hit@TP1=0.8250, hit@TP2=0.3603, PF=1.8740, ev_r=0.1753, maxDD_usd=23.18
- - preOS: actual=22.69 | p80=23.85 (<=60: PASS) | p95=28.95 (<=80: PASS) | n=1533
- - OS: actual=12.25 | p80=19.25 (<=60: PASS) | p95=24.66 (<=80: PASS) | n=322
- - ALL: actual=22.69 | p80=24.41 (<=60: PASS) | p95=28.61 (<=80: PASS) | n=1855
- - OVERALL=PASS

## 复现方法
1) 创建环境：`conda env create -f environment.yml` 或 `pip install -r requirements.txt`
2) 数据已包含在 `data/`（默认使用 `data/42swam/data_xauusd/xauusd_M5.csv`）
3) 运行：`scripts\run_019.ps1` 与 `scripts\run_020.ps1`

## 目录结构
- docs/：方法说明与审计清单
- reports/：019/020 报告与冻结摘要
- artifacts/019、artifacts/020：产物与稳定性证据
- experiments/：脚本副本（已支持环境变量路径注入）
- configs/：selected_config/thresholds/tp2_policy 等关键配置
- data/：复现所需数据
- scripts/：一键复现脚本

## 风险提示
本仓库仅用于研究与复现，不构成任何投资建议。
