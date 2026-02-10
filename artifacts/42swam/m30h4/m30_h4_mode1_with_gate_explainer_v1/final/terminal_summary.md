FINAL_PRINT_BLOCK
A) best_rules_long/short
long_rule: -2367 <= h4_cci20_int <= -1050 | complexity=1 | json=explainer/best_rules_long.json
  train: pass=80/133 (0.602) tp1=0.5625 win=0.5875 sum_pnl=699.04 sharpe_simple=0.3845
  test: pass=70/111 (0.631) tp1=0.5286 win=0.5571 sum_pnl=458.56 sharpe_simple=0.3423
  verify: pass=61/94 (0.649) tp1=0.5246 win=0.5246 sum_pnl=526.03 sharpe_simple=0.3580
short_rule: m30_range_mean6_int >= 2 | complexity=1 | json=explainer/best_rules_short.json
  train: pass=55/69 (0.797) tp1=0.5455 win=0.5636 sum_pnl=119.14 sharpe_simple=0.1686
  test: pass=62/95 (0.653) tp1=0.5645 win=0.5645 sum_pnl=226.82 sharpe_simple=0.2542
  verify: pass=55/65 (0.846) tp1=0.6000 win=0.6000 sum_pnl=373.94 sharpe_simple=0.4096
  combined_prefilter:
    segment=train
  prefilter: tp1=0.5556 win=0.5778 sum_pnl=818.18 avg_pnl=6.0606 profit_y=138.35 max_dd=-0.0546 sharpe_simple=0.3103 sharpe=1.1494 n=135 (L/S=80/55) freq_y=22.83 freq_ratio=0.668
    segment=test
  prefilter: tp1=0.5455 win=0.5606 sum_pnl=685.38 avg_pnl=5.1923 profit_y=120.53 max_dd=-0.0427 sharpe_simple=0.3041 sharpe=1.2472 n=132 (L/S=70/62) freq_y=23.21 freq_ratio=0.641
    segment=verify
  prefilter: tp1=0.5603 win=0.5603 sum_pnl=899.97 avg_pnl=7.7584 profit_y=238.72 max_dd=-0.0752 sharpe_simple=0.3729 sharpe=1.6088 n=116 (L/S=61/55) freq_y=30.77 freq_ratio=0.730

B) baseline vs prefilter
freq_min=0.40 downgrade=no
segment=test
  baseline: tp1=0.4806 win=0.4951 sum_pnl=703.24 avg_pnl=3.4138 profit_y=119.64 max_dd=-0.0564 sharpe_simple=0.2181 sharpe=1.1977 n=206 (L/S=111/95) freq_y=35.04
  prefilter: tp1=0.5455 win=0.5606 sum_pnl=685.38 avg_pnl=5.1923 profit_y=120.53 max_dd=-0.0427 sharpe_simple=0.3041 sharpe=1.2472 n=132 (L/S=70/62) freq_y=23.21 freq_ratio=0.641
  delta: tp1=0.0649 win=0.0655 sum_pnl=-17.86 avg_pnl=1.7785 profit_y=0.89 max_dd=0.0137 sharpe_simple=0.0860 sharpe=0.0495
segment=verify
  baseline: tp1=0.4780 win=0.4780 sum_pnl=823.65 avg_pnl=5.1802 profit_y=218.47 max_dd=-0.1306 sharpe_simple=0.2580 sharpe=1.3769 n=159 (L/S=94/65) freq_y=42.17
  prefilter: tp1=0.5603 win=0.5603 sum_pnl=899.97 avg_pnl=7.7584 profit_y=238.72 max_dd=-0.0752 sharpe_simple=0.3729 sharpe=1.6088 n=116 (L/S=61/55) freq_y=30.77 freq_ratio=0.730
  delta: tp1=0.0824 win=0.0824 sum_pnl=76.32 avg_pnl=2.5782 profit_y=20.24 max_dd=0.0555 sharpe_simple=0.1149 sharpe=0.2319