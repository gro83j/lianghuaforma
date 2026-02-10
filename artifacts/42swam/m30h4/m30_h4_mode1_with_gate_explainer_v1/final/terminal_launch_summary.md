FINAL_LAUNCH_BLOCK
A) deployment_mode
- prefilter_only

B) rules
- long_rule: -2367 <= h4_cci20_int <= -1050 | complexity=1
- short_rule: m30_range_mean6_int >= 2 | complexity=1

C) baseline vs prefilter (test/verify/lockbox)
segment=test
baseline: tp1=0.4806 win=0.4951 sum_pnl=703.24 profit_y=119.64 max_dd=-0.0564 sharpe_simple=0.2181 sharpe=1.1977 n_trades=206 freq_y=35.04 freq_ratio=1.000
prefilter: tp1=0.5455 win=0.5606 sum_pnl=685.38 profit_y=120.53 max_dd=-0.0427 sharpe_simple=0.3041 sharpe=1.2472 n_trades=132 freq_y=23.21 freq_ratio=0.641
segment=verify
baseline: tp1=0.4780 win=0.4780 sum_pnl=823.65 profit_y=218.47 max_dd=-0.1306 sharpe_simple=0.2580 sharpe=1.3769 n_trades=159 freq_y=42.17 freq_ratio=1.000
prefilter: tp1=0.5603 win=0.5603 sum_pnl=899.97 profit_y=238.72 max_dd=-0.0752 sharpe_simple=0.3729 sharpe=1.6088 n_trades=116 freq_y=30.77 freq_ratio=0.730
segment=lockbox
baseline: tp1=0.4737 win=0.4737 sum_pnl=639.74 profit_y=224.68 max_dd=-0.0719 sharpe_simple=0.2962 sharpe=1.3811 n_trades=114 freq_y=40.04 freq_ratio=1.000
prefilter: tp1=0.5556 win=0.5556 sum_pnl=670.96 profit_y=235.64 max_dd=-0.0320 sharpe_simple=0.4143 sharpe=1.6459 n_trades=81 freq_y=28.45 freq_ratio=0.711

D) files
- production_config.json path: \Users\junnanma\reports\m30_h4_mode1_with_gate_explainer_v1\final\production_config.json
- launch_readme.md path: \Users\junnanma\reports\m30_h4_mode1_with_gate_explainer_v1\final\launch_readme.md
- prefilter_only_eval output dir path: \Users\junnanma\reports\m30_h4_mode1_with_gate_explainer_v1\prefilter_only_eval
- best_rules_long.json path: \Users\junnanma\reports\m30_h4_mode1_with_gate_explainer_v1\explainer\best_rules_long.json
- best_rules_short.json path: \Users\junnanma\reports\m30_h4_mode1_with_gate_explainer_v1\explainer\best_rules_short.json