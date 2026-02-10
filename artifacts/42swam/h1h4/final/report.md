# Entry Quality Prefilter Exit Optimization Report

## Summary
- exit_base_params: tp1=0.18, tp2=0.54, sl=0.18
- b_pass_count: 80
- strict_count: 222
- relaxed_count: 0
- baseline_exitbase_verify_p95_mae: 1.550819

## Selected Candidates
- C01979 mode=ml_only tp1=0.20 tp2=0.66 sl=0.26 q_short=0.50 q_long=0.70 verify_tp1=0.783 verify_sharpe=0.856 verify_freq_y=30.02 verify_max_dd=-0.115 verify_p95_mae=1.206 verify_worst_fold_p95_mae=1.255
- C01982 mode=ml_only tp1=0.20 tp2=0.66 sl=0.26 q_short=0.50 q_long=0.85 verify_tp1=0.772 verify_sharpe=0.816 verify_freq_y=31.77 verify_max_dd=-0.114 verify_p95_mae=1.212 verify_worst_fold_p95_mae=1.279
- C01973 mode=ml_only tp1=0.20 tp2=0.66 sl=0.26 q_short=0.45 q_long=0.70 verify_tp1=0.783 verify_sharpe=0.859 verify_freq_y=28.77 verify_max_dd=-0.114 verify_p95_mae=1.208 verify_worst_fold_p95_mae=1.261

## Near-miss (test 0.63-0.65, verify stronger)
- count: 291

## Walk-forward rolling validation
- slices_csv: /Users/junnanma/trend_project/new_process/h1_h4/mode1_with_gate_entry_quality_prefilter_exitopt_v1/20251227_214851/final/rolling_slices.csv
- rolling_metrics_long_csv: /Users/junnanma/trend_project/new_process/h1_h4/mode1_with_gate_entry_quality_prefilter_exitopt_v1/20251227_214851/final/rolling_metrics_long.csv
- rolling_summary_csv: /Users/junnanma/trend_project/new_process/h1_h4/mode1_with_gate_entry_quality_prefilter_exitopt_v1/20251227_214851/final/rolling_summary.csv
- rolling_pareto_top1: L00164 (min_tp1=0.7333, min_sharpe_simple=0.3560, min_freq_y=26.09)

## Leakage guards & audit
- leakage_audit_csv: /Users/junnanma/trend_project/new_process/h1_h4/mode1_with_gate_entry_quality_prefilter_exitopt_v1/20251227_214851/final/leakage_audit.csv
- leakage_audit_md: /Users/junnanma/trend_project/new_process/h1_h4/mode1_with_gate_entry_quality_prefilter_exitopt_v1/20251227_214851/final/leakage_audit.md
- leakage_violation_total: 0
- threshold_fit_log_csv: /Users/junnanma/trend_project/new_process/h1_h4/mode1_with_gate_entry_quality_prefilter_exitopt_v1/20251227_214851/final/threshold_fit_log.csv
- walkforward_train_sizes_csv: /Users/junnanma/trend_project/new_process/h1_h4/mode1_with_gate_entry_quality_prefilter_exitopt_v1/20251227_214851/final/walkforward_train_sizes.csv
- slice_boundary_drops_csv: /Users/junnanma/trend_project/new_process/h1_h4/mode1_with_gate_entry_quality_prefilter_exitopt_v1/20251227_214851/final/slice_boundary_drops.csv
- slice_boundary_dropped_total: 17

## Rolling Pareto front summary
- pareto_rolling_3d_csv: /Users/junnanma/trend_project/new_process/h1_h4/mode1_with_gate_entry_quality_prefilter_exitopt_v1/20251227_214851/final/pareto_rolling_3d.csv
- top10_by_rolling_robust_score_csv: /Users/junnanma/trend_project/new_process/h1_h4/mode1_with_gate_entry_quality_prefilter_exitopt_v1/20251227_214851/final/top10_by_rolling_robust_score.csv

## Recommended candidates to forward-test (MT5)
- L00164 min_tp1=0.7333 min_sharpe_simple=0.3560 min_freq_y=26.09
- L00167 min_tp1=0.7333 min_sharpe_simple=0.3262 min_freq_y=25.09
- C01950 min_tp1=0.6667 min_sharpe_simple=0.3113 min_freq_y=25.09

## Notes
- ML gate uses HistGradientBoostingClassifier on H1/H4 int features.
- Rule-only runs are evaluated only on exit_base params for comparison.
- MAE is computed on H1 OHLC and normalized by SL distance.
- intersection_topk empty: top5 primary objective and top3 freq do not overlap.

## Path1 Candidate Pool
- source_csv: /Users/junnanma/trend_project/new_process/h1_h4/mode1_with_gate_entry_quality_prefilter_exitopt_v1/20251227_214851/final/b_pass_candidates.csv
- pool_count: 257
- local_refine_count: 200
- filters: tp1>=0.65, freq_y>=20, n>=80, verify_max_dd>=-0.20, verify_p95_mae<=1.30, worst_fold_p95_mae<=1.35
- candidate_pool_csv: /Users/junnanma/trend_project/new_process/h1_h4/mode1_with_gate_entry_quality_prefilter_exitopt_v1/20251227_214851/final/candidate_pool.csv
- local_refine_candidates_csv: /Users/junnanma/trend_project/new_process/h1_h4/mode1_with_gate_entry_quality_prefilter_exitopt_v1/20251227_214851/final/local_refine_candidates.csv
- top10_by_min_sharpe_csv: /Users/junnanma/trend_project/new_process/h1_h4/mode1_with_gate_entry_quality_prefilter_exitopt_v1/20251227_214851/final/top10_by_min_sharpe.csv
- top10_by_min_sharpe_md: /Users/junnanma/trend_project/new_process/h1_h4/mode1_with_gate_entry_quality_prefilter_exitopt_v1/20251227_214851/final/top10_by_min_sharpe.md
- top10_ids: L00063, L00189, L00077, L00089, L00001, L00012, L00066, L00093, L00101, L00003

## Pareto Summary
- pareto_verify_count: 31
- pareto_robust_count: 23
- local_refine_count: 200
- verify_ranges: tp1[0.7603,0.8333] sharpe[0.5323,1.1459] freq[20.01,40.03]
- robust_ranges: tp1[0.6806,0.7891] sharpe[0.0980,0.6345] freq[24.02,40.03]

## Top3 Overlay Paths
- /Users/junnanma/trend_project/new_process/h1_h4/mode1_with_gate_entry_quality_prefilter_exitopt_v1/20251227_214851/plots/verify_overlay_equity_top1.png
- /Users/junnanma/trend_project/new_process/h1_h4/mode1_with_gate_entry_quality_prefilter_exitopt_v1/20251227_214851/plots/verify_overlay_equity_top2.png
- /Users/junnanma/trend_project/new_process/h1_h4/mode1_with_gate_entry_quality_prefilter_exitopt_v1/20251227_214851/plots/verify_overlay_equity_top3.png
## Top3 Timeseries Paths
- /Users/junnanma/trend_project/new_process/h1_h4/mode1_with_gate_entry_quality_prefilter_exitopt_v1/20251227_214851/plots/verify_overlay_timeseries_top1.csv
- /Users/junnanma/trend_project/new_process/h1_h4/mode1_with_gate_entry_quality_prefilter_exitopt_v1/20251227_214851/plots/verify_overlay_timeseries_top2.csv
- /Users/junnanma/trend_project/new_process/h1_h4/mode1_with_gate_entry_quality_prefilter_exitopt_v1/20251227_214851/plots/verify_overlay_timeseries_top3.csv
