# H4/D1 Optimize Pareto v3.6_e2_localsearch_fold1_diag Summary
- wf_folds: [('2014-01-01', '2016-12-31'), ('2017-01-01', '2019-12-31'), ('2020-01-01', '2023-12-31')]
- stage1_total: 160
- stage1_topn: 117
- max_evals: 160
- hard_constraints: per_fold_min_n>=8, wf_side_ok, wf_mdd_worst_pct>=-0.30, wf_winrate_min>=0.35, wf_sharpe_min>=-0.20
- rank_keys: folds_pass_win65, folds_pass_sh80, wf_sharpe_med, wf_sharpe_min, fold1_win_rate_pnl, fold1_sharpe_raw, all_tpm, wf_mdd_worst_pct
- fold1_counterfactual: {'lock0': {'win_rate_pnl': 0.425, 'sharpe_raw': -0.1184756042631535}, 'lockX': {'lock_pts': 50, 'win_rate_pnl': 0.9, 'sharpe_raw': 1.110551535330311}}
- worst_fold=Fold1 sharpe_raw=1.111 win_rate_pnl=0.900 n=40
