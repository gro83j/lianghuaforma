# Final Attribution Report

## Evidence Sources
- Feature deciles: feature_deciles_top20.csv
- Feature importance: feature_importance_long.csv / feature_importance_short.csv

## Best OR Rules (Long/Short)
### long
rules_or: entry_rel_fib100 > -0.919211 and h4_di20p <= 16.759650 and h4_rsi14 > 28.643888 OR entry_rel_fib100 <= -0.919211 and h4_di14p <= 15.869530 and m30_atr14_ratio <= 1.057682 OR entry_rel_fib100 > -0.919211 and h4_di20p > 16.759650 and m30_roc10 > 0.016078 OR entry_rel_fib100 > -0.919211 and h4_di20p <= 16.759650 and h4_rsi14 <= 28.643888
coverage=0.7532, precision=0.7000, trades=170, success=119
by_year:
- 2010: coverage=0.0506, precision=0.8000, trades=10
- 2011: coverage=0.0570, precision=0.5625, trades=16
- 2012: coverage=0.0443, precision=0.4667, trades=15
- 2013: coverage=0.0380, precision=0.7500, trades=8
- 2014: coverage=0.0253, precision=0.5000, trades=8
- 2015: coverage=0.1392, precision=0.8462, trades=26
- 2016: coverage=0.0380, precision=0.8571, trades=7
- 2017: coverage=0.0127, precision=0.4000, trades=5
- 2018: coverage=0.0443, precision=0.7000, trades=10
- 2019: coverage=0.0633, precision=0.8333, trades=12
- 2020: coverage=0.0759, precision=0.7500, trades=16
- 2021: coverage=0.0316, precision=0.4167, trades=12
- 2022: coverage=0.0823, precision=0.8667, trades=15
- 2023: coverage=0.0506, precision=0.8000, trades=10

### short
rules_or: entry_rel_fib100 <= 0.821800 OR entry_rel_fib100 > 0.821800 and m30_di14p > 22.825665 and m30_cci20 > 61.421772 OR entry_rel_fib100 > 0.821800 and m30_di14p <= 22.825665 OR entry_rel_fib100 > 0.821800 and m30_di14p > 22.825665 and m30_cci20 <= 61.421772 and m30_roc10 <= -0.017321
coverage=0.9204, precision=0.6420, trades=162, success=104
by_year:
- 2010: coverage=0.0973, precision=0.7333, trades=15
- 2011: coverage=0.0619, precision=0.5833, trades=12
- 2012: coverage=0.0177, precision=0.5000, trades=4
- 2013: coverage=0.0354, precision=0.5714, trades=7
- 2014: coverage=0.0177, precision=0.2000, trades=10
- 2015: coverage=0.0708, precision=0.6667, trades=12
- 2016: coverage=0.0354, precision=0.6667, trades=6
- 2017: coverage=0.0619, precision=0.6364, trades=11
- 2018: coverage=0.0265, precision=1.0000, trades=3
- 2019: coverage=0.0708, precision=0.4444, trades=18
- 2020: coverage=0.2743, precision=0.7561, trades=41
- 2021: coverage=0.0442, precision=0.7143, trades=7
- 2022: coverage=0.0531, precision=0.6667, trades=9
- 2023: coverage=0.0531, precision=0.8571, trades=7
