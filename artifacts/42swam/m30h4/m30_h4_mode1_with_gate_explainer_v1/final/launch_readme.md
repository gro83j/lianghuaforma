# Launch Readme (Prefilter Only)

## Rationale
- lockbox prefilter sharpe_simple 0.4143 > ml_A 0.4135
- ml_A reduces sharpe/sum_pnl/freq versus prefilter-only on lockbox

## Deployment Logic
- baseline signal triggers; apply long_rule/short_rule prefilter first
- if rule fails, discard the trade; otherwise keep as-is

## Risk Notes
- frequency drops on lockbox (freq_ratio=0.711)

## Future Upgrade Path
- ML can be reintroduced only as an optional ranking layer
- must include downgrade/rollback when verify sharpe_simple degrades
