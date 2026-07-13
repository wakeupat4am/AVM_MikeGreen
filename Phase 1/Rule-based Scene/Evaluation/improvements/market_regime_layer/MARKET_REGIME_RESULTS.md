# Market Regime Layer Results

## Summary

- Added a leakage-safe market-regime layer driven by local `price_per_m2` quantile dispersion.
- Tuned the regime definition thresholds on validation without using price-derived features in the classifier.
- Compared two AVM integrations: hard regime grouping and probability-based residual adjustment.

## Best Overall Result

- Best overall among the compared options: `current hybrid fallback` with Median_APE `14.63` and HitRate_20_pct `59.83`.

## Regime Classifier

- Validation accuracy `0.6715`, macro-F1 `0.5047`, log-loss `0.7243`.
- Test accuracy `0.6584`, macro-F1 `0.4919`, log-loss `0.7417`.

## AVM Comparison

- Hard regime routing vs current hybrid Median_APE delta: 12.15 points.
- Probability adjustment vs current hybrid Median_APE delta: 0.77 points.

## Threshold Choice

- Selected thresholds: minimum_group_size `1000`, dispersion_ratio_min `2.0`, iqr_ratio_min `1.3`, dispersion_log_min `0.55`.

## Interpretation

- If probability adjustment beats hard routing, the regime signal is useful but should remain a soft calibration layer instead of replacing local context.
- If hard routing underperforms, that supports the caution that pooling low-tail or high-tail rows across regions can destroy local valuation structure.
- Keep all previous outputs for comparison; this layer is an additive experiment, not a replacement of the current hybrid pipeline yet.