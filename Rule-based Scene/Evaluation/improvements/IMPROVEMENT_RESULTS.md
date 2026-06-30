# Improvement Results

## What Changed

- Added train-only local quantile outlier handling based on `price_per_m2` with leakage-safe fallback hierarchy.
- Added robust LightGBM objective comparison with weighted training on `log_price`.
- Added region, subtype, and region-subtype LightGBM families plus validation-based hybrid routing.
- Expanded evaluation to subtype, region, price band, area band, source, and outlier band.

## Best Overall Model

- Best overall on current run: `region subtype model` with Median_APE `15.00` and HitRate_20_pct `59.48`.

## Improvement Readout

- Hybrid improved subtypes: alley_frontage_house, alley_house, frontage_house, normal_house, townhouse_project, villa.
- Hybrid improved regions: bà rịa vũng tàu, bình dương, cần thơ, huế, hải phòng, khánh hòa, long an, tp.hcm, tphcm, đà nẵng.
- Hybrid vs current baseline Median_APE delta: -2.29 points.

## Interpretation

- If hybrid beats the global baseline, the next production step is to keep the global model as the safe default and route only the groups where local validation proves a gain.
- If a subtype or region does not improve on validation, it should stay on the parent model to avoid overfitting noisy local segments.