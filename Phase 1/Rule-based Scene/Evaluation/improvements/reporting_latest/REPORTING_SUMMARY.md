# Latest Improvement Reporting Summary

## Overall

- Best raw model after address normalization retraining: `region subtype model` with Median_APE `15.73` and HitRate_20_pct `58.11`.
- Best practical routed model: `hybrid fallback` with Median_APE `18.96` and HitRate_20_pct `51.84`.
- Compared with the original `lightgbm global`, the refreshed hybrid improves Median_APE by `3.41` points and HitRate_20_pct by `6.25` points.

## Address Normalization Effect

- Province aliases such as `tp.hcm`, `tphcm`, and `hồ chí minh` are now merged into a single canonical label.
- Malformed province buckets created from noisy full listing phrases were removed from `addr_last_1`.
- District aliases such as `q.hải châu`, `hải châu`, and `quận hải châu` were partially collapsed into cleaner canonical buckets.
- Routing groups in the hybrid policy dropped to `363` because fragmented local groups were merged into cleaner region buckets.

## Market-Regime Layer

- Best option inside the market-regime experiment after retraining: `current hybrid fallback` with Median_APE `18.96`.
- Classifier validation accuracy `0.7100`, test accuracy `0.7012`.

## Top Subtype Gains of Hybrid vs Global

- frontage_house: Median_APE gain `3.58` points.
- normal_house: Median_APE gain `3.24` points.
- villa: Median_APE gain `3.09` points.
- townhouse_project: Median_APE gain `3.08` points.
- alley_frontage_house: Median_APE gain `1.97` points.

## Top Region Gains of Hybrid vs Global (count >= 100)

- khánh hòa: Median_APE gain `4.49` points.
- đồng nai: Median_APE gain `4.16` points.
- cần thơ: Median_APE gain `3.94` points.
- long an: Median_APE gain `3.83` points.
- hà nội: Median_APE gain `3.28` points.