# Deduplication Audit

- Input rows: `704,370`
- Canonical deduplicated rows: `638,287`
- Removed duplicate rows if using canonical dataset: `66,083`
- Duplicate clusters: `52,237`
- Rows inside duplicate clusters: `118,320`
- Conflicting duplicate clusters with price spread >= 1.25x: `1,559`
- Rows inside conflicting duplicate clusters: `4,310`

## Method

- Normalize Vietnamese title/address/location text by lowercasing, removing accents/punctuation, and removing common listing words.
- Block candidates by canonical province, district, ward, street, property subtype, area bin, and title fingerprint.
- Link rows only when area is close and title/address similarity is high.
- Save both canonical-observed and median-price deduplicated datasets.

## Recommended Training Input

- Use `deduplicated_median_price_modeling_dataset.parquet` for the next robust retrain because it keeps one row per cluster and uses cluster median price to reduce duplicated label noise.
- Keep `clustered_all_rows.parquet` for audit and optional sample weighting experiments.

## Config

```json
{
  "input_path": "/Users/dienmayhaituyet/Documents/AVM/Rule-based Scene/draft/pre_retrain_feature_fixes/fixed_landed_residential_house_modeling_dataset.parquet",
  "input_rows": 704370,
  "clustered_rows": 704370,
  "deduplicated_rows": 638287,
  "removed_duplicate_rows": 66083,
  "duplicate_clusters": 52237,
  "rows_in_duplicate_clusters": 118320,
  "conflicting_duplicate_clusters": 1559,
  "rows_in_conflicting_duplicate_clusters": 4310,
  "area_tolerance": 0.08,
  "price_ratio_limit": 2.0,
  "max_pairwise_block": 250,
  "random_state": 42
}
```