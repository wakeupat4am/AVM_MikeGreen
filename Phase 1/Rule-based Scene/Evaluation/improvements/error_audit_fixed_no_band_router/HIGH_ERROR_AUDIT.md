# High Error Audit - Fixed Clean No Band Router

- Prediction file: `/Users/dienmayhaituyet/Documents/AVM/Rule-based Scene/Evaluation/improvements/region_subtype_models_pre_retrain_feature_fixes_no_band_router/predictions.csv`
- Test rows audited: `140,874`
- Rows with abs_pct_error >= 100%: `4,259` (3.02%)
- Median abs pct error: `13.84%`
- Mean abs pct error: `31.77%`

## Most Common Issue Flags In Top 1000 Errors
- `possible_missing_feature_issue`: 490/1,000 rows (49.0%)
- `flag_structural_missing_high`: 476/1,000 rows (47.6%)
- `possible_bad_label_or_outlier`: 425/1,000 rows (42.5%)
- `flag_extreme_ppm2`: 375/1,000 rows (37.5%)
- `possible_wrong_property_type`: 275/1,000 rows (27.5%)
- `flag_land_only_terms`: 206/1,000 rows (20.6%)
- `flag_luxury_or_large`: 177/1,000 rows (17.7%)
- `possible_duplicate_noise`: 93/1,000 rows (9.3%)

## Most Common Issue Flags When Error >= 100%
- `possible_missing_feature_issue`: 2,120/4,259 rows (49.8%)
- `flag_structural_missing_high`: 2,069/4,259 rows (48.6%)
- `possible_bad_label_or_outlier`: 1,214/4,259 rows (28.5%)
- `possible_wrong_property_type`: 1,164/4,259 rows (27.3%)
- `flag_land_only_terms`: 917/4,259 rows (21.5%)
- `flag_extreme_ppm2`: 878/4,259 rows (20.6%)
- `flag_luxury_or_large`: 699/4,259 rows (16.4%)
- `flag_commercial_terms`: 317/4,259 rows (7.4%)

## Generated Files

- `top_200_high_error_review.csv`: first manual review file.
- `top_1000_high_error_review.csv`: larger manual review file.
- `errors_over_100pct.csv`: all extreme errors.
- `issue_flag_summary.csv`: issue-flag counts by subset.
- `error_band_distribution.csv`: error distribution by APE band.
- `error_by_property_subtype_refined.csv`, `error_by_addr_last_1.csv`, `error_by_source_family.csv`: concentration summaries.