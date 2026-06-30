# Rule-based Scene Summary

## Structure

All rule-based work now lives under [Rule-based Scene](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene).

Main folders:

- [Rule-based Scene/data/raw](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/data/raw)
- [Rule-based Scene/data/processed](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/data/processed)
- [Rule-based Scene/notebooks](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/notebooks)
- [Rule-based Scene/Evaluation](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/Evaluation)
- [Rule-based Scene/Visualizations](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/Visualizations)

Notebook sequence:

1. [01_process_data.ipynb](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/notebooks/01_process_data.ipynb)
2. [02_train_baseline.ipynb](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/notebooks/02_train_baseline.ipynb)
3. [03_train_ridge.ipynb](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/notebooks/03_train_ridge.ipynb)
4. [04_train_lightgbm.ipynb](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/notebooks/04_train_lightgbm.ipynb)

## Current run scope

The raw full dataset is stored at:

- [mick_df_full_raw.csv](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/data/raw/mick_df_full_raw.csv)

The current rule-based scene execution uses a reproducible sample cap of `250,000` raw rows so the full notebook sequence can run end-to-end in VS Code without stalling on preprocessing and export.

Processed output files:

- [full_cleaned_dataset.parquet](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/data/processed/full_cleaned_dataset.parquet)
- [landed_residential_house_modeling_dataset.parquet](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/data/processed/landed_residential_house_modeling_dataset.parquet)
- [train_dataset.parquet](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/data/processed/train_dataset.parquet)
- [test_dataset.parquet](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/data/processed/test_dataset.parquet)
- [metadata.json](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/data/processed/metadata.json)

Processed run summary:

- raw rows used: `250,000`
- cleaned rows: `240,803`
- modeling rows: `131,902`
- train rows: `105,521`
- test rows: `26,381`
- split method: `stratified_random`

## What the scene does

1. Ingest raw listing data.
2. Clean and standardize text, price, area, frontage, road width, room counts, legal fields, and address proxies.
3. Classify which rows belong to `landed_residential_house`.
4. Classify subgroup:
   - `normal_house`
   - `alley_house`
   - `alley_frontage_house`
   - `frontage_house`
   - `villa`
   - `townhouse_project`
   - `shophouse`
5. Build global features and subgroup-specific feature maps.
6. Train all models on both targets:
   - `log(price)`
   - `log(price per m2)`
7. Evaluate on:
   - overall test set
   - subtype
   - city/province
   - price range
   - area range
   - source
8. Generate comparison charts, trade-off charts, and bias charts.

## Evaluation outputs

Per-model results:

- [Evaluation/baseline](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/Evaluation/baseline)
- [Evaluation/ridge](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/Evaluation/ridge)
- [Evaluation/lightgbm](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/Evaluation/lightgbm)

Combined evaluation tables:

- [Evaluation/combined](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/Evaluation/combined)

Visualization code and outputs:

- [generate_visualizations.py](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/Visualizations/generate_visualizations.py)
- [visualizations](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/Visualizations/visualizations)

## Main results

Overall test-set metrics from the current run:

### Total price target

- Baseline global: `Median_APE = 31.99`, `HitRate_10 = 17.66%`, `HitRate_20 = 33.47%`
- Ridge global: `Median_APE = 29.91`, `HitRate_10 = 17.88%`, `HitRate_20 = 34.69%`
- LightGBM global: `Median_APE = 22.36`, `HitRate_10 = 24.58%`, `HitRate_20 = 45.59%`
- LightGBM subtype: `Median_APE = 22.14`, `HitRate_10 = 25.37%`, `HitRate_20 = 46.16%`

### Price-per-m2 target

- Baseline global: `Median_APE = 31.99`, `HitRate_10 = 17.63%`, `HitRate_20 = 33.45%`
- Ridge global: `Median_APE = 26.96`, `HitRate_10 = 19.86%`, `HitRate_20 = 38.25%`
- LightGBM global: `Median_APE = 22.30`, `HitRate_10 = 24.93%`, `HitRate_20 = 45.92%`
- LightGBM subtype: `Median_APE = 22.30`, `HitRate_10 = 25.33%`, `HitRate_20 = 45.85%`

Interpretation:

- `LightGBM` is the strongest model family in this rule-based scene.
- For total price, subtype-specific LightGBM slightly improves the overall hit rates compared with the global LightGBM.
- For price per m2, the global and subtype LightGBM variants are very close overall, but some subtypes still benefit from specialization.

## Subtype routing highlights for LightGBM

From [Evaluation/lightgbm/model_selection.csv](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/Evaluation/lightgbm/model_selection.csv):

- `price` target:
  - subtype model preferred for `alley_house`, `frontage_house`, `shophouse`
  - global model preferred for `alley_frontage_house`, `normal_house`, `townhouse_project`, `villa`
- `price_per_m2` target:
  - subtype model preferred for `frontage_house`, `shophouse`, `villa`
  - global model preferred for `alley_frontage_house`, `alley_house`, `normal_house`, `townhouse_project`

This means the scene already supports both:

- one global valuation model for the whole parent group
- one subgroup-specific model where subgroup behavior is measurably different

## Next use

This folder is the rule-based baseline scene.
The natural next phase is to build a separate `NLP-Classifier Scene` for:

- parent-group classification
- subtype classification
- confidence scoring for noisy listings

Then feed those outputs into the valuation layer already built here.
