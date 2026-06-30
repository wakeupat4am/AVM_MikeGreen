# AVM Project Detailed Status

## 1. Current Scope

This project is an Automated Valuation Model for Vietnamese landed residential houses.

Current working scene:
- `Rule-based Scene`

Current modeling objective:
- Predict residential house value using structured listing data
- Main production-style target now emphasized: `log_price`
- Supporting target from earlier stage: `log(price_per_m2)`

Current in-scope property family:
- `landed_residential_house`

Current in-scope subtypes:
- `normal_house`
- `alley_house`
- `alley_frontage_house`
- `frontage_house`
- `villa`
- `townhouse_project`
- `shophouse`

Out-of-scope groups excluded by rule-based logic:
- apartment / condo style listings
- land-only listings
- pure commercial listings
- unknown groups that do not fit the landed residential house definition cleanly

## 2. Project Folder Structure

Main project folder:
- [Rule-based Scene](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene)

Key subfolders:
- [data/raw](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/data/raw): raw input CSV
- [data/processed](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/data/processed): cleaned and modeling-ready parquet files
- [src](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/src): reusable processing and training code
- [notebooks](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/notebooks): VS Code runnable entry notebooks
- [Evaluation](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/Evaluation): saved metrics, predictions, and reports
- [Visualizations](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/Visualizations): chart generation and output images

Important source files:
- [common.py](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/src/common.py): shared preprocessing, feature definitions, evaluation, split, model builders
- [process_data.py](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/src/process_data.py): data processing pipeline entry
- [train_baseline.py](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/src/train_baseline.py): simple rule/median baseline
- [train_ridge.py](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/src/train_ridge.py): ridge regression benchmark
- [train_lightgbm.py](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/src/train_lightgbm.py): original LightGBM stage
- [outlier_handling.py](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/src/outlier_handling.py): local outlier cleaning and weighting
- [train_robust_lightgbm.py](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/src/train_robust_lightgbm.py): robust global LightGBM
- [train_region_subtype_models.py](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/src/train_region_subtype_models.py): region/subtype/hybrid models

## 3. Current Data Assets

Raw source file:
- [mick_df_full_raw.csv](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/data/raw/mick_df_full_raw.csv)

Processed artifacts:
- [full_cleaned_dataset.parquet](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/data/processed/full_cleaned_dataset.parquet)
- [landed_residential_house_modeling_dataset.parquet](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/data/processed/landed_residential_house_modeling_dataset.parquet)
- [train_dataset.parquet](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/data/processed/train_dataset.parquet)
- [test_dataset.parquet](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/data/processed/test_dataset.parquet)
- [metadata.json](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/data/processed/metadata.json)

Current processed counts from `metadata.json`:
- Clean rows: `240,803`
- Modeling rows: `131,902`
- Train rows: `105,521`
- Test rows: `26,381`
- Split method: `stratified_random`

Subtype counts in train:
- `frontage_house`: `37,431`
- `alley_house`: `19,887`
- `normal_house`: `16,863`
- `alley_frontage_house`: `12,100`
- `villa`: `9,035`
- `townhouse_project`: `8,535`
- `shophouse`: `1,670`

## 4. Data Cleaning and Feature Engineering Status

### 4.1 Ingestion and normalization

Implemented in [common.py](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/src/common.py):
- text normalization to lower-case and compact whitespace
- Vietnamese numeric parsing
- Vietnamese price parsing to VND
- address splitting into coarse location tokens
- duplicate removal
- listing date parsing

### 4.2 Price cleaning

Implemented:
- parse price text such as `tỷ`, `triệu`, price per square meter text
- convert to `price_vnd`
- derive `price_per_m2`
- derive `price_per_m2_million`
- remove invalid / impossible low-quality rows in modeling stage

### 4.3 Address cleaning

Current address decomposition is rule-based and coarse:
- `addr_last_1`: last address token, typically city/province level proxy
- `addr_last_2`: second-last token, usually district-level proxy
- `addr_last_3`: third-last token, usually ward-level proxy
- `street_proxy`: fourth-last token, rough street/location proxy

Important limitation:
- address strings are noisy and not fully standardized
- some bad address text currently leaks into region labels
- this is visible in evaluation where a few “city/province” groups are actually malformed full listing phrases

### 4.4 Property classification

Current approach:
- rule-based subtype classification from title/address/description text

Current classification output columns:
- `property_subtype`
- `property_group`

Current definition of landed residential house:
- any row whose subtype is one of the seven in-scope residential house subtypes

### 4.5 Engineered structured features

Current numeric feature families include:
- size and geometry: `area_m2`, `width_m`, `length_m`, `frontage_m`, `road_width_m`
- building details: `num_floors`, `bedrooms`, `bathrooms`, `floor_area_m2`
- text-derived binary signals: frontage, alley, business potential, car access, legal, project, urban area, new/renovated
- date features: listing year and month
- interaction features: frontage x road width, floors x area, area x frontage, etc.

Current categorical feature families include:
- subtype
- location proxies
- legal status
- direction
- source
- project name

## 5. Current Notebooks

Existing notebooks:
- [01_process_data.ipynb](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/notebooks/01_process_data.ipynb)
- [02_train_baseline.ipynb](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/notebooks/02_train_baseline.ipynb)
- [03_train_ridge.ipynb](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/notebooks/03_train_ridge.ipynb)
- [04_train_lightgbm.ipynb](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/notebooks/04_train_lightgbm.ipynb)
- [05_outlier_handling.ipynb](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/notebooks/05_outlier_handling.ipynb)
- [06_train_robust_lightgbm.ipynb](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/notebooks/06_train_robust_lightgbm.ipynb)
- [07_train_region_subtype_models.ipynb](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/notebooks/07_train_region_subtype_models.ipynb)

Notebook design:
- thin entrypoints
- main logic lives in `src/`
- intended to run cleanly in VS Code without duplicating business logic

## 6. Current Modeling Stages

### 6.1 Baseline

File:
- [train_baseline.py](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/src/train_baseline.py)

Method:
- group median price per square meter
- convert back to total price using area

Purpose:
- sanity-check benchmark

### 6.2 Ridge regression

File:
- [train_ridge.py](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/src/train_ridge.py)

Method:
- regularized linear benchmark
- supports both global and subtype-specific training

Purpose:
- interpretable benchmark

### 6.3 Original LightGBM stage

File:
- [train_lightgbm.py](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/src/train_lightgbm.py)

Method:
- global LightGBM
- subtype-specific LightGBM
- trained for both `log_price` and `log_price_per_m2`

Purpose:
- first strong nonlinear benchmark

### 6.4 Improvement stage: local outlier handling

File:
- [outlier_handling.py](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/src/outlier_handling.py)

Implemented:
- impossible-row removal:
  - `price <= 0`
  - `area <= 0`
  - `price_per_m2 <= 0`
  - unrealistic floors/bedrooms/bathrooms/frontage/road width
- local quantile hierarchy:
  - district + subtype
  - city/province + subtype
  - subtype
  - global
- quantiles computed:
  - `Q01`, `Q05`, `Q25`, `Q50`, `Q75`, `Q95`, `Q99`
- local flags:
  - `is_low_local_outlier`
  - `is_high_local_outlier`
  - `is_extreme_local_outlier`
  - `local_ppm2_band`
- sample weights:
  - `Q25–Q75`: `1.0`
  - `Q05–Q25` or `Q75–Q95`: `0.9`
  - `Q01–Q05` or `Q95–Q99`: `0.5`
  - below `Q01` or above `Q99`: `0.2`

Important design decision:
- `price_per_m2` is used for cleaning, filtering, weighting, and diagnostics
- it is not used as an input feature when the target is total `price`

### 6.5 Improvement stage: robust LightGBM

File:
- [train_robust_lightgbm.py](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/src/train_robust_lightgbm.py)

Implemented:
- target: `log_price`
- objective comparison:
  - `regression_l2`
  - `regression_l1`
  - `huber`
  - `quantile` with alpha `0.5`
- training uses local `sample_weight`
- leakage columns excluded:
  - `price`
  - `price_vnd`
  - `price_per_m2`
  - `price_per_m2_million`
  - `log_price`
  - `log_price_per_m2`
  - target-derived / prediction / outlier columns

### 6.6 Improvement stage: region/subtype/hybrid LightGBM

File:
- [train_region_subtype_models.py](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/src/train_region_subtype_models.py)

Implemented families:
- `global model`
- `region model`
- `subtype model`
- `region subtype model`
- `hybrid fallback`

Current routing logic:
- if group size `< 500`: fallback to global
- if group size `500–2000`: prefer region, compare with subtype where available
- if group size `>= 2000`: allow region-subtype model
- final choice depends on validation `Median_APE`

Current routing result counts from [routing_policy.csv](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/Evaluation/improvements/region_subtype_models/routing_policy.csv):
- `global`: `722` groups
- `region`: `35` groups
- `subtype`: `2` groups
- `region_subtype`: `3` groups

Interpretation:
- most fine-grained groups are still too sparse or too noisy to beat the safer fallback
- region-level specialization is currently the strongest practical gain

## 7. Current Evaluation Design

Core metrics used:
- `MAE`
- `RMSE`
- `MAPE`
- `Median_APE`
- `HitRate_10_pct`
- `HitRate_20_pct`
- `Bias_pct`
- `OvervaluationRate_pct`
- `UndervaluationRate_pct`

Evaluation slices currently supported:
- overall test set
- by subtype
- by city/province or region proxy
- by price range
- by area range
- by source
- improvement stage also includes by outlier band

Saved evaluation folders:
- [Evaluation/baseline](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/Evaluation/baseline)
- [Evaluation/ridge](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/Evaluation/ridge)
- [Evaluation/lightgbm](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/Evaluation/lightgbm)
- [Evaluation/improvements](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/Evaluation/improvements)
- [Evaluation/combined](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/Evaluation/combined)

## 8. Current Main Results

### 8.1 Original LightGBM baseline

From [lightgbm/all_metrics.csv](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/Evaluation/lightgbm/all_metrics.csv), overall `price`, `lightgbm global`:
- `MAE`: `3.086B VND`
- `RMSE`: `11.823B VND`
- `MAPE`: `40.76%`
- `Median_APE`: `22.36%`
- `HitRate_10_pct`: `24.58%`
- `HitRate_20_pct`: `45.59%`
- `Bias_pct`: `16.25%`

### 8.2 Improvement stage final comparison

From [final_comparison.csv](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/Evaluation/improvements/final_comparison.csv):

`lightgbm global`:
- `Median_APE`: `22.36%`
- `HitRate_20_pct`: `45.59%`

`robust lightgbm global`:
- `Median_APE`: `22.66%`
- `HitRate_20_pct`: `45.41%`

`region model`:
- `Median_APE`: `17.98%`
- `HitRate_20_pct`: `53.93%`

`subtype model`:
- `Median_APE`: `22.36%`
- `HitRate_20_pct`: `46.01%`

`region subtype model`:
- `Median_APE`: `15.00%`
- `HitRate_20_pct`: `59.48%`

`hybrid fallback`:
- `Median_APE`: `20.07%`
- `HitRate_20_pct`: `49.88%`

### 8.3 Current interpretation

What is clearly working:
- region-aware modeling helps materially
- region-subtype modeling is strongest where groups are large enough
- hybrid fallback beats the original global LightGBM while staying safer than forcing local models everywhere

What did not help enough on this run:
- robust global weighting/objective change alone did not beat the old global LightGBM baseline
- subtype-only specialization is weaker than region specialization

## 9. Current Charts and Reporting Assets

Chart generation:
- [generate_visualizations.py](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/Visualizations/generate_visualizations.py)

Saved charts:
- [Visualizations/visualizations](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/Visualizations/visualizations)

Current chart types available:
- metric comparison charts across models
- level-based charts by subtype, source, price range, area range, city/province
- tradeoff charts such as `Median_APE` vs `HitRate_20_pct`
- bias charts

Current improvement summary report:
- [IMPROVEMENT_RESULTS.md](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/Evaluation/improvements/IMPROVEMENT_RESULTS.md)

## 10. Current Known Problems

### 10.1 Address noise

Problem:
- location strings are not standardized enough
- some rows produce malformed region buckets

Impact:
- noisy region segmentation
- bad geospatial proxies
- misleading per-region evaluation on tiny corrupted groups

### 10.2 Label noise in prices

Problem:
- listing prices may be stale, negotiated, duplicated, promotional, or text-parsed imperfectly

Impact:
- large unavoidable residual noise
- high `MAPE` and `RMSE`
- unstable tails and bias in expensive segments

### 10.3 Group sparsity

Problem:
- many `region + subtype` combinations are too small

Impact:
- local models cannot safely generalize
- routing often falls back to global or region models

### 10.4 Class imbalance

Problem:
- `frontage_house` dominates the dataset
- `shophouse` is relatively small

Impact:
- subtype-specific learning quality is uneven

### 10.5 Feature missingness

Problem:
- bedrooms, bathrooms, frontage, road width, and floor details are missing for many listings

Impact:
- some subgroup models train on thinner signal than intended

### 10.6 Region label contamination

Observed symptom:
- some entries inside `by_city_province` are actually full listing phrases rather than province names

Impact:
- evaluation slices at region level are partially polluted

## 11. Current Answers to Key Business Questions

### Is the project already doing end-to-end AVM?

Yes, for the rule-based scene:
- ingest raw data
- clean and standardize
- define landed residential house and subtype groups
- engineer features
- train baseline/ridge/LightGBM/improved LightGBM variants
- evaluate and visualize

### Is the model currently production-ready?

Not yet.

Reason:
- address quality and local grouping quality still need improvement
- noise in labels is still large
- local model routing works, but only a small number of fine-grained groups justify region-subtype specialization

### What is the strongest current deploy candidate?

Current practical candidate:
- `hybrid fallback`

Reason:
- better than old global LightGBM
- safer than blindly using the most granular local model everywhere

### What is the strongest raw modeling result?

Best raw score in the current comparison:
- `region subtype model`

Caution:
- it is not the safest universal deployment choice because it only applies well to sufficiently large groups

## 12. Current Recommended Next Stage

Highest-value next work:
- improve address normalization
- merge equivalent city/province labels such as `tp.hcm`, `tphcm`, and other duplicates
- add geocoding / map / POI features
- create NLP classifier scene to replace or support subtype rule-based labeling
- add leakage-safe time-based validation where listing chronology is reliable
- audit extreme price rows and duplicated near-identical listings
- compare total price target vs price-per-m2 target in a unified routing setup

Recommended NLP integration direction:
- build a separate `NLP-Classifier Scene`
- first classify whether a listing is landed residential house
- then classify subtype probabilities
- feed subtype probabilities or confidence into AVM routing

## 13. Files Most Important for Continuing the Project

If continuing data logic:
- [common.py](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/src/common.py)
- [process_data.py](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/src/process_data.py)

If continuing outlier and robust modeling:
- [outlier_handling.py](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/src/outlier_handling.py)
- [train_robust_lightgbm.py](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/src/train_robust_lightgbm.py)

If continuing local model routing:
- [train_region_subtype_models.py](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/src/train_region_subtype_models.py)
- [routing_policy.csv](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/Evaluation/improvements/region_subtype_models/routing_policy.csv)

If reviewing latest performance:
- [final_comparison.csv](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/Evaluation/improvements/final_comparison.csv)
- [IMPROVEMENT_RESULTS.md](/Users/dienmayhaituyet/Documents/AVM/Rule-based%20Scene/Evaluation/improvements/IMPROVEMENT_RESULTS.md)

## 14. Bottom-Line Status

The project is no longer just an initial notebook experiment. It now has:
- a structured folder layout
- reusable processing code
- multiple model families
- saved evaluation outputs
- visualization outputs
- an implemented improvement stage with local outlier handling and region-aware modeling

Current overall judgment:
- the rule-based AVM pipeline is functional and materially improved
- the biggest remaining bottlenecks are address quality, local label noise, and sparse fine-grained groups
- the next major maturity jump should come from better location intelligence and an NLP-assisted classifier layer
