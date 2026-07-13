# Phase 2 — Hà Nội Transaction/Inventory Data

**Goal of Phase 2:** move from Phase 1's scraped *asking-price listings* toward
**transaction/inventory data** with richer features (geocoordinates, construction year, facade,
road access) — and use it to **test Phase 1's central hypothesis**:

> *Phase 1's ~12–13% accuracy ceiling was a **data-quality + missing-feature** limit (noisy labels,
> no fine location, no construction year), not a **model** limit.*

**Headline conclusion:** the hypothesis is **confirmed with an honest boundary.** Clean
transaction-inventory data reaches Phase-1 accuracy (**~13% Median APE**) on **70× less data** —
proving data quality was the lever. But the *specific* "fine-location feature breaks the ceiling"
claim is **not statistically significant** here (location is already captured by lat/long + ward,
and the SOLD sample is too small to prove small effects).

---

## 1. Dataset

`MikeGreen2710/transaction_data_26_07_10` — a Hà Nội secondary-market brokerage inventory.

| | |
|---|---|
| Raw | 11,252 rows × 45 columns |
| **Cleaned** | **10,493 rows × 36 columns** |
| Scope | Hà Nội only, secondary market, landed houses |
| Location levels | 16 districts, 197 wards, 732 streets |
| Price (asking) | median 6.95B VND; ~155M VND/m² |

### Problems acknowledged up front (see `notebooks/01_transaction_eda.ipynb`)
1. **No realized transaction price.** `import_price`, `original_price`, `minimum_selling_price`,
   `seller_expected_net_income` are **100% empty**. The only usable price is
   `reference_selling_price` (an **asking** price). So Phase 2 is *cleaner asking-price data*, not
   sold-price data.
2. **Only ~194 SOLD records** (95% of rows are `REVOKED`). The realized-sale sample is tiny.
3. Data-quality issues: 8 fully-empty columns, ~10% invalid coordinates (0/90), garbage
   construction years (0, 2119), garbage waist-widths, ~746 duplicate `property_id` re-listings,
   mixed date formats, `district_name` prefix inconsistency (`Q.` vs `Quận`).

### The upside (why Phase 2 was worth running)
- **`construction_year` 97% usable** (was 100% *missing* in Phase 1).
- **Real geocoordinates** for ~88% of rows → street/neighbourhood location (Phase 1's missing lever).
- Richer structure: `facade_width`, `main_road_dist`, `shape_name`, waist widths.

---

## 2. Cleaning pipeline (`src/clean_transaction_data.py`)

Risk-removal, leakage-safe (`filter_summary.csv` records rows removed per step):

| Step | Effect |
|---|---|
| Prune dead / redundant / near-empty / internal columns | −17 columns |
| **Deduplicate by `property_id`** — keep **SOLD first, then latest `last_modified`** | 11,252 → 10,506 |
| Drop non-positive price / area ≤ 10 m² | 10,506 → 10,493 |
| Clip garbage numerics → **NULL** (construction year, floors, facade, waist widths) | ~95 values nulled |
| Null invalid coordinates (Hà Nội bbox) + `has_geo` flag | 88% valid geo |
| Two-sided price/m² plausibility → **flag** (not delete), 20M–800M band | 70 flagged (0.7%) |
| Parse dates, derive recency; keep `available_for_sale_status` for availability tracking | — |

Availability/sold state is preserved via `status` (SOLD / REVOKED / ON_STOCK / LOCKED) +
`available_for_sale_status` (AVAILABLE / NOT_AVAILABLE).

---

## 3. Key EDA findings

- **Price distribution by status** (`figs/price_dist_by_status.png`): **SOLD and REVOKED overlap
  almost perfectly** (median 5.7B vs 6.8B) — so REVOKED is a sound proxy population for training.
  **AVAILABLE is the pricey tail** (median 18B) — unsold expensive inventory.
- **Address diversity** is dense at **ward level** (median ~53 records/ward, 59% of wards ≥ 30) —
  fine enough for ward-level location, *finer than Phase 1 could go*. Streets are too sparse
  (median 5) → better used via coordinates.
- **District price gradient** (`figs/price_dist_by_district.png`): 3.5× range from central
  Hoàn Kiếm (280M/m²) to peripheral Đông Anh (79M/m²), but **~53% internal spread per district** —
  district alone is too coarse; ward + geo are needed below it.
- **Small data → do NOT split into per-subgroup models.** With ~10K rows (biggest ward ~236),
  separate local models overfit. Location is used as **features + soft shrinkage**, not hard splits.

---

## 4. The 3-model ablation (`src/train_ablation_models.py`)

Train on **REVOKED**, lock a 20% internal test, evaluate externally on **SOLD** (the truth).
5-fold ward-grouped CV for internal stability. Each model changes one thing vs A.

### Results — MAE (billion ₫) · MAPE · Median APE

**Internal REVOKED test (n = 1,979)**
| Model | MAE | MAPE | Median APE |
|---|--:|--:|--:|
| A — REVOKED (base) | 2.42 | 19.0% | 13.0% |
| B — REVOKED + AVAILABLE | 2.39 | 19.2% | 12.6% |
| C — REVOKED + geo-KNN prior | 2.41 | 19.1% | 13.1% |

**External SOLD (n = 194)**
| Model | MAE | MAPE | Median APE | Hit ≤20% |
|---|--:|--:|--:|--:|
| A — REVOKED (base) | 1.94 | 24.1% | 16.3% | 57.7% |
| B — REVOKED + AVAILABLE | 1.96 | 24.6% | 17.1% | 57.7% |
| **C — REVOKED + geo prior** | **1.87** | **23.6%** | **15.3%** | **62.4%** |

- **A vs B — adding AVAILABLE hurts.** B is worse on SOLD and noisier in CV — the pricey inventory
  shifts the model. *Do not augment with AVAILABLE.*
- **A vs C — geo prior is directionally better** on SOLD, but see the robustness check below.

---

## 5. Robustness — paired bootstrap (`src/train_geo_features.py`)

Because SOLD has only **194 samples**, the A-vs-C difference was bootstrapped (5,000 paired
resamples) for a confidence interval:

| Comparison | Improvement | 95% CI | P(better) | Verdict |
|---|--:|---|--:|---|
| **C − A** (simple geo prior) | +1.25 pts | **[−0.57, +3.53]** | 89% | **CI includes 0 → NOT significant** |
| D − A (rich out-of-fold geo) | −0.18 pts | [−2.49, +2.58] | 42% | not significant |

**The confidence interval does not exclude 0** — the geo prior's ~1-point gain is **within noise**.
It is *directionally* positive (89% of resamples favour it) but **cannot be claimed as a
statistically significant improvement** on this sample.

---

## 6. Geo-KNN study (`figs/geo_*`)

- **K trade-off** (`geo_knn_k_tradeoff.png`): location *alone* (lat/long only) predicts price/m² to
  **~22% Median APE**, best around **K≈8**, flat K=5–35 (robust). A strong signal, but not
  sufficient alone (~1.7× the full model's error).
- **Radius trade-off** (`geo_radius_tradeoff.png`): sweet spot **~200–300 m** (96–98% coverage);
  beyond 500 m it over-smooths.
- **Rich out-of-fold features** (`local_median/p10/p90_ppsqm`, `neighbor_count`, `neighbor_iqr`,
  `nearest_neighbor_distance`, distance-weighted, radius→KNN fallback) swept over K∈{5,8,12,20} ×
  radius∈{200,250,300}: internal MedAPE stayed **12.9–13.5%**, SOLD **15.5–17.0%** — **flat, no
  significant gain**. Reason: **lat/long + ward already saturate the location signal**, so an
  explicit geo prior adds little.

---

## 7. Verdict on the Phase 1 hypothesis

| Claim | Verdict |
|---|---|
| Clean data + rich features lift the accuracy ceiling | ✅ **Confirmed** — ~13% Median APE on ~8K clean rows, matching Phase 1's national model built on 700K. |
| The specific "fine-location / geo-KNN prior breaks the ceiling" | ⚠️ **Inconclusive** — not statistically significant (CI [−0.6, +3.5]); location is already captured by lat/long + ward. |

**Net:** Phase 2 corroborates the *core* Phase 1 thesis — **data quality is the lever, not the
model** — while showing that, once fine location is present, further geo engineering yields no
provable gain.

---

## 8. Binding constraints (what limits Phase 2 now)

1. **SOLD sample size (n=194)** — too small to resolve ~1-point effects; would need ~thousands.
2. **Asking, not transaction, prices** — the realized-price fields are empty; this is cleaner
   asking-price modelling, not sold-price modelling.
3. **Location signal is saturated** — coordinates + ward already capture it; more geo features
   don't help.

**The next real lever is data acquisition (more SOLD records, realized transaction prices), not
more modelling or feature engineering** — every experiment here points there.

---

## 9. Time factor & re-listing dynamics (`src/train_time_model.py`)

### The problem: prices move, and the model ignored it
Hà Nội prices rose steadily across the listing window: median price/m² climbed **147M → 184M
(+25%) from Sep-2023 to May-2024** (~+4.1M/m² per month; +11% year-on-year). A model that treats
all listings as one timeless market anchors near the middle of that range and therefore
**under-values later listings** — inflating their apparent over-pricing (`figs/price_time_series.png`).

### Re-listing lifecycle (the over-price → revoke → re-price cycle)
The raw data records the same property re-listed over time (691 properties, 1,437 rows). Of the
406 that changed price: **72% were re-listed *higher* (median +15%, tracking the rising market)**
and **28% were *cut* (some −30 to −43%)** — properties listed too high, left unsold, `REVOKED`,
then re-listed lower (e.g. a Thanh Xuân house 35B → 20B). This is the over-pricing dynamic the
model is meant to catch *before* a listing wastes months on the market.

### Fix: normalized time features + leakage-safe validation
- Added time features from `last_modified_date`: **`lm_year`, `lm_month`, `lm_tindex`** (months
  since Jan-2023).
- **Kept re-listings** (no dedup) so the model sees the same property at different times, and used
  **`GroupKFold` on `property_id`** so a house never appears in both train and validation (the
  leakage that would otherwise flatter the score).

| Grouped-CV (by property_id, leakage-safe) | Median APE | Hit20 |
|---|--:|--:|
| Baseline (no time) | 13.60 ± 0.28 | 64.8 |
| **+ time features** | **13.12 ± 0.48** | **66.7** |

Time features contribute ~9% importance and improve honest CV accuracy (`lm_tindex`/`lm_month`
matter most). More strikingly, on the AVAILABLE set the time-aware model cut the apparent
over-ask from **~+47% to ~+16% median** (over>30% from ~62% → **36%**) — **confirming that a large
share of the "over-pricing" was actually market appreciation** the earlier model ignored.

### Honest caveat
AVAILABLE listings were `last_modified` in **2025–2026**, *outside* the training range (2023–2024),
so the time adjustment for AVAILABLE is **extrapolation** — directionally right (the market rose)
but not calibrated (no 2025–26 transactions to verify the magnitude). The +16% is a better estimate
than +47%, but its exact level is uncertain until later transaction data exists.

---

## 10. Reproduce

```bash
cd "Phase 2/src"
python3 clean_transaction_data.py       # -> data/transaction_cleaned.{parquet,csv}, filter_summary.csv
python3 train_ablation_models.py        # 3-model ablation -> results_ablation.csv
python3 train_geo_features.py           # bootstrap CI + out-of-fold geo-feature sweep
```
EDA notebook: `notebooks/01_transaction_eda.ipynb` (runs top-to-bottom; charts inline).
Figures: `figs/`.

## 11. Layout
```
Phase 2/
├── data/            transaction_cleaned.{parquet,csv} · filter_summary.csv · geo_clusters.csv
├── notebooks/       01_transaction_eda.ipynb (+ builder)
├── src/             clean_transaction_data.py · train_ablation_models.py · train_geo_features.py
├── figs/            price/status/district distributions · geo-KNN trade-offs · cluster map
└── results_ablation.csv
```
