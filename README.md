# AVM — Automated Valuation Model for Vietnamese Landed Residential Houses

An end-to-end **Automated Valuation Model (AVM)** that predicts the market value of Vietnamese
landed residential houses from structured listing data. The project has moved past "chase a
single accuracy number" and is now organised as a **segmented, business-defensible reliability
system**: it reports where it can be trusted, explains where its errors come from, and flags the
cases it should not answer confidently.

All active work lives under [`Phase 1/Rule-based Scene/`](Phase%201/Rule-based%20Scene).

> **Phase 2** — a follow-up on cleaner Hà Nội **transaction/inventory data** (with geocoordinates
> and construction year) that **tests this project's central hypothesis**: it reaches Phase-1
> accuracy (~13% Median APE) on **70× less data**, confirming *data quality was the ceiling, not the
> model*. Full write-up: [`Phase 2/README.md`](Phase%202/README.md).

---

## Table of contents

1. [What this project is](#1-what-this-project-is)
2. [Project direction & framing](#2-project-direction--framing)
3. [Data & pipeline](#3-data--pipeline)
4. [Models & results](#4-models--results)
5. [The bias story — read this before judging the numbers](#5-the-bias-story)
6. [Calibration layer: shrinkage routing + prediction intervals](#6-calibration-layer)
7. [Regional reliability report — the business defense](#7-regional-reliability-report)
8. [Final model — cleaned training and the stability defense](#8-final-model)
9. [How to reproduce](#9-how-to-reproduce)
10. [Repository layout](#10-repository-layout)
11. [Limitations & next steps](#11-limitations--next-steps)

---

## 1. What this project is

- **Goal:** estimate a house's value from a structured listing (area, floors, frontage, road
  width, location, subtype, text-derived flags, …).
- **Primary target:** `log_price` (log of price in VND). Supporting target: `log(price_per_m2)`.
- **In-scope property family:** `landed_residential_house`, across seven subtypes —
  `frontage_house`, `alley_house`, `normal_house`, `alley_frontage_house`, `townhouse_project`,
  `villa`, `shophouse`.
- **Headline result:** the final model (shrinkage blend, cleaned training — Section 8) values a
  **normal, sensible asset to ~12.7% typical error** with **~65% of estimates within 20%**, is
  **statistically unbiased**, ships with a **per-listing confidence interval**, and **declines to
  guess** on listings outside its validity domain (price-flagged or below 20M VND/m²).

---

## 2. Project direction & framing

Early work focused on lifting overall accuracy through better models (baseline → Ridge →
LightGBM → robust LightGBM → regional/subtype routing → market-regime layer → CatBoost
experiments). That worked up to a point, but **further modeling now yields only fractions of a
percentage point** — the return on more tuning has flattened.

Following advisor feedback, the project was **re-framed for a real-world product**. The right
question is not *"what is the average error over the whole dataset?"* but:

> **"For which listings, and in which markets, can we trust the valuation — and what does the
> system do when it can't?"**

Concretely this means:

1. **Segment the problem, defend per segment.** Report accuracy **per region** (and per subtype),
   because a national average hides that some markets are prediction-ready today while others are
   data-limited. A product is defended market-by-market, not by one blended figure.
2. **Explain the errors instead of only minimising them.** Show that large errors are **not
   random model failure** but a small, identifiable population of anomalous / mislabeled listings.
3. **Defend production behaviour on new cases.** Argue — with evidence — that for a *normal*
   incoming listing the system predicts **sharply**, and that it **knows when it cannot**, so
   anomalies are flagged rather than silently mis-valued.

Sections 5–8 deliver each of these; Section 8 defines the final model and its validity domain.

---

## 3. Data & pipeline

Raw listings are cleaned, parsed, deduplicated, and feature-engineered into a modeling table.

| Stage | Rows |
|---|---|
| Cleaned listings | 813,215 |
| Modeling rows (landed residential, valid price/area) | 704,370 |
| Train / Test (stratified random split by subtype) | 563,496 / 140,874 |

Subtype mix (modeling rows): `frontage_house` 171k · `normal_house` 152k · `alley_house` 80k ·
`alley_frontage_house` 55k · `townhouse_project` 52k · `villa` 44k · `shophouse` 10k.

**Data-quality work applied** (in `src/common.py` and the pipeline scripts):

- **Address normalization** — merges province/district tone-variant and abbreviation aliases
  (`tp.hcm`/`tphcm`/`hồ chí minh` → one label; `hoà`→`hòa`, etc.) so location buckets stop
  fragmenting.
- **Near-duplicate removal** — collapses the same listing re-posted many times (matched on
  address + area bucket + price bucket), keeping the most recent.
- **Recency weighting** — down-weights stale listings (2025 → 1.0, older years progressively
  lower) during training.
- **Text feature extraction** — pulls structured flags (frontage, alley, business, car access,
  corner, villa, legal, project, …) from the free-text description.
- **Suspicious-price auditing** — flags/removes implausible prices (placeholders, sequential
  digits, sub-500M "tỷ" mismatches). ~8k rows removed in the cleaned variant.

---

## 4. Models & results

All models predict `log_price` with LightGBM (objective `regression_l1`) and back-transform with
`expm1`. Evaluated on the same 140,874-row test set.

| Model | Median_APE | HitRate_20 | MAPE | Role |
|---|---|---|---|---|
| global | 21.96% | 46.5% | 44.0% | baseline |
| subtype | 20.45% | 49.2% | 42.6% | one model per subtype |
| region | 15.36% | 59.0% | 34.1% | one model per province |
| hybrid fallback (hard router) | 15.39% | 58.3% | 36.1% | previous production model |
| **shrinkage blend** | **14.84%** | **59.5%** | **34.5%** | **production architecture** |
| region×subtype | 14.01% | 61.3% | 32.1% | best *raw* score (dense groups only) |

The **final shipped model** is the shrinkage blend retrained on cleaned data — see Section 8 for
its numbers on the serving population (12.74% Median_APE / 64.6% HitRate_20) and the stability
comparison that backs them.

**Two "best" models, two meanings:**

- **`region×subtype`** has the best raw score but only covers groups with enough data
  (≥ 2,000 training rows) — it is not a complete product on its own.
- **`shrinkage blend`** is the best model that **values every listing** and is what should ship
  (Section 6).

**Metric definitions** (`src/common.py::metric_row`):

- **Median_APE** — median absolute percentage error. *The headline metric for a skewed target.*
- **HitRate_20** — share of estimates within 20% of the true price.
- **MAPE** — *mean* absolute percentage error; inflated by a heavy right tail (see Section 5).
- **Bias_pct / Median_Bias_pct / Bias_pct_trimmed10** — mean / median / 10%-trimmed-mean signed
  error; direction of systematic over/under-valuation.

---

## 5. The bias story

**Read this before concluding the model is "off by 14%."** It isn't.

The raw `Bias_pct` looks alarming (+13% to +20% across models), but it is a **reporting artifact,
not a modeling defect**:

| Bias measure (region×subtype model) | Value |
|---|---|
| Raw mean bias | +13.9% |
| **Median bias** | **−0.29%** |
| 10%-trimmed mean bias | +0.63% |

Because the models predict the **median** (L1 loss), `median(pred/true) ≈ 1.00` — the model is
**centered**. The raw mean bias is a *mean-of-ratios* pulled up by a thin right tail of listings
where the model predicts far above a suspiciously low listed price (relative error is bounded
below by −100% but unbounded above, so it is structurally positive on skewed data).

**Forcing the raw bias to zero is actively harmful:** the multiplicative correction needed
(×0.877) pushes Median_APE from 12.8% → 16.8% and HitRate_20 from 64% → 58%. So the project
deliberately **does not** apply a naive smearing/bias correction. Instead it **reports robust bias
metrics** (`Median_Bias_pct`, `Bias_pct_trimmed10`) that tell the truth: the model is calibrated;
the tail is a data-quality problem, addressed by flagging (Section 7), not by distorting
predictions.

The same tail is why **MAPE (~35%) ≫ Median_APE (~15%)**. Steer by Median_APE and HitRate_20 for a
skewed target like house prices; treat MAPE as a *label-quality* indicator.

---

## 6. Calibration layer

Two additions turn the routed model into a shippable, self-aware system
([`src/train_region_subtype_calibrated.py`](Phase%201/Rule-based%20Scene/src/train_region_subtype_calibrated.py)).

### Hierarchical shrinkage routing (replaces the hard fallback)

Instead of hard-selecting one model per group, the estimate is **blended up the hierarchy** in log
space — `global → region → region×subtype` — each level pulled in by its own sample size via the
empirical-Bayes weight **`w = n / (n + K)`** (K tuned on validation). Sparse cells lean on their
parent; data-rich cells trust their local model; the transition is smooth, not a cliff at a
threshold.

**Result vs the old hard router (same 140,874-row test set):**

| Metric | Hard router | Shrinkage blend | Change | Significance |
|---|---|---|---|---|
| Median_APE | 15.39% | **14.84%** | −0.55 pp | 95% CI [−0.62, −0.48] |
| HitRate_20 | 58.31% | **59.45%** | +1.14 pp | McNemar p < 0.001 |

Real and statistically solid (net **+1,607 listings** moved into the ±20% band), though modest in
absolute terms — consistent with the diminishing returns that motivated the re-framing.

### Prediction intervals (P10–P90)

Two LightGBM quantile models produce a per-listing **80% interval**. Empirical coverage on the test
set is **76.7%** (target 80%). This interval is the backbone of the production defense — it is a
signal available **at inference time, before the true price is known**.

---

## 7. Regional reliability report

Full report: [`Phase 1/Rule-based Scene/Evaluation/regional_report/README.md`](Phase%201/Rule-based%20Scene/Evaluation/regional_report/README.md).
Per-region datasets and metrics live in that folder's `regions/` subtree (48 regions with ≥ 100
test listings reported individually, covering ~99% of listings; the rest pooled).

### 7a. Segmented reliability — accuracy is not uniform, and we say so

| Region | Test listings | Median_APE | HitRate_20 |
|---|---|---|---|
| Đà Nẵng | 26,787 | 11.3% | 69% |
| Hải Phòng | 13,952 | 12.8% | 65% |
| Bình Dương | 13,663 | 12.8% | 64% |
| Khánh Hòa | 8,034 | 11.2% | 66% |
| Đồng Nai | 12,978 | 14.6% | 59% |
| **Hồ Chí Minh** | 14,904 | **23.7%** | **44%** |

Median bias ≈ 0 in every region — no market is systematically over- or under-valued. The
segmentation also **surfaces the weak spot**: Hồ Chí Minh is a high-variance market the blended
average was hiding — a concrete, business-relevant lead (it likely needs district-level modeling).

### 7b. Error forensics — where large errors actually come from

Comparing normal listings (APE ≤ 20%) against large-error listings (APE > 50%):

> **29.2%** of large-error listings sit in the extreme price/m² tail, versus only **5.1%** of
> normal listings — a **~6× enrichment**.

Large errors are dominated by listings whose *stated price per m² is implausible* (underpriced
re-posts, placeholder prices, data-entry errors). **In these cases the listed price — not the
model — is the anomaly.** A normally-priced listing rarely lands in the error tail. This is the
"misunderstanding" the raw MAPE creates: it *sounds* unreliable, but that number is inflated by bad
labels, not by bad predictions on ordinary homes.

### 7c. Production defense — behaviour on a new case

The model's own interval width strongly predicts its real error, enabling a three-tier operating
policy that needs **no access to the true price**:

| Tier | Share | Median_APE | Within 20% | Action |
|---|---|---|---|---|
| **A — auto-serve** (narrow interval) | 40% | **9.6%** | 75% | return value automatically |
| B — serve with range | 40% | 17.6% | 54% | return value **with** P10–P90 band |
| C — flag for review (wide interval) | 20% | 28.9% | 39% | route to human review |

Complementary anomaly check: when a listing's asking price falls **inside** its predicted interval
(76.7% of listings) Median_APE is **10.9%**; when it falls **outside** (23%) it jumps to **40%**.

**Why this is defensible in a real product:** the dangerous failure mode of an AVM is not "it is
sometimes wrong" — every AVM is — but "it is *confidently* wrong without warning." This system is
not. Its error is concentrated in a population it can **identify up front** from interval width and
price plausibility. A new, normal listing lands in Tier A/B and is valued sharply; a new anomalous
listing lands in Tier C and is flagged. *Sharp where confident, cautious where not* — that is the
property that makes it safe to ship.

---

## 8. Final model

**The final model is the shrinkage blend retrained on cleaned data**: the training set minus
(a) listings flagged by the suspicious-price audit and (b) listings priced below **20M VND/m²**
(563,496 → 457,562 rows). The test set was left untouched, so the cleaned training is directly
comparable to the full training on the **identical 140,874 test rows**.

### The stability comparison (`Evaluation/improvements/final_model_stability/`)

Scored on the *sensible-asset* population — listings at ≥ 20M VND/m² and not price-flagged, i.e.
"a normal, make-sense case of asset" (**81% of the test set**, 114,540 listings):

| Model | Median_APE | HitRate_20 | Median bias |
|---|---|---|---|
| Trained on full data | 13.19% | 63.8% | −2.16% |
| **Trained on cleaned data (final)** | **12.74%** | **64.6%** | **−0.19%** |

Three things make this defensible:

1. **The result is stable.** Two trainings on very different data (one with 106k noisy rows, one
   without) land within **0.45 points** of each other on the sensible population, and the cleaned
   training is the better and *better-centered* of the two (median bias −0.19% vs −2.16%). The
   model's accuracy on normal assets does not depend on the noise — it is a property of the signal.
2. **Cleaning helps where the noise lived.** On sensible listings in previously noise-heavy rural
   markets, the cleaned training improves sharply: Long An 15.1% → **11.0%**, Hòa Bình 24.7% →
   **14.8%**, Tiền Giang 23.6% → **17.5%** Median_APE. Its prediction intervals are also
   *narrower* (median width 76% vs ~90%) while keeping **76.3%** coverage on the serving
   population.
3. **The validity domain is explicit — and enforced.** On the excluded population (sub-20M/m² or
   flagged listings, 19% of test) the cleaned model is deliberately not valid (it never saw such
   assets and grossly over-values them). This is not a hidden weakness but a **defined serving
   boundary**: a listing below 20M/m² or price-flagged is routed to Tier C / human review by the
   same input checks used for cleaning — the model only answers where it has proven, stable
   accuracy. *When a normal, sensible asset comes in, the system predicts sharply (typical error
   ~12.7%, ~65% within 20%); when an out-of-domain one comes in, it declines to guess.*

Reproduce: `python3 train_region_subtype_calibrated.py --output-suffix clean_ppm2_20m
--drop-suspicious --min-ppm2 20000000`, then `python3 compare_final_model_stability.py`.

---

## 9. How to reproduce

From `Phase 1/Rule-based Scene/src/` (expects processed parquet in `data/processed/`):

```bash
# 1. Train the deployable model: shrinkage routing + prediction intervals
python3 train_region_subtype_calibrated.py

# 2. Build the region-segmented reliability report (datasets, metrics, README)
python3 build_regional_report.py
```

All figures in the reports are computed from test predictions at build time — the documents cannot
drift from the numbers.

---

## 10. Repository layout

```
Phase 1/Rule-based Scene/
├── src/                          reusable logic
│   ├── common.py                 features, cleaning, metrics (incl. robust bias)
│   ├── outlier_handling.py       leakage-safe local price/m² outlier bands
│   ├── train_region_subtype_models.py       regional/subtype families + hard router
│   ├── train_region_subtype_calibrated.py   shrinkage routing + prediction intervals
│   ├── build_regional_report.py             region-segmented reliability report
│   └── … data-quality, dedup, text-extraction, CatBoost experiment scripts
├── notebooks/                    thin entrypoints 01–09
├── data/processed/               parquet artifacts (gitignored — reproducible & large)
├── Evaluation/
│   ├── improvements/             per-experiment metrics & predictions
│   │   └── region_subtype_calibrated/   the production model's results
│   └── regional_report/          ← the segmented reliability report + per-region data
└── Visualizations/               chart generation
```

---

## 11. Limitations & next steps

- **Thin regions stay uncertain.** Small provinces have wide intervals and larger errors; the
  policy correctly sends more of them to Tier C. More regional data is the only real fix.
- **The error tail is label noise, contained but not cleaned.** Flagging limits the damage;
  cleaning the underlying suspicious-price listings (already underway) is the upstream fix and the
  single biggest lever on raw MAPE.
- **Random split.** All numbers are on a stratified *random* test split. A **time-based split**
  (hold out the most recent months) is the next step to prove the model holds on *future* listings.
- **Hồ Chí Minh** is the clearest modeling follow-up — a high-variance market that likely needs
  district-level segmentation rather than one city-level model.

---

*This project values Vietnamese residential houses. It is built to be honest about its own
uncertainty: it predicts sharply where the data supports it, and it says so — and flags the rest —
where the data does not.*
