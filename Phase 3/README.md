# Phase 3 — Hà Nội Sellability & Mispricing Report

**Goal:** turn the combined Hà Nội data into a **listing-triage engine** — for any house, estimate
the price it would actually *clear* at, and flag its asking price as **over- / fair- / under-priced**.
The dataset is weak for absolute valuation (asking prices only, tiny realized-sale sample) but strong
for **relative mispricing**, so that is what Phase 3 delivers.

---

## 1. Dataset — merge of both sources

`build_phase3_dataset.py` maps Phase 1 Hà Nội scraped listings onto the **Phase 2 cleaned schema**
and appends them → `data/phase3_merged_dataset.parquet`.

| source (`dataset_source`) | rows | what it is |
|---|--:|---|
| `phase2_transaction` | 10,493 | brokerage inventory with a lifecycle status (SOLD/REVOKED/…) |
| `phase1_listing` | 17,307 | scraped Hà Nội asking listings (landed houses) |
| **Total** | **27,800** | 36 Phase-2 columns + `dataset_source` |

Phase-1 listings get an availability label from their date (as requested):
**AVAILABLE if `last_modified` ≥ Oct 2025 (or 2026), else UNKNOWN** → 307 AVAILABLE, 17,000 UNKNOWN.

> ⚠️ **The single most important caveat:** *every price in the data is an **asking** price.* There is
> no realized transaction price anywhere; `SOLD` is a **label** (a deal happened), not a sale value.
> All accuracy figures below are agreement with **asking** prices; `SOLD` simply marks asking prices
> that cleared the market.

---

## 2. Status policy — what each status is used for

| status | n | role | action |
|---|--:|---|---|
| **SOLD** | 194 | asking that **cleared** — gold label | train (up-weighted) + positive class |
| **REVOKED** | 9,941 | asking that **didn't clear** | train (bulk) + negative class |
| **AVAILABLE** (+**ON_STOCK** 345) | 652 | live **unsold** inventory | **score, don't train** |
| **LOCKED** | 13 | deal in progress | fold into SOLD |
| **UNKNOWN** | 17,000 | scraped listings, noisy, mostly undated, no geo | **score + comparable prior, never a label** |

Rules baked into `build_phase3_modeling.py`:
- **Train on** SOLD (+LOCKED, ×5 weight) + REVOKED, suspicious prices removed.
- **Label from** SOLD (=1) vs REVOKED (=0) only.
- **Score** AVAILABLE and UNKNOWN (never trained on — Phase 2 proved adding unsold inventory *hurts*).

---

## 3. The lean common-feature model (why not the full one)

The three groups **do not share features** — UNKNOWN has none of Phase 2's rich columns:

| feature | TRAIN (SOLD+REV) | AVAILABLE | UNKNOWN |
|---|--:|--:|--:|
| total_area | 100% | 100% | 100% |
| number_of_floors | 100% | 78% | 72% |
| facade_width | 100% | 66% | 42% |
| construction_year | 97% | 53% | **0%** |
| latitude / longitude | 88% | 44% | **0%** |
| main_road_dist | 100% | 53% | **0%** |
| ward_name | 100% | 98% | 55% |

A full model would learn to lean on geo / construction_year (88–100% in training) then **impute them
for 100% of UNKNOWN** — a silent failure. So Phase 3 uses the **lean common set**:
`total_area, number_of_floors, facade_width, district_name, ward_name`. Phase 2 already showed geo
adds **no statistically significant gain**, so the accuracy cost is small and the model applies
*honestly* to UNKNOWN. Model: **LightGBM, L1 (median-calibrated), log target, SOLD ×5**.

---

## 4. Temporal treatment (the decisive step)

Hà Nội prices **rose ~136%** across the window (`figs/phase3_price_timeseries.png`): the market index
`I(t)` = monthly median ₫/m² climbs **135M → 319M/m² (2023-09 → 2026)**. The transaction pool is
**2023–24-heavy**, but the inventory we score is **2025–26** — so an un-timed model anchors in the
*cheaper* market and **overstates over-pricing**.

`train_phase3_clearing_time.py` fixes this with a **deterministic market-index adjustment**
(transparent, non-extrapolating — a learned tree time-feature cannot extrapolate):

1. **Build** the smoothed monthly index `I(t)` from all dated members.
2. **Deflate** every training price to a reference month: `price_ref = price · I(ref)/I(t)` → the model
   predicts the clearing price *at the reference date*.
3. **Re-inflate** each prediction to the listing's own date: `clearing(t) = pred_ref · I(t)/I(ref)`,
   then compare to the asking (also at `t`). Dates beyond the data are **clamped** (no extrapolation);
   undated UNKNOWN rows fall back to the reference level and are flagged `time_adjusted=False`.

---

## 5. Results

### 5.1 Validation (out-of-fold, leakage-safe GroupKFold by `property_id`)

| model | set | MedAPE | MAPE | Hit20 |
|---|---|--:|--:|--:|
| no-time | all SOLD+REVOKED | 18.7% | 26.2 | 52.2 |
| **time-aware** | all SOLD+REVOKED | **18.0%** | 25.2 | **54.1** |
| **time-aware** | SOLD only (truth-ish) | 18.6% | 26.6 | 51.5 |

Lean model is ~18% MedAPE (vs Phase 2's ~13% full model — the price of dropping geo for UNKNOWN
compatibility). Time adjustment gives a modest, honest lift.

### 5.2 Scoring — the temporal treatment transforms the verdict

**AVAILABLE (n=612 clean, full-feature → higher trust)**

| | no-time | **time-aware** |
|---|--:|--:|
| median asking vs clearing | **+48%** | **+1%** |
| strongly over (>30%) | 62% | 29% |
| over-priced (>10%) | 73% | 42% |
| fair (±10%) | 9% | 17% |
| under-priced | 18% | 41% |

**UNKNOWN (n=16,369 clean, feature-poor + noisy → lower trust)**

| | no-time | **time-aware** |
|---|--:|--:|
| median asking vs clearing | +21% | +14% |
| strongly over (>30%) | 43% | 39% |
| fair | 16% | 17% |
| under-priced | 26% | 29% |

UNKNOWN barely moves because **only 792 of 17,000 are dated** — we honestly cannot time-place the
other 16,208, so they stay at the reference level.

### 5.3 Comparable cross-check (model-independent)

`unknown_address_feature_matches.csv`: **4,629 UNKNOWN listings (68% of the ward-locatable ones)** have
a feature-identical comparable (same ward + area ±3% + floors) in the transaction pool. On the
high-confidence street-matched subset, UNKNOWN asks a **median +13–28% above** comparable SOLD/REVOKED
prices — a second, independent confirmation of systematic over-asking. **Trust an UNKNOWN flag only
where the model and the comparable agree.**

### 5.4 Prediction intervals — the decision-grade layer (`train_phase3_intervals.py`)

A point clearing price ± an arbitrary 10% band is untrustworthy when the model's own error is ~18%
(the ±10% flag sits *inside* the noise). So the model is wrapped in a **calibrated band [P10, P50, P90]**
via **Conformalized Quantile Regression (CQR)**, time-aware, calibrated **per district** (global
fallback for thin ones). A listing is flagged only by where its asking falls relative to the band.

**Coverage is validated** (out-of-fold, target 80%):

| | value |
|---|---|
| raw quantile coverage | 69.5% (under-covers) |
| **CQR-calibrated coverage** | **80.0%** (on target) |
| per-district coverage | **all 80.0–80.3%** |
| median band width (P90/P10) | ×2.19 — widens where uncertainty is real: **Hoàn Kiếm ×6.2**, Nam Từ Liêm ×1.8 |

**Interval-based flags** (asking `< P10` under · in-band `fair` · `> P90` over · `> P90×1.3` strongly-over):

| | AVAILABLE (n=652) | UNKNOWN (n=17,000) |
|---|--:|--:|
| UNDERPRICED (< P10 floor) | 20.1% | 13.3% |
| **fair (inside 80% band)** | **61.2%** | **61.5%** |
| OVERPRICED (> P90 ceiling) | 9.4% | 12.9% |
| STRONGLY_OVER (> P90 ×1.3) | 9.4% | 12.2% |

**Why this matters:** the point model called AVAILABLE ~29–62% over-priced; the interval model says only
**~19% clear the P90 ceiling** — the rest fall *inside* a calibrated clearing band. That gap was false
confidence. Now each flag means "confidently outside the 80% clearing range," with a controlled
false-positive rate — exactly what a triage engine needs. Example: Thanh Xuân 74m² asking 34.0B, band
`[13.0, 34.7]` → correctly **fair**, though a point model at 18.1B would have screamed "+88% over."

---

## 6. Insights

1. **Available inventory is not ¾ over-priced — it's roughly fair once you credit the market.**
   The headline "+48%" collapses to **+1% median** after time adjustment. The apparent over-pricing was
   mostly the model being anchored in the 2023–24 market while the listings live in 2025–26.
   ~**29% remain genuinely strongly over-priced** — those are the robust flags.
2. **The market, not mispricing, is the dominant force.** +136% index over the window swamps the
   ±10–30% mispricing signal. Any valuation here that ignores time is wrong by ~40%.
3. **SOLD ≈ REVOKED confirmed a third way** (distribution, KDE, and now the model): the market clears at
   the low end; unsold inventory is the pricier tail.
4. **UNKNOWN behaves like unsold listing inventory** — over-asking confirmed by distribution, KDE, and
   comparable-matching — but its **own prices are too noisy to trust row-by-row** (26–29% "underpriced"
   is largely data noise, not bargains).
5. **Location is saturated by district+ward.** Adding geo didn't significantly help; the lean model is
   the right honesty/accuracy trade-off given 61% of rows have no coordinates.

### Honest limits
- **~18% MedAPE**, not 13% — the lean model's cost. Absolute valuations are ±18%, use the **band/flag**,
  not the point.
- **AVAILABLE's 41% "under-priced" is time-extrapolation-optimistic** — re-inflating into thin 2026 index
  months pushes clearing prices up. Treat AVAILABLE UNDERPRICED as "review," not "confirmed bargain."
- **Everything is asking-price modelling.** Realized transaction prices remain the true ceiling.

---

## 7. Step-by-step — reproduce the whole report

```bash
cd "Phase 3/src"

# 1. Merge Phase 1 Hà Nội listings into the Phase 2 schema  -> phase3_merged_dataset.parquet
python3 build_phase3_dataset.py

# 2. Assemble modeling splits per the status policy (train + 2 scoring frames, with weights/labels)
python3 build_phase3_modeling.py

# 3a. Baseline clearing model (no time)  -> phase3_{available,unknown}_predictions.csv
python3 train_phase3_clearing.py

# 3b. TIME-AWARE clearing model (market-index adjust) -> *_predictions_timeadj.csv
python3 train_phase3_clearing_time.py

# 3c. PREDICTION INTERVALS (CQR, per-district coverage) -> *_intervals.csv   [decision-grade; use these]
python3 train_phase3_intervals.py

# 4. Figures
python3 plot_price_kde.py                 # price density by status / by district
python3 plot_kde_district_by_status.py    # district x status density facets
python3 plot_timeseries.py                # market index / composition over time
```

**Pipeline in one line:** merge → split by status policy → lean model, time-adjusted → clearing price →
compare to asking → over/fair/under flag, gated by confidence (AVAILABLE full-feature; UNKNOWN needs the
comparable cross-check).

### Key outputs
| file | contents |
|---|---|
| `phase3_merged_dataset.parquet` | 27,800-row merged table (Phase-2 schema + `dataset_source`) |
| **`phase3_available_intervals.csv`** | AVAILABLE: asking, **P10/P50/P90 band**, over-P90 %, calibrated flag |
| **`phase3_unknown_intervals.csv`** | UNKNOWN: same (lower trust; `time_adjusted` marks the 792 dated) |
| `phase3_interval_coverage.csv` | per-district empirical coverage + median band width |
| `phase3_*_predictions_timeadj.csv` | point time-aware predictions (pre-interval) |
| `unknown_address_feature_matches.csv` | 4,629 UNKNOWN↔transaction comparable pairs w/ price gap |
| `figs/phase3_*` | status/district density curves, district×status facets, market time series |

---

## 8. Next steps (ranked)
1. **"Will it sell?" classifier** from the SOLD-vs-REVOKED label — the one thing this dataset has *real*
   labels for; sidesteps the asking-price ceiling entirely.
2. **Full-feature model for AVAILABLE** (it has 44% geo / 53% year) to recover ~13% accuracy there.
3. **Fuse model interval flag + comparable flag** into one confidence-tiered verdict for UNKNOWN.
4. **Geocode UNKNOWN** addresses to recover the location feature and place them on the map.
5. **Acquire realized transaction prices** — the true accuracy ceiling; every experiment points here.

*Done:* status policy · lean model · temporal (market-index) treatment · **calibrated prediction
intervals (CQR, 80% coverage validated per district)**.

## 9. Layout
```
Phase 3/
├── data/   phase3_merged_dataset.parquet · phase3_*_intervals.csv · phase3_interval_coverage.csv
│            phase3_*_predictions_timeadj.csv · unknown_address_feature_matches.csv
├── src/    build_phase3_dataset.py · build_phase3_modeling.py
│            train_phase3_clearing[_time].py · train_phase3_intervals.py · plot_*.py
├── figs/   phase3_price_kde_* · phase3_price_dist_* · phase3_price_timeseries.png
└── README.md   (this report)
```
