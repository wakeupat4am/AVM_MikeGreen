# Phase 3 — Hà Nội Sellability & Mispricing Report

**Goal:** turn the combined Hà Nội data into a **listing-triage engine** — for any house, estimate
the price it would actually *clear* at, and flag its asking price as **over- / fair- / under-priced**,
with a calibrated confidence band. The dataset is weak for absolute valuation (asking prices only,
tiny realized-sale sample) but strong for **relative mispricing**, so that is what Phase 3 delivers.

> ### ⚠️ Read this first — headline numbers were revised by a real-time backtest (2026-07)
> Sections 1–6.4 below were validated with **random GroupKFold**, which lets the model see the
> future and flatters every metric. A **rolling-origin backtest** ([§6.5](#65-real-time-backtest--the-defensibility-test), `backtest_rolling_origin.py`)
> re-measured everything as it would run in real time. The honest figures:
> - **MedAPE ~21% overall (not 18%)** — and 27–37% in 2025 as the market drifts from the training pool.
> - **v1's 80% interval coverage was an artifact** — it delivered only 67–77% under a time split. The
>   fixed **v2 bands** (`train_phase3_intervals_v2.py`) hold **81.6–95.5%** coverage in real time.
> - **The market rose +111% (hedonic index), not +136%** (raw median) — the gap was composition drift.
>
> **For decisions, use the v2 bands and [`MODEL_CARD.md`](MODEL_CARD.md), not the v1 tables below.**
> The original sections are kept for the analysis narrative.

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
**AVAILABLE if `last_modified` ≥ Oct 2025 (or 2026), else UNKNOWN** → 307 AVAILABLE, 17,000 UNKNOWN
(792 dated before Oct-2025 + 16,208 undated). No 2026 listings exist (scrape ends 2025-12).

> ⚠️ **The single most important caveat:** *every price in the data is an **asking** price.* There is
> no realized transaction price anywhere; `SOLD` is a **label** (a deal happened), not a sale value.
> All accuracy figures below are agreement with **asking** prices; `SOLD` simply marks asking prices
> that cleared the market.

---

## 2. Status policy — what each status is used for

| status | n | what it is | role | action |
|---|--:|---|---|---|
| **SOLD** | 194 | asking that **cleared** — gold label | train + positive class | up-weight ×5 |
| **REVOKED** | 9,941 | asking that **didn't clear**, later pulled | train (bulk) + negative class | keep |
| **AVAILABLE** (+**ON_STOCK** 345) | 652 | live **unsold** inventory | score, not train | merge ON_STOCK in |
| **LOCKED** | 13 | deal in progress | fold into SOLD | negligible |
| **UNKNOWN** | 17,000 | scraped listings, noisy, mostly undated, no geo | score + comparable prior | never a label |

Baked into `build_phase3_modeling.py`: **train on** SOLD (+LOCKED, ×5 weight) + REVOKED, suspicious
prices removed; **label from** SOLD (=1) vs REVOKED (=0) only; **score** AVAILABLE and UNKNOWN. Phase 2
proved that training on unsold inventory (AVAILABLE) *hurts* — it drags the model toward the asking level.

**What `LOCKED` means** (undocumented; inferred from all 13 rows): all are `NOT_AVAILABLE`, all have a
**SIGNED** brokerage contract, priced in the normal (clearing) range (median 6.5B) — i.e. a property
**frozen mid-transaction** (reserved / under deposit), between "available" and "sold". So it is treated
as effectively clearing → folded into SOLD. It is 0.05% of rows, so the choice is immaterial.

---

## 3. Market structure — what the data shows (figures interpreted)

### 3.1 Price by status (`figs/phase3_price_kde_by_status.png`, `..._dist_by_status.png`)
Median asking price, low → high: **SOLD 5.7B ≈ REVOKED 6.8B** (near-overlapping density peaks) →
**UNKNOWN 10.5B** → **AVAILABLE 14.2B** (after merging ON_STOCK's 18.5B tail). The clear signal:
**the market clears at the low end; unsold / listed inventory is the pricier tail.** This SOLD ≈ REVOKED
overlap is what makes REVOKED a sound training proxy, and AVAILABLE/UNKNOWN the "asking, not clearing" pools.

### 3.2 Price by district (`figs/phase3_price_kde_by_district.png`, district×status facets)
A clean **~4× median gradient**, peripheral → central:

| Hoài Đức | Hoàng Mai / Long Biên | Hà Đông / Nam Từ Liêm | Đống Đa / Ba Đình | Thanh Xuân | Tây Hồ | Cầu Giấy | Hoàn Kiếm |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 5.0B | 6.5B | 8.3–8.5B | 10.3–10.5B | 11.0B | 13.0B | 15.0B | **21.0B** |

Within-district spread is large everywhere (Hoàn Kiếm's 5th–95th spans ~2B→170B), so **district alone is
too coarse — ward + district is the location unit.** The district×status facets show the same
REVOKED-left / AVAILABLE-right ordering *inside* most districts, not just in aggregate.

### 3.3 Price over time (`figs/phase3_price_timeseries.png`)
Using `last_modified_date` across all members (2023-09 → 2026-07): median asking **price/m² rose
~147M → ~350M (+~140%)**; the smoothed market index used by the model runs **135M → 319M/m² (+136%)**.
Composition (bottom panel) is honest about coverage: **2023-09 → 2024-mid is thick transaction data**
(peak 1,700 rows/mo), **2025 thins and adds scraped listings**, **2026 rests on ~20–50 rows/mo** — so
the late uptrend is real in direction but low-confidence in level. **16,208 undated listings are excluded**
(can't be placed on the axis). This appreciation is the single biggest force in the data and motivates §5.

---

## 4. The lean common-feature model (why not the full one)

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

A full model would learn to lean on geo / construction_year (88–100% in training) then **impute them for
100% of UNKNOWN** — a silent failure. So Phase 3 uses the **lean common set**:
`total_area, number_of_floors, facade_width, district_name, ward_name`. Phase 2 showed geo adds **no
statistically significant gain**, so the accuracy cost is small and the model applies *honestly* to
UNKNOWN. Model: **LightGBM, L1 (median-calibrated), log target, SOLD ×5**.

---

## 5. Temporal treatment (the decisive step)

The market rose **+136%** (§3.3), but the transaction pool is **2023–24-heavy** while the inventory we
score is **2025–26** — so an un-timed model anchors in the *cheaper* market and **overstates over-pricing.**

`train_phase3_clearing_time.py` fixes this with a **deterministic market-index adjustment** (a learned
tree time-feature cannot extrapolate):

1. **Build** the smoothed monthly index `I(t)` from all dated members.
2. **Deflate** every training price to a reference month: `price_ref = price · I(ref)/I(t)` → the model
   predicts the clearing price *at the reference date*.
3. **Re-inflate** each prediction to the listing's own date: `clearing(t) = pred_ref · I(t)/I(ref)`,
   then compare to the asking (also at `t`). Dates beyond the data are **clamped** (no extrapolation);
   undated UNKNOWN rows fall back to the reference level and are flagged `time_adjusted=False`.

---

## 6. Results

### 6.1 Validation (out-of-fold, leakage-safe GroupKFold by `property_id`)

| model | set | MedAPE | MAPE | Hit20 |
|---|---|--:|--:|--:|
| no-time | all SOLD+REVOKED | 18.7% | 26.2 | 52.2 |
| **time-aware** | all SOLD+REVOKED | **18.0%** | 25.2 | **54.1** |
| time-aware | SOLD only (truth-ish) | 18.6% | 26.6 | 51.5 |

Lean model ≈18% MedAPE (vs Phase 2's ~13% full model — the price of dropping geo for UNKNOWN
compatibility). Time adjustment gives a modest, honest lift.

### 6.2 Scoring — the temporal treatment transforms the verdict

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

### 6.3 Comparable cross-check (model-independent)

`unknown_address_feature_matches.csv`: **4,629 UNKNOWN listings (68% of the ward-locatable ones)** have
a feature-identical comparable (same ward + area ±3% + floors) in the transaction pool; 1,351 also match
street name. The asking-vs-comparable price gap by matched status (street-matched, plausible prices):

| matched against | median gap | reading |
|---|--:|---|
| **SOLD** | **+28%** | UNKNOWN asks ~30% above where identical houses *cleared* |
| **REVOKED** | **+20%** | above the revoked/asking pool too |
| **AVAILABLE** | **~0%** | same level as other unsold inventory |
| ON_STOCK | −20% | below the priciest live inventory |
| **overall** | **+13% (median), +18% (trimmed mean)** | |

A second, independent confirmation of systematic over-asking. The matched extremes (+1000% / −99%)
also expose **heavy price noise** in the scraped listings — another reason UNKNOWN is a scoring target,
not a label. **Trust an UNKNOWN flag only where the model and the comparable agree.**

### 6.4 Prediction intervals — the decision-grade layer (`train_phase3_intervals.py`)

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

**Example bands (AVAILABLE):**

| district | ward | m² | asking B | P10 | P50 | P90 | flag |
|---|---|--:|--:|--:|--:|--:|---|
| Ba Đình | Cống Vị | 40 | 360.0 | 8.4 | 13.2 | 19.0 | STRONGLY_OVER *(also a data error)* |
| Thanh Xuân | T.X. Trung | 74 | 34.0 | 13.0 | 18.1 | **34.7** | **fair** *(point model would say "+88% over")* |
| Đống Đa | Hàng Bột | 34 | 7.0 | **7.29** | 10.2 | 16.3 | UNDERPRICED |

**Why this matters:** the point model called AVAILABLE ~29–62% over-priced; the interval model says only
**~19% clear the P90 ceiling** — the rest fall *inside* a calibrated clearing band. That gap was false
confidence. Now each flag means "confidently outside the 80% clearing range," with a controlled
false-positive rate — exactly what a triage engine needs.

---

## 6.5 Real-time backtest — the defensibility test

Everything above uses **random GroupKFold**, and the index above was smoothed with a *centered* window
over the whole timeline — both let the model peek at the future. But the model's whole pitch is
temporal (deflate 2023–24, re-inflate into 2025–26), so it must be judged the way it would run: **train
on the past, predict the future.** `backtest_rolling_origin.py` does exactly that — stand at origin
month `T`, use only data ≤ `T` (model *and* index), predict SOLD+REVOKED of `T+1..T+3`, slide `T`
quarterly across 2024-03→2025-09 — and pits five index variants against each other.

### 6.5.1 Point accuracy is worse than random-fold said — honestly ~21%, drifting to 37%

| origin (predict next 3 mo) | no-time | leaky (README §5) | causal nowcast | **hedonic nowcast** |
|---|--:|--:|--:|--:|
| 2024-03 | 19.5 | 18.9 | 19.0 | **18.9** |
| 2024-09 | 24.9 | 25.2 | 25.8 | **25.6** |
| 2024-12 | 32.2 | 34.3 | 35.3 | **33.4** |
| 2025-06 | 32.0 | 27.2 | 27.5 | **27.5** |
| 2025-09 | 43.9 | 37.3 | 37.0 | **36.3** |
| **weighted overall** | 21.7 | 21.6 | 21.7 | **21.5** |

Two lessons: **(a)** the real-time MedAPE is **~21%, and 27–37% once the market drifts** — the "18%"
was random-fold optimism; **(b)** the time adjustment earns its keep exactly when the market has moved
(2025-09: 43.9%→36.3%), and the **hedonic index wins** at almost every hard origin. The *leaky* variant
(current README method) is within noise of the causal ones — so the look-ahead was in the **fold**, not
the index; the index architecture is sound.

### 6.5.2 The v1 80% band did NOT hold — and the fix that does (v2)

Under a **temporal** calibration split, v1's CQR band covered only **67–77%** at most origins — the
80.0% in §6.4 was a random-fold artifact. `backtest_cqr_fix.py` grids the fix (calibration window,
recency weight, nominal level) and selects **split-conformal on a trailing window at nominal 90%**, the
drift margin that clears 80% at every origin:

| origin | v1 real-time coverage | **v2 real-time coverage** |
|---|--:|--:|
| 2024-03 | 76.3% | **87.7%** |
| 2024-12 | 69.9% | 84.0% |
| 2025-03 | 89.1% | 95.5% |
| 2025-09 | 67.1% | **81.6%** |
| **range** | **67–89% (mostly < 80)** ❌ | **81.6–95.5% (never < 80)** ✅ |

`train_phase3_intervals_v2.py` is the production scorer. Honest cost: median band widens ×2.19→×4.5 —
that width **is** the real drift uncertainty v1 hid. Result on inventory: AVAILABLE **81% fair / 10%
over** (was 61% / 19%), UNKNOWN **82% fair**.

### 6.5.3 Does the verdict survive index error? (sensitivity)

The 2025–26 index rests on 20–50 rows/month, so `sensitivity_index_error.py` perturbs the late index by
±20% and re-counts v2 flags:

| index error | AVAILABLE `fair` | AVAILABLE `UNDERPRICED` | AVAILABLE `STRONGLY_OVER` |
|---|--:|--:|--:|
| −20% | 75.0% | 6.4% | 9.2% |
| **0%** | **81.0%** | 8.9% | 5.1% |
| +20% | 80.8% | 12.7% | 2.9% |

**The `fair` verdict is robust (75–81% across ±20%); the *tails* are index-sensitive.** So: trust the
`fair` flag; treat `UNDERPRICED`/`STRONGLY_OVER` on 2025+ listings as "review," not "confirmed."

---

## 7. Insights

1. **Available inventory is not ¾ over-priced — it's roughly fair once you credit the market.** The
   headline "+48%" collapses to **+1% median** after time adjustment; the interval model confirms **~61%
   fair, only ~19% above the P90 ceiling.** The apparent over-pricing was mostly the model anchored in the
   2023–24 market while the listings live in 2025–26.
2. **The market, not mispricing, is the dominant force.** +136% appreciation swamps the ±10–30% mispricing
   signal. Any valuation here that ignores time is wrong by ~40%.
3. **SOLD ≈ REVOKED, confirmed four ways** (distribution, KDE, district×status facets, and the model): the
   market clears at the low end; unsold inventory is the pricier tail.
4. **UNKNOWN behaves like unsold listing inventory** — over-asking confirmed by distribution, KDE, *and*
   comparable-matching (+20–28% over SOLD/REVOKED) — but its **own prices are too noisy to trust
   row-by-row** (the 13–29% "underpriced" is largely data noise, not bargains).
5. **Location is saturated by district+ward.** Adding geo didn't significantly help; the lean model is the
   right honesty/accuracy trade-off given 61% of rows have no coordinates.
6. **Uncertainty must be modelled, not assumed.** With ~18% point error, only a *calibrated band* makes the
   flags trustworthy — the CQR layer is what turns "directionally useful" into "decision-grade."

### Honest limits
- **~18% MedAPE**, not 13% — the lean model's cost. Absolute valuations are ±18%; use the **band/flag**,
  not the point.
- **AVAILABLE's ~41% "under-priced" (point model) is time-extrapolation-optimistic** — re-inflating into
  thin 2026 index months pushes clearing prices up. The interval model tempers this (20% under P10). Treat
  AVAILABLE UNDERPRICED as "review," not "confirmed bargain."
- **Everything is asking-price modelling.** Realized transaction prices remain the true ceiling.

---

## 8. Step-by-step — reproduce the whole report

```bash
cd "Phase 3/src"

# 1. Merge Phase 1 Hà Nội listings into the Phase 2 schema  -> phase3_merged_dataset.parquet
python3 build_phase3_dataset.py

# 2. Assemble modeling splits per the status policy (train + 2 scoring frames, weights/labels)
python3 build_phase3_modeling.py

# 3a. Baseline clearing model (no time)                    -> phase3_{available,unknown}_predictions.csv
python3 train_phase3_clearing.py

# 3b. TIME-AWARE clearing model (market-index adjust)      -> *_predictions_timeadj.csv
python3 train_phase3_clearing_time.py

# 3c. PREDICTION INTERVALS v1 (CQR, per-district; random-fold — see §6.5 caveat)
python3 train_phase3_intervals.py

# 3d. REAL-TIME BACKTEST (rolling-origin) + drift-proof CQR selection + PRODUCTION bands
python3 backtest_rolling_origin.py        # real-time MedAPE + v1 coverage gap        (§6.5.1–2)
python3 backtest_cqr_fix.py               # grid-select the drift-proof CQR config     (§6.5.2)
python3 train_phase3_intervals_v2.py      # PRODUCTION bands -> *_intervals_v2.csv   [use these]
python3 sensitivity_index_error.py        # index-error robustness table               (§6.5.3)

# 4. Figures
python3 plot_price_kde.py                 # price density by status / by district
python3 plot_kde_district_by_status.py    # district x status density facets
python3 plot_price_distributions.py       # boxplot views (status / district)
python3 plot_timeseries.py                # market index / composition over time
```

**Pipeline in one line:** merge → split by status policy → lean model, time-adjusted → calibrated
clearing band → compare asking to the band → over/fair/under flag, gated by confidence (AVAILABLE
full-feature; UNKNOWN needs the comparable cross-check).

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

## 9. Next steps (ranked)
1. **"Will it sell?" classifier** from the SOLD-vs-REVOKED label — the one thing this dataset has *real*
   labels for; sidesteps the asking-price ceiling entirely.
2. **Anchor the hedonic index to a published Hà Nội price index** (CBRE / Savills / batdongsan) — an
   independent external check on the +111% trajectory.
3. **Full-feature model for AVAILABLE** (it has 44% geo / 53% year) to recover accuracy there.
4. **Fuse the interval flag + comparable flag** into one confidence-tiered verdict for UNKNOWN.
5. **Acquire realized transaction prices** — the true accuracy ceiling; every experiment points here.

*Done:* merge · status policy (incl. LOCKED) · market-structure EDA · lean model · temporal
(market-index) treatment · CQR intervals · **rolling-origin backtest (real-time MedAPE + coverage)** ·
**drift-proof CQR v2 + hedonic index** · **index-error sensitivity** · **[`MODEL_CARD.md`](MODEL_CARD.md)**.

## 10. Layout
```
Phase 3/
├── data/   phase3_merged_dataset.parquet · phase3_*_intervals.csv · phase3_*_intervals_v2.csv
│            phase3_backtest_results.csv · phase3_backtest_coverage.csv · phase3_cqr_fix_grid.csv
│            phase3_sensitivity_index.csv · phase3_*_predictions_timeadj.csv
│            phase3_interval_coverage.csv · unknown_address_feature_matches.csv
├── src/    build_phase3_dataset.py · build_phase3_modeling.py
│            train_phase3_clearing[_time].py · train_phase3_intervals[_v2].py
│            backtest_rolling_origin.py · backtest_cqr_fix.py · sensitivity_index_error.py · plot_*.py
├── figs/   phase3_price_kde_* · phase3_price_dist_* · phase3_price_timeseries.png
├── MODEL_CARD.md   (intended use, validity domain, real-time metrics, limits — read first)
└── README.md   (this report)
```
