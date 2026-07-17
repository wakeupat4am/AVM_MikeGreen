# Model Card — Phase 3 Hà Nội Listing-Triage Engine

*Last updated: 2026-07-17. Owner: AVM project. Status: validated by rolling-origin backtest.*

This card is the first thing a reviewer or buyer should read. It states what the model is for,
where it is valid, how well it works **under real-time conditions** (not random-fold), and where it
must not be trusted. Every number here comes from the reproducible scripts in `src/`.

---

## 1. What it is

A **triage engine** for Hà Nội landed-house listings. For any listing it returns a **calibrated
clearing-price band `[P10, P50, P90]`** and a flag — `UNDERPRICED` / `fair` / `OVERPRICED` /
`STRONGLY_OVER` — by where the asking price falls relative to the band.

It is **not** an absolute appraiser. Every price in the training data is an *asking* price; `SOLD`
is a label (a deal happened), not a realized sale value. The product is **relative mispricing with
a calibrated confidence band**, which the data supports; single-number valuation, which it does not.

## 2. Intended use / out of scope

| Intended | Out of scope |
|---|---|
| Rank/triage live inventory by mispricing | Legal/collateral valuation of a single property |
| Flag listings confidently outside the 80% clearing band | Absolute price quote without a band |
| Hà Nội landed houses (nhà đất) | Apartments, land-only plots, other cities |
| Listings priced roughly 1–200B with sane area | Prices/areas outside the training envelope |

**Validity domain:** Hà Nội, landed houses, features within the training range. Outside it the model
declines rather than extrapolates (undated rows fall back to the reference-month index level and are
flagged `time_adjusted=False`).

## 3. Data

| source | rows | role |
|---|--:|---|
| Phase 2 brokerage inventory (has lifecycle status) | 10,493 | train (SOLD ×5 + REVOKED) + labels |
| Phase 1 scraped Hà Nội asking listings | 17,307 | score (AVAILABLE / UNKNOWN) |
| **merged** | **27,800** | — |

Train on **SOLD (194, +LOCKED 13, up-weighted ×5) + REVOKED (9,941)**; score **AVAILABLE (652)** and
**UNKNOWN (17,000)**. Training on unsold inventory was shown (Phase 2) to drag predictions toward the
asking level, so it is excluded. Lean feature set — `total_area, number_of_floors, facade_width,
district_name, ward_name` — because UNKNOWN has 0% geo/construction-year; a richer model would impute
those for 100% of UNKNOWN (silent failure). Model: LightGBM, log target, L1/quantile.

## 4. Market-time adjustment (the dominant force)

Hà Nội prices rose sharply over 2023–2026, while the transaction pool is 2023–24-heavy and the scored
inventory is 2025–26. Ignoring this overstates over-pricing by ~40%. v2 corrects it with a **hedonic
index** — month coefficients of `log(price/m²) ~ month + district + log(area) + floors + facade`,
built causally (only data ≤ cutoff). Train prices are deflated to a reference month; predictions are
re-inflated to each listing's own month.

> The hedonic index measures **+111% (×2.11)** over 2023-09→2026-07 — *not* the +136% a raw median
> shows. The gap is composition drift (transaction data early, scraped listings late); the hedonic
> index controls it out. Use +111% as the defensible figure.

## 5. Performance — **real-time (rolling-origin backtest)**

Validated by walk-forward: train on data ≤ month T, predict SOLD+REVOKED of months T+1..T+3, slide T
quarterly 2024-03→2025-09 (`backtest_rolling_origin.py`). This is the honest number; the random-fold
"18%" reported in earlier drafts benefited from look-ahead and should not be quoted.

| metric | value | note |
|---|--:|---|
| MedAPE, early period (2024) | **~19–25%** | market still near training level |
| MedAPE, late period (2025) | **~27–37%** | degrades as the market drifts from training |
| MedAPE, weighted overall | **~21.5%** (hedonic) | best of 5 index variants |
| Hit20, overall | ~48% | |

**Point predictions degrade over time — this is expected and is why the product is the band, not the
point.** The time adjustment earns its keep exactly when the market has drifted: at origin 2025-09 it
cuts MedAPE from 43.9% (no adjustment) to 36.3% (hedonic).

## 6. Calibrated bands — coverage **holds under drift** (v2)

v1's CQR band promised 80% coverage but a temporal split revealed it delivered only **67–77%** in real
time (random-fold coverage was an artifact). v2 fixes this: split-conformal calibration on a trailing
window of the most recent training months, at **nominal 90%** — a drift safety margin selected by
backtest (`backtest_cqr_fix.py`).

| | v1 (random-fold) | **v2 (drift-proof)** |
|---|--:|--:|
| real-time coverage (per origin) | 67–89%, mostly **< 80%** ❌ | **81.6–95.5%, never < 80%** ✅ |
| median band width P90/P10 | ×2.19 (optimistic) | **×4.5 (honest)** |
| AVAILABLE flagged `fair` | 61% | **81%** |
| AVAILABLE flagged over / strongly-over | 19% | **10%** |

The wider band is not a worse model — it is the true uncertainty under drift that v1 hid. For a triage
product this is the selling point: a flag now means "confidently outside the 80% clearing range," with
a controlled false-positive rate.

## 7. Robustness — sensitivity to late-index error

The 2025–26 index months rest on 20–50 rows/month, so their level is low-confidence. Perturbing the
late index by ±20% (`sensitivity_index_error.py`, on v2):

- **`fair` share is stable: AVAILABLE 75–81%, UNKNOWN 81–82%** → the headline verdict survives.
- The **tails move**: AVAILABLE `UNDERPRICED` 6→13%, `STRONGLY_OVER` 9→3%.

→ **Trust the `fair` flag; treat `UNDERPRICED`/`STRONGLY_OVER` on *2025+* listings as "review," not
"confirmed," because they are sensitive to the thin late index.**

## 8. Known limitations

1. **Asking-price modelling.** No realized transaction prices exist anywhere; accuracy is agreement
   with asking prices, and `SOLD` is a label. Realized prices remain the true ceiling.
2. **Real-time MedAPE ~21% overall, worse (27–37%) in 2025.** Use the band/flag, not the point.
3. **UNKNOWN is feature-poor and noisy** (0% geo, mostly undated). Its own asking prices carry heavy
   noise; a flag is trustworthy only where the model **and** the comparable cross-check agree.
4. **Tail flags on 2025+ listings are index-sensitive** (§7).
5. **Thin late data.** 2026 rests on ~2–13 rows/month; the index direction is real, the *level* is
   uncertain, and coverage is widest there (Hoàn Kiếm P90/P10 up to ×6).

## 9. Update & monitoring policy

- **Re-fit + re-calibrate** monthly as new dated rows arrive; the trailing calibration window and
  hedonic index refresh automatically.
- **Re-run `backtest_rolling_origin.py`** whenever the pipeline changes; publish the new real-time
  MedAPE and per-origin coverage — do not report random-fold numbers.
- **Watch coverage drift:** if backtest coverage at the latest origin falls below 80%, raise the CQR
  nominal or lengthen the calibration window (both gridded in `backtest_cqr_fix.py`).

## 10. Reproduce

```bash
cd "Phase 3/src"
python3 build_phase3_dataset.py            # merge -> phase3_merged_dataset.parquet
python3 build_phase3_modeling.py           # status-policy splits
python3 backtest_rolling_origin.py         # real-time performance (§5) + v1 coverage gap (§6)
python3 backtest_cqr_fix.py                # select the drift-proof CQR config (§6)
python3 train_phase3_intervals_v2.py       # PRODUCTION bands -> *_intervals_v2.csv
python3 sensitivity_index_error.py         # robustness table (§7)
```
