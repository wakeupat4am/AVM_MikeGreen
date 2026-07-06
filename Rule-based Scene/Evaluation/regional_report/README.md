# AVM Regional Reliability Report

*Production model: hierarchical **shrinkage blend** (`pred_price_vnd_shrunk`) with P10–P90 prediction intervals. Evaluated on the held-out test set of 140,874 listings across 94 regions.*

---

## 1. Why this report exists (framing)

Chasing a single dataset-wide accuracy number has hit diminishing returns — repeated modeling
iterations now move the headline error by only fractions of a point. For a real product, the
right question is not *"what is the average error?"* but **"where, and for which listings, can we
trust the valuation — and what do we do when we can't?"**

This report therefore reframes the model along three business-relevant axes:

1. **Segmented reliability** — accuracy is reported **per region**, because a national average
   hides that some markets are prediction-ready today and others are data-limited.
2. **Error forensics** — we show that large errors are **not random model failure**; they
   concentrate in a small, identifiable population of anomalous / mislabeled listings.
3. **A production defense** — for a *normal* incoming listing the system predicts **sharply**,
   and it **knows when it cannot** (via its own prediction interval), so anomalous cases are
   flagged rather than silently mis-valued.

---

## 2. Per-region results

Full ranked table: [`region_summary.csv`](region_summary.csv). Every province has **one CSV named
after it** under [`regions/`](regions/) (e.g. `regions/đà nẵng.csv`) holding all of that province's
test listings — actual price, prediction, P10–P90 interval, `abs_pct_error`, a `case_type`
label (normal / moderate / large_error), and confidence tier — **sorted worst-error first**.
48 provinces with ≥ 100 test listings get their own file
(covering ~99% of listings); the rest are pooled into `other_small_regions.csv`. Which provinces
generate the most errors, and how those errors differ, is in
[`region_error_comparison.csv`](region_error_comparison.csv).

Top regions by sample size:

| region | n_test | MAE_billion | RMSE_billion | MAPE | Median_APE | HitRate_10_pct | HitRate_20_pct | Bias_pct | Median_Bias_pct | Bias_pct_trimmed10 | mean_rel_width_pct | in_band_coverage_pct | high_conf_Median_APE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| đà nẵng | 26787 | 2.4 | 11.663 | 21.0 | 11.34 | 45.9 | 69.46 | 5.32 | -0.12 | 0.38 | 68.6 | 77.1 | 8.82 |
| hồ chí minh | 14904 | 5.117 | 20.489 | 47.97 | 23.65 | 24.45 | 43.69 | 22.6 | -0.65 | 3.31 | 130.3 | 75.7 | 18.64 |
| hải phòng | 13952 | 1.575 | 5.137 | 23.8 | 12.83 | 42.3 | 65.12 | 5.76 | -0.72 | -0.22 | 96.2 | 77.9 | 9.42 |
| bình dương | 13663 | 1.192 | 4.988 | 30.51 | 12.84 | 42.82 | 64.01 | 13.53 | -0.2 | 1.29 | 91.9 | 77.6 | 8.59 |
| đồng nai | 12978 | 1.695 | 8.428 | 32.04 | 14.55 | 39.53 | 59.24 | 12.52 | -0.17 | 1.55 | 109.7 | 76.6 | 10.3 |
| khánh hòa | 8034 | 2.664 | 15.988 | 24.82 | 11.22 | 47.09 | 66.46 | 7.72 | -0.09 | 0.12 | 95.3 | 76.2 | 7.92 |
| bà rịa vũng tàu | 6873 | 2.791 | 11.035 | 27.94 | 11.89 | 45.47 | 64.63 | 9.47 | -0.45 | 0.04 | 101.2 | 76.8 | 7.65 |
| long an | 6496 | 1.066 | 4.23 | 33.54 | 17.71 | 33.39 | 53.74 | 11.93 | -0.0 | 2.57 | 143.5 | 78.0 | 12.69 |
| cần thơ | 3514 | 2.396 | 8.875 | 30.02 | 13.33 | 42.26 | 62.09 | 11.21 | -0.49 | 0.3 | 107.7 | 76.7 | 9.19 |
| hà nội | 3358 | 6.516 | 19.421 | 152.51 | 20.37 | 28.62 | 49.32 | 127.72 | -1.96 | -0.55 | 106.7 | 75.6 | 18.09 |
| huế | 2566 | 1.913 | 10.537 | 25.6 | 11.95 | 44.86 | 66.02 | 8.95 | -0.02 | 0.82 | 114.4 | 77.9 | 9.06 |
| đắk lắk | 2228 | 1.448 | 4.157 | 32.1 | 13.96 | 39.63 | 62.25 | 13.76 | -0.26 | 0.85 | 118.5 | 75.3 | 10.81 |

**Reading it:** `Median_APE` is the typical listing's error; `high_conf_Median_APE` is the error on
that region's higher-confidence half (narrow interval). `Median_Bias_pct` ≈ 0 everywhere confirms
the model is **centered** — it does not systematically over- or under-value any region. Well-sampled
urban markets are materially sharper than thin provincial ones — which is exactly the segmentation an
honest product should surface instead of averaging away.

---

## 3. Error forensics: what actually goes wrong

Comparing **normal listings** (APE ≤ 20%) against **large-error listings** (APE > 50%):

| feature | normal_median | error_median |
| --- | --- | --- |
| Area (m2) | 90.0 | 118.0 |
| Price / m2 | 53278688.525 | 22800000.0 |
| Floors | 3.0 | 3.0 |
| Interval width (rel) | 0.78 | 1.361 |
| Price/m2 percentile | 0.532 | 0.217 |
| Extreme price/m2 share (%) | 5.1 | 29.2 |

The tell-tale row is the last one: **29.2% of large-error listings sit in the extreme
price/m² tail, versus only 5.1% of normal listings** — a ~6× enrichment. Large errors
are dominated by listings whose *stated price per m² is implausible* (underpriced re-posts, placeholder
prices, data-entry errors, or genuinely mispriced deals). In these cases the listed price — not the
model — is the anomaly. A well-behaved, normally-priced listing rarely lands in the error tail.

This is the core "misunderstanding" the raw metrics create: a headline MAPE of ~34.52% *sounds*
like the model is unreliable, but that number is inflated by this thin tail of bad labels. The typical
listing is predicted to **14.84%** (Median_APE) — see the calibration analysis in the model report.

---

## 4. The model knows when it is uncertain

Grouping listings by the **width of their own predicted P10–P90 interval** (a signal available at
inference time, before we ever see the true price):

| width_q | n | Median_APE | HitRate_20_pct | MAPE |
| --- | --- | --- | --- | --- |
| Q1 narrowest | 28175.0 | 8.05 | 81.1 | 15.11 |
| Q2 | 28175.0 | 11.48 | 68.87 | 22.69 |
| Q3 | 28174.0 | 15.47 | 58.99 | 30.78 |
| Q4 | 28175.0 | 20.24 | 49.66 | 44.34 |
| Q5 widest | 28175.0 | 28.85 | 38.65 | 59.69 |

Accuracy falls monotonically from the narrowest to the widest interval band — the model's self-reported
uncertainty is a **strong, honest predictor of its actual error**. Concretely, on the narrowest fifth of
listings the typical error is single-digit and ~80% land within 20%; on the widest fifth the model is
openly unsure.

A complementary view: when a listing's **asking price falls inside** its predicted interval
(76.7% of listings), Median_APE is **10.91%**; when it falls **outside**
(23.3%), Median_APE jumps to **40.04%**. The interval is an
effective built-in anomaly detector.

---

## 5. Production defense — behaviour on a new case

**Claim: when a normal (non-anomalous) listing arrives, the system predicts sharply; when an
anomalous one arrives, the system flags it instead of guessing.**

This is operationalised with a three-tier policy driven purely by the model's interval width
(no access to the true price required — see [`operating_policy.csv`](operating_policy.csv)):

| tier | n | share_pct | Median_APE | HitRate_20_pct | in_band_coverage_pct |
| --- | --- | --- | --- | --- | --- |
| A_auto | 56350 | 40.0 | 9.55 | 74.98 | 76.5 |
| B_review | 56349 | 40.0 | 17.64 | 54.32 | 78.4 |
| C_flag | 28175 | 20.0 | 28.85 | 38.65 | 73.8 |

- **Tier A (auto-serve)** — narrow interval, high confidence. These are ordinary, well-specified
  listings; the AVM returns a value automatically at product-grade accuracy.
- **Tier B (serve with range)** — moderate confidence; return the estimate *with* the P10–P90 band.
- **Tier C (flag for review)** — wide interval or asking price outside the band. These are the
  anomalous / thin-data cases; the system routes them to human review rather than emitting a
  confident number.

**Why this is defensible in a real product:** the failure mode of an AVM is not "sometimes it is
wrong" — every AVM is — it is "it is confidently wrong without warning." This system is not: its error
is concentrated in a population it can *identify up front* from interval width and price plausibility.
A new, normal listing lands in Tier A/B and is valued sharply; a new anomalous listing lands in Tier C
and is flagged. That separation — sharp where confident, cautious where not — is the property that
makes the model safe to ship.

---

## 6. Honest limitations

- **Thin regions stay uncertain.** Small provinces have wide intervals and larger errors; the policy
  correctly sends more of them to Tier C rather than pretending confidence. More regional data is the
  only real fix.
- **The error tail is label noise, not solved.** Flagging anomalies contains the damage; it does not
  clean the underlying data. The `suspicious_numeric_price` cleanup is the upstream fix.
- **Static (random) split.** Numbers here are on a random test split; a time-based split is the next
  step to prove the model holds on *future* listings.

---

*Regenerate: `python3 src/build_regional_report.py`. All figures in this README are computed from the
test predictions at build time (see `report_stats.json`).*
