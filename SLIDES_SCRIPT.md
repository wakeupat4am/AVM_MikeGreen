# AVM Pitch — Slide Script

*A slide-by-slide narrative for presenting the Vietnamese landed-house AVM. Each slide has:*
- **Message** — the single takeaway (put this as the slide headline)
- **Show** — what goes on the slide (numbers / visual)
- **Why** — the reason for the task, or the reason to believe (this is what you *say* out loud)

The spine of the whole talk: **We don't sell a number. We sell a number you can trust — and a
system that knows when not to give you one.**

---

## SECTION A — THE PROBLEM & THE DATA

### Slide 1 — Title
- **Message:** Automated Valuation Model for Vietnamese Landed Residential Houses
- **Show:** Title, your name, one line: *"Trustworthy house valuations — sharp when confident, honest when not."*
- **Why:** Set the theme early: the product isn't accuracy alone, it's *trustworthy* accuracy.

### Slide 2 — The problem
- **Message:** Valuing a house from a listing is hard because the *price label itself* is noisy.
- **Show:** A house photo + 3 near-identical listings at very different prices. "Same size, same district — 1.8B vs 4.3B."
- **Figure:** `Rule-based Scene/Visualizations/pitch_figures/fig_price_dist.png` — the price distribution.
  Median **4.3B** vs mean **7.6B**: the gap *is* the skew, and it's why we report the median.
- **Why:** Frame the core difficulty up front so every later design choice has a reason. This also
  pre-empts the "why isn't it 99% accurate?" question — because the ground truth is noisy.

### Slide 3 — The data
- **Message:** 813K cleaned listings → 704K in-scope houses → 563K train / 141K test.
- **Show:** Funnel of row counts (813K → 704K → 563K/141K); the subtype-mix bar chart.
- **Figure:** `Rule-based Scene/Visualizations/pitch_figures/fig_subtype_mix.png` — 704K houses across
  7 subtypes (frontage 171K · normal 152K · alley 80K · alley-frontage 55K · townhouse 52K · villa 44K · shophouse 10K).
- **Why:** Show scale and that the split is held-out (no peeking). Credibility starts with honest data volume.

---

## SECTION B — THE CLEANING PIPELINE (one slide, task → reason)

### Slide 4 — Data cleaning: every step exists for a reason
- **Message:** We remove noise that would otherwise be learned as signal.
- **Show:** A table — Task → Why:

| Cleaning task | Why we do it |
|---|---|
| **Address normalization** (`tp.hcm`/`tphcm`/`hồ chí minh` → one) | Location is the strongest price driver; spelling variants split one market into many and weaken it. |
| **Near-duplicate removal** (same listing re-posted) | Re-posts let one property vote many times and leak between train/test. |
| **Recency weighting** (2025 = 1.0, older less) | Prices move; stale listings shouldn't dominate. |
| **Text feature extraction** (frontage, business, corner…) | Recovers signal that lives only in the description. |
| **Suspicious-price audit** (~8K removed) | Placeholder / mistyped prices are label errors, not cheap houses. |
| **Price/m² floor (< 20M dropped)** | Below-market outliers distort the residential signal. |
| **Scope filter — hotels, offices, apartments** | Different asset class, different pricing logic (rooms/revenue) — out of scope for a *house* AVM. |

- **Why (say this):** "Each row here is a decision we can defend: we only remove what is provably
  noise or out of scope. Nothing is dropped to make the numbers look better — we later *prove* the
  cleaning is accuracy-neutral on real houses."

---

## SECTION C — THE MODEL (task → reason)

### Slide 5 — Why this model architecture
- **Message:** One national model is too generic; one model per tiny area overfits. We blend.
- **Show:** The hierarchy: Global → Region → Region×Subtype, with the shrinkage weight `w = n/(n+K)`.
- **Why:** Explain hierarchical shrinkage in one line: *"A local model is trusted only as much as
  its data supports — data-rich segments use their local model, thin ones lean on the broader one,
  smoothly."* This is the intellectual core; give it its own slide.

### Slide 6 — The model beats the obvious alternative
- **Message:** Soft shrinkage beats hard "pick one model" routing — and it's statistically real.
- **Show:**

| Router | Median_APE | Hit ≤20% |
|---|--:|--:|
| Hard fallback (pick one) | 15.39% | 58.3% |
| **Shrinkage blend (ours)** | **14.84%** | **59.5%** |

  *"+1,607 listings moved inside ±20%; McNemar p < 0.001."*
- **Figure:** `Rule-based Scene/Visualizations/pitch_figures/fig_model_compare.png` — Median_APE across
  all six models; shrinkage blend (dark) is the best *deployable* one.
- **Why:** Shows you didn't just build a model — you compared it and proved the gain isn't luck.

---

## SECTION D — THE RESULTS

### Slide 7 — Headline results
- **Message:** On a normal, in-scope house: ~12.7% typical error, ~65% within 20%.
- **Show:** Big numbers: **Median_APE 12.6% ± 0.05**, **HitRate_20 65.2% ± 0.2**, on the served population
  (mean ± std across 5 random splits). Add the calibration scatter.
- **Figure:** `Rule-based Scene/Visualizations/pitch_figures/fig_pred_vs_actual.png` — predicted vs
  actual, density hugging the y = x line.
- **Why:** Lead with the metric that matters for a skewed target (median, not mean). Say why:
  *"We report the typical listing's error, because a few bad labels inflate the average."*

---

## SECTION E — WHY BELIEVE IT (the trust case — 4 pillars)

### Slide 8 — Reason to believe #1: The model is unbiased
- **Message:** The scary "+14% bias" is a reporting artifact — the model is centered.
- **Show:**

| Bias measure | Value |
|---|--:|
| Raw mean bias | +13.9% |
| **Median bias** | **−0.3%** |
| Trimmed mean bias | +0.6% |

  Plus: "median bias ≈ 0 in *every region*, and −0.02% ± 0.02 across 5 splits."
- **Figure:** `Rule-based Scene/Visualizations/pitch_figures/fig_bias.png` — raw bias (orange) towers,
  median & trimmed bias (green) sit at zero.
- **Why:** Pre-empt the #1 attack. *"A mean-of-ratios on skewed data is structurally positive.
  Forcing it to zero would wreck accuracy (12.8% → 16.8%), so we report robust bias instead — and it
  proves the model doesn't systematically over- or under-value anything."*

### Slide 9 — Reason to believe #2: The big errors are bad *labels*, not bad *predictions*
- **Message:** When the model looks very wrong, the listing is usually the one that's wrong.
- **Show:** Worst-error table:

| Listed | Model says | Listing |
|--:|--:|---|
| 0.24B | 128B | "Khách sạn MT Lê Thánh Tôn" @ 1.5M/m² (impossible) |
| 0.12B | 25B | mặt-phố Ba Đình house @ 2M/m² (mistyped) |

  *"29% of large errors sit in the extreme-price tail vs 5% of normal listings."*
- **Figure:** `Rule-based Scene/Visualizations/pitch_figures/fig_error_vs_normal.png` — 29.2% vs 5.1%
  extreme-price share (a 6× gap).
- **Why:** The most persuasive slide. *"The model reconstructs the true value and refuses to believe
  a broken price — that's a feature, not a failure. Every one of our top-30 errors is a data error we
  already flag."*

### Slide 10 — Reason to believe #3: The model knows when it's uncertain
- **Message:** Its own confidence predicts its accuracy — so it can decide when to auto-serve.
- **Show:** The tier table:

| Tier | Share | Median_APE | Action |
|---|--:|--:|---|
| **A** | 40% | 9.6% | AUTO_VALUATION |
| B | 40% | 17.6% | value + range |
| C | 20% | 28.9% | REVIEW |

- **Figure:** `Rule-based Scene/Visualizations/pitch_figures/fig_confidence_tiers.png` — Median_APE
  climbs 9.6% → 17.6% → 28.9% from Tier A to C.
- **Why:** *"We don't pretend to be equally sure about every house. On the 40% it's most confident
  about, error is single-digit — and it hands the hard 20% to a human instead of guessing."*

### Slide 11 — Reason to believe #4: It's stable and honest about where it's weak
- **Message:** Retraining on very different data barely moves the result; and we name our weak spot.
- **Show:** Per-region bar chart — Đà Nẵng 11.3%, Hải Phòng 12.8% … **HCMC 23.7% (flagged red)**.
  Plus the 5-seed stability strip and: "trained with vs without 106K noisy rows → within 0.45 pts."
- **Figure:** `Rule-based Scene/Visualizations/pitch_figures/fig_region_ape.png` (per-region, HCMC red)
  and `fig_stability.png` (Median_APE across 5 splits, band = ±0.05).
- **Why:** *"A model you can trust is one that tells you where *not* to trust it. HCMC is a
  high-variance market — we surface it instead of hiding it in a national average. And the numbers
  hold across five random splits — not a lucky draw."*

---

## SECTION F — THE PRODUCT (the demo)

### Slide 12 — The serving contract
- **Message:** Every valuation is gated: sharp where confident, decline where out of scope.
- **Show:** Flow diagram: `listing → [gate: commercial? apartment? price<20M? outside band?] → REVIEW`
  or `→ predict + interval + tier → AUTO / RANGE`.
- **Why:** *"The dangerous failure of an AVM isn't being wrong — it's being *confidently* wrong. Ours
  can't: it declines assets it wasn't built for."*

### Slide 13 — Live demo (this sells it)
- **Message:** Watch it value a normal house — and refuse a mispriced one.
- **Show:** Three real outputs side by side:
  ```
  Normal house      Underpriced (real 9.5B)   Apartment
  Prediction 6.7B   Prediction 6.1B           Prediction 4.7B
  Interval 4.3–7.7  Interval 4.9–7.0          —
  Tier A            (asking 9.5B outside band) —
  AUTO_VALUATION    Decision: REVIEW          Decision: REVIEW (out of scope)
  actual 6.3B (5%)  → declined, would've been 36% wrong
  ```
- **Why:** The middle case is the whole pitch: *"It was about to be 36% wrong — and it stopped
  itself."* Run it live if you can (`score_listing.py`).

---

## SECTION G — HONESTY & ROADMAP

### Slide 14 — What we don't claim (build trust by naming limits)
- **Message:** We know the ceiling and why it exists.
- **Show:** Three bullets: (1) prices for identical-on-paper houses vary ~30% — a data limit, not a
  model flaw; (2) 86% of listings are undated → time-validation is the next rigor step; (3) we predict
  *asking* prices, not transactions.
- **Why:** *"Naming the ceiling is what makes the numbers below it credible."* Judges/buyers trust a
  presenter who volunteers limits.

### Slide 15 — Roadmap to break the ceiling
- **Message:** The next gains come from data, not tuning.
- **Show:** Prioritized: (1) calibrate intervals to exact 80% (conformal); (2) transaction-price
  labels; (3) street-level geocoding + condition features; (4) time-based validation.
- **Why:** Shows you know exactly where value is left — the difference between a finished experiment
  and a live roadmap.

### Slide 16 — Close
- **Message:** A valuation you can act on — and a system that says so.
- **Show:** One line: *"Sharp where confident. Honest where not. Silent where it shouldn't guess."*
- **Why:** End on the theme you opened with. That symmetry is what they'll remember.

---

## APPENDIX — Q&A ammunition (keep as backup slides)

- **"Isn't 12.7% error high?"** → Two identical-on-paper houses list 30% apart; we're already well
  below the data's own noise floor. Densest segments floor at ~10% — that's the ceiling of listing data.
- **"Your MAPE is 21% though."** → MAPE is a mean, inflated by the label-noise tail; Median_APE (the
  typical house) is 12.7%. For a skewed target, median is the honest metric.
- **"How do I know it's not overfitting?"** → Held-out test set; shrinkage prevents thin-segment
  overfit by design; retraining on very different data moved results < 0.5 pts.
- **"What about hotels/apartments in your data?"** → Detected and excluded from scope at serving time;
  we proved excluding them from training is accuracy-neutral on real houses.
- **"Why should I trust one number?"** → You shouldn't — that's why every valuation ships with an
  interval and a confidence tier, and low-confidence ones are routed to review.

---

## DETAILED RESULTS (backup / appendix tables)

### D1 — Data funnel
| Stage | Rows |
|---|--:|
| Cleaned listings | 813,215 |
| In-scope modeling rows | 704,370 |
| Train / Test (stratified) | 563,496 / 140,874 |
| Removed: suspicious price | 6,357 |
| Removed: commercial (hotel/office) | 18,657 |
| Removed: price/m² < 20M | 97,810 |
| **Final training set** | **440,672** |

### D2 — Model comparison (test set, Median_APE)
| Model | Median_APE | HitRate_20 | MAPE | Role |
|---|--:|--:|--:|---|
| Global | 21.96% | 46.5% | 44.0% | baseline |
| Subtype | 20.45% | 49.2% | 42.6% | per subtype |
| Region | 15.36% | 59.0% | 34.1% | per province |
| Hybrid (hard router) | 15.39% | 58.3% | 36.1% | previous prod |
| **Shrinkage blend** | **14.84%** | **59.5%** | 34.5% | **production** |
| Region×Subtype | 14.01% | 61.3% | 32.1% | best raw (dense only) |

### D3 — Final model on the served population, with error bars (5 random splits)
| Metric | Mean ± Std |
|---|---|
| Median_APE | **12.57% ± 0.05** |
| HitRate_20 | **65.16% ± 0.20** |
| HitRate_10 | 43.14% ± 0.09 |
| MAPE | 20.98% ± 0.06 |
| Median bias | −0.02% ± 0.02 |

### D4 — Bias (region×subtype model)
| Measure | Value |
|---|--:|
| Raw mean bias | +13.9% |
| **Median bias** | **−0.29%** |
| 10%-trimmed mean bias | +0.63% |

### D5 — Confidence tiers (operating policy)
| Tier | Action | Share | Median_APE | HitRate_20 |
|---|---|--:|--:|--:|
| A | AUTO_VALUATION | 40% | 9.6% | 75.0% |
| B | value + range | 40% | 17.6% | 54.3% |
| C | REVIEW | 20% | 28.9% | 38.7% |

### D6 — Per-region accuracy (top markets by volume)
| Region | Test n | Median_APE | HitRate_20 |
|---|--:|--:|--:|
| Đà Nẵng | 26,787 | 11.3% | 69% |
| Hải Phòng | 13,952 | 12.8% | 65% |
| Bình Dương | 13,663 | 12.8% | 64% |
| **Hồ Chí Minh** | 14,904 | **23.7%** | **44%** |
| Đồng Nai | 12,978 | 14.6% | 59% |
| Khánh Hòa | 8,034 | 11.2% | 66% |
| Long An | 6,496 | 17.7% | 54% |

### D7 — Error forensics
- 29.2% of large-error listings (APE > 50%) sit in the extreme price/m² tail vs **5.1%** of normal listings (6×).
- Worst-30 errors overall: 24 suspicious-price, 5 sub-20M, 1 commercial — **0 genuine model failures**.
- In-band coverage: when asking price ∈ [P10, P90] (77% of listings) Median_APE = 10.9%; outside → 40%.

---

## FIGURE INDEX
All in `Rule-based Scene/Visualizations/pitch_figures/` (regenerate: `python3 src/make_pitch_figures.py`).

| File | Slide | Shows |
|---|---|---|
| `fig_price_dist.png` | 2 | price distribution; median 4.3B vs mean 7.6B (the skew) |
| `fig_subtype_mix.png` | 3 | 704K houses across 7 subtypes |
| `fig_model_compare.png` | 6 | Median_APE across 6 models |
| `fig_pred_vs_actual.png` | 7 | predicted vs actual, density on y = x |
| `fig_bias.png` | 8 | raw bias vs median/trimmed (≈0) |
| `fig_error_vs_normal.png` | 9 | extreme-price share: 29.2% vs 5.1% |
| `fig_confidence_tiers.png` | 10 | Median_APE by tier A/B/C |
| `fig_region_ape.png` | 11 | per-region accuracy, HCMC flagged red |
| `fig_stability.png` | 11 | Median_APE across 5 splits (±0.05) |
