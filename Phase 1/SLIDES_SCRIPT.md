# From Noisy Listings to Trustworthy Valuations

*Pitch script for the Vietnamese landed-house AVM. Six sections. Each slide has:*
- **Message** — the headline (say it, put it on the slide)
- **Show** — content + which figure(s) to place (all in `Rule-based Scene/Visualizations/pitch_figures/`)
- **Say** — the talking points / the *why*

**Through-line:** we don't sell a number — we sell a number you can trust, from a system that
knows when *not* to give you one.

---

## TITLE SLIDE
- **From Noisy Listings to Trustworthy Valuations**
- Subtitle: *An Automated Valuation Model for Vietnamese landed residential houses.*
- One line: *"Sharp where confident. Honest where not. Silent where it shouldn't guess."*

---

## SECTION 1 — Dataset reality *(the data is the reason the task is hard)*

**Main message: The raw data is large, but noisy and uneven.**

- **Show (4 figures):**
  - `fig_price_dist.png` — prices are right-skewed; **median 4.3B vs mean 7.6B**.
  - `fig_missingness.png` — most structural fields are half-empty; **built_year 100% missing**.
  - `fig_subtype_mix.png` — 7 subtypes, **frontage 171K … shophouse 10K** (imbalanced).
  - `fig_region_mix.png` — a few metros dominate; many provinces are data-thin.
- **Noisy-label examples** (put this small table on the slide — real rows):

| Listed price | Price/m² | What it actually is |
|---|---|---|
| 0.24B | 1.5M | a hotel ("khách sạn MT Lê Thánh Tôn") — impossible price |
| 0.12B | 2M | a mặt-phố Ba Đình house — mistyped (dropped digits) |
| 5.9B | 59M | a "Green Pearl" **apartment** — wrong asset class |

- **Say:** "We have 704K in-scope houses — real scale. But three things make this hard, and every
  later design choice answers one of them: **(1)** the price *label itself* is often wrong or
  implausible; **(2)** the features are badly incomplete — `built_year` is entirely missing, half the
  structural fields are gaps; **(3)** the data is uneven — a few subtypes and a few cities dominate.
  This is why we can't just fit a model and quote one accuracy number."

---

## SECTION 2 — Cleaning pipeline *(risk removal, not "data prep")*

**Main message: We remove noise the model should not learn.**

- **Show — frame every step as a risk removed:**

| Cleaning step | Risk removed |
|---|---|
| Duplicate removal | train/test **leakage** (same listing on both sides) |
| Suspicious-price audit | **mistyped / placeholder labels** |
| Price/m² floor (< 20M dropped) | **impossible / below-market labels** |
| Commercial + apartment filter | **wrong asset class** (hotels, offices, condos) |
| Address normalization | **fragmented location signal** (`tp.hcm`=`tphcm`=`hồ chí minh`) |
| Recency weighting | **stale market signal** (old prices down-weighted) |

- Data funnel: **813K cleaned → 704K in-scope → 440K final training rows.**
- **Say:** "We don't 'prepare' data — we *remove risk*. Each row here maps a cleaning step to a
  specific failure it prevents. And critically, we **proved** this cleaning is accuracy-neutral on
  real houses (retraining with vs without 100K+ noisy rows moved results < 0.5 pts) — so nothing was
  dropped to flatter the numbers. We remove only what is provably noise or out of scope."

---

## SECTION 3 — Feature design *(real signals, not the answer)*

**Main message: The model sees real listing signals, not the answer.**

- **Show — features in five groups:**
  - **Location:** province / district / ward (`addr_last_1/2/3`), street, project name.
  - **Physical structure:** area, floors, frontage, road width, width, length, bedrooms, bathrooms.
  - **Text-extracted:** flags mined from the description — frontage, business, corner, car-access,
    legal, villa, project, urban.
  - **Interaction:** floors×area, frontage×road-width, floors×frontage, area×frontage …
  - **EXCLUDED — leakage:** `price_per_m2`, `log_price_per_m2`, and anything price-derived.
- **Say (make the exclusion the punchline):** "The most important features on this slide are the
  ones we **threw away**. Anything derived from price is excluded — if the model could see
  price-per-m², it would be reading the answer, and our test scores would be a lie. It sees the
  *listing*, never the *price*. That's what makes every number after this honest."

---

## SECTION 4 — Final model architecture *(a gated valuation system)*

**Main message: The AVM is not one model; it is a gated valuation system.**

- **Show — one horizontal flow (draw as boxes + arrows):**

```
 log(price)          predict the MEDIAN        blend by data size         uncertainty         decide
     │                      │                         │                        │                 │
 log-transform  →  L1 LightGBM base  →  Global → Region → Region×Subtype  →  P10–P90 quantile  →  Confidence tier
 (skew, %-errors)  (robust to noise;    (hierarchical shrinkage:            interval            A → SERVE
                    exact back-transform) w = n/(n+K), K=250)                                    B → SERVE + RANGE
                                                                                                 C → REVIEW
                                                                        + Scope guardrails (hotel / apartment /
                                                                          price<20M / outside band) → REVIEW
```

- **One line per box (say these):**
  - **log price** — model proportional errors, not absolute.
  - **L1 LightGBM** — predict the *median* → robust to bad labels, and the back-transform is exact (this is *why* bias ≈ 0).
  - **Global / Region / Region×Subtype** — three nested models: general → local → most-specific.
  - **Shrinkage blend** — trust the local model only as much as its data supports (`n/(n+K)`); thin segments lean on their parent.
  - **Quantile interval** — a plausible price *range* per house, not just a point.
  - **Confidence tier + guardrails** — auto-serve the confident, defer the uncertain, reject the out-of-scope.
- **Say:** "It's not 'a LightGBM model.' It's a pipeline that ends in a *decision* — value it, value it
  with a range, or send it to review. The model is allowed to abstain."

---

## SECTION 5 — Results

**Main message: Accuracy improves, and confidence predicts error.**

Present in this order:

**5.1 — Baseline vs final model.** Figure `fig_model_compare.png`.
- Global baseline **22.0%** → shrinkage blend **14.8%** Median_APE; beats the previous hard router
  (15.4%) with p < 0.001.

**5.2 — Headline served-population metrics** (with error bars from 5 splits). Figure `fig_pred_vs_actual.png`.
- **Median_APE 12.6% ± 0.05**, **HitRate_20 65.2% ± 0.2** — the typical in-scope house.
- The scatter shows density hugging the perfect-prediction line.

**5.3 — Confidence-tier performance.** Figure `fig_confidence_tiers.png`.
- Tier A (40%) **9.6%** · Tier B (40%) **17.6%** · Tier C (20%) **28.9%** — the model's own confidence
  tracks its real error.

**5.4 — Regional weakness + stability.** Figures `fig_region_ape.png` and `fig_stability.png`.
- Most markets 11–15%; **Hồ Chí Minh 23.7% (flagged)**. Numbers hold across 5 random splits (±0.05).

- **Say:** "Two results, not one. First, accuracy — we roughly halved the baseline error and beat the
  prior model with statistical significance. Second, and more important for a product: **the model's
  confidence predicts its accuracy** — that's what lets us automate the easy 40% and escalate the hard 20%."

---

## SECTION 6 — Why trust it

**Main message: The system is designed to know when not to guess.**

- **Five trust proofs (each with its evidence):**
  1. **The model is unbiased.** `fig_bias.png` — raw bias is an artifact; **median bias ≈ 0**
     (−0.02% ± 0.02), and ≈ 0 in every region. It doesn't systematically over- or under-value.
  2. **Bad errors are usually bad labels.** `fig_error_vs_normal.png` — 29% of large errors sit in the
     extreme-price tail vs 5% of normal listings; the worst-30 errors are **0 genuine model failures**.
  3. **Confidence tiers work.** (ref `fig_confidence_tiers.png`) — it earns the right to auto-serve.
  4. **Out-of-scope cases are rejected.** Live: a hotel and an apartment both return **REVIEW**, not a
     fake valuation. *(Demo `score_listing.py` here if you can.)*
  5. **Weak regions are flagged honestly.** (ref `fig_region_ape.png`) — HCMC is surfaced, not hidden.
- **Close:** "A model you can trust isn't one that's never wrong — every AVM is wrong sometimes. It's
  one that is **never confidently wrong without warning**. Ours knows when not to guess. *From noisy
  listings, to trustworthy valuations.*"

---

## APPENDIX — Detailed results (backup slides)

### D1 — Data funnel
| Stage | Rows |
|---|--:|
| Cleaned listings | 813,215 |
| In-scope modeling rows | 704,370 |
| Train / Test | 563,496 / 140,874 |
| Final training set (after cleaning) | 440,672 |

### D2 — Model comparison (Median_APE)
| Model | Median_APE | HitRate_20 | Role |
|---|--:|--:|---|
| Global | 22.0% | 46.5% | baseline |
| Region | 15.4% | 59.0% | per province |
| Hybrid (hard router) | 15.4% | 58.3% | previous prod |
| **Shrinkage blend** | **14.8%** | **59.5%** | **production** |
| Region×Subtype | 14.0% | 61.3% | best raw (dense only) |

### D3 — Final model, served population, 5-split error bars
| Metric | Mean ± Std |
|---|---|
| Median_APE | 12.57% ± 0.05 |
| HitRate_20 | 65.16% ± 0.20 |
| Median bias | −0.02% ± 0.02 |

### D4 — Confidence tiers
| Tier | Action | Share | Median_APE |
|---|---|--:|--:|
| A | AUTO_VALUATION | 40% | 9.6% |
| B | value + range | 40% | 17.6% |
| C | REVIEW | 20% | 28.9% |

### D5 — Q&A ammunition
- **"Isn't 12.7% high?"** → identical-on-paper houses list ~30% apart; we're below the data's own noise floor (dense cells floor at ~10%).
- **"Your MAPE is 21%."** → MAPE is a mean inflated by the label-noise tail; Median_APE (typical house) is 12.6%.
- **"Overfitting?"** → held-out test; shrinkage prevents thin-segment overfit; stable to ±0.05 across 5 splits.
- **"Hotels/apartments?"** → detected and rejected at serving; excluding them from training is accuracy-neutral.

---

## FIGURE INDEX
Regenerate with `python3 src/make_pitch_figures.py`.

| File | Section | Shows |
|---|---|---|
| `fig_price_dist.png` | 1 | right-skew; median 4.3B vs mean 7.6B |
| `fig_missingness.png` | 1 | % missing per feature; built_year 100% |
| `fig_subtype_mix.png` | 1 | 704K houses across 7 subtypes |
| `fig_region_mix.png` | 1 | listings by region (uneven) |
| `fig_model_compare.png` | 5.1 | Median_APE across 6 models |
| `fig_pred_vs_actual.png` | 5.2 | predicted vs actual on y = x |
| `fig_confidence_tiers.png` | 5.3 / 6 | Median_APE by tier A/B/C |
| `fig_region_ape.png` | 5.4 / 6 | per-region accuracy, HCMC flagged |
| `fig_stability.png` | 5.4 | Median_APE across 5 splits (±0.05) |
| `fig_bias.png` | 6 | raw vs median/trimmed bias (≈0) |
| `fig_error_vs_normal.png` | 6 | extreme-price share 29.2% vs 5.1% |
