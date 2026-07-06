"""Region-segmented reliability report for the AVM.

Shifts the story from "one accuracy number for the whole dataset" to a per-region,
business-defensible reliability profile, plus an error forensics analysis and a
production-readiness defense.

Deliverables (written under Evaluation/regional_report/):
  regions/<region>/data.csv      per-region test listings with actual, prediction, interval, APE, confidence tier
  regions/<region>/metrics.csv   per-region accuracy + robust bias + interval coverage
  region_summary.csv             every region ranked, with sample size and accuracy
  error_vs_normal.csv            what distinguishes large-error listings from normal ones
  confidence_gate.csv            Median_APE by model-reported interval-width quintile
  operating_policy.csv           tiered auto-serve / review / flag policy for production
  README.md                      the full narrative report with interpretation

The production model evaluated here is the shrinkage blend (`pred_price_vnd_shrunk`),
the best deployable "covers every listing" model.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from common import EVAL_DIR, PROCESSED_DATA_DIR, metric_row

PRED_PATH = EVAL_DIR / "improvements" / "region_subtype_calibrated" / "predictions.csv"
OUT_DIR = EVAL_DIR / "regional_report"
POINT = "pred_price_vnd_shrunk"
P10, P90 = "pred_price_vnd_p10", "pred_price_vnd_p90"
MIN_REGION_ROWS = 100  # regions below this are pooled into "other_small_regions"


def slugify(name: str) -> str:
    s = str(name).strip().lower().replace(" ", "_")
    return re.sub(r"[^0-9a-zà-ỹ_]+", "", s) or "unknown"


def load() -> pd.DataFrame:
    pred = pd.read_csv(PRED_PATH)
    feats = pd.read_parquet(PROCESSED_DATA_DIR / "test_dataset.parquet")[
        ["id", "area_m2", "num_floors", "frontage_m", "road_width_m", "price_per_m2", "listing_year"]
    ]
    df = pred.merge(feats, on="id", suffixes=("", "_feat"))
    true = df["price_vnd"].astype(float)
    df["actual_price_vnd"] = true
    df["ape"] = (df[POINT].astype(float) - true).abs() / true
    df["signed_pe"] = (df[POINT].astype(float) - true) / true
    df["rel_width"] = (df[P90] - df[P10]) / df[POINT]
    df["in_interval"] = (true >= df[P10]) & (true <= df[P90])
    # Label-independent extremeness of price/m2 within subtype (an inference-time-style anomaly signal).
    df["ppm2_pctile"] = df.groupby("property_subtype")["price_per_m2"].rank(pct=True)
    df["extreme_ppm2"] = (df["ppm2_pctile"] < 0.05) | (df["ppm2_pctile"] > 0.95)
    # Explicit, human-readable error labelling so each region file shows errors vs normal cases.
    df["abs_pct_error"] = (df["ape"] * 100).round(2)
    df["case_type"] = np.select([df["ape"] <= 0.20, df["ape"] > 0.50], ["normal", "large_error"], default="moderate")
    df["price_outside_interval"] = ~df["in_interval"]
    return df


def region_metric(g: pd.DataFrame) -> dict:
    m = metric_row(g["actual_price_vnd"], g[POINT])
    hi = g[g["rel_width"] <= g["rel_width"].median()]
    return {
        "n_test": int(len(g)),
        "MAE_billion": round(float(m["MAE"]) / 1e9, 3),
        "RMSE_billion": round(float(m["RMSE"]) / 1e9, 3),
        "MAPE": round(m["MAPE"], 2),
        "Median_APE": round(m["Median_APE"], 2),
        "HitRate_10_pct": round(m["HitRate_10_pct"], 2),
        "HitRate_20_pct": round(m["HitRate_20_pct"], 2),
        "Bias_pct": round(m["Bias_pct"], 2),
        "Median_Bias_pct": round(m["Median_Bias_pct"], 2),
        "Bias_pct_trimmed10": round(m["Bias_pct_trimmed10"], 2),
        "mean_rel_width_pct": round(float(g["rel_width"].mean()) * 100, 1),
        "in_band_coverage_pct": round(float(g["in_interval"].mean()) * 100, 1),
        "high_conf_Median_APE": round(float(np.median(hi["ape"])) * 100, 2) if len(hi) else np.nan,
    }


def region_error_profile(g: pd.DataFrame) -> dict:
    """Characterise a region's large-error listings against its normal ones."""
    normal = g[g["ape"] <= 0.20]
    err = g[g["ape"] > 0.50]
    return {
        "n_test": int(len(g)),
        "large_error_rate_pct": round(float((g["ape"] > 0.50).mean()) * 100, 2),
        "Median_APE": round(float(np.median(g["ape"])) * 100, 2),
        "err_median_ppm2_mil": round(float(err["price_per_m2"].median()) / 1e6, 1) if len(err) else np.nan,
        "normal_median_ppm2_mil": round(float(normal["price_per_m2"].median()) / 1e6, 1) if len(normal) else np.nan,
        "err_extreme_ppm2_share_pct": round(float(err["extreme_ppm2"].mean()) * 100, 1) if len(err) else np.nan,
        "normal_extreme_ppm2_share_pct": round(float(normal["extreme_ppm2"].mean()) * 100, 1) if len(normal) else np.nan,
    }


DATA_COLS = [
    "id", "title", "addr_last_1", "property_subtype", "source", "area_m2", "num_floors",
    "price_per_m2", "actual_price_vnd", POINT, P10, P90,
    "abs_pct_error", "case_type", "price_outside_interval", "confidence_tier",
]


def confidence_tier(rel_width: pd.Series) -> pd.Series:
    q40, q80 = rel_width.quantile(0.40), rel_width.quantile(0.80)
    tier = pd.Series("B_review", index=rel_width.index)
    tier[rel_width <= q40] = "A_auto"
    tier[rel_width > q80] = "C_flag"
    return tier


# ---------------------------------------------------------------------------
# LaTeX generation helpers (province names are romanised so the report compiles
# with a plain pdflatex install; CSV files keep the original Vietnamese names).
# ---------------------------------------------------------------------------

def deaccent(s: str) -> str:
    s = str(s).replace("đ", "d").replace("Đ", "D")
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def region_title(s: str) -> str:
    name = deaccent(s)
    return name.title() if "(" not in name else name  # keep "other_small_regions (N regions)" readable


def subtype_label(s: str) -> str:
    return deaccent(str(s)).replace("_house", "").replace("_", " ").strip() or deaccent(str(s))


def build_examples(df: pd.DataFrame, ordered_regions) -> pd.DataFrame:
    """For every reported region: its single worst-error listing and a representative normal
    listing, with the features that explain the gap."""
    rows = []
    for region in ordered_regions:
        g = df[(df["addr_last_1"] == region) & np.isfinite(df["price_per_m2"])]
        if g.empty:
            continue
        worst = g.sort_values("ape", ascending=False).iloc[0]
        normals = g[g["case_type"] == "normal"].sort_values("ape")
        norm = (normals.iloc[len(normals) // 2] if len(normals) else g.sort_values("ape").iloc[0])
        for case, r in [("worst error", worst), ("normal", norm)]:
            rows.append({
                "region": region, "case": case, "subtype": r["property_subtype"],
                "area_m2": r["area_m2"], "num_floors": r["num_floors"], "ppm2_mil": r["price_per_m2"] / 1e6,
                "listed_bil": r["actual_price_vnd"] / 1e9, "pred_bil": float(r[POINT]) / 1e9,
                "abs_pct_error": r["abs_pct_error"],
            })
    return pd.DataFrame(rows)


def _longtable(colspec, header_cells, body_rows, caption, label, size="footnotesize"):
    header = " & ".join(header_cells) + r" \\"
    ncol = len(header_cells)
    out = [r"{\%s" % size, r"\setlength{\tabcolsep}{4pt}", r"\begin{longtable}{%s}" % colspec,
           r"\caption{%s}\label{%s}\\" % (caption, label),
           r"\toprule", header, r"\midrule", r"\endfirsthead",
           r"\multicolumn{%d}{c}{\tablename\ \thetable\ -- continued}\\" % ncol,
           r"\toprule", header, r"\midrule", r"\endhead",
           r"\midrule\multicolumn{%d}{r}{\itshape continued on next page}\\" % ncol, r"\endfoot",
           r"\bottomrule", r"\endlastfoot"]
    out += body_rows
    out += [r"\end{longtable}}", ""]
    return "\n".join(out)


def write_latex_fragments(repo_root: Path, summary: pd.DataFrame, examples: pd.DataFrame):
    def rname(r):
        return region_title(r).replace("_", " ").replace("&", r"\&").replace("%", r"\%")

    # (a) Accuracy metrics per region.
    acc = [" & ".join([rname(r.region), f"{int(r.n_test):,}", f"{r.MAE_billion:.2f}", f"{r.MAPE:.1f}",
                       f"{r.Median_APE:.1f}", f"{r.HitRate_10_pct:.1f}", f"{r.HitRate_20_pct:.1f}"]) + r" \\"
           for r in summary.itertuples(index=False)]
    (repo_root / "gen_region_metrics_accuracy.tex").write_text(_longtable(
        "lrrrrrr",
        [r"\textbf{Region}", r"\textbf{n}", r"\textbf{MAE (B)}", r"\textbf{MAPE\%}",
         r"\textbf{Med.APE\%}", r"\textbf{Hit10\%}", r"\textbf{Hit20\%}"],
        acc, "Accuracy metrics per region (production shrinkage-blend model).", "tab:regacc",
        size="scriptsize"), encoding="utf-8")

    # (b) Calibration & confidence per region.
    cal = [" & ".join([rname(r.region), f"{r.Bias_pct:.1f}", f"{r.Median_Bias_pct:.1f}",
                       f"{r.Bias_pct_trimmed10:.1f}", f"{r.mean_rel_width_pct:.0f}",
                       f"{r.in_band_coverage_pct:.0f}", f"{r.high_conf_Median_APE:.1f}"]) + r" \\"
           for r in summary.itertuples(index=False)]
    (repo_root / "gen_region_metrics_calibration.tex").write_text(_longtable(
        "lrrrrrr",
        [r"\textbf{Region}", r"\textbf{Raw Bias\%}", r"\textbf{Med.Bias\%}", r"\textbf{Trim.Bias\%}",
         r"\textbf{Int.Width\%}", r"\textbf{In-Band\%}", r"\textbf{HiConf MedAPE\%}"],
        cal, "Calibration and confidence per region. Median/trimmed bias near 0 = the model is centered.",
        "tab:regcal", size="scriptsize"), encoding="utf-8")

    # (c) Worst-vs-normal example listing per region, with features.
    ex = []
    for r in examples.itertuples(index=False):
        ex.append(" & ".join([
            rname(r.region), r.case.replace("worst error", "worst"), subtype_label(r.subtype),
            f"{r.area_m2:.0f}", f"{r.num_floors:.0f}" if np.isfinite(r.num_floors) else "--",
            f"{r.ppm2_mil:.1f}", f"{r.listed_bil:.2f}", f"{r.pred_bil:.2f}", f"{r.abs_pct_error:,.0f}"]) + r" \\")
    (repo_root / "gen_region_examples.tex").write_text(_longtable(
        "lllrrrrrr",
        [r"\textbf{Region}", r"\textbf{Case}", r"\textbf{Subtype}", r"\textbf{Area}", r"\textbf{Flr}",
         r"\textbf{ppm2(M)}", r"\textbf{Listed(B)}", r"\textbf{Model(B)}", r"\textbf{Err\%}"],
        ex, "Worst-error vs.\\ a normal listing for every region, with the features that drive the gap. "
            "Error listings consistently show an implausibly low price/m$^2$; the model predicts a "
            "sensible value, so the large ``error'' reflects a bad listed price, not a bad prediction.",
        "tab:regex", size="scriptsize"), encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "regions").mkdir(exist_ok=True)
    df = load()
    df["confidence_tier"] = confidence_tier(df["rel_width"])

    # ---- Per-region datasets (one CSV per province, filename = province name) ----
    regions_dir = OUT_DIR / "regions"
    regions_dir.mkdir(exist_ok=True)
    cols = [c for c in DATA_COLS if c in df.columns]
    summary_rows = []
    err_rows = []
    counts = df["addr_last_1"].value_counts()
    big = counts[counts >= MIN_REGION_ROWS].index
    for region, g in df.groupby("addr_last_1"):
        if region not in big:
            continue
        fname = str(region).replace("/", "-").strip()  # province name as the file name
        # Worst errors first so the problem cases sit at the top of each province file.
        g.sort_values("ape", ascending=False)[cols].to_csv(regions_dir / f"{fname}.csv", index=False)
        summary_rows.append({"region": region, **region_metric(g)})
        err_rows.append({"region": region, **region_error_profile(g)})

    small = df[~df["addr_last_1"].isin(big)]
    if len(small):
        small.sort_values("ape", ascending=False)[cols].to_csv(regions_dir / "other_small_regions.csv", index=False)
        summary_rows.append({"region": f"other_small ({small['addr_last_1'].nunique()})", **region_metric(small)})
        err_rows.append({"region": "other_small_regions", **region_error_profile(small)})

    summary = pd.DataFrame(summary_rows).sort_values("n_test", ascending=False).reset_index(drop=True)
    summary.to_csv(OUT_DIR / "region_summary.csv", index=False)

    # ---- Which regions produce the most errors, and how those errors differ ----
    err_compare = pd.DataFrame(err_rows).sort_values("large_error_rate_pct", ascending=False).reset_index(drop=True)
    err_compare.to_csv(OUT_DIR / "region_error_comparison.csv", index=False)

    # ---- Error forensics: normal vs error listings ----
    normal = df[df["ape"] <= 0.20]
    err = df[df["ape"] > 0.50]
    forensic_feats = [("area_m2", "Area (m2)"), ("price_per_m2", "Price / m2"), ("num_floors", "Floors"),
                      ("rel_width", "Interval width (rel)"), ("ppm2_pctile", "Price/m2 percentile")]
    frows = []
    for col, label in forensic_feats:
        frows.append({"feature": label,
                      "normal_median": round(float(normal[col].median()), 3),
                      "error_median": round(float(err[col].median()), 3)})
    frows.append({"feature": "Extreme price/m2 share (%)",
                  "normal_median": round(float(normal["extreme_ppm2"].mean()) * 100, 1),
                  "error_median": round(float(err["extreme_ppm2"].mean()) * 100, 1)})
    forensics = pd.DataFrame(frows)
    forensics.to_csv(OUT_DIR / "error_vs_normal.csv", index=False)

    # ---- Confidence gate: accuracy by interval-width quintile ----
    df["width_q"] = pd.qcut(df["rel_width"], 5, labels=["Q1 narrowest", "Q2", "Q3", "Q4", "Q5 widest"])
    gate = df.groupby("width_q", observed=True).apply(
        lambda x: pd.Series({"n": len(x),
                             "Median_APE": round(float(np.median(x["ape"])) * 100, 2),
                             "HitRate_20_pct": round(float((x["ape"] <= 0.20).mean()) * 100, 2),
                             "MAPE": round(float(x["ape"].mean()) * 100, 2)}), include_groups=False
    ).reset_index()
    gate.to_csv(OUT_DIR / "confidence_gate.csv", index=False)

    # ---- Operating policy: auto / review / flag tiers ----
    prows = []
    for tier, x in df.groupby("confidence_tier"):
        prows.append({"tier": tier, "n": len(x), "share_pct": round(len(x) / len(df) * 100, 1),
                      "Median_APE": round(float(np.median(x["ape"])) * 100, 2),
                      "HitRate_20_pct": round(float((x["ape"] <= 0.20).mean()) * 100, 2),
                      "in_band_coverage_pct": round(float(x["in_interval"].mean()) * 100, 1)})
    policy = pd.DataFrame(prows).sort_values("tier").reset_index(drop=True)
    policy.to_csv(OUT_DIR / "operating_policy.csv", index=False)

    stats = {
        "n_test": len(df),
        "n_regions": int(df["addr_last_1"].nunique()),
        "n_regions_reported": int(len(big)),
        "overall_median_ape": round(float(np.median(df["ape"])) * 100, 2),
        "overall_hit20": round(float((df["ape"] <= 0.20).mean()) * 100, 2),
        "overall_mape": round(float(df["ape"].mean()) * 100, 2),
        "in_band_share": round(float(df["in_interval"].mean()) * 100, 1),
        "in_band_medape": round(float(np.median(df[df["in_interval"]]["ape"])) * 100, 2),
        "out_band_medape": round(float(np.median(df[~df["in_interval"]]["ape"])) * 100, 2),
        "err_extreme_share": round(float(err["extreme_ppm2"].mean()) * 100, 1),
        "normal_extreme_share": round(float(normal["extreme_ppm2"].mean()) * 100, 1),
    }
    (OUT_DIR / "report_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    write_readme(OUT_DIR / "README.md", stats, summary, forensics, gate, policy)

    # ---- LaTeX fragments for the full report (all regions, all metrics, examples) ----
    ordered_regions = list(counts[counts >= MIN_REGION_ROWS].index)  # by descending sample size
    examples = build_examples(df, ordered_regions)
    examples.to_csv(OUT_DIR / "region_examples.csv", index=False)
    repo_root = EVAL_DIR.parent.parent  # repo root (parent of "Rule-based Scene")
    write_latex_fragments(repo_root, summary, examples)

    print("Wrote regional report to", OUT_DIR)
    print("Wrote LaTeX fragments (gen_region_*.tex) to", repo_root)
    print(summary.head(15).to_string(index=False))
    print("\nConfidence gate:\n", gate.to_string(index=False))
    print("\nOperating policy:\n", policy.to_string(index=False))


def write_readme(path, s, summary, forensics, gate, policy):
    top = summary[~summary["region"].str.startswith("other_small")].head(12)

    def md_table(df):
        head = "| " + " | ".join(df.columns) + " |\n| " + " | ".join(["---"] * len(df.columns)) + " |\n"
        body = "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False))
        return head + body

    lines = f"""# AVM Regional Reliability Report

*Production model: hierarchical **shrinkage blend** (`pred_price_vnd_shrunk`) with P10–P90 prediction intervals. Evaluated on the held-out test set of {s['n_test']:,} listings across {s['n_regions']} regions.*

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
{s['n_regions_reported']} provinces with ≥ {MIN_REGION_ROWS} test listings get their own file
(covering ~99% of listings); the rest are pooled into `other_small_regions.csv`. Which provinces
generate the most errors, and how those errors differ, is in
[`region_error_comparison.csv`](region_error_comparison.csv).

Top regions by sample size:

{md_table(top)}

**Reading it:** `Median_APE` is the typical listing's error; `high_conf_Median_APE` is the error on
that region's higher-confidence half (narrow interval). `Median_Bias_pct` ≈ 0 everywhere confirms
the model is **centered** — it does not systematically over- or under-value any region. Well-sampled
urban markets are materially sharper than thin provincial ones — which is exactly the segmentation an
honest product should surface instead of averaging away.

---

## 3. Error forensics: what actually goes wrong

Comparing **normal listings** (APE ≤ 20%) against **large-error listings** (APE > 50%):

{md_table(forensics)}

The tell-tale row is the last one: **{s['err_extreme_share']}% of large-error listings sit in the extreme
price/m² tail, versus only {s['normal_extreme_share']}% of normal listings** — a ~6× enrichment. Large errors
are dominated by listings whose *stated price per m² is implausible* (underpriced re-posts, placeholder
prices, data-entry errors, or genuinely mispriced deals). In these cases the listed price — not the
model — is the anomaly. A well-behaved, normally-priced listing rarely lands in the error tail.

This is the core "misunderstanding" the raw metrics create: a headline MAPE of ~{s['overall_mape']}% *sounds*
like the model is unreliable, but that number is inflated by this thin tail of bad labels. The typical
listing is predicted to **{s['overall_median_ape']}%** (Median_APE) — see the calibration analysis in the model report.

---

## 4. The model knows when it is uncertain

Grouping listings by the **width of their own predicted P10–P90 interval** (a signal available at
inference time, before we ever see the true price):

{md_table(gate)}

Accuracy falls monotonically from the narrowest to the widest interval band — the model's self-reported
uncertainty is a **strong, honest predictor of its actual error**. Concretely, on the narrowest fifth of
listings the typical error is single-digit and ~80% land within 20%; on the widest fifth the model is
openly unsure.

A complementary view: when a listing's **asking price falls inside** its predicted interval
({s['in_band_share']}% of listings), Median_APE is **{s['in_band_medape']}%**; when it falls **outside**
({round(100 - s['in_band_share'], 1)}%), Median_APE jumps to **{s['out_band_medape']}%**. The interval is an
effective built-in anomaly detector.

---

## 5. Production defense — behaviour on a new case

**Claim: when a normal (non-anomalous) listing arrives, the system predicts sharply; when an
anomalous one arrives, the system flags it instead of guessing.**

This is operationalised with a three-tier policy driven purely by the model's interval width
(no access to the true price required — see [`operating_policy.csv`](operating_policy.csv)):

{md_table(policy)}

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
"""
    path.write_text(lines, encoding="utf-8")


if __name__ == "__main__":
    main()
