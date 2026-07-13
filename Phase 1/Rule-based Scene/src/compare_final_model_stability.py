"""Stability comparison for the final model (shrinkage blend).

Compares two trainings of the identical pipeline on the SAME 140,874 test rows:

  A. full        - trained on the full training set (region_subtype_calibrated)
  B. cleaned     - trained after dropping suspicious-price rows and rows with
                   price_per_m2 < 20M VND (region_subtype_calibrated_clean_ppm2_20m)

and scores both on three test slices:

  all listings           - the whole test set
  sensible listings      - price_per_m2 >= 20M and not flagged suspicious
                           ("a normal, make-sense case of asset")
  below-20M listings     - the excluded population, reported for honesty

Output: Evaluation/improvements/final_model_stability/ (comparison.csv, by_region_sensible.csv)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import EVAL_DIR, metric_row

FULL_DIR = EVAL_DIR / "improvements" / "region_subtype_calibrated"
CLEAN_DIR = EVAL_DIR / "improvements" / "region_subtype_calibrated_clean_ppm2_20m"
OUT_DIR = EVAL_DIR / "improvements" / "final_model_stability"
POINT = "pred_price_vnd_shrunk"
MIN_PPM2 = 20e6


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    a = pd.read_csv(FULL_DIR / "predictions.csv")[["id", "addr_last_1", "property_subtype", "price_vnd", "price_per_m2", POINT]]
    b = pd.read_csv(CLEAN_DIR / "predictions.csv")[["id", POINT]].rename(columns={POINT: f"{POINT}_clean"})
    df = a.merge(b, on="id", validate="one_to_one")
    sus_ids = set(pd.read_csv(EVAL_DIR.parent / "data" / "suspicious_numeric_price_rows.csv.gz")["id"])
    df["sensible"] = (df["price_per_m2"] >= MIN_PPM2) & (~df["id"].isin(sus_ids))

    slices = {
        "all listings": df,
        "sensible (ppm2>=20M, not suspicious)": df[df["sensible"]],
        "below-20M / suspicious (excluded pop.)": df[~df["sensible"]],
    }
    rows = []
    for name, s in slices.items():
        for model, col in (("full-trained", POINT), ("clean-trained", f"{POINT}_clean")):
            m = metric_row(s["price_vnd"], s[col])
            rows.append({"test_slice": name, "model": model, "n": m["count"],
                         "Median_APE": round(m["Median_APE"], 2), "HitRate_20_pct": round(m["HitRate_20_pct"], 2),
                         "HitRate_10_pct": round(m["HitRate_10_pct"], 2), "MAPE": round(m["MAPE"], 2),
                         "Median_Bias_pct": round(m["Median_Bias_pct"], 2)})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT_DIR / "comparison.csv", index=False)
    print(comparison.to_string(index=False))

    # Per-region stability on the sensible slice: do the two trainings agree everywhere?
    sens = df[df["sensible"]]
    reg_rows = []
    for region, g in sens.groupby("addr_last_1"):
        if len(g) < 100:
            continue
        mf = metric_row(g["price_vnd"], g[POINT])
        mc = metric_row(g["price_vnd"], g[f"{POINT}_clean"])
        reg_rows.append({"region": region, "n": len(g),
                         "Median_APE_full": round(mf["Median_APE"], 2),
                         "Median_APE_clean": round(mc["Median_APE"], 2),
                         "delta": round(mc["Median_APE"] - mf["Median_APE"], 2)})
    by_region = pd.DataFrame(reg_rows).sort_values("n", ascending=False)
    by_region.to_csv(OUT_DIR / "by_region_sensible.csv", index=False)
    print("\nPer-region deltas on sensible slice (clean - full), extremes:")
    print(by_region.nsmallest(5, "delta").to_string(index=False))
    print(by_region.nlargest(5, "delta").to_string(index=False))
    agree = (by_region["delta"].abs() <= 1.0).mean() * 100
    print(f"\nregions where |delta Median_APE| <= 1.0 pt: {agree:.0f}%")


if __name__ == "__main__":
    main()
