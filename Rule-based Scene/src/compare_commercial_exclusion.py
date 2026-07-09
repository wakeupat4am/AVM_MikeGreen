"""Does excluding hotel/commercial listings from training improve the residential AVM?

Compares, on the IDENTICAL residential test slice (sensible price and not commercial):
  A. current final   - trained with commercial rows in (region_subtype_calibrated_clean_ppm2_20m)
  B. residential-only - trained with commercial rows excluded (..._clean_no_commercial)

The test set is scored on residential rows only, because commercial listings are out of scope
for a residential AVM (flagged / routed to review, not served).
"""

from __future__ import annotations

import pandas as pd

from common import EVAL_DIR, is_commercial_listing, metric_row, PROCESSED_DATA_DIR

A_DIR = EVAL_DIR / "improvements" / "region_subtype_calibrated_clean_ppm2_20m"
B_DIR = EVAL_DIR / "improvements" / "region_subtype_calibrated_clean_no_commercial"
OUT = EVAL_DIR / "improvements" / "commercial_exclusion_comparison.csv"
POINT = "pred_price_vnd_shrunk"
MIN_PPM2 = 20e6


def main():
    a = pd.read_csv(A_DIR / "predictions.csv")[["id", "addr_last_1", "price_vnd", "price_per_m2", POINT]]
    b = pd.read_csv(B_DIR / "predictions.csv")[["id", POINT]].rename(columns={POINT: f"{POINT}_B"})
    df = a.merge(b, on="id", validate="one_to_one")

    # Rebuild the commercial flag on the test set (needs text/structural columns).
    feats = pd.read_parquet(PROCESSED_DATA_DIR / "test_dataset.parquet")[
        ["id", "all_text_lower", "num_floors", "area_m2"]]
    df = df.merge(feats, on="id", how="left")
    df["commercial"] = is_commercial_listing(df)
    sus = set(pd.read_csv(EVAL_DIR.parent / "data" / "suspicious_numeric_price_rows.csv.gz")["id"])
    df["residential_sensible"] = (df["price_per_m2"] >= MIN_PPM2) & (~df["commercial"]) & (~df["id"].isin(sus))

    slices = {
        "residential sensible (served population)": df[df["residential_sensible"]],
        "commercial (out of scope, for reference)": df[df["commercial"]],
        "all test rows": df,
    }
    rows = []
    for name, s in slices.items():
        for model, col in (("A: commercial-in", POINT), ("B: commercial-excluded", f"{POINT}_B")):
            m = metric_row(s["price_vnd"], s[col])
            rows.append({"test_slice": name, "model": model, "n": m["count"],
                         "Median_APE": round(m["Median_APE"], 2), "HitRate_20_pct": round(m["HitRate_20_pct"], 2),
                         "HitRate_10_pct": round(m["HitRate_10_pct"], 2), "MAPE": round(m["MAPE"], 2),
                         "MAE_billion": round(float(m["MAE"]) / 1e9, 3), "Median_Bias_pct": round(m["Median_Bias_pct"], 2)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))

    served = df[df["residential_sensible"]]
    da = (served[f"{POINT}_B"] - served["price_vnd"]).abs() / served["price_vnd"]
    db = (served[POINT] - served["price_vnd"]).abs() / served["price_vnd"]
    print(f"\nServed population: n={len(served)}")
    print(f"  excluding commercial changes Median_APE by "
          f"{(da.median() - db.median()) * 100:+.2f} pts (B - A)")
    print(f"  rows improved by B: {(da < db).mean() * 100:.1f}%  worse: {(da > db).mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
