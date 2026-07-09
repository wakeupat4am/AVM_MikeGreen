"""Repeated-split stability: error bars on the final model's headline metrics.

Re-splits the full modeling dataset with 5 different random seeds, retrains the shrinkage-blend
components each time (same cleaning as the final model: drop suspicious + price/m2 >= 20M), and
evaluates on the served population (sensible, in-scope, non-commercial). Reports mean +/- std so
the headline numbers come with error bars instead of a single lucky split.

Note (honest framing): this measures SPLIT variance, not temporal drift. With ~140k test rows the
variance is expected to be small; the point is to show the numbers are not a one-split fluke.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import EVAL_DIR, PROCESSED_DATA_DIR, is_commercial_listing, metric_row
from outlier_handling import prepare_outlier_artifacts
from train_region_subtype_models import (
    MIN_REGION_ROWS, MIN_REGION_SUBTYPE_ROWS, REGION_COL, SUBTYPE_COL,
    fit_model, load_robust_params, non_leakage_global_features, non_leakage_subtype_features,
    predict_with_model,
)
from train_region_subtype_calibrated import apply_train_cleaning, shrink_blend

SEEDS = [0, 1, 2, 3, 4]
SHRINK_K = 250.0
MIN_PPM2 = 20e6
OUT_DIR = EVAL_DIR / "improvements" / "repeated_split_stability"


def fit_and_eval(df: pd.DataFrame, seed: int, params: dict, sus_ids: set) -> dict:
    tr_raw, te = train_test_split(df, test_size=0.2, random_state=seed, stratify=df[SUBTYPE_COL])
    tr_raw = apply_train_cleaning(tr_raw.copy(), MIN_PPM2, drop_suspicious=True)
    art = prepare_outlier_artifacts(tr_raw, {"test": te})
    tr, te = art.train_df, art.other_dfs["test"].copy()
    gnum, gcat = non_leakage_global_features(tr)
    submap = non_leakage_subtype_features(tr)

    gm, gc = fit_model(tr, gnum, gcat, params)
    te["pred_price_vnd_global"] = predict_with_model(gm, te, gc)

    te["pred_price_vnd_region"] = np.nan
    for region, g in tr.groupby(REGION_COL):
        if len(g) < MIN_REGION_ROWS:
            continue
        m, c = fit_model(g, gnum, gcat, params)
        mask = te[REGION_COL] == region
        if mask.any():
            te.loc[mask, "pred_price_vnd_region"] = predict_with_model(m, te.loc[mask], c)

    te["pred_price_vnd_region_subtype"] = np.nan
    for (region, subtype), g in tr.groupby([REGION_COL, SUBTYPE_COL]):
        if len(g) < MIN_REGION_SUBTYPE_ROWS:
            continue
        fd = submap.get(subtype, {"numeric": gnum, "categorical": gcat})
        m, c = fit_model(g, fd["numeric"], fd["categorical"], params)
        mask = (te[REGION_COL] == region) & (te[SUBTYPE_COL] == subtype)
        if mask.any():
            te.loc[mask, "pred_price_vnd_region_subtype"] = predict_with_model(m, te.loc[mask], c)

    region_sizes = tr.groupby(REGION_COL).size().to_dict()
    rs_sizes = tr.groupby([REGION_COL, SUBTYPE_COL]).size().to_dict()
    te["pred"] = shrink_blend(te, region_sizes, rs_sizes, SHRINK_K)

    te["commercial"] = is_commercial_listing(te)
    served = te[(te["price_per_m2"] >= MIN_PPM2) & (~te["id"].isin(sus_ids)) & (~te["commercial"])]
    m_all = metric_row(te["price_vnd"], te["pred"])
    m_srv = metric_row(served["price_vnd"], served["pred"])
    return {
        "seed": seed, "test_n": len(te), "served_n": len(served),
        "served_Median_APE": round(m_srv["Median_APE"], 3),
        "served_HitRate_20": round(m_srv["HitRate_20_pct"], 3),
        "served_HitRate_10": round(m_srv["HitRate_10_pct"], 3),
        "served_MAPE": round(m_srv["MAPE"], 3),
        "served_Median_Bias": round(m_srv["Median_Bias_pct"], 3),
        "all_Median_APE": round(m_all["Median_APE"], 3),
        "all_HitRate_20": round(m_all["HitRate_20_pct"], 3),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROCESSED_DATA_DIR / "landed_residential_house_modeling_dataset.parquet")
    params = load_robust_params()
    sus_ids = set(pd.read_csv(EVAL_DIR.parent / "data" / "suspicious_numeric_price_rows.csv.gz")["id"])

    rows = []
    for seed in SEEDS:
        print(f"=== seed {seed} ===", flush=True)
        rows.append(fit_and_eval(df, seed, params, sus_ids))
        print(rows[-1], flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "per_seed_metrics.csv", index=False)

    metrics = ["served_Median_APE", "served_HitRate_20", "served_HitRate_10", "served_MAPE",
               "served_Median_Bias", "all_Median_APE", "all_HitRate_20"]
    summary = pd.DataFrame({
        "metric": metrics,
        "mean": [round(res[m].mean(), 3) for m in metrics],
        "std": [round(res[m].std(ddof=1), 3) for m in metrics],
        "min": [round(res[m].min(), 3) for m in metrics],
        "max": [round(res[m].max(), 3) for m in metrics],
    })
    summary.to_csv(OUT_DIR / "summary_mean_std.csv", index=False)
    print("\n=== PER SEED ===\n", res.to_string(index=False))
    print("\n=== MEAN +/- STD across 5 seeds ===\n", summary.to_string(index=False))
    print("\nHeadline (served population):")
    r = summary.set_index("metric")
    print(f"  Median_APE = {r.loc['served_Median_APE','mean']:.2f}% +/- {r.loc['served_Median_APE','std']:.2f}")
    print(f"  HitRate_20 = {r.loc['served_HitRate_20','mean']:.2f}% +/- {r.loc['served_HitRate_20','std']:.2f}")


if __name__ == "__main__":
    main()
