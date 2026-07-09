"""Fit the final production model and save everything needed to score a single new listing.

Trains the shrinkage-blend components (global + per-region + per-region-subtype LightGBM models),
the P10/P90 quantile models, the group sizes for shrinkage weighting, and the confidence-tier
thresholds, and pickles them to models/final_scorer.joblib.

Cleaning matches the final model: drop suspicious-price rows and rows below 20M VND/m2.
(Commercial listings are handled as an inference-time gate in score_listing.py, not here.)

Run once:  python3 build_final_scorer.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from common import EVAL_DIR, load_processed_train_test
from outlier_handling import prepare_outlier_artifacts
from train_region_subtype_models import (
    MIN_REGION_ROWS, MIN_REGION_SUBTYPE_ROWS, REGION_COL, SUBTYPE_COL,
    fit_model, load_robust_params, non_leakage_global_features, non_leakage_subtype_features,
)
from train_region_subtype_calibrated import apply_train_cleaning, fit_quantile_models, SHRINK_K_CANDIDATES

OUT = EVAL_DIR.parent / "models" / "final_scorer.joblib"
SHRINK_K = 250.0
MIN_PPM2 = 20e6


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    train_df, _ = load_processed_train_test()
    train_df = apply_train_cleaning(train_df, MIN_PPM2, drop_suspicious=True, drop_commercial=False)
    params = load_robust_params()

    art = prepare_outlier_artifacts(train_df, {})
    tr = art.train_df
    gnum, gcat = non_leakage_global_features(tr)
    submap = non_leakage_subtype_features(tr)

    print("fitting global ...")
    global_model, global_cols = fit_model(tr, gnum, gcat, params)

    print("fitting region models ...")
    region_models = {}
    for region, g in tr.groupby(REGION_COL):
        if len(g) < MIN_REGION_ROWS:
            continue
        region_models[region] = fit_model(g, gnum, gcat, params)  # (pipe, cols)

    print("fitting region x subtype models ...")
    rs_models = {}
    for (region, subtype), g in tr.groupby([REGION_COL, SUBTYPE_COL]):
        if len(g) < MIN_REGION_SUBTYPE_ROWS:
            continue
        fd = submap.get(subtype, {"numeric": gnum, "categorical": gcat})
        rs_models[(region, subtype)] = fit_model(g, fd["numeric"], fd["categorical"], params)

    print("fitting quantile models ...")
    qm = fit_quantile_models(tr, gnum, gcat, params)

    region_sizes = tr.groupby(REGION_COL).size().to_dict()
    rs_sizes = tr.groupby([REGION_COL, SUBTYPE_COL]).size().to_dict()

    # Confidence-tier thresholds from this model's own interval widths on a training sample.
    samp = tr.sample(min(30000, len(tr)), random_state=0)
    p10 = np.expm1(qm["p10"][0].predict(samp[qm["p10"][1]]))
    p90 = np.expm1(qm["p90"][0].predict(samp[qm["p90"][1]]))
    point = np.expm1(global_model.predict(samp[global_cols]))
    rw = (np.maximum(p10, p90) - np.minimum(p10, p90)) / np.clip(point, 1, None)
    tier_q40, tier_q80 = float(np.quantile(rw, 0.40)), float(np.quantile(rw, 0.80))

    artifact = {
        "global": (global_model, global_cols),
        "region": region_models,
        "region_subtype": rs_models,
        "p10": qm["p10"], "p90": qm["p90"],
        "region_sizes": region_sizes, "region_subtype_sizes": rs_sizes,
        "shrink_k": SHRINK_K,
        "features": {"numeric": gnum, "categorical": gcat},
        "tier_q40": tier_q40, "tier_q80": tier_q80,
        "min_ppm2": MIN_PPM2,
        "region_col": REGION_COL, "subtype_col": SUBTYPE_COL,
    }
    joblib.dump(artifact, OUT)
    print(f"\nSaved scorer -> {OUT}")
    print(f"  regions={len(region_models)} region_subtype={len(rs_models)} "
          f"tier_q40={tier_q40:.3f} tier_q80={tier_q80:.3f}")


if __name__ == "__main__":
    main()
