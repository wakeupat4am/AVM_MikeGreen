"""Region/subtype AVM with hierarchical shrinkage + prediction intervals.

This extends `train_region_subtype_models` with three additions requested during the
"bias fix" review:

1. Robust bias reporting (handled in common.metric_row): Median_Bias_pct and
   Bias_pct_trimmed10. The raw Bias_pct is a mean-of-ratios inflated by a right tail of
   underpriced/mislabeled listings; the L1 models are median-calibrated, so those robust
   views sit near 0. We keep the naive expm1 back-transform (it is correct for the median)
   and deliberately do NOT apply Duan smearing, which would push predictions up and make
   the bias worse without helping Median_APE.

2. Hierarchical shrinkage (`pred_price_vnd_shrunk`): instead of hard-routing each group to a
   single model, we blend global -> region -> region_subtype in log space, each level pulled
   in by its own training size via weight n/(n+K). K is tuned on the validation split. This
   is a soft, empirical-Bayes version of the existing hybrid router and is gentler on sparse
   region x subtype cells.

3. Prediction intervals (`pred_price_vnd_p10/p90`): two global LightGBM quantile models give
   an ~80% interval per listing; we report empirical coverage and median relative width.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import (
    EVAL_DIR,
    RANDOM_STATE,
    build_lgbm_pipeline,
    evaluate_custom_levels,
    is_commercial_listing,
    list_metric_columns,
    load_processed_train_test,
    metric_row,
    save_level_outputs,
    save_predictions,
    standard_group_definitions,
)
from outlier_handling import prepare_outlier_artifacts
from train_region_subtype_models import (
    MIN_REGION_ROWS,
    MIN_REGION_SUBTYPE_ROWS,
    REGION_COL,
    SUBTYPE_COL,
    choose_policy,
    fit_model,
    load_robust_params,
    non_leakage_global_features,
    non_leakage_subtype_features,
    predict_with_model,
    train_family_models,
)

MODEL_DIR = EVAL_DIR / "improvements" / "region_subtype_calibrated"
SHRINK_K_CANDIDATES = [250, 500, 1000, 2000, 4000, 8000]
QUANTILE_LOW = 0.1
QUANTILE_HIGH = 0.9


def shrink_blend(
    df: pd.DataFrame,
    region_sizes: Dict[str, int],
    region_subtype_sizes: Dict[Tuple[str, str], int],
    k: float,
) -> np.ndarray:
    """Sequential empirical-Bayes shrinkage in log1p space: global -> region -> region_subtype.

    Each local level is blended toward the running estimate with weight n/(n+k), so sparse
    groups lean on their parent and data-rich groups trust their local model.
    """
    base = np.log1p(df["pred_price_vnd_global"].to_numpy(dtype=float))
    log_est = base.copy()

    # Region level.
    reg = df["pred_price_vnd_region"].to_numpy(dtype=float)
    has_reg = np.isfinite(reg)
    n_reg = df[REGION_COL].map(region_sizes).fillna(0.0).to_numpy(dtype=float)
    a_reg = n_reg / (n_reg + k)
    log_reg = np.where(has_reg, np.log1p(np.clip(reg, 0, None)), log_est)
    log_est = np.where(has_reg, a_reg * log_reg + (1.0 - a_reg) * log_est, log_est)

    # Region x subtype level.
    rs = df["pred_price_vnd_region_subtype"].to_numpy(dtype=float)
    has_rs = np.isfinite(rs)
    keys = list(zip(df[REGION_COL], df[SUBTYPE_COL]))
    n_rs = np.array([region_subtype_sizes.get(key, 0) for key in keys], dtype=float)
    a_rs = n_rs / (n_rs + k)
    log_rs = np.where(has_rs, np.log1p(np.clip(rs, 0, None)), log_est)
    log_est = np.where(has_rs, a_rs * log_rs + (1.0 - a_rs) * log_est, log_est)

    return np.expm1(log_est)


def tune_shrinkage_k(
    val_df: pd.DataFrame,
    region_sizes: Dict[str, int],
    region_subtype_sizes: Dict[Tuple[str, str], int],
    candidates: List[float],
) -> Tuple[float, pd.DataFrame]:
    rows = []
    for k in candidates:
        preds = shrink_blend(val_df, region_sizes, region_subtype_sizes, k)
        m = metric_row(val_df["price_vnd"], preds)
        rows.append({"k": k, "Median_APE": m["Median_APE"], "HitRate_20_pct": m["HitRate_20_pct"], "Bias_pct": m["Bias_pct"]})
    sweep = pd.DataFrame(rows)
    best_k = float(sweep.sort_values(["Median_APE", "HitRate_20_pct"], ascending=[True, False]).iloc[0]["k"])
    return best_k, sweep


def fit_quantile_models(tr: pd.DataFrame, numeric: List[str], categorical: List[str], params: Dict):
    quantile_models = {}
    for tag, alpha in (("p10", QUANTILE_LOW), ("p90", QUANTILE_HIGH)):
        q_params = dict(params)
        q_params["objective"] = "quantile"
        q_params["alpha"] = alpha
        model = build_lgbm_pipeline(numeric, categorical, q_params)
        feature_cols = numeric + categorical
        model.fit(tr[feature_cols], tr["log_price"], model__sample_weight=tr["sample_weight"].values)
        quantile_models[tag] = (model, feature_cols)
    return quantile_models


def retrain_predict_calibrated(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: Dict,
    routing_policy: pd.DataFrame,
    shrink_k: float,
):
    """Mirror of train_region_subtype_models.retrain_and_predict, extended with the shrinkage
    blend, quantile intervals, and the group sizes needed to compute shrinkage on test rows."""
    artifacts = prepare_outlier_artifacts(train_df, {"test": test_df})
    tr = artifacts.train_df
    te = artifacts.other_dfs["test"].copy()
    global_num, global_cat = non_leakage_global_features(tr)
    subtype_feature_map = non_leakage_subtype_features(tr)

    global_model, global_cols = fit_model(tr, global_num, global_cat, params)
    te["pred_price_vnd_global"] = predict_with_model(global_model, te, global_cols)

    te["pred_price_vnd_region"] = np.nan
    for region, region_train in tr.groupby(REGION_COL):
        if len(region_train) < MIN_REGION_ROWS:
            continue
        region_model, region_cols = fit_model(region_train, global_num, global_cat, params)
        region_mask = te[REGION_COL] == region
        if region_mask.any():
            te.loc[region_mask, "pred_price_vnd_region"] = predict_with_model(region_model, te.loc[region_mask], region_cols)

    te["pred_price_vnd_subtype"] = np.nan
    for subtype, subtype_train in tr.groupby(SUBTYPE_COL):
        if len(subtype_train) < MIN_REGION_ROWS:
            continue
        feature_dict = subtype_feature_map.get(subtype, {"numeric": global_num, "categorical": global_cat})
        subtype_model, subtype_cols = fit_model(subtype_train, feature_dict["numeric"], feature_dict["categorical"], params)
        subtype_mask = te[SUBTYPE_COL] == subtype
        if subtype_mask.any():
            te.loc[subtype_mask, "pred_price_vnd_subtype"] = predict_with_model(subtype_model, te.loc[subtype_mask], subtype_cols)

    te["pred_price_vnd_region_subtype"] = np.nan
    for row in routing_policy.itertuples(index=False):
        if row.train_rows < MIN_REGION_SUBTYPE_ROWS:
            continue
        mask = (tr[REGION_COL] == row.addr_last_1) & (tr[SUBTYPE_COL] == row.property_subtype)
        group_train = tr.loc[mask]
        if group_train.empty:
            continue
        feature_dict = subtype_feature_map.get(row.property_subtype, {"numeric": global_num, "categorical": global_cat})
        group_model, group_cols = fit_model(group_train, feature_dict["numeric"], feature_dict["categorical"], params)
        test_mask = (te[REGION_COL] == row.addr_last_1) & (te[SUBTYPE_COL] == row.property_subtype)
        if test_mask.any():
            te.loc[test_mask, "pred_price_vnd_region_subtype"] = predict_with_model(group_model, te.loc[test_mask], group_cols)

    # Hard hybrid router (kept as the apples-to-apples baseline).
    policy_lookup = routing_policy.set_index([REGION_COL, SUBTYPE_COL])["chosen_model"].to_dict()
    te["selected_model"] = [policy_lookup.get((r, s), "global") for r, s in zip(te[REGION_COL], te[SUBTYPE_COL])]
    te["pred_price_vnd_hybrid"] = te["pred_price_vnd_global"]
    for level, col in (("region", "pred_price_vnd_region"), ("subtype", "pred_price_vnd_subtype")):
        sel = te["selected_model"] == level
        te.loc[sel, "pred_price_vnd_hybrid"] = te.loc[sel, col].fillna(te.loc[sel, "pred_price_vnd_global"])
    sel = te["selected_model"] == "region_subtype"
    te.loc[sel, "pred_price_vnd_hybrid"] = (
        te.loc[sel, "pred_price_vnd_region_subtype"]
        .fillna(te.loc[sel, "pred_price_vnd_region"])
        .fillna(te.loc[sel, "pred_price_vnd_global"])
    )

    # Hierarchical shrinkage blend.
    region_sizes = tr.groupby(REGION_COL).size().to_dict()
    region_subtype_sizes = tr.groupby([REGION_COL, SUBTYPE_COL]).size().to_dict()
    te["pred_price_vnd_shrunk"] = shrink_blend(te, region_sizes, region_subtype_sizes, shrink_k)

    # Prediction intervals from global quantile models.
    quantile_models = fit_quantile_models(tr, global_num, global_cat, params)
    p10 = predict_with_model(quantile_models["p10"][0], te, quantile_models["p10"][1])
    p90 = predict_with_model(quantile_models["p90"][0], te, quantile_models["p90"][1])
    lo = np.minimum(p10, p90)  # guard against quantile crossing
    hi = np.maximum(p10, p90)
    te["pred_price_vnd_p10"] = lo
    te["pred_price_vnd_p90"] = hi

    return te


def interval_coverage(te: pd.DataFrame, point_col: str = "pred_price_vnd_shrunk") -> Dict[str, float]:
    true = te["price_vnd"].to_numpy(dtype=float)
    lo = te["pred_price_vnd_p10"].to_numpy(dtype=float)
    hi = te["pred_price_vnd_p90"].to_numpy(dtype=float)
    point = te[point_col].to_numpy(dtype=float)
    mask = np.isfinite(true) & np.isfinite(lo) & np.isfinite(hi) & (true > 0) & np.isfinite(point) & (point > 0)
    true, lo, hi, point = true[mask], lo[mask], hi[mask], point[mask]
    covered = (true >= lo) & (true <= hi)
    rel_width = (hi - lo) / point
    return {
        "target_coverage_pct": (QUANTILE_HIGH - QUANTILE_LOW) * 100,
        "empirical_coverage_pct": float(np.mean(covered) * 100),
        "median_rel_width_pct": float(np.median(rel_width) * 100),
        "count": int(mask.sum()),
    }


def evaluate_family(df_eval: pd.DataFrame, pred_col: str, model_name: str, out_dir: Path) -> pd.DataFrame:
    levels = evaluate_custom_levels(
        df_eval=df_eval,
        actual_col="price_vnd",
        pred_col=pred_col,
        model_name=model_name,
        target_name="price",
        group_definitions=standard_group_definitions(df_eval),
    )
    save_level_outputs(levels, out_dir)
    return pd.concat(levels.values(), ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-suffix", type=str, default="")
    parser.add_argument("--data-path", type=str, default=None)
    # Training-data cleaning flags. They filter the TRAIN split only; the test set is left
    # untouched so runs with different cleaning stay comparable on identical test rows.
    parser.add_argument("--min-ppm2", type=float, default=None,
                        help="Drop training rows with price_per_m2 below this (VND).")
    parser.add_argument("--drop-suspicious", action="store_true",
                        help="Drop training rows flagged in data/suspicious_numeric_price_rows.csv.gz.")
    parser.add_argument("--drop-commercial", action="store_true",
                        help="Drop hotel/office/commercial listings (common.is_commercial_listing).")
    return parser.parse_args()


def apply_train_cleaning(train_df: pd.DataFrame, min_ppm2: float | None, drop_suspicious: bool,
                         drop_commercial: bool = False) -> pd.DataFrame:
    n0 = len(train_df)
    if drop_suspicious:
        sus_path = EVAL_DIR.parent / "data" / "suspicious_numeric_price_rows.csv.gz"
        sus_ids = set(pd.read_csv(sus_path)["id"])
        train_df = train_df[~train_df["id"].isin(sus_ids)]
        print(f"drop-suspicious: removed {n0 - len(train_df)} training rows")
    if drop_commercial:
        n1 = len(train_df)
        train_df = train_df[~is_commercial_listing(train_df)]
        print(f"drop-commercial: removed {n1 - len(train_df)} training rows")
    if min_ppm2 is not None:
        n1 = len(train_df)
        train_df = train_df[train_df["price_per_m2"] >= min_ppm2]
        print(f"min-ppm2 {min_ppm2:.0f}: removed {n1 - len(train_df)} training rows")
    print(f"training rows: {n0} -> {len(train_df)}")
    return train_df


def main():
    args = parse_args()
    output_dir = MODEL_DIR if not args.output_suffix else MODEL_DIR.parent / f"{MODEL_DIR.name}_{args.output_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_processed_train_test(args.data_path)
    train_df = apply_train_cleaning(train_df, args.min_ppm2, args.drop_suspicious, args.drop_commercial)
    params = load_robust_params()

    # Validation split for routing policy + shrinkage-K tuning.
    train_split, val_split = train_test_split(train_df, test_size=0.2, random_state=RANDOM_STATE, stratify=train_df[SUBTYPE_COL])
    artifacts = prepare_outlier_artifacts(train_split, {"validation": val_split})
    tr_enriched = artifacts.train_df
    val_enriched = artifacts.other_dfs["validation"]

    family = train_family_models(tr_enriched, val_enriched.copy(), params)
    routing_policy = choose_policy(family["validation_predictions"], family["train_group_sizes"])
    routing_policy.to_csv(output_dir / "routing_policy.csv", index=False)

    val_pred = family["validation_predictions"]
    region_sizes_val = tr_enriched.groupby(REGION_COL).size().to_dict()
    region_subtype_sizes_val = tr_enriched.groupby([REGION_COL, SUBTYPE_COL]).size().to_dict()
    best_k, sweep = tune_shrinkage_k(val_pred, region_sizes_val, region_subtype_sizes_val, SHRINK_K_CANDIDATES)
    sweep.to_csv(output_dir / "shrinkage_k_sweep.csv", index=False)
    print(f"Selected shrinkage K = {best_k}")

    te = retrain_predict_calibrated(train_df, test_df, params, routing_policy, best_k)

    family_metrics = []
    eval_specs = [
        ("pred_price_vnd_global", "global model", "global_model"),
        ("pred_price_vnd_region", "region model", "region_model"),
        ("pred_price_vnd_subtype", "subtype model", "subtype_model"),
        ("pred_price_vnd_region_subtype", "region subtype model", "region_subtype_model"),
        ("pred_price_vnd_hybrid", "hybrid fallback", "hybrid_fallback"),
        ("pred_price_vnd_shrunk", "shrinkage blend", "shrinkage_blend"),
    ]
    for pred_col, model_name, folder in eval_specs:
        if te[pred_col].notna().any():
            family_metrics.append(evaluate_family(te, pred_col, model_name, output_dir / "price" / folder))
    family_metrics_df = pd.concat(family_metrics, ignore_index=True)
    family_metrics_df.to_csv(output_dir / "all_metrics.csv", index=False)

    # Final comparison at the overall test level, including robust bias columns.
    overall = family_metrics_df[family_metrics_df["level"] == "overall_test_set"].copy()
    comparison = overall[["model"] + list_metric_columns()].reset_index(drop=True)
    comparison.to_csv(output_dir / "final_comparison.csv", index=False)

    coverage = interval_coverage(te, "pred_price_vnd_shrunk")
    (output_dir / "interval_coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")

    save_predictions(
        te[[c for c in [
            "id", "title", REGION_COL, SUBTYPE_COL, "source", "price_vnd", "price_per_m2",
            "selected_model", "pred_price_vnd_global", "pred_price_vnd_region", "pred_price_vnd_subtype",
            "pred_price_vnd_region_subtype", "pred_price_vnd_hybrid", "pred_price_vnd_shrunk",
            "pred_price_vnd_p10", "pred_price_vnd_p90",
        ] if c in te.columns]],
        output_dir / "predictions.csv",
    )

    (output_dir / "config.json").write_text(
        json.dumps({
            "shrink_k": best_k,
            "shrink_k_candidates": SHRINK_K_CANDIDATES,
            "quantile_low": QUANTILE_LOW,
            "quantile_high": QUANTILE_HIGH,
            "data_path": args.data_path,
            "min_ppm2": args.min_ppm2,
            "drop_suspicious": args.drop_suspicious,
            "drop_commercial": args.drop_commercial,
            "train_rows_after_cleaning": int(len(train_df)),
            "interval_coverage": coverage,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Saved calibrated region/subtype results to", output_dir)
    print(comparison.to_string(index=False))
    print("Interval coverage:", coverage)


if __name__ == "__main__":
    main()
