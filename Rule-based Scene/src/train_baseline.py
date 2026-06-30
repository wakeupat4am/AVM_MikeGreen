from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import EVAL_DIR, TARGET_DEFS, evaluate_levels, load_processed_train_test, save_level_outputs, save_predictions


MODEL_DIR = EVAL_DIR / "baseline"


def fit_group_median(train_df: pd.DataFrame, group_cols):
    median_map = train_df.groupby(group_cols)["price_per_m2"].median()
    global_median = train_df["price_per_m2"].median()
    return median_map, global_median


def predict_group_median(df: pd.DataFrame, median_map, global_median, group_cols):
    keys = list(zip(*[df[c].values for c in group_cols]))
    preds = []
    for key in keys:
        lookup_key = key[0] if len(group_cols) == 1 else key
        preds.append(median_map.get(lookup_key, global_median))
    preds = np.asarray(preds, dtype=float)
    pred_price = preds * df["area_m2"].values
    return pred_price, preds


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train_df, test_df = load_processed_train_test()
    test_pred = test_df.copy()

    global_group_cols = [c for c in ["addr_last_2", "property_subtype"] if c in train_df.columns]
    global_map, global_default = fit_group_median(train_df, global_group_cols)
    global_price, global_ppm2 = predict_group_median(test_df, global_map, global_default, global_group_cols)
    test_pred["pred_price_vnd_global"] = global_price
    test_pred["pred_price_per_m2_global"] = global_ppm2

    test_pred["pred_price_vnd_subtype"] = np.nan
    test_pred["pred_price_per_m2_subtype"] = np.nan
    for subtype, tr_sub in train_df.groupby("property_subtype"):
        te_mask = test_df["property_subtype"] == subtype
        if te_mask.sum() == 0:
            continue
        subtype_cols = [c for c in ["addr_last_2"] if c in train_df.columns]
        median_map, default = fit_group_median(tr_sub, subtype_cols)
        pred_price, pred_ppm2 = predict_group_median(test_df.loc[te_mask], median_map, default, subtype_cols)
        test_pred.loc[te_mask, "pred_price_vnd_subtype"] = pred_price
        test_pred.loc[te_mask, "pred_price_per_m2_subtype"] = pred_ppm2

    all_level_frames = []
    for target_name, target_def in TARGET_DEFS.items():
        for scope_name, pred_col in [
            ("baseline global", f"{target_def['prediction_col']}_global"),
            ("baseline subtype", f"{target_def['prediction_col']}_subtype"),
        ]:
            level_outputs = evaluate_levels(test_pred, target_def["actual_col"], pred_col, scope_name, target_name)
            out_dir = MODEL_DIR / target_name / scope_name.replace(" ", "_")
            save_level_outputs(level_outputs, out_dir)
            all_level_frames.extend(level_outputs.values())

    save_predictions(
        test_pred[
            [
                c
                for c in [
                    "id",
                    "title",
                    "property_subtype",
                    "addr_last_1",
                    "addr_last_2",
                    "source",
                    "price_vnd",
                    "price_per_m2",
                    "pred_price_vnd_global",
                    "pred_price_per_m2_global",
                    "pred_price_vnd_subtype",
                    "pred_price_per_m2_subtype",
                ]
                if c in test_pred.columns
            ]
        ],
        MODEL_DIR / "predictions.csv",
    )
    all_metrics = pd.concat(all_level_frames, ignore_index=True)
    all_metrics.to_csv(MODEL_DIR / "all_metrics.csv", index=False)
    (MODEL_DIR / "config.json").write_text(json.dumps({"global_group_cols": global_group_cols}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved baseline results to", MODEL_DIR)


if __name__ == "__main__":
    main()
