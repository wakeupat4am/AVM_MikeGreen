from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import (
    EVAL_DIR,
    MIN_ROWS_FOR_SUBTYPE_MODEL,
    TARGET_DEFS,
    evaluate_levels,
    global_features,
    load_processed_train_test,
    optimize_ridge,
    save_level_outputs,
    save_predictions,
    subtype_features,
    summarize_model_selection,
    build_ridge_pipeline,
)


MODEL_DIR = EVAL_DIR / "ridge"


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train_df, test_df = load_processed_train_test()
    global_num, global_cat = global_features(train_df)
    subtype_feature_map = subtype_features(train_df)
    feature_cols_global = global_num + global_cat
    test_pred = test_df.copy()

    tuning_outputs = {}
    all_level_frames = []

    for target_name, target_def in TARGET_DEFS.items():
        best_alpha, tuning_df = optimize_ridge(train_df, global_num, global_cat, target_def["target_col"], target_def["actual_col"])
        tuning_outputs[target_name] = {"best_alpha": best_alpha}
        tuning_df.to_csv(MODEL_DIR / f"tuning_{target_name}.csv", index=False)

        global_model = build_ridge_pipeline(global_num, global_cat, best_alpha)
        global_model.fit(train_df[feature_cols_global], train_df[target_def["target_col"]])
        test_pred[f"{target_def['prediction_col']}_global"] = np.expm1(global_model.predict(test_df[feature_cols_global]))

        test_pred[f"{target_def['prediction_col']}_subtype"] = np.nan
        for subtype, feature_dict in subtype_feature_map.items():
            tr_sub = train_df[train_df["property_subtype"] == subtype].copy()
            te_mask = test_df["property_subtype"] == subtype
            if len(tr_sub) < MIN_ROWS_FOR_SUBTYPE_MODEL or te_mask.sum() == 0:
                continue
            num, cat = feature_dict["numeric"], feature_dict["categorical"]
            if not (num + cat):
                continue
            subtype_model = build_ridge_pipeline(num, cat, best_alpha)
            subtype_model.fit(tr_sub[num + cat], tr_sub[target_def["target_col"]])
            test_pred.loc[te_mask, f"{target_def['prediction_col']}_subtype"] = np.expm1(subtype_model.predict(test_df.loc[te_mask, num + cat]))

        for scope_name, pred_col in [("ridge global", f"{target_def['prediction_col']}_global"), ("ridge subtype", f"{target_def['prediction_col']}_subtype")]:
            level_outputs = evaluate_levels(test_pred, target_def["actual_col"], pred_col, scope_name, target_name)
            out_dir = MODEL_DIR / target_name / scope_name.replace(" ", "_")
            save_level_outputs(level_outputs, out_dir)
            all_level_frames.extend(level_outputs.values())

    all_metrics = pd.concat(all_level_frames, ignore_index=True)
    all_metrics.to_csv(MODEL_DIR / "all_metrics.csv", index=False)

    selection_rows = []
    for target_name in TARGET_DEFS:
        by_subtype = all_metrics[(all_metrics["level"] == "by_subtype") & (all_metrics["target"] == target_name)]
        selection_rows.append(summarize_model_selection(all_metrics, by_subtype, "ridge", target_name))
    pd.concat(selection_rows, ignore_index=True).to_csv(MODEL_DIR / "model_selection.csv", index=False)

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
                    "pred_price_vnd_subtype",
                    "pred_price_per_m2_global",
                    "pred_price_per_m2_subtype",
                ]
                if c in test_pred.columns
            ]
        ],
        MODEL_DIR / "predictions.csv",
    )
    (MODEL_DIR / "config.json").write_text(json.dumps(tuning_outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved ridge results to", MODEL_DIR)


if __name__ == "__main__":
    main()
