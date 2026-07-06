from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from common import evaluate_custom_levels, load_processed_train_test, save_level_outputs, standard_group_definitions
from outlier_handling import prepare_outlier_artifacts
from train_region_subtype_models import fit_model, non_leakage_global_features


MODEL_DIR = Path("/Users/dienmayhaituyet/Documents/AVM/Rule-based Scene/Evaluation/improvements/address_combo_comparison")
MIN_ROWS = 500
FAST_N_ESTIMATORS = 200
GROUP_SETS = [
    ["addr_last_1", "property_subtype"],
    ["addr_last_2", "property_subtype"],
    ["addr_last_3", "property_subtype"],
    ["addr_last_1", "addr_last_2", "property_subtype"],
    ["addr_last_1", "addr_last_3", "property_subtype"],
    ["addr_last_2", "addr_last_3", "property_subtype"],
    ["addr_last_1", "addr_last_2", "addr_last_3", "property_subtype"],
]


def combo_name(group_cols: List[str]) -> str:
    addr_cols = [col.replace("addr_last_", "addr") for col in group_cols if col.startswith("addr_last_")]
    return "_plus_".join(addr_cols + ["subtype"])


def load_saved_metrics(name: str) -> pd.DataFrame:
    model_dir = MODEL_DIR / name
    frames = []
    for csv_path in sorted(model_dir.glob("*.csv")):
        if csv_path.name == "predictions.csv":
            continue
        frames.append(pd.read_csv(csv_path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def run_grouped_model(tr: pd.DataFrame, te: pd.DataFrame, params: Dict, group_cols: List[str], numeric_features: List[str], categorical_features: List[str]):
    name = combo_name(group_cols)
    pred_col = f"pred_{name}"
    te[pred_col] = te["pred_price_vnd_global_fallback"]

    group_sizes = tr.groupby(group_cols).size().rename("train_rows").reset_index()
    test_groups = te.groupby(group_cols).size().rename("test_rows").reset_index()
    eligible = (
        group_sizes.merge(test_groups, on=group_cols, how="inner")
        .query("train_rows >= @MIN_ROWS and test_rows > 0")
        .copy()
    )
    used_rows = 0
    used_groups = 0

    print(f"Running {name}: {len(eligible)} eligible groups")

    for row in eligible.itertuples(index=False):
        mask_tr = pd.Series(True, index=tr.index)
        mask_te = pd.Series(True, index=te.index)
        for col in group_cols:
            value = getattr(row, col)
            mask_tr &= tr[col] == value
            mask_te &= te[col] == value
        if mask_te.sum() == 0:
            continue
        group_train = tr.loc[mask_tr]
        model, feature_cols = fit_model(group_train, numeric_features, categorical_features, params)
        te.loc[mask_te, pred_col] = np.expm1(model.predict(te.loc[mask_te, feature_cols]))
        used_rows += int(mask_te.sum())
        used_groups += 1

    level_outputs = evaluate_custom_levels(
        df_eval=te,
        actual_col="price_vnd",
        pred_col=pred_col,
        model_name=name,
        target_name="price",
        group_definitions=standard_group_definitions(te),
    )
    save_level_outputs(level_outputs, MODEL_DIR / name)

    coverage = {
        "model": name,
        "group_definition": " + ".join(group_cols),
        "eligible_groups": int(len(eligible)),
        "used_groups_with_test_rows": int(used_groups),
        "test_rows_using_granular_model": int(used_rows),
        "test_rows_total": int(len(te)),
        "granular_coverage_pct": used_rows / len(te) * 100,
    }
    return pd.concat(level_outputs.values(), ignore_index=True), coverage


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    config = json.loads((Path("/Users/dienmayhaituyet/Documents/AVM/Rule-based Scene/Evaluation/improvements/robust_lightgbm/config.json")).read_text())
    params = config["best_params"].copy()
    params["n_estimators"] = min(int(params.get("n_estimators", FAST_N_ESTIMATORS)), FAST_N_ESTIMATORS)

    train_df, test_df = load_processed_train_test()
    artifacts = prepare_outlier_artifacts(train_df, {"test": test_df})
    tr = artifacts.train_df
    te = artifacts.other_dfs["test"].copy()

    numeric_features, categorical_features = non_leakage_global_features(tr)
    global_model, global_cols = fit_model(tr, numeric_features, categorical_features, params)
    te["pred_price_vnd_global_fallback"] = np.expm1(global_model.predict(te[global_cols]))

    all_metrics = []
    coverage_rows = []
    for group_cols in GROUP_SETS:
        name = combo_name(group_cols)
        saved_overall = MODEL_DIR / name / "overall_test_set.csv"
        if saved_overall.exists():
            print(f"Skipping {name}: using saved outputs")
            metrics_df = load_saved_metrics(name)
            coverage = None
            partial_coverage_path = MODEL_DIR / "coverage_summary_partial.csv"
            if partial_coverage_path.exists():
                partial_coverage = pd.read_csv(partial_coverage_path)
                matched = partial_coverage[partial_coverage["model"] == name]
                if not matched.empty:
                    coverage = matched.iloc[0].to_dict()
            if coverage is None:
                group_sizes = tr.groupby(group_cols).size().rename("train_rows").reset_index()
                test_groups = te.groupby(group_cols).size().rename("test_rows").reset_index()
                eligible = (
                    group_sizes.merge(test_groups, on=group_cols, how="inner")
                    .query("train_rows >= @MIN_ROWS and test_rows > 0")
                    .copy()
                )
                coverage = {
                    "model": name,
                    "group_definition": " + ".join(group_cols),
                    "eligible_groups": int(len(eligible)),
                    "used_groups_with_test_rows": int(len(eligible)),
                    "test_rows_using_granular_model": int(eligible["test_rows"].sum()),
                    "test_rows_total": int(len(te)),
                    "granular_coverage_pct": float(eligible["test_rows"].sum() / len(te) * 100),
                }
        else:
            metrics_df, coverage = run_grouped_model(tr, te, params, group_cols, numeric_features, categorical_features)
        all_metrics.append(metrics_df)
        coverage_rows.append(coverage)

        partial_metrics = pd.concat(all_metrics, ignore_index=True)
        partial_metrics.to_csv(MODEL_DIR / "all_metrics_partial.csv", index=False)
        pd.DataFrame(coverage_rows).to_csv(MODEL_DIR / "coverage_summary_partial.csv", index=False)

    all_metrics_df = pd.concat(all_metrics, ignore_index=True)
    all_metrics_df.to_csv(MODEL_DIR / "all_metrics.csv", index=False)

    coverage_df = pd.DataFrame(coverage_rows)
    coverage_df.to_csv(MODEL_DIR / "coverage_summary.csv", index=False)

    overall = all_metrics_df[all_metrics_df["level"] == "overall_test_set"][
        [
            "model",
            "count",
            "MAE",
            "RMSE",
            "MAPE",
            "Median_APE",
            "HitRate_10_pct",
            "HitRate_20_pct",
            "Bias_pct",
            "OvervaluationRate_pct",
            "UndervaluationRate_pct",
        ]
    ].copy()
    overall = overall.merge(coverage_df, on="model", how="left")
    overall = overall.sort_values(["Median_APE", "HitRate_20_pct"], ascending=[True, False]).reset_index(drop=True)
    overall.to_csv(MODEL_DIR / "overall_comparison.csv", index=False)

    subtype = all_metrics_df[all_metrics_df["level"] == "by_subtype"][["group", "model", "count", "Median_APE", "HitRate_20_pct", "Bias_pct"]].copy()
    subtype_pivot = subtype.pivot_table(index=["group", "count"], columns="model", values=["Median_APE", "HitRate_20_pct", "Bias_pct"])
    subtype_pivot.columns = [f"{metric}__{model}" for metric, model in subtype_pivot.columns]
    subtype_pivot.reset_index().rename(columns={"group": "subtype"}).to_csv(MODEL_DIR / "by_subtype_comparison.csv", index=False)

    region = all_metrics_df[(all_metrics_df["level"] == "by_region") & (all_metrics_df["count"] >= 100)][["group", "model", "count", "Median_APE", "HitRate_20_pct", "Bias_pct"]].copy()
    region_pivot = region.pivot_table(index=["group", "count"], columns="model", values=["Median_APE", "HitRate_20_pct", "Bias_pct"])
    region_pivot.columns = [f"{metric}__{model}" for metric, model in region_pivot.columns]
    region_pivot.reset_index().rename(columns={"group": "region"}).to_csv(MODEL_DIR / "by_region_comparison.csv", index=False)

    prediction_cols = [
        "id",
        "title",
        "addr_last_1",
        "addr_last_2",
        "addr_last_3",
        "property_subtype",
        "price_vnd",
        "pred_price_vnd_global_fallback",
    ] + [f"pred_{combo_name(cols)}" for cols in GROUP_SETS]
    te[[col for col in prediction_cols if col in te.columns]].to_csv(MODEL_DIR / "predictions.csv", index=False)
    print("Saved address combo comparison to", MODEL_DIR)


if __name__ == "__main__":
    main()
