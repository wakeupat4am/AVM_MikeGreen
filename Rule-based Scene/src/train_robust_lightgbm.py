from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import (
    EVAL_DIR,
    RANDOM_STATE,
    build_lgbm_pipeline,
    evaluate_custom_levels,
    global_features,
    list_metric_columns,
    load_processed_train_test,
    metric_row,
    save_level_outputs,
    save_predictions,
    standard_group_definitions,
)
from outlier_handling import prepare_outlier_artifacts, quantile_diagnostics


MODEL_DIR = EVAL_DIR / "improvements" / "robust_lightgbm"
LEAKAGE_COLUMNS = {
    "price",
    "price_vnd",
    "price_per_m2",
    "price_per_m2_million",
    "log_price",
    "log_price_per_m2",
    "pred_price_vnd",
    "pred_price_per_m2",
    "Q01",
    "Q05",
    "Q25",
    "Q50",
    "Q75",
    "Q95",
    "Q99",
    "sample_weight",
    "local_ppm2_band",
    "is_low_local_outlier",
    "is_high_local_outlier",
    "is_extreme_local_outlier",
    "quantile_source",
}
OBJECTIVE_GRID = [
    {"objective_name": "regression_l2", "params": {"objective": "regression_l2"}},
    {"objective_name": "regression_l1", "params": {"objective": "regression_l1"}},
    {"objective_name": "huber", "params": {"objective": "huber", "alpha": 0.9}},
    {"objective_name": "quantile_alpha_0_5", "params": {"objective": "quantile", "alpha": 0.5}},
]
BASE_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
}


def non_leakage_features(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric, categorical = global_features(df)
    numeric = [col for col in numeric if col not in LEAKAGE_COLUMNS and not col.startswith("pred_")]
    categorical = [col for col in categorical if col not in LEAKAGE_COLUMNS and not col.startswith("pred_")]
    return numeric, categorical


def build_params(extra_params: Dict) -> Dict:
    params = BASE_PARAMS.copy()
    params.update(extra_params)
    return params


def fit_weighted_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    params: Dict,
) -> Tuple[object, np.ndarray]:
    feature_cols = numeric_features + categorical_features
    model = build_lgbm_pipeline(numeric_features, categorical_features, params)
    model.fit(
        train_df[feature_cols],
        train_df["log_price"],
        model__sample_weight=train_df["sample_weight"].values,
    )
    preds = np.expm1(model.predict(val_df[feature_cols]))
    return model, preds


def evaluate_prediction_frame(df_eval: pd.DataFrame, pred_col: str, model_name: str) -> Dict[str, pd.DataFrame]:
    return evaluate_custom_levels(
        df_eval=df_eval,
        actual_col="price_vnd",
        pred_col=pred_col,
        model_name=model_name,
        target_name="price",
        group_definitions=standard_group_definitions(df_eval),
    )


def flatten_level_outputs(level_outputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(level_outputs.values(), ignore_index=True)


def compare_objectives(train_df: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
    tr_df, val_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=train_df["property_subtype"],
    )
    artifacts = prepare_outlier_artifacts(tr_df, {"validation": val_df})
    tr_enriched = artifacts.train_df
    val_enriched = artifacts.other_dfs["validation"]
    numeric_features, categorical_features = non_leakage_features(tr_enriched)

    rows = []
    for objective_spec in OBJECTIVE_GRID:
        params = build_params(objective_spec["params"])
        _, preds = fit_weighted_model(tr_enriched, val_enriched, numeric_features, categorical_features, params)
        metrics = metric_row(val_enriched["price_vnd"], preds)
        rows.append(
            {
                "objective_name": objective_spec["objective_name"],
                "params": json.dumps(params, sort_keys=True),
                **metrics,
            }
        )

    results = pd.DataFrame(rows).sort_values(
        ["Median_APE", "HitRate_20_pct", "RMSE"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    best_params = json.loads(results.iloc[0]["params"])
    return best_params, results


def baseline_summary_frame() -> pd.DataFrame:
    baseline_path = EVAL_DIR / "lightgbm" / "all_metrics.csv"
    if not baseline_path.exists():
        return pd.DataFrame()
    baseline = pd.read_csv(baseline_path)
    baseline = baseline[(baseline["target"] == "price") & (baseline["model"] == "lightgbm global")].copy()
    baseline["experiment"] = "current_baseline"
    return baseline


def save_overall_comparison(best_row: pd.Series, out_dir: Path):
    baseline = baseline_summary_frame()
    rows = []
    if not baseline.empty:
        base_overall = baseline[baseline["level"] == "overall_test_set"].iloc[0]
        rows.append(
            {
                "experiment": "current_baseline",
                "model": "lightgbm global",
                **{metric: base_overall[metric] for metric in list_metric_columns()},
            }
        )
    rows.append(
        {
            "experiment": "robust_lightgbm",
            "model": "robust lightgbm global",
            **{metric: best_row[metric] for metric in list_metric_columns()},
        }
    )
    pd.DataFrame(rows).to_csv(out_dir / "overall_comparison.csv", index=False)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train_df, test_df = load_processed_train_test()

    best_params, tuning_df = compare_objectives(train_df)
    tuning_df.to_csv(MODEL_DIR / "tuning_results.csv", index=False)

    artifacts = prepare_outlier_artifacts(train_df, {"test": test_df})
    train_enriched = artifacts.train_df
    test_enriched = artifacts.other_dfs["test"].copy()
    numeric_features, categorical_features = non_leakage_features(train_enriched)
    feature_cols = numeric_features + categorical_features

    model = build_lgbm_pipeline(numeric_features, categorical_features, best_params)
    model.fit(
        train_enriched[feature_cols],
        train_enriched["log_price"],
        model__sample_weight=train_enriched["sample_weight"].values,
    )
    test_enriched["pred_price_vnd_robust_global"] = np.expm1(model.predict(test_enriched[feature_cols]))

    level_outputs = evaluate_prediction_frame(test_enriched, "pred_price_vnd_robust_global", "robust lightgbm global")
    save_level_outputs(level_outputs, MODEL_DIR / "price" / "robust_lightgbm_global")
    all_metrics = flatten_level_outputs(level_outputs)
    all_metrics.to_csv(MODEL_DIR / "all_metrics.csv", index=False)

    save_predictions(
        test_enriched[
            [
                col
                for col in [
                    "id",
                    "title",
                    "property_subtype",
                    "addr_last_1",
                    "addr_last_2",
                    "source",
                    "price_vnd",
                    "price_per_m2",
                    "local_ppm2_band",
                    "sample_weight",
                    "pred_price_vnd_robust_global",
                ]
                if col in test_enriched.columns
            ]
        ],
        MODEL_DIR / "predictions.csv",
    )

    quantile_diagnostics(artifacts.quantile_tables).to_csv(MODEL_DIR / "outlier_diagnostics.csv", index=False)
    (MODEL_DIR / "config.json").write_text(
        json.dumps(
            {
                "best_params": best_params,
                "feature_columns": feature_cols,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    save_overall_comparison(all_metrics[all_metrics["level"] == "overall_test_set"].iloc[0], MODEL_DIR)
    print("Saved robust LightGBM results to", MODEL_DIR)


if __name__ == "__main__":
    main()
