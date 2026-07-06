from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import (
    EVAL_DIR,
    RANDOM_STATE,
    evaluate_custom_levels,
    list_metric_columns,
    load_processed_train_test,
    save_level_outputs,
    save_predictions,
    standard_group_definitions,
)
from market_regime import (
    REGIME_CLASSES,
    REGIME_CLASSES_LOGLOSS,
    append_probabilities,
    apply_adjustment_model,
    choose_thresholds,
    fit_regime_artifacts,
    hard_group_training_sets,
    regime_level_groups,
    train_adjustment_model,
    train_regime_classifier,
    predict_regime_probabilities,
)
from train_catboost_region_subtype import (
    CATBOOST_PARAMS,
    choose_policy,
    fit_model,
    non_leakage_global_features,
    predict_with_model,
    retrain_and_predict,
    train_family_models,
)


MODEL_DIR = EVAL_DIR / "improvements" / "catboost_market_regime_layer"


def build_validation_hybrid_predictions(validation_predictions: pd.DataFrame, routing_policy: pd.DataFrame) -> pd.DataFrame:
    out = validation_predictions.copy()
    policy_lookup = routing_policy.set_index(["addr_last_1", "property_subtype"])["chosen_model"].to_dict()
    out["selected_model"] = out.apply(lambda row: policy_lookup.get((row["addr_last_1"], row["property_subtype"]), "global"), axis=1)
    out["pred_price_vnd_hybrid"] = out["pred_price_vnd_global"]
    out.loc[out["selected_model"] == "region", "pred_price_vnd_hybrid"] = (
        out.loc[out["selected_model"] == "region", "pred_price_vnd_region"]
        .fillna(out.loc[out["selected_model"] == "region", "pred_price_vnd_global"])
    )
    out.loc[out["selected_model"] == "subtype", "pred_price_vnd_hybrid"] = (
        out.loc[out["selected_model"] == "subtype", "pred_price_vnd_subtype"]
        .fillna(out.loc[out["selected_model"] == "subtype", "pred_price_vnd_global"])
    )
    out.loc[out["selected_model"] == "region_subtype", "pred_price_vnd_hybrid"] = (
        out.loc[out["selected_model"] == "region_subtype", "pred_price_vnd_region_subtype"]
        .fillna(out.loc[out["selected_model"] == "region_subtype", "pred_price_vnd_region"])
        .fillna(out.loc[out["selected_model"] == "region_subtype", "pred_price_vnd_global"])
    )
    return out


def train_regime_pool_model(train_df: pd.DataFrame, params: Dict):
    numeric_features, categorical_features = non_leakage_global_features(train_df)
    return fit_model(train_df, numeric_features, categorical_features, params)


def predict_regime_pool(model, feature_cols: List[str], df: pd.DataFrame) -> np.ndarray:
    return np.expm1(model.predict(df))


def train_hard_regime_models(train_df: pd.DataFrame, params: Dict):
    training_sets = hard_group_training_sets(train_df)
    models = {}
    for group_name, group_df in training_sets.items():
        if len(group_df) < 300:
            continue
        model, feature_cols = train_regime_pool_model(group_df, params)
        models[group_name] = {"model": model, "feature_cols": feature_cols, "rows": len(group_df)}
    return models, training_sets


def assign_hard_predictions(df: pd.DataFrame, hard_models: Dict[str, Dict]) -> pd.DataFrame:
    out = df.copy()
    out["pred_price_vnd_hard_regime"] = np.nan
    route_map = {
        "low_tail": "low_tail_group",
        "normal": "normal_group",
        "high_tail": "high_tail_group",
        "stable_normal": "stable_normal_group",
    }
    for regime_label, model_name in route_map.items():
        if model_name not in hard_models:
            continue
        mask = out["predicted_regime_label"] == regime_label
        if not mask.any():
            continue
        bundle = hard_models[model_name]
        out.loc[mask, "pred_price_vnd_hard_regime"] = predict_regime_pool(bundle["model"], bundle["feature_cols"], out.loc[mask])
    return out


def prepare_comparison_levels(df_eval: pd.DataFrame) -> Dict:
    groups = standard_group_definitions(df_eval)
    groups.update(regime_level_groups(df_eval))
    return groups


def evaluate_prediction(df_eval: pd.DataFrame, pred_col: str, model_name: str, out_dir: Path) -> pd.DataFrame:
    levels = evaluate_custom_levels(
        df_eval=df_eval,
        actual_col="price_vnd",
        pred_col=pred_col,
        model_name=model_name,
        target_name="price",
        group_definitions=prepare_comparison_levels(df_eval),
    )
    save_level_outputs(levels, out_dir)
    return pd.concat(levels.values(), ignore_index=True)


def overall_row(metrics_df: pd.DataFrame, model_name: str) -> pd.Series:
    return metrics_df[(metrics_df["level"] == "overall_test_set") & (metrics_df["model"] == model_name)].iloc[0]


def build_final_comparison(current_hybrid_metrics, hard_metrics, soft_metrics, trimmed_metrics) -> pd.DataFrame:
    rows = []
    for metrics_df, model_name, experiment in [
        (current_hybrid_metrics, "current hybrid fallback", "current_hybrid_fallback"),
        (hard_metrics, "hard regime routing", "hard_regime_grouping"),
        (soft_metrics, "probability adjustment", "probability_adjustment"),
        (trimmed_metrics, "trimmed hybrid local", "trimmed_hybrid_local"),
    ]:
        row = overall_row(metrics_df, model_name)
        rows.append({"experiment": experiment, "model": model_name, **{metric: row[metric] for metric in list_metric_columns()}})
    return pd.DataFrame(rows)


def fit_base_hybrid_for_validation(train_df, validation_df, params, exclude_num_floors_bundle=False):
    family = train_family_models(train_df, validation_df.copy(), params, exclude_num_floors_bundle=exclude_num_floors_bundle)
    routing_policy = choose_policy(family["validation_predictions"], family["train_group_sizes"])
    val_with_hybrid = build_validation_hybrid_predictions(family["validation_predictions"], routing_policy)
    return val_with_hybrid, routing_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-suffix", type=str, default="")
    parser.add_argument("--exclude-num-floors-bundle", action="store_true")
    parser.add_argument("--params-json", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = MODEL_DIR if not args.output_suffix else MODEL_DIR.parent / f"{MODEL_DIR.name}_{args.output_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df, test_df = load_processed_train_test()
    params = CATBOOST_PARAMS.copy()
    if args.params_json:
        params.update(json.loads(args.params_json))

    regime_train_df, regime_val_df = train_test_split(
        train_df, test_size=0.2, random_state=RANDOM_STATE, stratify=train_df["property_subtype"],
    )

    best_thresholds, threshold_tuning_df = choose_thresholds(regime_train_df, regime_val_df)
    threshold_tuning_df.to_csv(output_dir / "threshold_tuning.csv", index=False)

    validation_artifacts = fit_regime_artifacts(regime_train_df, {"validation": regime_val_df}, best_thresholds)
    validation_classifier, validation_feature_cols = train_regime_classifier(
        validation_artifacts.train_df, exclude_num_floors_bundle=args.exclude_num_floors_bundle,
    )
    val_probs = predict_regime_probabilities(validation_classifier, validation_artifacts.other_dfs["validation"], validation_feature_cols)
    validation_df = append_probabilities(validation_artifacts.other_dfs["validation"], val_probs)
    val_prob_for_metrics = val_probs.copy()
    val_prob_for_metrics.columns = [col.replace("prob_", "") for col in val_prob_for_metrics.columns]
    classifier_val_metrics = {
        "accuracy": float((validation_df["predicted_regime_label"] == validation_df["true_regime_label"]).mean()),
        "macro_f1": float(
            __import__("sklearn.metrics").metrics.f1_score(validation_df["true_regime_label"], validation_df["predicted_regime_label"], average="macro")
        ),
        "log_loss": float(
            __import__("sklearn.metrics").metrics.log_loss(
                validation_df["true_regime_label"],
                val_prob_for_metrics[REGIME_CLASSES_LOGLOSS],
                labels=REGIME_CLASSES_LOGLOSS,
            )
        ),
    }

    val_with_hybrid, validation_routing_policy = fit_base_hybrid_for_validation(
        validation_artifacts.train_df, validation_df.copy(), params, exclude_num_floors_bundle=args.exclude_num_floors_bundle,
    )
    val_with_hybrid["base_pred_log_price"] = np.log1p(val_with_hybrid["pred_price_vnd_hybrid"])
    adjustment_model, adjustment_feature_cols = train_adjustment_model(val_with_hybrid)

    full_artifacts = fit_regime_artifacts(train_df, {"test": test_df}, best_thresholds)
    full_classifier, full_feature_cols = train_regime_classifier(
        full_artifacts.train_df, exclude_num_floors_bundle=args.exclude_num_floors_bundle,
    )
    test_probs = predict_regime_probabilities(full_classifier, full_artifacts.other_dfs["test"], full_feature_cols)
    test_regime_df = append_probabilities(full_artifacts.other_dfs["test"], test_probs)

    family_full = train_family_models(
        validation_artifacts.train_df, validation_df.copy(), params, exclude_num_floors_bundle=args.exclude_num_floors_bundle,
    )
    full_routing_policy = choose_policy(family_full["validation_predictions"], family_full["train_group_sizes"])
    test_hybrid_pred = retrain_and_predict(train_df, test_df, params, full_routing_policy, exclude_num_floors_bundle=args.exclude_num_floors_bundle)
    hybrid_saved = test_hybrid_pred[[c for c in [
        "id", "pred_price_vnd_hybrid", "pred_price_vnd_global",
        "pred_price_vnd_region", "pred_price_vnd_subtype", "pred_price_vnd_region_subtype", "selected_model",
    ] if c in test_hybrid_pred.columns]].copy()
    test_hybrid_df = test_regime_df.merge(hybrid_saved, on="id", how="left")
    test_hybrid_df["base_pred_log_price"] = np.log1p(test_hybrid_df["pred_price_vnd_hybrid"])

    hard_models, training_sets = train_hard_regime_models(full_artifacts.train_df, params)
    hard_test_df = assign_hard_predictions(test_hybrid_df, hard_models)
    hard_test_df["pred_price_vnd_hard_regime"] = hard_test_df["pred_price_vnd_hard_regime"].fillna(hard_test_df["pred_price_vnd_hybrid"])

    hard_test_df["pred_price_vnd_trimmed_hybrid"] = np.nan
    if "trimmed_local_group" in training_sets and len(training_sets["trimmed_local_group"]) >= 500:
        trimmed_model, trimmed_cols = train_regime_pool_model(training_sets["trimmed_local_group"], params)
        hard_test_df["pred_price_vnd_trimmed_hybrid"] = predict_regime_pool(trimmed_model, trimmed_cols, hard_test_df)
    else:
        hard_test_df["pred_price_vnd_trimmed_hybrid"] = hard_test_df["pred_price_vnd_hybrid"]

    test_adjustment = apply_adjustment_model(adjustment_model, test_hybrid_df, adjustment_feature_cols)
    hard_test_df["pred_price_vnd_probability_adjustment"] = np.expm1(hard_test_df["base_pred_log_price"] + test_adjustment)

    test_prob_for_metrics = test_probs.copy()
    test_prob_for_metrics.columns = [col.replace("prob_", "") for col in test_prob_for_metrics.columns]
    classifier_test_metrics = {
        "accuracy": float((hard_test_df["predicted_regime_label"] == hard_test_df["true_regime_label"]).mean()),
        "macro_f1": float(
            __import__("sklearn.metrics").metrics.f1_score(hard_test_df["true_regime_label"], hard_test_df["predicted_regime_label"], average="macro")
        ),
        "log_loss": float(
            __import__("sklearn.metrics").metrics.log_loss(
                hard_test_df["true_regime_label"],
                test_prob_for_metrics[REGIME_CLASSES_LOGLOSS],
                labels=REGIME_CLASSES_LOGLOSS,
            )
        ),
    }
    pd.DataFrame([classifier_val_metrics]).to_csv(output_dir / "classifier_validation_metrics.csv", index=False)
    pd.DataFrame([classifier_test_metrics]).to_csv(output_dir / "classifier_test_metrics.csv", index=False)

    current_hybrid_metrics = evaluate_prediction(hard_test_df, "pred_price_vnd_hybrid", "current hybrid fallback", output_dir / "price" / "current_hybrid_fallback")
    hard_metrics = evaluate_prediction(hard_test_df, "pred_price_vnd_hard_regime", "hard regime routing", output_dir / "price" / "hard_regime_routing")
    soft_metrics = evaluate_prediction(hard_test_df, "pred_price_vnd_probability_adjustment", "probability adjustment", output_dir / "price" / "probability_adjustment")
    trimmed_metrics = evaluate_prediction(hard_test_df, "pred_price_vnd_trimmed_hybrid", "trimmed hybrid local", output_dir / "price" / "trimmed_hybrid_local")

    all_metrics = pd.concat([current_hybrid_metrics, hard_metrics, soft_metrics, trimmed_metrics], ignore_index=True)
    all_metrics.to_csv(output_dir / "all_metrics.csv", index=False)
    comparison_df = build_final_comparison(current_hybrid_metrics, hard_metrics, soft_metrics, trimmed_metrics)
    comparison_df.to_csv(output_dir / "final_comparison.csv", index=False)

    save_predictions(
        hard_test_df[[c for c in [
            "id", "title", "addr_last_1", "addr_last_2", "property_subtype", "source",
            "price_vnd", "price_per_m2", "true_regime_label", "predicted_regime_label",
            "group_dispersion_type", "dispersion_score",
            "prob_low_tail", "prob_normal", "prob_high_tail", "prob_stable_normal",
            "pred_price_vnd_hybrid", "pred_price_vnd_hard_regime",
            "pred_price_vnd_probability_adjustment", "pred_price_vnd_trimmed_hybrid",
        ] if c in hard_test_df.columns]],
        output_dir / "predictions.csv",
    )

    group_rows = [{"training_group": g, "rows": len(df)} for g, df in training_sets.items()]
    pd.DataFrame(group_rows).to_csv(output_dir / "training_group_sizes.csv", index=False)
    (output_dir / "config.json").write_text(
        json.dumps({"params": params, "best_thresholds": best_thresholds, "regime_classes": REGIME_CLASSES, "exclude_num_floors_bundle": args.exclude_num_floors_bundle}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Saved CatBoost market regime layer results to", output_dir)


if __name__ == "__main__":
    main()
