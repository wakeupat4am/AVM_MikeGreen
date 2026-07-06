from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import EVAL_DIR, RANDOM_STATE, evaluate_custom_levels, load_processed_train_test, save_level_outputs, save_predictions, standard_group_definitions
from market_regime import (
    REGIME_CLASSES,
    REGIME_CLASSES_LOGLOSS,
    append_probabilities,
    apply_adjustment_model,
    choose_thresholds,
    fit_regime_artifacts,
    train_adjustment_model,
    train_regime_classifier,
    predict_regime_probabilities,
)
from train_catboost_region_subtype import CATBOOST_PARAMS, choose_policy, train_family_models


MODEL_DIR = EVAL_DIR / "improvements" / "catboost_market_regime_probability_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-suffix", type=str, default="")
    parser.add_argument("--exclude-num-floors-bundle", action="store_true")
    parser.add_argument("--hybrid-predictions", type=str, required=True)
    parser.add_argument("--params-json", type=str, default="")
    return parser.parse_args()


def prepare_comparison_levels(df_eval: pd.DataFrame):
    return standard_group_definitions(df_eval)


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


def fit_base_hybrid_for_validation(train_df: pd.DataFrame, validation_df: pd.DataFrame, params: dict, exclude_num_floors_bundle: bool):
    family = train_family_models(
        train_df,
        validation_df.copy(),
        params,
        exclude_num_floors_bundle=exclude_num_floors_bundle,
    )
    routing_policy = choose_policy(family["validation_predictions"], family["train_group_sizes"])
    out = family["validation_predictions"].copy()
    policy_lookup = routing_policy.set_index(["addr_last_1", "property_subtype"])["chosen_model"].to_dict()
    out["selected_model"] = out.apply(lambda row: policy_lookup.get((row["addr_last_1"], row["property_subtype"]), "global"), axis=1)
    out["pred_price_vnd_hybrid"] = out["pred_price_vnd_global"]
    out.loc[out["selected_model"] == "region", "pred_price_vnd_hybrid"] = out.loc[out["selected_model"] == "region", "pred_price_vnd_region"].fillna(
        out.loc[out["selected_model"] == "region", "pred_price_vnd_global"]
    )
    out.loc[out["selected_model"] == "subtype", "pred_price_vnd_hybrid"] = out.loc[out["selected_model"] == "subtype", "pred_price_vnd_subtype"].fillna(
        out.loc[out["selected_model"] == "subtype", "pred_price_vnd_global"]
    )
    out.loc[out["selected_model"] == "region_subtype", "pred_price_vnd_hybrid"] = out.loc[
        out["selected_model"] == "region_subtype", "pred_price_vnd_region_subtype"
    ].fillna(out.loc[out["selected_model"] == "region_subtype", "pred_price_vnd_region"]).fillna(
        out.loc[out["selected_model"] == "region_subtype", "pred_price_vnd_global"]
    )
    return out


def main():
    args = parse_args()
    output_dir = MODEL_DIR if not args.output_suffix else MODEL_DIR.parent / f"{MODEL_DIR.name}_{args.output_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_processed_train_test()
    params = CATBOOST_PARAMS.copy()
    if args.params_json:
        params.update(json.loads(args.params_json))

    regime_train_df, regime_val_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=train_df["property_subtype"],
    )

    best_thresholds, threshold_tuning_df = choose_thresholds(regime_train_df, regime_val_df)
    threshold_tuning_df.to_csv(output_dir / "threshold_tuning.csv", index=False)

    validation_artifacts = fit_regime_artifacts(regime_train_df, {"validation": regime_val_df}, best_thresholds)
    validation_classifier, validation_feature_cols = train_regime_classifier(
        validation_artifacts.train_df,
        exclude_num_floors_bundle=args.exclude_num_floors_bundle,
    )
    val_probs = predict_regime_probabilities(validation_classifier, validation_artifacts.other_dfs["validation"], validation_feature_cols)
    validation_df = append_probabilities(validation_artifacts.other_dfs["validation"], val_probs)
    val_with_hybrid = fit_base_hybrid_for_validation(
        validation_artifacts.train_df,
        validation_df.copy(),
        params,
        exclude_num_floors_bundle=args.exclude_num_floors_bundle,
    )
    val_with_hybrid["base_pred_log_price"] = np.log1p(val_with_hybrid["pred_price_vnd_hybrid"])
    adjustment_model, adjustment_feature_cols = train_adjustment_model(val_with_hybrid)

    full_artifacts = fit_regime_artifacts(train_df, {"test": test_df}, best_thresholds)
    full_classifier, full_feature_cols = train_regime_classifier(
        full_artifacts.train_df,
        exclude_num_floors_bundle=args.exclude_num_floors_bundle,
    )
    test_probs = predict_regime_probabilities(full_classifier, full_artifacts.other_dfs["test"], full_feature_cols)
    test_regime_df = append_probabilities(full_artifacts.other_dfs["test"], test_probs)

    hybrid_saved = pd.read_csv(args.hybrid_predictions)
    keep_cols = [
        col
        for col in [
            "id",
            "pred_price_vnd_hybrid",
            "pred_price_vnd_global",
            "pred_price_vnd_region",
            "pred_price_vnd_subtype",
            "pred_price_vnd_region_subtype",
            "selected_model",
        ]
        if col in hybrid_saved.columns
    ]
    test_hybrid_df = test_regime_df.merge(hybrid_saved[keep_cols], on="id", how="left")
    test_hybrid_df["base_pred_log_price"] = np.log1p(test_hybrid_df["pred_price_vnd_hybrid"])
    test_adjustment = apply_adjustment_model(adjustment_model, test_hybrid_df, adjustment_feature_cols)
    test_hybrid_df["pred_price_vnd_probability_adjustment"] = np.expm1(test_hybrid_df["base_pred_log_price"] + test_adjustment)

    current_hybrid_metrics = evaluate_prediction(
        test_hybrid_df,
        "pred_price_vnd_hybrid",
        "current hybrid fallback",
        output_dir / "price" / "current_hybrid_fallback",
    )
    probability_metrics = evaluate_prediction(
        test_hybrid_df,
        "pred_price_vnd_probability_adjustment",
        "probability adjustment",
        output_dir / "price" / "probability_adjustment",
    )
    all_metrics = pd.concat([current_hybrid_metrics, probability_metrics], ignore_index=True)
    all_metrics.to_csv(output_dir / "all_metrics.csv", index=False)
    all_metrics[all_metrics["level"] == "overall_test_set"].to_csv(output_dir / "final_comparison.csv", index=False)

    save_predictions(
        test_hybrid_df[
            [
                col
                for col in [
                    "id",
                    "title",
                    "addr_last_1",
                    "addr_last_2",
                    "property_subtype",
                    "source",
                    "price_vnd",
                    "price_per_m2",
                    "true_regime_label",
                    "predicted_regime_label",
                    "prob_low_tail",
                    "prob_normal",
                    "prob_high_tail",
                    "prob_stable_normal",
                    "pred_price_vnd_hybrid",
                    "pred_price_vnd_probability_adjustment",
                ]
                if col in test_hybrid_df.columns
            ]
        ],
        output_dir / "predictions.csv",
    )

    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "params": params,
                "best_thresholds": best_thresholds,
                "exclude_num_floors_bundle": args.exclude_num_floors_bundle,
                "hybrid_predictions": args.hybrid_predictions,
                "regime_classes": REGIME_CLASSES,
                "regime_logloss_classes": REGIME_CLASSES_LOGLOSS,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Saved CatBoost market regime probability-only results to", output_dir)


if __name__ == "__main__":
    main()
