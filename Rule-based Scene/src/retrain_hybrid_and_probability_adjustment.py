from __future__ import annotations

import argparse
import json
import time
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
from train_region_subtype_models import load_robust_params, retrain_and_predict


_START = time.time()


def log(message: str) -> None:
    elapsed = int(time.time() - _START)
    minutes, seconds = divmod(elapsed, 60)
    print(f"[{minutes:02d}:{seconds:02d}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-suffix", type=str, default="classification_fix_v2")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument(
        "--routing-policy",
        type=str,
        default=str(EVAL_DIR / "improvements" / "region_subtype_models" / "routing_policy.csv"),
    )
    return parser.parse_args()


def evaluate_prediction(df_eval: pd.DataFrame, pred_col: str, model_name: str, out_dir: Path) -> pd.DataFrame:
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


def main():
    args = parse_args()
    output_dir = EVAL_DIR / "improvements" / f"hybrid_probability_refresh_{args.output_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log("Loading train/test data and routing policy")
    routing_policy = pd.read_csv(args.routing_policy)
    train_df, test_df = load_processed_train_test(args.data_path)
    params = load_robust_params()

    regime_train_df, regime_val_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=train_df["property_subtype"],
    )

    log("Tuning market-regime thresholds")
    best_thresholds, threshold_tuning_df = choose_thresholds(regime_train_df, regime_val_df)
    threshold_tuning_df.to_csv(output_dir / "threshold_tuning.csv", index=False)

    log("Training validation regime classifier")
    validation_artifacts = fit_regime_artifacts(regime_train_df, {"validation": regime_val_df}, best_thresholds)
    validation_classifier, validation_feature_cols = train_regime_classifier(validation_artifacts.train_df)
    val_probs = predict_regime_probabilities(validation_classifier, validation_artifacts.other_dfs["validation"], validation_feature_cols)
    validation_df = append_probabilities(validation_artifacts.other_dfs["validation"], val_probs)

    log("Retraining hybrid fallback for validation residuals")
    val_hybrid = retrain_and_predict(validation_artifacts.train_df, validation_df, params, routing_policy)
    val_hybrid["base_pred_log_price"] = np.log1p(val_hybrid["pred_price_vnd_hybrid"])
    log("Training probability adjustment model")
    adjustment_model, adjustment_feature_cols = train_adjustment_model(val_hybrid)

    log("Training full regime classifier")
    full_artifacts = fit_regime_artifacts(train_df, {"test": test_df}, best_thresholds)
    full_classifier, full_feature_cols = train_regime_classifier(full_artifacts.train_df)
    test_probs = predict_regime_probabilities(full_classifier, full_artifacts.other_dfs["test"], full_feature_cols)
    test_regime_df = append_probabilities(full_artifacts.other_dfs["test"], test_probs)

    log("Retraining hybrid fallback on full train set")
    test_hybrid = retrain_and_predict(train_df, test_regime_df, params, routing_policy)
    test_hybrid["base_pred_log_price"] = np.log1p(test_hybrid["pred_price_vnd_hybrid"])
    log("Applying probability adjustment")
    test_adjustment = apply_adjustment_model(adjustment_model, test_hybrid, adjustment_feature_cols)
    test_hybrid["pred_price_vnd_probability_adjustment"] = np.expm1(test_hybrid["base_pred_log_price"] + test_adjustment)

    log("Evaluating hybrid fallback and probability adjustment")
    hybrid_metrics = evaluate_prediction(
        test_hybrid,
        "pred_price_vnd_hybrid",
        "hybrid fallback",
        output_dir / "price" / "hybrid_fallback",
    )
    probability_metrics = evaluate_prediction(
        test_hybrid,
        "pred_price_vnd_probability_adjustment",
        "probability adjustment",
        output_dir / "price" / "probability_adjustment",
    )
    all_metrics = pd.concat([hybrid_metrics, probability_metrics], ignore_index=True)
    all_metrics.to_csv(output_dir / "all_metrics.csv", index=False)
    all_metrics[all_metrics["level"] == "overall_test_set"].to_csv(output_dir / "final_comparison.csv", index=False)

    save_predictions(
        test_hybrid[
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
                    "selected_model",
                    "pred_price_vnd_global",
                    "pred_price_vnd_region",
                    "pred_price_vnd_subtype",
                    "pred_price_vnd_region_subtype",
                    "pred_price_vnd_hybrid",
                    "pred_price_vnd_probability_adjustment",
                ]
                if col in test_hybrid.columns
            ]
        ],
        output_dir / "predictions.csv",
    )

    config = {
        "data_path": args.data_path,
        "routing_policy": args.routing_policy,
        "best_thresholds": best_thresholds,
        "regime_classes": REGIME_CLASSES,
        "regime_logloss_classes": REGIME_CLASSES_LOGLOSS,
    }
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Saved hybrid/probability refresh results to {output_dir}")


if __name__ == "__main__":
    main()
