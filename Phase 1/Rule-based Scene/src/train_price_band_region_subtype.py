from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import EVAL_DIR, RANDOM_STATE, evaluate_custom_levels, load_processed_train_test, save_level_outputs, save_predictions, standard_group_definitions
from outlier_handling import prepare_outlier_artifacts
from train_region_subtype_models import (
    SUBTYPE_COL,
    choose_policy,
    load_robust_params,
    retrain_and_predict,
    train_family_models,
)


BANDS = [
    ("lt_2b", "<2B", None, 2_000_000_000),
    ("2b_to_10_5b", "2B-10.5B", 2_000_000_000, 10_500_000_000),
    ("gt_10_5b", ">10.5B", 10_500_000_000, None),
]

_START = time.time()


def log(message: str) -> None:
    elapsed = int(time.time() - _START)
    minutes, seconds = divmod(elapsed, 60)
    print(f"[{minutes:02d}:{seconds:02d}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-suffix", type=str, default="classification_fix_v2")
    return parser.parse_args()


def band_mask(df: pd.DataFrame, low: int | None, high: int | None) -> pd.Series:
    if low is None:
        return df["price_vnd"] < high
    if high is None:
        return df["price_vnd"] > low
    return (df["price_vnd"] >= low) & (df["price_vnd"] <= high)


def split_train_validation(train_df: pd.DataFrame):
    stratify = train_df[SUBTYPE_COL] if train_df[SUBTYPE_COL].value_counts().min() >= 2 else None
    return train_test_split(
        train_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )


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
    output_dir = EVAL_DIR / "improvements" / f"oracle_price_band_region_subtype_{args.output_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log("Loading processed train/test data")
    train_df, test_df = load_processed_train_test()
    params = load_robust_params()

    all_metrics = []
    all_predictions = []
    band_summary_rows = []

    for band_id, band_label, low, high in BANDS:
        band_dir = output_dir / band_id
        band_dir.mkdir(parents=True, exist_ok=True)

        train_band = train_df.loc[band_mask(train_df, low, high)].copy()
        test_band = test_df.loc[band_mask(test_df, low, high)].copy()
        log(f"Band {band_label}: train={len(train_band):,}, test={len(test_band):,}")

        train_split, val_split = split_train_validation(train_band)
        artifacts = prepare_outlier_artifacts(train_split, {"validation": val_split})
        tr_enriched = artifacts.train_df
        val_enriched = artifacts.other_dfs["validation"]

        log(f"Band {band_label}: training validation family and routing policy")
        family = train_family_models(tr_enriched, val_enriched.copy(), params)
        routing_policy = choose_policy(family["validation_predictions"], family["train_group_sizes"])
        routing_policy.to_csv(band_dir / "routing_policy.csv", index=False)

        log(f"Band {band_label}: retraining on band train and predicting band test")
        test_pred = retrain_and_predict(train_band, test_band, params, routing_policy)
        test_pred["price_band_id"] = band_id
        test_pred["price_band_label"] = band_label

        band_summary_rows.append(
            {
                "price_band_id": band_id,
                "price_band_label": band_label,
                "train_rows": len(train_band),
                "test_rows": len(test_band),
                "routing_groups": len(routing_policy),
                "region_subtype_predicted_rows": int(test_pred["pred_price_vnd_region_subtype"].notna().sum()),
                "region_subtype_coverage_pct": float(test_pred["pred_price_vnd_region_subtype"].notna().mean() * 100),
            }
        )

        rs_metrics = evaluate_prediction(
            test_pred,
            "pred_price_vnd_region_subtype",
            f"region subtype model ({band_label})",
            band_dir / "price" / "region_subtype_model",
        )
        hybrid_metrics = evaluate_prediction(
            test_pred,
            "pred_price_vnd_hybrid",
            f"hybrid fallback ({band_label})",
            band_dir / "price" / "hybrid_fallback",
        )
        for metrics in [rs_metrics, hybrid_metrics]:
            metrics.insert(0, "price_band_id", band_id)
            metrics.insert(1, "price_band_label", band_label)
            all_metrics.append(metrics)

        save_predictions(
            test_pred[
                [
                    col
                    for col in [
                        "id",
                        "title",
                        "price_band_id",
                        "price_band_label",
                        "addr_last_1",
                        "addr_last_2",
                        "property_subtype",
                        "source",
                        "price_vnd",
                        "price_per_m2",
                        "local_ppm2_band",
                        "sample_weight",
                        "selected_model",
                        "pred_price_vnd_global",
                        "pred_price_vnd_region",
                        "pred_price_vnd_subtype",
                        "pred_price_vnd_region_subtype",
                        "pred_price_vnd_hybrid",
                    ]
                    if col in test_pred.columns
                ]
            ],
            band_dir / "predictions.csv",
        )
        all_predictions.append(test_pred)

    all_metrics_df = pd.concat(all_metrics, ignore_index=True)
    all_metrics_df.to_csv(output_dir / "all_metrics.csv", index=False)

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    save_predictions(
        predictions_df[
            [
                col
                for col in [
                    "id",
                    "title",
                    "price_band_id",
                    "price_band_label",
                    "addr_last_1",
                    "addr_last_2",
                    "property_subtype",
                    "source",
                    "price_vnd",
                    "price_per_m2",
                    "local_ppm2_band",
                    "sample_weight",
                    "selected_model",
                    "pred_price_vnd_region_subtype",
                    "pred_price_vnd_hybrid",
                ]
                if col in predictions_df.columns
            ]
        ],
        output_dir / "predictions.csv",
    )

    final_comparison = all_metrics_df[all_metrics_df["level"] == "overall_test_set"].copy()
    final_comparison.to_csv(output_dir / "final_comparison_by_band.csv", index=False)

    band_summary = pd.DataFrame(band_summary_rows)
    band_summary.to_csv(output_dir / "band_summary.csv", index=False)

    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "bands": [
                    {"price_band_id": band_id, "price_band_label": band_label, "low": low, "high": high}
                    for band_id, band_label, low, high in BANDS
                ],
                "note": "Oracle experiment: true price is used to assign train/test rows to price bands.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"Saved oracle price-band region/subtype results to {output_dir}")


if __name__ == "__main__":
    main()
