from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import EVAL_DIR, RANDOM_STATE, load_processed_train_test, metric_row
from outlier_handling import prepare_outlier_artifacts
from train_region_subtype_models import fit_model, load_robust_params, non_leakage_global_features
from train_structural_ablation import LENGTH_BUNDLE, NUM_FLOORS_BUNDLE, STRUCTURAL_FEATURES, WIDTH_BUNDLE, sample_frame


MODEL_DIR = EVAL_DIR / "improvements" / "structural_ablation_global"
EXPERIMENTS = [
    {"name": "current_full", "include": STRUCTURAL_FEATURES, "description": "Current feature set with structural variables."},
    {"name": "base_without_structural", "include": set(), "description": "Remove num_floors, width, length and floors-derived bundle."},
    {"name": "plus_num_floors", "include": NUM_FLOORS_BUNDLE, "description": "Base + num_floors bundle only."},
    {"name": "plus_width", "include": WIDTH_BUNDLE, "description": "Base + width only."},
    {"name": "plus_length", "include": LENGTH_BUNDLE, "description": "Base + length only."},
    {"name": "plus_width_length", "include": WIDTH_BUNDLE | LENGTH_BUNDLE, "description": "Base + width and length."},
    {"name": "plus_num_floors_width_length", "include": STRUCTURAL_FEATURES, "description": "Base + all structural features."},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-max-rows", type=int, default=None)
    parser.add_argument("--test-max-rows", type=int, default=None)
    parser.add_argument("--output-suffix", type=str, default="")
    return parser.parse_args()


def select_features(train_df: pd.DataFrame, include_features: set[str]):
    global_num, global_cat = non_leakage_global_features(train_df)
    if include_features == STRUCTURAL_FEATURES:
        return global_num, global_cat
    filtered_num = [col for col in global_num if col not in STRUCTURAL_FEATURES or col in include_features]
    return filtered_num, global_cat


def main():
    args = parse_args()
    output_dir = MODEL_DIR if not args.output_suffix else MODEL_DIR.parent / f"{MODEL_DIR.name}_{args.output_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_processed_train_test()
    train_df = sample_frame(train_df, args.train_max_rows, "property_subtype")
    test_df = sample_frame(test_df, args.test_max_rows, "property_subtype")
    artifacts = prepare_outlier_artifacts(train_df, {"test": test_df})
    tr = artifacts.train_df
    te = artifacts.other_dfs["test"].copy()
    params = load_robust_params()

    summary_rows = []
    for experiment in EXPERIMENTS:
        experiment_dir = output_dir / experiment["name"]
        experiment_dir.mkdir(parents=True, exist_ok=True)

        numeric_features, categorical_features = select_features(tr, experiment["include"])
        model, feature_cols = fit_model(tr, numeric_features, categorical_features, params)
        te_exp = te.copy()
        te_exp["pred_price_vnd"] = pd.Series(np.expm1(model.predict(te_exp[feature_cols])), index=te_exp.index)
        metrics = metric_row(te_exp["price_vnd"], te_exp["pred_price_vnd"])

        summary_rows.append(
            {
                "experiment": experiment["name"],
                "description": experiment["description"],
                "feature_bundle": ",".join(sorted(experiment["include"])) if experiment["include"] else "none",
                "count": int(len(te_exp)),
                **metrics,
            }
        )

        pd.DataFrame([metrics]).to_csv(experiment_dir / "overall_metrics.csv", index=False)
        te_exp[["id", "property_subtype", "addr_last_1", "price_vnd", "pred_price_vnd"]].to_csv(
            experiment_dir / "predictions.csv", index=False
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["Median_APE", "HitRate_20_pct"], ascending=[True, False]).reset_index(drop=True)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    serializable_experiments = [
        {**experiment, "include": sorted(experiment["include"])}
        for experiment in EXPERIMENTS
    ]
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "random_state": RANDOM_STATE,
                "train_rows": len(tr),
                "test_rows": len(te),
                "experiments": serializable_experiments,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Saved structural ablation global results to", output_dir)


if __name__ == "__main__":
    main()
