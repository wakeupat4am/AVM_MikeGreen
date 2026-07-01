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
    evaluate_custom_levels,
    list_metric_columns,
    load_processed_train_test,
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
    fit_model,
    load_robust_params,
    median_ape_for_slice,
    non_leakage_global_features,
    non_leakage_subtype_features,
    predict_with_model,
)


MODEL_DIR = EVAL_DIR / "improvements" / "structural_ablation"
NUM_FLOORS_BUNDLE = {"num_floors", "floor_area_m2", "floors_x_area", "floors_x_frontage", "floors_x_business"}
WIDTH_BUNDLE = {"width_m"}
LENGTH_BUNDLE = {"length_m"}
STRUCTURAL_FEATURES = NUM_FLOORS_BUNDLE | WIDTH_BUNDLE | LENGTH_BUNDLE

EXPERIMENTS = [
    {"name": "current_full", "include": STRUCTURAL_FEATURES, "description": "Current feature set with structural variables."},
    {"name": "base_without_structural", "include": set(), "description": "Remove num_floors, width, length and floors-derived bundle."},
    {"name": "plus_num_floors", "include": NUM_FLOORS_BUNDLE, "description": "Base + num_floors bundle only."},
    {"name": "plus_width", "include": WIDTH_BUNDLE, "description": "Base + width only."},
    {"name": "plus_length", "include": LENGTH_BUNDLE, "description": "Base + length only."},
    {"name": "plus_width_length", "include": WIDTH_BUNDLE | LENGTH_BUNDLE, "description": "Base + width and length."},
    {"name": "plus_num_floors_width_length", "include": STRUCTURAL_FEATURES, "description": "Base + all structural features."},
]


def select_feature_sets(
    train_df: pd.DataFrame,
    include_features: set[str],
) -> Tuple[List[str], List[str], Dict[str, Dict[str, List[str]]]]:
    global_num, global_cat = non_leakage_global_features(train_df)
    subtype_map = non_leakage_subtype_features(train_df)

    if include_features == STRUCTURAL_FEATURES:
        return global_num, global_cat, subtype_map

    allowed_structural = set(include_features)
    filtered_global_num = [col for col in global_num if col not in STRUCTURAL_FEATURES or col in allowed_structural]
    filtered_subtype_map: Dict[str, Dict[str, List[str]]] = {}
    for subtype, feature_dict in subtype_map.items():
        filtered_subtype_map[subtype] = {
            "numeric": [col for col in feature_dict["numeric"] if col not in STRUCTURAL_FEATURES or col in allowed_structural],
            "categorical": feature_dict["categorical"],
        }
    return filtered_global_num, global_cat, filtered_subtype_map


def train_family_models_custom(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    params: Dict,
    global_num: List[str],
    global_cat: List[str],
    subtype_feature_map: Dict[str, Dict[str, List[str]]],
    min_region_rows: int,
    min_region_subtype_rows: int,
):
    global_model, global_cols = fit_model(train_df, global_num, global_cat, params)
    val_df["pred_price_vnd_global"] = predict_with_model(global_model, val_df, global_cols)

    region_models = {}
    val_df["pred_price_vnd_region"] = np.nan
    for region, region_train in train_df.groupby(REGION_COL):
        if len(region_train) < min_region_rows:
            continue
        region_model, region_cols = fit_model(region_train, global_num, global_cat, params)
        region_models[region] = {"model": region_model, "feature_cols": region_cols, "rows": len(region_train)}
        region_mask = val_df[REGION_COL] == region
        if region_mask.any():
            val_df.loc[region_mask, "pred_price_vnd_region"] = predict_with_model(region_model, val_df.loc[region_mask], region_cols)

    subtype_models = {}
    val_df["pred_price_vnd_subtype"] = np.nan
    for subtype, subtype_train in train_df.groupby(SUBTYPE_COL):
        if len(subtype_train) < min_region_rows:
            continue
        feature_dict = subtype_feature_map.get(subtype, {"numeric": global_num, "categorical": global_cat})
        subtype_model, subtype_cols = fit_model(subtype_train, feature_dict["numeric"], feature_dict["categorical"], params)
        subtype_models[subtype] = {"model": subtype_model, "feature_cols": subtype_cols, "rows": len(subtype_train)}
        subtype_mask = val_df[SUBTYPE_COL] == subtype
        if subtype_mask.any():
            val_df.loc[subtype_mask, "pred_price_vnd_subtype"] = predict_with_model(subtype_model, val_df.loc[subtype_mask], subtype_cols)

    region_subtype_models = {}
    val_df["pred_price_vnd_region_subtype"] = np.nan
    group_sizes = train_df.groupby([REGION_COL, SUBTYPE_COL]).size().rename("train_rows").reset_index()
    for row in group_sizes.itertuples(index=False):
        if row.train_rows < min_region_subtype_rows:
            continue
        mask = (train_df[REGION_COL] == row.addr_last_1) & (train_df[SUBTYPE_COL] == row.property_subtype)
        group_train = train_df.loc[mask]
        feature_dict = subtype_feature_map.get(row.property_subtype, {"numeric": global_num, "categorical": global_cat})
        group_model, group_cols = fit_model(group_train, feature_dict["numeric"], feature_dict["categorical"], params)
        region_subtype_models[(row.addr_last_1, row.property_subtype)] = {
            "model": group_model,
            "feature_cols": group_cols,
            "rows": int(row.train_rows),
        }
        val_mask = (val_df[REGION_COL] == row.addr_last_1) & (val_df[SUBTYPE_COL] == row.property_subtype)
        if val_mask.any():
            val_df.loc[val_mask, "pred_price_vnd_region_subtype"] = predict_with_model(group_model, val_df.loc[val_mask], group_cols)

    return {
        "global": {"model": global_model, "feature_cols": global_cols},
        "region": region_models,
        "subtype": subtype_models,
        "region_subtype": region_subtype_models,
        "validation_predictions": val_df,
        "train_group_sizes": group_sizes,
    }


def choose_policy_custom(
    validation_predictions: pd.DataFrame,
    train_group_sizes: pd.DataFrame,
    min_region_rows: int,
    min_region_subtype_rows: int,
) -> pd.DataFrame:
    rows = []
    keys = validation_predictions[[REGION_COL, SUBTYPE_COL]].drop_duplicates()
    group_size_lookup = {
        (row.addr_last_1, row.property_subtype): int(row.train_rows)
        for row in train_group_sizes.itertuples(index=False)
    }
    for row in keys.itertuples(index=False):
        region = row.addr_last_1
        subtype = row.property_subtype
        mask = (validation_predictions[REGION_COL] == region) & (validation_predictions[SUBTYPE_COL] == subtype)
        train_rows = group_size_lookup.get((region, subtype), 0)

        candidate_scores = {"global": median_ape_for_slice(validation_predictions, "pred_price_vnd_global", mask)}
        if validation_predictions.loc[mask, "pred_price_vnd_region"].notna().any():
            candidate_scores["region"] = median_ape_for_slice(validation_predictions, "pred_price_vnd_region", mask)
        if validation_predictions.loc[mask, "pred_price_vnd_subtype"].notna().any():
            candidate_scores["subtype"] = median_ape_for_slice(validation_predictions, "pred_price_vnd_subtype", mask)
        if validation_predictions.loc[mask, "pred_price_vnd_region_subtype"].notna().any():
            candidate_scores["region_subtype"] = median_ape_for_slice(validation_predictions, "pred_price_vnd_region_subtype", mask)

        if train_rows < min_region_rows:
            chosen_model = "global"
        elif train_rows < min_region_subtype_rows:
            chosen_model = "region" if "region" in candidate_scores else "global"
            if "subtype" in candidate_scores and candidate_scores["subtype"] < candidate_scores[chosen_model]:
                chosen_model = "subtype"
        else:
            chosen_model = "region" if "region" in candidate_scores else "global"
            if "subtype" in candidate_scores and candidate_scores["subtype"] < candidate_scores[chosen_model]:
                chosen_model = "subtype"
            if "region_subtype" in candidate_scores and candidate_scores["region_subtype"] < candidate_scores[chosen_model]:
                chosen_model = "region_subtype"

        rows.append(
            {
                REGION_COL: region,
                SUBTYPE_COL: subtype,
                "train_rows": train_rows,
                "chosen_model": chosen_model,
            }
        )
    return pd.DataFrame(rows)


def retrain_and_predict_custom(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: Dict,
    routing_policy: pd.DataFrame,
    global_num: List[str],
    global_cat: List[str],
    subtype_feature_map: Dict[str, Dict[str, List[str]]],
    min_region_rows: int,
    min_region_subtype_rows: int,
) -> pd.DataFrame:
    train_artifacts = prepare_outlier_artifacts(train_df, {"test": test_df})
    tr = train_artifacts.train_df
    te = train_artifacts.other_dfs["test"].copy()

    global_model, global_cols = fit_model(tr, global_num, global_cat, params)
    te["pred_price_vnd_global"] = predict_with_model(global_model, te, global_cols)

    te["pred_price_vnd_region"] = np.nan
    for region, region_train in tr.groupby(REGION_COL):
        if len(region_train) < min_region_rows:
            continue
        region_model, region_cols = fit_model(region_train, global_num, global_cat, params)
        region_mask = te[REGION_COL] == region
        if region_mask.any():
            te.loc[region_mask, "pred_price_vnd_region"] = predict_with_model(region_model, te.loc[region_mask], region_cols)

    te["pred_price_vnd_subtype"] = np.nan
    for subtype, subtype_train in tr.groupby(SUBTYPE_COL):
        if len(subtype_train) < min_region_rows:
            continue
        feature_dict = subtype_feature_map.get(subtype, {"numeric": global_num, "categorical": global_cat})
        subtype_model, subtype_cols = fit_model(subtype_train, feature_dict["numeric"], feature_dict["categorical"], params)
        subtype_mask = te[SUBTYPE_COL] == subtype
        if subtype_mask.any():
            te.loc[subtype_mask, "pred_price_vnd_subtype"] = predict_with_model(subtype_model, te.loc[subtype_mask], subtype_cols)

    te["pred_price_vnd_region_subtype"] = np.nan
    eligible_groups = routing_policy[routing_policy["train_rows"] >= min_region_subtype_rows]
    for row in eligible_groups.itertuples(index=False):
        mask = (tr[REGION_COL] == row.addr_last_1) & (tr[SUBTYPE_COL] == row.property_subtype)
        group_train = tr.loc[mask]
        feature_dict = subtype_feature_map.get(row.property_subtype, {"numeric": global_num, "categorical": global_cat})
        group_model, group_cols = fit_model(group_train, feature_dict["numeric"], feature_dict["categorical"], params)
        test_mask = (te[REGION_COL] == row.addr_last_1) & (te[SUBTYPE_COL] == row.property_subtype)
        if test_mask.any():
            te.loc[test_mask, "pred_price_vnd_region_subtype"] = predict_with_model(group_model, te.loc[test_mask], group_cols)

    policy_lookup = routing_policy.set_index([REGION_COL, SUBTYPE_COL])["chosen_model"].to_dict()
    te["selected_model"] = te.apply(lambda row: policy_lookup.get((row[REGION_COL], row[SUBTYPE_COL]), "global"), axis=1)
    te["pred_price_vnd_hybrid"] = te["pred_price_vnd_global"]
    te.loc[te["selected_model"] == "region", "pred_price_vnd_hybrid"] = te.loc[te["selected_model"] == "region", "pred_price_vnd_region"].fillna(
        te.loc[te["selected_model"] == "region", "pred_price_vnd_global"]
    )
    te.loc[te["selected_model"] == "subtype", "pred_price_vnd_hybrid"] = te.loc[te["selected_model"] == "subtype", "pred_price_vnd_subtype"].fillna(
        te.loc[te["selected_model"] == "subtype", "pred_price_vnd_global"]
    )
    te.loc[te["selected_model"] == "region_subtype", "pred_price_vnd_hybrid"] = te.loc[
        te["selected_model"] == "region_subtype", "pred_price_vnd_region_subtype"
    ].fillna(te.loc[te["selected_model"] == "region_subtype", "pred_price_vnd_region"]).fillna(
        te.loc[te["selected_model"] == "region_subtype", "pred_price_vnd_global"]
    )
    return te


def evaluate_experiment(df_eval: pd.DataFrame, experiment_name: str, out_dir: Path) -> pd.DataFrame:
    levels = evaluate_custom_levels(
        df_eval=df_eval,
        actual_col="price_vnd",
        pred_col="pred_price_vnd_hybrid",
        model_name=experiment_name,
        target_name="price",
        group_definitions=standard_group_definitions(df_eval),
    )
    save_level_outputs(levels, out_dir)
    return pd.concat(levels.values(), ignore_index=True)


def sample_frame(df: pd.DataFrame, max_rows: int | None, stratify_col: str) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df.copy()
    sample_frac = max_rows / len(df)
    sampled_index = (
        df.groupby(stratify_col, group_keys=False)
        .apply(lambda part: part.sample(max(1, int(round(len(part) * sample_frac))), random_state=RANDOM_STATE).index)
        .explode()
        .astype(int)
        .tolist()
    )
    sampled = df.loc[sampled_index].copy().reset_index(drop=True)
    if len(sampled) > max_rows:
        sampled = sampled.sample(max_rows, random_state=RANDOM_STATE).reset_index(drop=True)
    return sampled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-max-rows", type=int, default=None)
    parser.add_argument("--test-max-rows", type=int, default=None)
    parser.add_argument("--output-suffix", type=str, default="")
    parser.add_argument("--threshold-scale", type=float, default=1.0)
    parser.add_argument("--experiments", nargs="*", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = MODEL_DIR if not args.output_suffix else MODEL_DIR.parent / f"{MODEL_DIR.name}_{args.output_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_processed_train_test()
    train_df = sample_frame(train_df, args.train_max_rows, SUBTYPE_COL)
    test_df = sample_frame(test_df, args.test_max_rows, SUBTYPE_COL)
    params = load_robust_params()
    min_region_rows = max(50, int(round(MIN_REGION_ROWS * args.threshold_scale)))
    min_region_subtype_rows = max(200, int(round(MIN_REGION_SUBTYPE_ROWS * args.threshold_scale)))

    train_split, val_split = train_test_split(
        train_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=train_df[SUBTYPE_COL],
    )

    selected_experiments = EXPERIMENTS
    if args.experiments:
        wanted = set(args.experiments)
        selected_experiments = [experiment for experiment in EXPERIMENTS if experiment["name"] in wanted]
        if not selected_experiments:
            raise ValueError(f"No experiments matched: {sorted(wanted)}")

    summary_rows = []
    all_metrics = []
    for experiment in selected_experiments:
        experiment_dir = output_dir / experiment["name"]
        experiment_dir.mkdir(parents=True, exist_ok=True)

        train_artifacts = prepare_outlier_artifacts(train_split, {"validation": val_split})
        tr_enriched = train_artifacts.train_df
        val_enriched = train_artifacts.other_dfs["validation"].copy()
        global_num, global_cat, subtype_map = select_feature_sets(tr_enriched, experiment["include"])

        family = train_family_models_custom(
            tr_enriched,
            val_enriched,
            params,
            global_num,
            global_cat,
            subtype_map,
            min_region_rows,
            min_region_subtype_rows,
        )
        routing_policy = choose_policy_custom(
            family["validation_predictions"],
            family["train_group_sizes"],
            min_region_rows,
            min_region_subtype_rows,
        )
        routing_policy.to_csv(experiment_dir / "routing_policy.csv", index=False)

        test_pred = retrain_and_predict_custom(
            train_df,
            test_df,
            params,
            routing_policy,
            global_num,
            global_cat,
            subtype_map,
            min_region_rows,
            min_region_subtype_rows,
        )
        metrics_df = evaluate_experiment(test_pred, experiment["name"], experiment_dir / "price" / "hybrid_fallback")
        metrics_df.to_csv(experiment_dir / "all_metrics.csv", index=False)
        all_metrics.append(metrics_df)

        overall = metrics_df[metrics_df["level"] == "overall_test_set"].iloc[0]
        summary_rows.append(
            {
                "experiment": experiment["name"],
                "description": experiment["description"],
                "feature_bundle": ",".join(sorted(experiment["include"])) if experiment["include"] else "none",
                "count": int(overall["count"]),
                **{metric: overall[metric] for metric in list_metric_columns()},
            }
        )

        save_predictions(
            test_pred[
                [
                    col
                    for col in [
                        "id",
                        "title",
                        REGION_COL,
                        SUBTYPE_COL,
                        "price_vnd",
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
            experiment_dir / "predictions.csv",
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["Median_APE", "HitRate_20_pct"], ascending=[True, False]).reset_index(drop=True)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    pd.concat(all_metrics, ignore_index=True).to_csv(output_dir / "all_metrics.csv", index=False)
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "experiments": EXPERIMENTS,
                "selected_experiments": [experiment["name"] for experiment in selected_experiments],
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "min_region_rows": min_region_rows,
                "min_region_subtype_rows": min_region_subtype_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Saved structural ablation results to", output_dir)


if __name__ == "__main__":
    main()
