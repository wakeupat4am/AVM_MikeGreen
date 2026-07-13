from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

from common import (
    EVAL_DIR,
    RANDOM_STATE,
    evaluate_custom_levels,
    global_features,
    list_metric_columns,
    load_processed_train_test,
    metric_row,
    save_level_outputs,
    save_predictions,
    standard_group_definitions,
    subtype_features,
)
from outlier_handling import prepare_outlier_artifacts


MODEL_DIR = EVAL_DIR / "improvements" / "catboost_region_subtype_models"
MIN_REGION_ROWS = 500
MIN_REGION_SUBTYPE_ROWS = 2000
REGION_COL = "addr_last_1"
SUBTYPE_COL = "property_subtype"
NUM_FLOORS_BUNDLE = {"num_floors", "floor_area_m2", "floors_x_area", "floors_x_frontage", "floors_x_business"}

CATBOOST_PARAMS: Dict = {
    "iterations": 600,
    "learning_rate": 0.03,
    "depth": 6,
    "loss_function": "MAE",
    "l2_leaf_reg": 3.0,
    "min_data_in_leaf": 20,
    "random_seed": RANDOM_STATE,
    "thread_count": -1,
    "verbose": 0,
}


class CatBoostModel:
    """Thin wrapper so callers get a (model, feature_cols) interface identical to LightGBM."""

    def __init__(self, numeric_features: List[str], categorical_features: List[str], params: Dict):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.feature_cols = numeric_features + categorical_features
        self._cat_idx = [self.feature_cols.index(c) for c in categorical_features]
        self._model = CatBoostRegressor(**params)

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.feature_cols].copy()
        for c in self.categorical_features:
            X[c] = X[c].fillna("__NA__").astype(str)
        for c in self.numeric_features:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        return X

    def fit(self, df: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray | None = None):
        X = self._prepare(df)
        pool = Pool(X, label=y.values, cat_features=self._cat_idx, weight=sample_weight)
        self._model.fit(pool)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._prepare(df)
        return self._model.predict(X)


def non_leakage_global_features(df: pd.DataFrame, exclude_num_floors_bundle: bool = False) -> Tuple[List[str], List[str]]:
    from train_region_subtype_models import LEAKAGE_COLUMNS
    numeric, categorical = global_features(df)
    numeric = [col for col in numeric if not (exclude_num_floors_bundle and col in NUM_FLOORS_BUNDLE)]
    return (
        [col for col in numeric if col not in LEAKAGE_COLUMNS and not col.startswith("pred_")],
        [col for col in categorical if col not in LEAKAGE_COLUMNS and not col.startswith("pred_")],
    )


def non_leakage_subtype_features(df: pd.DataFrame, exclude_num_floors_bundle: bool = False) -> Dict[str, Dict[str, List[str]]]:
    from train_region_subtype_models import LEAKAGE_COLUMNS
    feature_map = subtype_features(df)
    cleaned = {}
    for subtype, feature_dict in feature_map.items():
        cleaned[subtype] = {
            "numeric": [
                col for col in feature_dict["numeric"]
                if col not in LEAKAGE_COLUMNS and not col.startswith("pred_")
                and not (exclude_num_floors_bundle and col in NUM_FLOORS_BUNDLE)
            ],
            "categorical": [
                col for col in feature_dict["categorical"]
                if col not in LEAKAGE_COLUMNS and not col.startswith("pred_")
            ],
        }
    return cleaned


def fit_model(
    train_df: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    params: Dict,
) -> Tuple[CatBoostModel, List[str]]:
    model = CatBoostModel(numeric_features, categorical_features, params)
    model.fit(train_df, train_df["log_price"], sample_weight=train_df["sample_weight"].values)
    return model, model.feature_cols


def predict_with_model(model: CatBoostModel, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    return np.expm1(model.predict(df))


def median_ape_for_slice(df_eval: pd.DataFrame, pred_col: str, mask: pd.Series) -> float:
    subset = df_eval.loc[mask]
    metrics = metric_row(subset["price_vnd"], subset[pred_col])
    return float(metrics["Median_APE"]) if pd.notna(metrics["Median_APE"]) else np.inf


def train_family_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    params: Dict,
    exclude_num_floors_bundle: bool = False,
):
    global_num, global_cat = non_leakage_global_features(train_df, exclude_num_floors_bundle=exclude_num_floors_bundle)
    subtype_feature_map = non_leakage_subtype_features(train_df, exclude_num_floors_bundle=exclude_num_floors_bundle)

    global_model, global_cols = fit_model(train_df, global_num, global_cat, params)
    val_df["pred_price_vnd_global"] = predict_with_model(global_model, val_df, global_cols)

    region_models = {}
    val_df["pred_price_vnd_region"] = np.nan
    for region, region_train in train_df.groupby(REGION_COL):
        if len(region_train) < MIN_REGION_ROWS:
            continue
        region_model, region_cols = fit_model(region_train, global_num, global_cat, params)
        region_models[region] = {"model": region_model, "feature_cols": region_cols, "rows": len(region_train)}
        region_mask = val_df[REGION_COL] == region
        if region_mask.any():
            val_df.loc[region_mask, "pred_price_vnd_region"] = predict_with_model(region_model, val_df.loc[region_mask], region_cols)

    subtype_models = {}
    val_df["pred_price_vnd_subtype"] = np.nan
    for subtype, subtype_train in train_df.groupby(SUBTYPE_COL):
        if len(subtype_train) < MIN_REGION_ROWS:
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
        if row.train_rows < MIN_REGION_SUBTYPE_ROWS:
            continue
        mask = (train_df[REGION_COL] == row.addr_last_1) & (train_df[SUBTYPE_COL] == row.property_subtype)
        group_train = train_df.loc[mask]
        feature_dict = subtype_feature_map.get(row.property_subtype, {"numeric": global_num, "categorical": global_cat})
        group_model, group_cols = fit_model(group_train, feature_dict["numeric"], feature_dict["categorical"], params)
        region_subtype_models[(row.addr_last_1, row.property_subtype)] = {
            "model": group_model, "feature_cols": group_cols, "rows": int(row.train_rows),
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


def choose_policy(validation_predictions: pd.DataFrame, train_group_sizes: pd.DataFrame) -> pd.DataFrame:
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

        if train_rows < MIN_REGION_ROWS:
            chosen_model = "global"
            fallback_model = "global"
        elif train_rows < MIN_REGION_SUBTYPE_ROWS:
            fallback_model = "region" if "region" in candidate_scores else "global"
            chosen_model = fallback_model
            if "subtype" in candidate_scores and candidate_scores["subtype"] < candidate_scores[chosen_model]:
                chosen_model = "subtype"
        else:
            fallback_model = "region" if "region" in candidate_scores else "global"
            chosen_model = fallback_model
            if "subtype" in candidate_scores and candidate_scores["subtype"] < candidate_scores[chosen_model]:
                chosen_model = "subtype"
            if "region_subtype" in candidate_scores and candidate_scores["region_subtype"] < candidate_scores[chosen_model]:
                chosen_model = "region_subtype"

        rows.append({
            REGION_COL: region, SUBTYPE_COL: subtype, "train_rows": train_rows,
            "fallback_model": fallback_model, "chosen_model": chosen_model,
            "global_median_ape": candidate_scores.get("global"),
            "region_median_ape": candidate_scores.get("region"),
            "subtype_median_ape": candidate_scores.get("subtype"),
            "region_subtype_median_ape": candidate_scores.get("region_subtype"),
        })
    return pd.DataFrame(rows)


def retrain_and_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: Dict,
    routing_policy: pd.DataFrame,
    exclude_num_floors_bundle: bool = False,
) -> pd.DataFrame:
    train_artifacts = prepare_outlier_artifacts(train_df, {"test": test_df})
    tr = train_artifacts.train_df
    te = train_artifacts.other_dfs["test"].copy()
    global_num, global_cat = non_leakage_global_features(tr, exclude_num_floors_bundle=exclude_num_floors_bundle)
    subtype_feature_map = non_leakage_subtype_features(tr, exclude_num_floors_bundle=exclude_num_floors_bundle)

    global_model, global_cols = fit_model(tr, global_num, global_cat, params)
    te["pred_price_vnd_global"] = predict_with_model(global_model, te, global_cols)

    te["pred_price_vnd_region"] = np.nan
    for region, region_train in tr.groupby(REGION_COL):
        if len(region_train) < MIN_REGION_ROWS:
            continue
        region_model, region_cols = fit_model(region_train, global_num, global_cat, params)
        region_mask = te[REGION_COL] == region
        if region_mask.any():
            te.loc[region_mask, "pred_price_vnd_region"] = predict_with_model(region_model, te.loc[region_mask], region_cols)

    te["pred_price_vnd_subtype"] = np.nan
    for subtype, subtype_train in tr.groupby(SUBTYPE_COL):
        if len(subtype_train) < MIN_REGION_ROWS:
            continue
        feature_dict = subtype_feature_map.get(subtype, {"numeric": global_num, "categorical": global_cat})
        subtype_model, subtype_cols = fit_model(subtype_train, feature_dict["numeric"], feature_dict["categorical"], params)
        subtype_mask = te[SUBTYPE_COL] == subtype
        if subtype_mask.any():
            te.loc[subtype_mask, "pred_price_vnd_subtype"] = predict_with_model(subtype_model, te.loc[subtype_mask], subtype_cols)

    te["pred_price_vnd_region_subtype"] = np.nan
    for row in routing_policy.itertuples(index=False):
        if row.train_rows < MIN_REGION_SUBTYPE_ROWS:
            continue
        mask = (tr[REGION_COL] == row.addr_last_1) & (tr[SUBTYPE_COL] == row.property_subtype)
        group_train = tr.loc[mask]
        feature_dict = subtype_feature_map.get(row.property_subtype, {"numeric": global_num, "categorical": global_cat})
        group_model, group_cols = fit_model(group_train, feature_dict["numeric"], feature_dict["categorical"], params)
        test_mask = (te[REGION_COL] == row.addr_last_1) & (te[SUBTYPE_COL] == row.property_subtype)
        if test_mask.any():
            te.loc[test_mask, "pred_price_vnd_region_subtype"] = predict_with_model(group_model, te.loc[test_mask], group_cols)

    policy_lookup = routing_policy.set_index([REGION_COL, SUBTYPE_COL])["chosen_model"].to_dict()
    te["selected_model"] = te.apply(lambda r: policy_lookup.get((r[REGION_COL], r[SUBTYPE_COL]), "global"), axis=1)
    te["pred_price_vnd_hybrid"] = te["pred_price_vnd_global"]
    te.loc[te["selected_model"] == "region", "pred_price_vnd_hybrid"] = (
        te.loc[te["selected_model"] == "region", "pred_price_vnd_region"]
        .fillna(te.loc[te["selected_model"] == "region", "pred_price_vnd_global"])
    )
    te.loc[te["selected_model"] == "subtype", "pred_price_vnd_hybrid"] = (
        te.loc[te["selected_model"] == "subtype", "pred_price_vnd_subtype"]
        .fillna(te.loc[te["selected_model"] == "subtype", "pred_price_vnd_global"])
    )
    te.loc[te["selected_model"] == "region_subtype", "pred_price_vnd_hybrid"] = (
        te.loc[te["selected_model"] == "region_subtype", "pred_price_vnd_region_subtype"]
        .fillna(te.loc[te["selected_model"] == "region_subtype", "pred_price_vnd_region"])
        .fillna(te.loc[te["selected_model"] == "region_subtype", "pred_price_vnd_global"])
    )
    return te


def evaluate_family(df_eval: pd.DataFrame, pred_col: str, model_name: str, out_dir: Path) -> pd.DataFrame:
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

    train_split, val_split = train_test_split(
        train_df, test_size=0.2, random_state=RANDOM_STATE, stratify=train_df[SUBTYPE_COL],
    )
    artifacts = prepare_outlier_artifacts(train_split, {"validation": val_split})
    tr_enriched = artifacts.train_df
    val_enriched = artifacts.other_dfs["validation"]

    family = train_family_models(tr_enriched, val_enriched.copy(), params, exclude_num_floors_bundle=args.exclude_num_floors_bundle)
    routing_policy = choose_policy(family["validation_predictions"], family["train_group_sizes"])
    routing_policy.to_csv(output_dir / "routing_policy.csv", index=False)

    test_pred = retrain_and_predict(train_df, test_df, params, routing_policy, exclude_num_floors_bundle=args.exclude_num_floors_bundle)
    family_metrics = []
    family_metrics.append(evaluate_family(test_pred, "pred_price_vnd_global", "global model", output_dir / "price" / "global_model"))
    if test_pred["pred_price_vnd_region"].notna().any():
        family_metrics.append(evaluate_family(test_pred, "pred_price_vnd_region", "region model", output_dir / "price" / "region_model"))
    if test_pred["pred_price_vnd_subtype"].notna().any():
        family_metrics.append(evaluate_family(test_pred, "pred_price_vnd_subtype", "subtype model", output_dir / "price" / "subtype_model"))
    if test_pred["pred_price_vnd_region_subtype"].notna().any():
        family_metrics.append(evaluate_family(test_pred, "pred_price_vnd_region_subtype", "region subtype model", output_dir / "price" / "region_subtype_model"))
    family_metrics.append(evaluate_family(test_pred, "pred_price_vnd_hybrid", "hybrid fallback", output_dir / "price" / "hybrid_fallback"))
    family_metrics_df = pd.concat(family_metrics, ignore_index=True)
    family_metrics_df.to_csv(output_dir / "all_metrics.csv", index=False)

    save_predictions(
        test_pred[[c for c in [
            "id", "title", REGION_COL, SUBTYPE_COL, "source", "price_vnd", "price_per_m2",
            "local_ppm2_band", "sample_weight", "selected_model",
            "pred_price_vnd_global", "pred_price_vnd_region", "pred_price_vnd_subtype",
            "pred_price_vnd_region_subtype", "pred_price_vnd_hybrid",
        ] if c in test_pred.columns]],
        output_dir / "predictions.csv",
    )
    (output_dir / "config.json").write_text(
        json.dumps({"params": params, "exclude_num_floors_bundle": args.exclude_num_floors_bundle}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Saved CatBoost region/subtype model results to", output_dir)


if __name__ == "__main__":
    main()
