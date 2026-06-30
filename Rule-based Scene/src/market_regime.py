from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import RANDOM_STATE, metric_row
from outlier_handling import HIERARCHY, QUANTILE_COLUMNS, prepare_outlier_artifacts
from train_robust_lightgbm import LEAKAGE_COLUMNS


REGIME_CLASSES = ["low_tail", "normal", "high_tail", "stable_normal"]
REGIME_CLASSES_LOGLOSS = sorted(REGIME_CLASSES)
DEFAULT_THRESHOLD_GRID = [
    {"minimum_group_size": 200, "dispersion_ratio_min": 1.6, "iqr_ratio_min": 1.15, "dispersion_log_min": 0.35},
    {"minimum_group_size": 500, "dispersion_ratio_min": 1.8, "iqr_ratio_min": 1.20, "dispersion_log_min": 0.45},
    {"minimum_group_size": 500, "dispersion_ratio_min": 2.0, "iqr_ratio_min": 1.25, "dispersion_log_min": 0.55},
    {"minimum_group_size": 1000, "dispersion_ratio_min": 2.0, "iqr_ratio_min": 1.30, "dispersion_log_min": 0.55},
    {"minimum_group_size": 1000, "dispersion_ratio_min": 2.2, "iqr_ratio_min": 1.35, "dispersion_log_min": 0.65},
]
REGIME_NUMERIC_FEATURES = [
    "area_m2",
    "width_m",
    "length_m",
    "frontage_m",
    "road_width_m",
    "num_floors",
    "bedrooms",
    "bathrooms",
    "floor_area_m2",
    "listing_year",
    "listing_month",
    "is_business_text",
    "is_car_access_text",
    "is_frontage_text",
    "is_alley_text",
    "has_legal_text",
    "is_project_text",
    "is_urban_area_text",
    "is_new_house_text",
    "is_renovated_text",
]
REGIME_CATEGORICAL_FEATURES = [
    "legal",
    "door_direction",
    "source",
    "property_subtype",
    "addr_last_1",
    "addr_last_2",
    "addr_last_3",
    "street_proxy",
    "project_name",
]


@dataclass
class RegimeArtifacts:
    train_df: pd.DataFrame
    other_dfs: Dict[str, pd.DataFrame]
    group_tables: Dict[str, pd.DataFrame]
    thresholds: Dict[str, float]


def regime_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric = [col for col in REGIME_NUMERIC_FEATURES if col in df.columns and col not in LEAKAGE_COLUMNS]
    categorical = [col for col in REGIME_CATEGORICAL_FEATURES if col in df.columns and col not in LEAKAGE_COLUMNS]
    return numeric, categorical


def build_regime_classifier_pipeline(numeric_features: List[str], categorical_features: List[str]) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=25)),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )
    model = LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def build_adjustment_pipeline(numeric_features: List[str], categorical_features: List[str]) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=25)),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocess", preprocess), ("model", Ridge(alpha=3.0, random_state=RANDOM_STATE))])


def build_group_table(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if group_cols:
        quantiles = df.groupby(group_cols)["price_per_m2"].quantile([0.05, 0.25, 0.75, 0.95]).unstack().reset_index()
        quantiles = quantiles.rename(columns={0.05: "Q05", 0.25: "Q25", 0.75: "Q75", 0.95: "Q95"})
        counts = df.groupby(group_cols).size().rename("group_size").reset_index()
        table = quantiles.merge(counts, on=group_cols, how="left")
    else:
        q = df["price_per_m2"].quantile([0.05, 0.25, 0.75, 0.95])
        table = pd.DataFrame(
            [{"Q05": q.loc[0.05], "Q25": q.loc[0.25], "Q75": q.loc[0.75], "Q95": q.loc[0.95], "group_size": int(len(df)), "__global_key": "global"}]
        )
    table["dispersion_ratio"] = table["Q95"] / table["Q05"]
    table["iqr_ratio"] = table["Q75"] / table["Q25"]
    table["dispersion_log"] = np.log(table["Q95"]) - np.log(table["Q05"])
    return table


def fit_market_group_tables(train_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    train_valid = train_df.loc[train_df["price_per_m2"].notna() & (train_df["price_per_m2"] > 0)].copy()
    return {level_name: build_group_table(train_valid, group_cols) for level_name, group_cols in HIERARCHY}


def _merge_level(df: pd.DataFrame, table: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    working = df.copy()
    if not group_cols:
        working["__global_key"] = "global"
        group_cols = ["__global_key"]
    merged = working[group_cols].merge(table, on=group_cols, how="left")
    merged.index = df.index
    return merged


def attach_market_group_stats(df: pd.DataFrame, group_tables: Dict[str, pd.DataFrame], thresholds: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for col in ["Q05_regime", "Q25_regime", "Q75_regime", "Q95_regime", "group_size", "dispersion_ratio", "iqr_ratio", "dispersion_log"]:
        out[col] = np.nan
    out["effective_group_level"] = "missing"

    for level_name, group_cols in HIERARCHY:
        merged = _merge_level(out, group_tables[level_name], group_cols)
        eligible = merged["group_size"].fillna(0) >= thresholds["minimum_group_size"]
        missing = out["effective_group_level"] == "missing"
        mask = missing & eligible
        if mask.any():
            out.loc[mask, "Q05_regime"] = merged.loc[mask, "Q05"]
            out.loc[mask, "Q25_regime"] = merged.loc[mask, "Q25"]
            out.loc[mask, "Q75_regime"] = merged.loc[mask, "Q75"]
            out.loc[mask, "Q95_regime"] = merged.loc[mask, "Q95"]
            out.loc[mask, "group_size"] = merged.loc[mask, "group_size"]
            out.loc[mask, "dispersion_ratio"] = merged.loc[mask, "dispersion_ratio"]
            out.loc[mask, "iqr_ratio"] = merged.loc[mask, "iqr_ratio"]
            out.loc[mask, "dispersion_log"] = merged.loc[mask, "dispersion_log"]
            out.loc[mask, "effective_group_level"] = level_name

    if (out["effective_group_level"] == "missing").any():
        global_table = group_tables["global"].iloc[0]
        mask = out["effective_group_level"] == "missing"
        out.loc[mask, "Q05_regime"] = global_table["Q05"]
        out.loc[mask, "Q25_regime"] = global_table["Q25"]
        out.loc[mask, "Q75_regime"] = global_table["Q75"]
        out.loc[mask, "Q95_regime"] = global_table["Q95"]
        out.loc[mask, "group_size"] = global_table["group_size"]
        out.loc[mask, "dispersion_ratio"] = global_table["dispersion_ratio"]
        out.loc[mask, "iqr_ratio"] = global_table["iqr_ratio"]
        out.loc[mask, "dispersion_log"] = global_table["dispersion_log"]
        out.loc[mask, "effective_group_level"] = "global"
    return out


def label_market_regimes(df: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    stable_mask = (
        (out["dispersion_ratio"] < thresholds["dispersion_ratio_min"])
        | (out["iqr_ratio"] < thresholds["iqr_ratio_min"])
        | (out["dispersion_log"] < thresholds["dispersion_log_min"])
    )
    out["group_dispersion_type"] = np.where(
        stable_mask,
        "stable_group",
        np.where(out["dispersion_ratio"] >= thresholds["dispersion_ratio_min"] * 1.5, "high_dispersion_group", "normal_dispersion_group"),
    )
    out["dispersion_score"] = (
        np.log(out["dispersion_ratio"].clip(lower=1.0))
        + np.log(out["iqr_ratio"].clip(lower=1.0))
        + out["dispersion_log"].clip(lower=0)
    )
    out["true_regime_label"] = np.where(
        stable_mask,
        "stable_normal",
        np.where(
            out["price_per_m2"] < out["Q05_regime"],
            "low_tail",
            np.where(out["price_per_m2"] > out["Q95_regime"], "high_tail", "normal"),
        ),
    )
    return out


def fit_regime_artifacts(train_df: pd.DataFrame, other_dfs: Dict[str, pd.DataFrame], thresholds: Dict[str, float]) -> RegimeArtifacts:
    outlier_artifacts = prepare_outlier_artifacts(train_df, other_dfs)
    group_tables = fit_market_group_tables(outlier_artifacts.train_df)

    train_regime = attach_market_group_stats(outlier_artifacts.train_df, group_tables, thresholds)
    train_regime = label_market_regimes(train_regime, thresholds)

    enriched_others: Dict[str, pd.DataFrame] = {}
    for name, df_other in outlier_artifacts.other_dfs.items():
        df_regime = attach_market_group_stats(df_other, group_tables, thresholds)
        df_regime = label_market_regimes(df_regime, thresholds)
        enriched_others[name] = df_regime

    return RegimeArtifacts(train_df=train_regime, other_dfs=enriched_others, group_tables=group_tables, thresholds=thresholds)


def classifier_metrics(y_true: pd.Series, prob_df: pd.DataFrame) -> Dict[str, float]:
    preds = prob_df.idxmax(axis=1)
    return {
        "accuracy": accuracy_score(y_true, preds),
        "macro_f1": f1_score(y_true, preds, average="macro"),
        "log_loss": log_loss(y_true, prob_df[REGIME_CLASSES_LOGLOSS], labels=REGIME_CLASSES_LOGLOSS),
    }


def train_regime_classifier(train_df: pd.DataFrame):
    numeric_features, categorical_features = regime_feature_columns(train_df)
    feature_cols = numeric_features + categorical_features
    model = build_regime_classifier_pipeline(numeric_features, categorical_features)
    model.fit(train_df[feature_cols], train_df["true_regime_label"])
    return model, feature_cols


def predict_regime_probabilities(model, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    probs = model.predict_proba(df[feature_cols])
    classes = list(model.named_steps["model"].classes_)
    out = pd.DataFrame(probs, index=df.index, columns=[f"prob_{cls}" for cls in classes])
    for cls in REGIME_CLASSES:
        col = f"prob_{cls}"
        if col not in out.columns:
            out[col] = 0.0
    return out[[f"prob_{cls}" for cls in REGIME_CLASSES]]


def evaluate_threshold_config(train_df: pd.DataFrame, val_df: pd.DataFrame, thresholds: Dict[str, float]) -> Dict[str, float]:
    artifacts = fit_regime_artifacts(train_df, {"validation": val_df}, thresholds)
    classifier, feature_cols = train_regime_classifier(artifacts.train_df)
    prob_df = predict_regime_probabilities(classifier, artifacts.other_dfs["validation"], feature_cols)
    prob_df.columns = [col.replace("prob_", "") for col in prob_df.columns]
    metrics = classifier_metrics(artifacts.other_dfs["validation"]["true_regime_label"], prob_df)
    label_balance = artifacts.train_df["true_regime_label"].value_counts(normalize=True).to_dict()
    return {
        **thresholds,
        **metrics,
        "train_rows": len(artifacts.train_df),
        "validation_rows": len(artifacts.other_dfs["validation"]),
        "stable_share": label_balance.get("stable_normal", 0.0),
        "low_share": label_balance.get("low_tail", 0.0),
        "normal_share": label_balance.get("normal", 0.0),
        "high_share": label_balance.get("high_tail", 0.0),
    }


def choose_thresholds(train_df: pd.DataFrame, val_df: pd.DataFrame, threshold_grid: List[Dict[str, float]] | None = None) -> Tuple[Dict[str, float], pd.DataFrame]:
    threshold_grid = threshold_grid or DEFAULT_THRESHOLD_GRID
    rows = [evaluate_threshold_config(train_df, val_df, threshold_cfg) for threshold_cfg in threshold_grid]
    tuning_df = pd.DataFrame(rows).sort_values(["macro_f1", "log_loss", "accuracy"], ascending=[False, True, False]).reset_index(drop=True)
    best = tuning_df.iloc[0][["minimum_group_size", "dispersion_ratio_min", "iqr_ratio_min", "dispersion_log_min"]].to_dict()
    return best, tuning_df


def append_probabilities(df: pd.DataFrame, prob_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in prob_df.columns:
        out[col] = prob_df[col].values
    out["predicted_regime_label"] = prob_df.idxmax(axis=1).str.replace("prob_", "", regex=False)
    return out


def hard_group_training_sets(train_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    dispersive = train_df["group_dispersion_type"].isin(["normal_dispersion_group", "high_dispersion_group"])
    low = train_df.loc[dispersive & (train_df["true_regime_label"] == "low_tail")].copy()
    normal = train_df.loc[(train_df["true_regime_label"] == "normal")].copy()
    high = train_df.loc[dispersive & (train_df["true_regime_label"] == "high_tail")].copy()
    stable = train_df.loc[train_df["true_regime_label"] == "stable_normal"].copy()
    trimmed_local = train_df.loc[~(dispersive & train_df["true_regime_label"].isin(["low_tail", "high_tail"]))].copy()
    return {
        "low_tail_group": low,
        "normal_group": normal,
        "high_tail_group": high,
        "stable_normal_group": stable,
        "trimmed_local_group": trimmed_local,
    }


def adjustment_feature_frames(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    out = pd.DataFrame(index=df.index)
    numeric = [
        "prob_low_tail",
        "prob_normal",
        "prob_high_tail",
        "prob_stable_normal",
        "dispersion_score",
        "base_pred_log_price",
    ]
    categorical = ["group_dispersion_type", "addr_last_1", "property_subtype"]
    for col in numeric + categorical:
        out[col] = df[col] if col in df.columns else np.nan
    return out, numeric, categorical


def train_adjustment_model(train_df: pd.DataFrame):
    feat_df, numeric_features, categorical_features = adjustment_feature_frames(train_df)
    model = build_adjustment_pipeline(numeric_features, categorical_features)
    model.fit(feat_df[numeric_features + categorical_features], train_df["log_price"] - train_df["base_pred_log_price"])
    return model, numeric_features + categorical_features


def apply_adjustment_model(model, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    feat_df, _, _ = adjustment_feature_frames(df)
    adjustment = model.predict(feat_df[feature_cols])
    return adjustment


def regime_level_groups(df: pd.DataFrame) -> Dict[str, pd.Series]:
    return {
        "by_regime_true": df["true_regime_label"],
        "by_regime_predicted": df["predicted_regime_label"],
        "by_dispersion_type": df["group_dispersion_type"],
    }
