from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from common import (  # noqa: E402
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
from outlier_handling import prepare_outlier_artifacts  # noqa: E402
from train_region_subtype_models import (  # noqa: E402
    SUBTYPE_COL,
    choose_policy,
    load_robust_params,
    retrain_and_predict,
    train_family_models,
)

warnings.filterwarnings("ignore", category=UserWarning)


BANDS = [
    ("lt_2b", "<2B", None, 2_000_000_000),
    ("mid_2b_10_5b", "2B-10.5B", 2_000_000_000, 10_500_000_000),
    ("gt_10_5b", ">10.5B", 10_500_000_000, None),
]
BAND_IDS = [band[0] for band in BANDS]
BAND_LABELS = {band_id: label for band_id, label, _, _ in BANDS}
HIGH_PRICE_THRESHOLD = 10_500_000_000
START = time.time()

LEAKAGE_COLUMNS = {
    "price",
    "price_vnd",
    "price_per_m2",
    "price_per_m2_million",
    "log_price",
    "log_price_per_m2",
    "sample_weight",
    "local_ppm2_band",
    "true_regime_label",
    "predicted_regime_label",
}
STRUCTURAL_MISSING_SOURCE_COLUMNS = [
    "num_floors",
    "floor_area_m2",
    "frontage_m",
    "road_width_m",
    "width_m",
    "length_m",
]
ADDRESS_MISSING_SOURCE_COLUMNS = ["addr_last_2", "addr_last_3", "street_proxy"]


def log(message: str) -> None:
    elapsed = int(time.time() - START)
    minutes, seconds = divmod(elapsed, 60)
    print(f"[{minutes:02d}:{seconds:02d}] {message}", flush=True)


def band_mask(df: pd.DataFrame, low: int | None, high: int | None) -> pd.Series:
    if low is None:
        return df["price_vnd"] < high
    if high is None:
        return df["price_vnd"] > low
    return (df["price_vnd"] >= low) & (df["price_vnd"] <= high)


def assign_price_band(df: pd.DataFrame) -> pd.Series:
    labels = pd.Series(index=df.index, dtype="object")
    for band_id, _, low, high in BANDS:
        labels.loc[band_mask(df, low, high)] = band_id
    return labels


def add_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in STRUCTURAL_MISSING_SOURCE_COLUMNS:
        if col in out.columns:
            out[f"{col}_missing"] = out[col].isna().astype(int)
    for col in ADDRESS_MISSING_SOURCE_COLUMNS:
        if col in out.columns:
            normalized = out[col].astype("string").str.lower().fillna("missing")
            out[f"{col}_missing"] = normalized.isin(["", "missing", "nan", "none", "_"]).astype(int)
    return out


def classifier_features(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric, categorical = global_features(df)
    numeric = [col for col in numeric if col in df.columns and col not in LEAKAGE_COLUMNS and not col.startswith("pred_")]
    categorical = [col for col in categorical if col in df.columns and col not in LEAKAGE_COLUMNS and not col.startswith("pred_")]
    missing_flags = [col for col in df.columns if col.endswith("_missing")]
    numeric = numeric + [col for col in missing_flags if col not in numeric]
    return numeric, categorical


def build_band_classifier(numeric: List[str], categorical: List[str]) -> Pipeline:
    from lightgbm import LGBMClassifier

    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    model = LGBMClassifier(
        objective="multiclass",
        n_estimators=450,
        learning_rate=0.045,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


def predict_band_probabilities(model: Pipeline, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    probs = model.predict_proba(df[feature_cols])
    classes = list(model.named_steps["model"].classes_)
    out = pd.DataFrame(0.0, index=df.index, columns=[f"prob_{band_id}" for band_id in BAND_IDS])
    for i, class_name in enumerate(classes):
        out[f"prob_{class_name}"] = probs[:, i]
    return out


def split_train_validation(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stratify = train_df["price_band_id"].astype(str) + "|" + train_df[SUBTYPE_COL].astype(str)
    counts = stratify.value_counts()
    stratify = stratify.where(stratify.map(counts) >= 2)
    if stratify.isna().any():
        stratify = train_df["price_band_id"]
    return train_test_split(
        train_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )


def fit_band_policy(train_band: pd.DataFrame, params: Dict, band_dir: Path) -> pd.DataFrame:
    policy_train, policy_val = split_train_validation(train_band)
    artifacts = prepare_outlier_artifacts(policy_train, {"validation": policy_val})
    family = train_family_models(artifacts.train_df, artifacts.other_dfs["validation"].copy(), params)
    routing_policy = choose_policy(family["validation_predictions"], family["train_group_sizes"])
    routing_policy.to_csv(band_dir / "routing_policy.csv", index=False)
    return routing_policy


def train_band_models_and_predict(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    params: Dict,
    output_dir: Path,
    run_label: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    predictions = predict_df[["id", "price_vnd", "price_band_id"]].copy()
    policy_summaries = []

    for band_id, band_label, low, high in BANDS:
        band_dir = output_dir / run_label / band_id
        band_dir.mkdir(parents=True, exist_ok=True)
        train_band = train_df.loc[band_mask(train_df, low, high)].copy()
        log(f"{run_label}: training {band_label} hybrid model, train={len(train_band):,}, predict={len(predict_df):,}")
        routing_policy = fit_band_policy(train_band, params, band_dir)
        pred_band = retrain_and_predict(train_band, predict_df.copy(), params, routing_policy)
        col = f"pred_log_{band_id}"
        predictions[col] = np.log1p(np.clip(pred_band["pred_price_vnd_hybrid"].values, 1, None))
        save_predictions(
            pred_band[
                [
                    col_name
                    for col_name in [
                        "id",
                        "title",
                        "addr_last_1",
                        "addr_last_2",
                        "property_subtype",
                        "source",
                        "price_vnd",
                        "selected_model",
                        "pred_price_vnd_global",
                        "pred_price_vnd_region",
                        "pred_price_vnd_subtype",
                        "pred_price_vnd_region_subtype",
                        "pred_price_vnd_hybrid",
                    ]
                    if col_name in pred_band.columns
                ]
            ],
            band_dir / "predictions.csv",
        )
        policy_summaries.append(
            {
                "run_label": run_label,
                "price_band_id": band_id,
                "price_band_label": band_label,
                "train_rows": len(train_band),
                "routing_groups": len(routing_policy),
                "region_subtype_groups": int((routing_policy["chosen_model"] == "region_subtype").sum()),
                "region_groups": int((routing_policy["chosen_model"] == "region").sum()),
                "subtype_groups": int((routing_policy["chosen_model"] == "subtype").sum()),
                "global_groups": int((routing_policy["chosen_model"] == "global").sum()),
            }
        )

    return predictions, pd.DataFrame(policy_summaries)


def weighted_log_prediction(predictions: pd.DataFrame, probabilities: pd.DataFrame) -> pd.Series:
    weighted = np.zeros(len(predictions), dtype=float)
    for band_id in BAND_IDS:
        weighted += probabilities[f"prob_{band_id}"].values * predictions[f"pred_log_{band_id}"].values
    return pd.Series(weighted, index=predictions.index)


def hard_route_prediction(predictions: pd.DataFrame, probabilities: pd.DataFrame) -> pd.Series:
    prob_values = probabilities[[f"prob_{band_id}" for band_id in BAND_IDS]].values
    chosen_idx = np.argmax(prob_values, axis=1)
    logs = np.vstack([predictions[f"pred_log_{band_id}"].values for band_id in BAND_IDS]).T
    return pd.Series(logs[np.arange(len(logs)), chosen_idx], index=predictions.index)


def train_residual_corrector(val_df: pd.DataFrame, base_log_pred: pd.Series, probabilities: pd.DataFrame) -> Tuple[Pipeline, List[str], List[str]]:
    from lightgbm import LGBMRegressor

    residual_df = add_missing_indicators(val_df).copy()
    residual_df["base_log_pred"] = base_log_pred.values
    for col in probabilities.columns:
        residual_df[col] = probabilities[col].values
    residual_df["prob_high_or_mid"] = residual_df["prob_gt_10_5b"] + 0.35 * residual_df["prob_mid_2b_10_5b"]
    residual_df["target_residual"] = residual_df["log_price"] - residual_df["base_log_pred"]

    numeric, categorical = classifier_features(residual_df)
    numeric = numeric + ["base_log_pred", "prob_high_or_mid"] + list(probabilities.columns)
    numeric = [col for col in dict.fromkeys(numeric) if col in residual_df.columns and col != "target_residual"]
    categorical = [col for col in categorical if col in residual_df.columns]
    feature_cols = numeric + categorical

    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    model = LGBMRegressor(
        objective="regression_l1",
        n_estimators=300,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=90,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
    pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
    weights = 1.0 + 2.5 * residual_df["prob_gt_10_5b"].values
    pipe.fit(residual_df[feature_cols], residual_df["target_residual"], model__sample_weight=weights)
    return pipe, numeric, categorical


def apply_residual_corrector(
    model: Pipeline,
    numeric: List[str],
    categorical: List[str],
    df: pd.DataFrame,
    base_log_pred: pd.Series,
    probabilities: pd.DataFrame,
) -> pd.Series:
    pred_df = add_missing_indicators(df).copy()
    pred_df["base_log_pred"] = base_log_pred.values
    for col in probabilities.columns:
        pred_df[col] = probabilities[col].values
    pred_df["prob_high_or_mid"] = pred_df["prob_gt_10_5b"] + 0.35 * pred_df["prob_mid_2b_10_5b"]
    feature_cols = numeric + categorical
    adjustment = model.predict(pred_df[feature_cols])
    # Keep the calibration layer conservative; it should correct bias, not dominate the AVM.
    adjustment = np.clip(adjustment, -0.35, 0.35)
    return pd.Series(base_log_pred.values + adjustment, index=df.index)


def evaluate_prediction(
    df_eval: pd.DataFrame,
    pred_col: str,
    model_name: str,
    output_dir: Path,
) -> pd.DataFrame:
    levels = evaluate_custom_levels(
        df_eval=df_eval,
        actual_col="price_vnd",
        pred_col=pred_col,
        model_name=model_name,
        target_name="price",
        group_definitions=standard_group_definitions(df_eval),
    )
    save_level_outputs(levels, output_dir / model_name.replace(" ", "_"))
    return pd.concat(levels.values(), ignore_index=True)


def baseline_predictions(test_df: pd.DataFrame) -> pd.DataFrame:
    path = ROOT / "Evaluation" / "improvements" / "hybrid_probability_refresh_classification_fix_v2" / "predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    base = pd.read_csv(path)
    keep = [
        col
        for col in ["id", "pred_price_vnd_hybrid", "pred_price_vnd_probability_adjustment"]
        if col in base.columns
    ]
    return test_df[["id", "price_vnd"]].merge(base[keep], on="id", how="left")


def add_price_band_group(df: pd.DataFrame) -> pd.Series:
    return assign_price_band(df).map(BAND_LABELS)


def final_comparison(metrics_df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not baseline_df.empty:
        for model_name, col in [
            ("current hybrid fallback", "pred_price_vnd_hybrid"),
            ("current probability adjustment", "pred_price_vnd_probability_adjustment"),
        ]:
            if col in baseline_df.columns:
                rows.append({"model": model_name, **metric_row(baseline_df["price_vnd"], baseline_df[col])})

    for model_name in metrics_df["model"].dropna().unique():
        subset = metrics_df[(metrics_df["level"] == "overall_test_set") & (metrics_df["model"] == model_name)]
        if subset.empty:
            continue
        row = subset.iloc[0]
        rows.append({"model": model_name, **{metric: row[metric] for metric in list_metric_columns()}})
    return pd.DataFrame(rows)


def write_report(output_dir: Path, comparison: pd.DataFrame, by_band: pd.DataFrame, classifier_summary: Dict) -> None:
    best = comparison.sort_values(["Median_APE", "HitRate_20_pct"], ascending=[True, False]).iloc[0]
    try:
        comparison_table = comparison.to_markdown(index=False)
    except ImportError:
        comparison_table = "```csv\n" + comparison.to_csv(index=False) + "```"
    try:
        by_band_table = by_band.to_markdown(index=False)
    except ImportError:
        by_band_table = "```csv\n" + by_band.to_csv(index=False) + "```"

    lines = [
        "# Draft Price-Band Soft Routing Results",
        "",
        "This draft experiment adds leakage-safe price-band routing and a conservative residual correction layer.",
        "",
        "## Classifier",
        f"- Validation accuracy: {classifier_summary['validation_accuracy']:.4f}",
        f"- Test accuracy: {classifier_summary['test_accuracy']:.4f}",
        "- Classes: `<2B`, `2B-10.5B`, `>10.5B`.",
        "",
        "## Overall Comparison",
        comparison_table,
        "",
        "## Best Overall Model",
        f"- Best by Median_APE: `{best['model']}` with Median_APE={best['Median_APE']:.3f} and HitRate_20_pct={best['HitRate_20_pct']:.3f}.",
        "",
        "## By True Price Band",
        by_band_table,
        "",
        "## Caveat",
        "This draft is leakage-safe at inference because the router uses predicted band probabilities, not true test prices. The by-price-band table still uses true price only for evaluation slices.",
    ]
    (output_dir / "DRAFT_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    output_dir = ROOT / "draft" / "price_band_soft_routing"
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Writing draft outputs to {output_dir}")

    train_df, test_df = load_processed_train_test()
    train_df = add_missing_indicators(train_df)
    test_df = add_missing_indicators(test_df)
    train_df["price_band_id"] = assign_price_band(train_df)
    test_df["price_band_id"] = assign_price_band(test_df)
    train_df["price_band_label"] = train_df["price_band_id"].map(BAND_LABELS)
    test_df["price_band_label"] = test_df["price_band_id"].map(BAND_LABELS)

    dev_train, validation = split_train_validation(train_df)
    params = load_robust_params()

    numeric, categorical = classifier_features(dev_train)
    classifier_feature_cols = numeric + categorical
    classifier = build_band_classifier(numeric, categorical)
    log("Training leakage-safe price-band classifier")
    classifier.fit(dev_train[classifier_feature_cols], dev_train["price_band_id"])

    val_probs = predict_band_probabilities(classifier, validation, classifier_feature_cols)
    val_pred_band = val_probs.idxmax(axis=1).str.replace("prob_", "", regex=False)
    val_accuracy = accuracy_score(validation["price_band_id"], val_pred_band)
    val_report = classification_report(validation["price_band_id"], val_pred_band, output_dict=True)

    full_classifier = build_band_classifier(numeric, categorical)
    log("Training final price-band classifier on full train")
    full_classifier.fit(train_df[classifier_feature_cols], train_df["price_band_id"])
    test_probs = predict_band_probabilities(full_classifier, test_df, classifier_feature_cols)
    test_pred_band = test_probs.idxmax(axis=1).str.replace("prob_", "", regex=False)
    test_accuracy = accuracy_score(test_df["price_band_id"], test_pred_band)
    test_report = classification_report(test_df["price_band_id"], test_pred_band, output_dict=True)

    classifier_summary = {
        "validation_accuracy": float(val_accuracy),
        "test_accuracy": float(test_accuracy),
        "validation_confusion_matrix": confusion_matrix(validation["price_band_id"], val_pred_band, labels=BAND_IDS).tolist(),
        "test_confusion_matrix": confusion_matrix(test_df["price_band_id"], test_pred_band, labels=BAND_IDS).tolist(),
        "validation_report": val_report,
        "test_report": test_report,
        "classifier_features": classifier_feature_cols,
    }
    (output_dir / "classifier_summary.json").write_text(json.dumps(classifier_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    log("Training dev band-specific hybrid models for validation residuals")
    val_band_predictions, dev_policy_summary = train_band_models_and_predict(dev_train, validation, params, output_dir, "dev_validation_models")
    val_soft_log = weighted_log_prediction(val_band_predictions, val_probs)
    val_hard_log = hard_route_prediction(val_band_predictions, val_probs)

    log("Training residual correction layer from validation residuals")
    residual_model, residual_num, residual_cat = train_residual_corrector(validation, val_soft_log, val_probs)

    log("Training full band-specific hybrid models for test predictions")
    test_band_predictions, full_policy_summary = train_band_models_and_predict(train_df, test_df, params, output_dir, "full_test_models")
    test_soft_log = weighted_log_prediction(test_band_predictions, test_probs)
    test_hard_log = hard_route_prediction(test_band_predictions, test_probs)
    test_residual_log = apply_residual_corrector(residual_model, residual_num, residual_cat, test_df, test_soft_log, test_probs)

    pred = test_df[
        [
            col
            for col in [
                "id",
                "title",
                "addr_last_1",
                "addr_last_2",
                "addr_last_3",
                "street_proxy",
                "property_subtype",
                "source",
                "price_vnd",
                "price_per_m2",
                "area_m2",
                "num_floors",
                "floor_area_m2",
                "frontage_m",
                "road_width_m",
                "width_m",
                "length_m",
                "price_band_id",
                "price_band_label",
            ]
            if col in test_df.columns
        ]
    ].copy()
    for band_id in BAND_IDS:
        pred[f"prob_{band_id}"] = test_probs[f"prob_{band_id}"].values
        pred[f"pred_price_vnd_{band_id}_hybrid"] = np.expm1(test_band_predictions[f"pred_log_{band_id}"].values)
    pred["predicted_price_band_id"] = test_pred_band.values
    pred["predicted_price_band_label"] = pred["predicted_price_band_id"].map(BAND_LABELS)
    pred["pred_price_vnd_price_band_hard_route"] = np.expm1(test_hard_log.values)
    pred["pred_price_vnd_price_band_soft_route"] = np.expm1(test_soft_log.values)
    pred["pred_price_vnd_price_band_soft_residual"] = np.expm1(test_residual_log.values)
    save_predictions(pred, output_dir / "predictions.csv")

    metrics = []
    for model_name, pred_col in [
        ("draft price-band hard route", "pred_price_vnd_price_band_hard_route"),
        ("draft price-band soft route", "pred_price_vnd_price_band_soft_route"),
        ("draft price-band soft route + residual correction", "pred_price_vnd_price_band_soft_residual"),
    ]:
        metrics.append(evaluate_prediction(pred, pred_col, model_name, output_dir / "metrics"))
    metrics_df = pd.concat(metrics, ignore_index=True)
    metrics_df.to_csv(output_dir / "all_metrics.csv", index=False)

    baseline_df = baseline_predictions(test_df)
    comparison = final_comparison(metrics_df, baseline_df)
    comparison.to_csv(output_dir / "final_comparison.csv", index=False)

    by_band_rows = []
    if not baseline_df.empty:
        baseline_join = test_df[["id", "price_band_label"]].merge(baseline_df, on="id", how="left")
        for band_label, subset in baseline_join.groupby("price_band_label"):
            for model_name, pred_col in [
                ("current hybrid fallback", "pred_price_vnd_hybrid"),
                ("current probability adjustment", "pred_price_vnd_probability_adjustment"),
            ]:
                if pred_col in subset.columns:
                    by_band_rows.append({"price_band_label": band_label, "model": model_name, **metric_row(subset["price_vnd"], subset[pred_col])})
    for band_label, subset in pred.groupby("price_band_label"):
        for model_name, pred_col in [
            ("draft price-band hard route", "pred_price_vnd_price_band_hard_route"),
            ("draft price-band soft route", "pred_price_vnd_price_band_soft_route"),
            ("draft price-band soft route + residual correction", "pred_price_vnd_price_band_soft_residual"),
        ]:
            by_band_rows.append({"price_band_label": band_label, "model": model_name, **metric_row(subset["price_vnd"], subset[pred_col])})
    by_band = pd.DataFrame(by_band_rows)
    by_band.to_csv(output_dir / "comparison_by_true_price_band.csv", index=False)

    pd.concat([dev_policy_summary, full_policy_summary], ignore_index=True).to_csv(output_dir / "band_policy_summary.csv", index=False)
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "bands": [{"price_band_id": band_id, "label": label, "low": low, "high": high} for band_id, label, low, high in BANDS],
                "note": "Draft leakage-safe price-band routing. Classifier and residual correction use non-price features and predicted band probabilities; true price bands are used only for training labels/evaluation.",
                "residual_features_numeric": residual_num,
                "residual_features_categorical": residual_cat,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(output_dir, comparison, by_band, classifier_summary)
    log(f"Saved draft experiment to {output_dir}")


if __name__ == "__main__":
    main()
