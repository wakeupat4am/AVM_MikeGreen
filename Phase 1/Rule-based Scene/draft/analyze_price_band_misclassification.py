from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DRAFT_DIR = ROOT / "draft" / "price_band_soft_routing"
OUTPUT_DIR = DRAFT_DIR / "misclassification_analysis"

BAND_ORDER = ["lt_2b", "mid_2b_10_5b", "gt_10_5b"]
PROB_COLS = ["prob_lt_2b", "prob_mid_2b_10_5b", "prob_gt_10_5b"]
PREDICTION_COLS = [
    "pred_price_vnd_price_band_hard_route",
    "pred_price_vnd_price_band_soft_route",
    "pred_price_vnd_price_band_soft_residual",
    "pred_price_vnd_current_hybrid",
    "pred_price_vnd_current_probability_adjustment",
]
NUMERIC_FEATURES = [
    "area_m2",
    "num_floors",
    "floor_area_m2",
    "frontage_m",
    "road_width_m",
    "width_m",
    "length_m",
    "price_per_m2",
]
CATEGORICAL_FEATURES = [
    "addr_last_1",
    "addr_last_2",
    "addr_last_3",
    "street_proxy",
    "property_subtype",
    "source",
]
MISSING_SOURCE_COLS = [
    "num_floors",
    "floor_area_m2",
    "frontage_m",
    "road_width_m",
    "width_m",
    "length_m",
    "addr_last_2",
    "addr_last_3",
    "street_proxy",
]


def metric_row(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y = y_true.astype(float).to_numpy()
    p = y_pred.astype(float).to_numpy()
    mask = np.isfinite(y) & np.isfinite(p) & (y > 0)
    y = y[mask]
    p = p[mask]
    if len(y) == 0:
        return {
            "count": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "MAPE": np.nan,
            "Median_APE": np.nan,
            "HitRate_10_pct": np.nan,
            "HitRate_20_pct": np.nan,
            "Bias_pct": np.nan,
        }
    pe = (p - y) / y
    ape = np.abs(pe)
    return {
        "count": int(len(y)),
        "MAE": float(np.mean(np.abs(p - y))),
        "RMSE": float(np.sqrt(np.mean((p - y) ** 2))),
        "MAPE": float(np.mean(ape) * 100),
        "Median_APE": float(np.median(ape) * 100),
        "HitRate_10_pct": float(np.mean(ape <= 0.10) * 100),
        "HitRate_20_pct": float(np.mean(ape <= 0.20) * 100),
        "Bias_pct": float(np.mean(pe) * 100),
    }


def add_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in PREDICTION_COLS:
        if col not in out.columns:
            continue
        out[f"{col}_ape_pct"] = ((out[col] - out["price_vnd"]).abs() / out["price_vnd"]) * 100
        out[f"{col}_signed_pct"] = ((out[col] - out["price_vnd"]) / out["price_vnd"]) * 100
    out["router_is_correct"] = out["price_band_id"] == out["predicted_price_band_id"]
    out["router_confidence"] = out[PROB_COLS].max(axis=1)
    out["true_band_probability"] = out.apply(lambda row: row.get(f"prob_{row['price_band_id']}", np.nan), axis=1)
    out["predicted_band_probability"] = out.apply(lambda row: row.get(f"prob_{row['predicted_price_band_id']}", np.nan), axis=1)
    out["probability_margin"] = out["predicted_band_probability"] - out["true_band_probability"]
    for col in MISSING_SOURCE_COLS:
        if col not in out.columns:
            continue
        if out[col].dtype == "object":
            normalized = out[col].astype("string").str.lower().fillna("missing")
            out[f"{col}_missing"] = normalized.isin(["", "missing", "nan", "none", "_"]).astype(int)
        else:
            out[f"{col}_missing"] = out[col].isna().astype(int)
    return out


def load_predictions() -> pd.DataFrame:
    draft = pd.read_csv(DRAFT_DIR / "predictions.csv")
    current_path = ROOT / "Evaluation" / "improvements" / "hybrid_probability_refresh_classification_fix_v2" / "predictions.csv"
    if current_path.exists():
        current = pd.read_csv(
            current_path,
            usecols=["id", "pred_price_vnd_hybrid", "pred_price_vnd_probability_adjustment"],
        ).rename(
            columns={
                "pred_price_vnd_hybrid": "pred_price_vnd_current_hybrid",
                "pred_price_vnd_probability_adjustment": "pred_price_vnd_current_probability_adjustment",
            }
        )
        draft = draft.merge(current, on="id", how="left")
    return add_error_columns(draft)


def confusion_summary(df: pd.DataFrame) -> pd.DataFrame:
    true_counts = df.groupby("price_band_id").size().rename("true_band_rows")
    rows = []
    for (true_band, pred_band), subset in df.groupby(["price_band_id", "predicted_price_band_id"]):
        row = {
            "true_band": true_band,
            "predicted_band": pred_band,
            "count": len(subset),
            "pct_of_true_band": len(subset) / true_counts.loc[true_band] * 100,
            "router_is_correct": true_band == pred_band,
            "avg_router_confidence": subset["router_confidence"].mean(),
            "avg_true_band_probability": subset["true_band_probability"].mean(),
        }
        for pred_col in PREDICTION_COLS:
            if pred_col in subset.columns:
                row[f"{pred_col}_Median_APE"] = subset[f"{pred_col}_ape_pct"].median()
                row[f"{pred_col}_Bias_pct"] = subset[f"{pred_col}_signed_pct"].mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["true_band", "router_is_correct", "count"], ascending=[True, True, False])


def correct_vs_wrong_by_true_band(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for true_band, subset in df.groupby("price_band_id"):
        for status, group in subset.groupby("router_is_correct"):
            row = {
                "true_band": true_band,
                "router_status": "correct" if status else "wrong",
                "count": len(group),
                "share_of_true_band_pct": len(group) / len(subset) * 100,
                "avg_router_confidence": group["router_confidence"].mean(),
                "avg_true_band_probability": group["true_band_probability"].mean(),
            }
            for pred_col in PREDICTION_COLS:
                if pred_col in group.columns:
                    row[f"{pred_col}_Median_APE"] = group[f"{pred_col}_ape_pct"].median()
                    row[f"{pred_col}_HitRate_20_pct"] = (group[f"{pred_col}_ape_pct"] <= 20).mean() * 100
                    row[f"{pred_col}_Bias_pct"] = group[f"{pred_col}_signed_pct"].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def numeric_profiles(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for true_band, true_subset in df.groupby("price_band_id"):
        correct = true_subset[true_subset["router_is_correct"]]
        for pred_band, wrong in true_subset[~true_subset["router_is_correct"]].groupby("predicted_price_band_id"):
            for feature in NUMERIC_FEATURES + [f"{col}_missing" for col in MISSING_SOURCE_COLS if f"{col}_missing" in df.columns]:
                if feature not in df.columns:
                    continue
                correct_median = correct[feature].median()
                wrong_median = wrong[feature].median()
                rows.append(
                    {
                        "true_band": true_band,
                        "predicted_band": pred_band,
                        "feature": feature,
                        "correct_median": correct_median,
                        "misclassified_median": wrong_median,
                        "median_diff": wrong_median - correct_median if pd.notna(correct_median) and pd.notna(wrong_median) else np.nan,
                        "ratio_misclassified_to_correct": wrong_median / correct_median
                        if pd.notna(correct_median) and correct_median not in [0, np.nan]
                        else np.nan,
                        "correct_missing_pct": correct[feature].isna().mean() * 100,
                        "misclassified_missing_pct": wrong[feature].isna().mean() * 100,
                        "misclassified_count": len(wrong),
                    }
                )
    out = pd.DataFrame(rows)
    out["abs_log_ratio"] = np.log(out["ratio_misclassified_to_correct"].replace([np.inf, -np.inf], np.nan)).abs()
    return out.sort_values(["true_band", "predicted_band", "abs_log_ratio"], ascending=[True, True, False])


def categorical_profiles(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for true_band, true_subset in df.groupby("price_band_id"):
        for feature in CATEGORICAL_FEATURES:
            if feature not in true_subset.columns:
                continue
            total = true_subset.groupby(feature, dropna=False).size().rename("true_band_feature_rows")
            wrong = (
                true_subset[~true_subset["router_is_correct"]]
                .groupby([feature, "predicted_price_band_id"], dropna=False)
                .size()
                .rename("misclassified_rows")
                .reset_index()
            )
            for _, row in wrong.iterrows():
                feature_value = row[feature]
                denom = total.loc[feature_value] if feature_value in total.index else np.nan
                if pd.isna(denom) or denom < 30:
                    continue
                rows.append(
                    {
                        "true_band": true_band,
                        "predicted_band": row["predicted_price_band_id"],
                        "feature": feature,
                        "value": feature_value,
                        "true_band_feature_rows": int(denom),
                        "misclassified_rows": int(row["misclassified_rows"]),
                        "misclassified_rate_pct": row["misclassified_rows"] / denom * 100,
                    }
                )
    return pd.DataFrame(rows).sort_values(["true_band", "predicted_band", "misclassified_rate_pct"], ascending=[True, True, False])


def model_metrics_by_router_status(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pred_col in PREDICTION_COLS:
        if pred_col not in df.columns:
            continue
        for (true_band, status), subset in df.groupby(["price_band_id", "router_is_correct"]):
            rows.append(
                {
                    "true_band": true_band,
                    "router_status": "correct" if status else "wrong",
                    "model_prediction_column": pred_col,
                    **metric_row(subset["price_vnd"], subset[pred_col]),
                }
            )
    return pd.DataFrame(rows)


def worst_examples(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "id",
        "title",
        "price_band_label",
        "predicted_price_band_label",
        "router_confidence",
        "true_band_probability",
        "probability_margin",
        "addr_last_1",
        "addr_last_2",
        "addr_last_3",
        "street_proxy",
        "property_subtype",
        "source",
        "price_vnd",
        "area_m2",
        "num_floors",
        "floor_area_m2",
        "frontage_m",
        "road_width_m",
        "width_m",
        "length_m",
        "price_per_m2",
        "pred_price_vnd_current_hybrid",
        "pred_price_vnd_price_band_hard_route",
        "pred_price_vnd_price_band_soft_route",
        "pred_price_vnd_price_band_soft_residual",
        "pred_price_vnd_price_band_hard_route_ape_pct",
        "pred_price_vnd_price_band_hard_route_signed_pct",
        "pred_price_vnd_current_hybrid_ape_pct",
        "pred_price_vnd_current_hybrid_signed_pct",
    ]
    cols = [col for col in cols if col in df.columns]
    wrong = df[~df["router_is_correct"]].copy()
    parts = []
    for _, subset in wrong.groupby(["price_band_id", "predicted_price_band_id"]):
        parts.append(subset.sort_values("pred_price_vnd_price_band_hard_route_ape_pct", ascending=False).head(50)[cols])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def manual_review_sample(df: pd.DataFrame) -> pd.DataFrame:
    wrong = df[~df["router_is_correct"]].copy()
    wrong["review_priority_score"] = wrong["router_confidence"] * wrong["pred_price_vnd_price_band_hard_route_ape_pct"]
    sample_parts = []
    for _, subset in wrong.groupby(["price_band_id", "predicted_price_band_id"]):
        sample_parts.append(subset.sort_values("review_priority_score", ascending=False).head(25))
    sample = pd.concat(sample_parts, ignore_index=True) if sample_parts else wrong.head(0)
    keep = [
        "id",
        "title",
        "price_band_label",
        "predicted_price_band_label",
        "review_priority_score",
        "router_confidence",
        "prob_lt_2b",
        "prob_mid_2b_10_5b",
        "prob_gt_10_5b",
        "addr_last_1",
        "addr_last_2",
        "addr_last_3",
        "street_proxy",
        "property_subtype",
        "source",
        "price_vnd",
        "pred_price_vnd_price_band_hard_route",
        "pred_price_vnd_price_band_hard_route_ape_pct",
        "pred_price_vnd_price_band_hard_route_signed_pct",
        "area_m2",
        "num_floors",
        "floor_area_m2",
        "frontage_m",
        "road_width_m",
        "width_m",
        "length_m",
    ]
    return sample[[col for col in keep if col in sample.columns]]


def write_summary(df: pd.DataFrame) -> None:
    summary = {
        "total_rows": int(len(df)),
        "misclassified_rows": int((~df["router_is_correct"]).sum()),
        "misclassified_pct": float((~df["router_is_correct"]).mean() * 100),
        "accuracy": float(df["router_is_correct"].mean() * 100),
        "by_true_band": (
            df.groupby("price_band_id")["router_is_correct"]
            .agg(rows="count", correct_rows="sum", accuracy_pct=lambda s: s.mean() * 100)
            .reset_index()
            .to_dict(orient="records")
        ),
    }
    (OUTPUT_DIR / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_predictions()

    misclassified = df[~df["router_is_correct"]].copy()
    misclassified.to_csv(OUTPUT_DIR / "band_router_misclassified_cases.csv", index=False)

    confusion_summary(df).to_csv(OUTPUT_DIR / "band_router_confusion_summary.csv", index=False)
    correct_vs_wrong_by_true_band(df).to_csv(OUTPUT_DIR / "band_router_correct_vs_wrong_by_true_band.csv", index=False)
    numeric_profiles(df).to_csv(OUTPUT_DIR / "misclassification_pair_numeric_profile.csv", index=False)
    categorical_profiles(df).to_csv(OUTPUT_DIR / "misclassification_pair_categorical_profile.csv", index=False)
    model_metrics_by_router_status(df).to_csv(OUTPUT_DIR / "model_metrics_by_router_status.csv", index=False)
    worst_examples(df).to_csv(OUTPUT_DIR / "worst_examples_by_confusion_pair.csv", index=False)
    manual_review_sample(df).to_csv(OUTPUT_DIR / "manual_review_sample.csv", index=False)
    write_summary(df)

    print(f"Saved misclassification analysis to {OUTPUT_DIR}")
    print(f"Rows: {len(df):,}")
    print(f"Misclassified rows: {len(misclassified):,} ({len(misclassified) / len(df) * 100:.2f}%)")


if __name__ == "__main__":
    main()
