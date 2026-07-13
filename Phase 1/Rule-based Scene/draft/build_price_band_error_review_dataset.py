from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = (
    ROOT
    / "Evaluation"
    / "improvements"
    / "oracle_price_band_region_subtype_classification_fix_v2"
    / "trained_price_band_error_analysis"
)
OUT_DIR = ANALYSIS_DIR / "review_dataset"

ERROR_CASES = ANALYSIS_DIR / "price_band_model_error_cases.csv"
TERM_SUMMARY = ANALYSIS_DIR / "feature_term_error_summary.csv"
FEATURE_RANK = ANALYSIS_DIR / "feature_misunderstanding_rank.csv"
NUMERIC_PROFILE = ANALYSIS_DIR / "numeric_feature_good_vs_high_error_profile.csv"


REVIEW_FEATURES = [
    "addr_last_1",
    "addr_last_2",
    "addr_last_3",
    "street_proxy",
    "property_subtype",
    "source",
    "local_ppm2_band",
    "selected_model",
]
STRUCTURAL_COLS = ["num_floors", "floor_area_m2", "frontage_m", "road_width_m", "width_m", "length_m"]


def safe_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.lower().fillna("missing")


def add_issue_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_high_error_20pct"] = (out["ape_pct"] > 20).astype(int)
    out["street_proxy_missing_flag"] = safe_text(out["street_proxy"]).isin(["missing", "", "nan", "none", "_"]).astype(int)
    out["addr_last_2_missing_flag"] = safe_text(out["addr_last_2"]).isin(["missing", "", "nan", "none", "_"]).astype(int)
    out["addr_last_3_missing_flag"] = safe_text(out["addr_last_3"]).isin(["missing", "", "nan", "none", "_"]).astype(int)
    for col in STRUCTURAL_COLS:
        if col in out.columns:
            out[f"{col}_missing_flag"] = out[col].isna().astype(int)

    out["structural_missing_count"] = out[[f"{col}_missing_flag" for col in STRUCTURAL_COLS if f"{col}_missing_flag" in out.columns]].sum(axis=1)
    out["broad_subtype_flag"] = out["property_subtype"].isin(["frontage_house", "normal_house", "townhouse_project"]).astype(int)
    out["global_model_flag"] = (out["selected_model"] == "global").astype(int)
    out["local_tail_flag"] = out["local_ppm2_band"].isin(["below_q01", "q01_q05", "q95_q99", "above_q99"]).astype(int)
    out["needs_address_fix_flag"] = ((out["street_proxy_missing_flag"] == 1) | (out["addr_last_2_missing_flag"] == 1)).astype(int)
    out["needs_structural_fix_flag"] = (out["structural_missing_count"] >= 2).astype(int)
    return out


def build_term_lookup(term_summary: pd.DataFrame) -> pd.DataFrame:
    terms = term_summary.copy()
    terms["term_key"] = terms["price_band_label"].astype(str) + "|" + terms["feature"].astype(str) + "|" + terms["term"].astype(str)
    terms = terms.rename(
        columns={
            "rows": "term_rows",
            "high_error_rows": "term_high_error_rows",
            "high_error_rate_pct": "term_high_error_rate_pct",
            "median_ape_pct": "term_median_ape_pct",
            "bias_pct": "term_bias_pct",
        }
    )
    return terms[
        [
            "term_key",
            "term_rows",
            "term_high_error_rows",
            "term_high_error_rate_pct",
            "term_median_ape_pct",
            "term_bias_pct",
            "over_20pct_rate",
            "under_20pct_rate",
        ]
    ]


def attach_worst_term_stats(df: pd.DataFrame, term_lookup: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    term_stat_frames = []
    for feature in REVIEW_FEATURES:
        if feature not in out.columns:
            continue
        temp = out[["id", "price_band_label", feature]].copy()
        temp["term_key"] = temp["price_band_label"].astype(str) + "|" + feature + "|" + temp[feature].astype(str)
        temp["term_feature"] = feature
        temp = temp.merge(term_lookup, on="term_key", how="left")
        term_stat_frames.append(temp.drop(columns=[feature]))

    long_stats = pd.concat(term_stat_frames, ignore_index=True)
    long_stats["term_high_error_rows"] = long_stats["term_high_error_rows"].fillna(0)
    long_stats["term_high_error_rate_pct"] = long_stats["term_high_error_rate_pct"].fillna(0)

    by_count = long_stats.sort_values(["id", "term_high_error_rows", "term_high_error_rate_pct"], ascending=[True, False, False]).groupby("id").head(1)
    by_rate = (
        long_stats[long_stats["term_rows"].fillna(0) >= 100]
        .sort_values(["id", "term_high_error_rate_pct", "term_high_error_rows"], ascending=[True, False, False])
        .groupby("id")
        .head(1)
    )

    by_count = by_count.rename(
        columns={
            "term_feature": "top_error_count_feature",
            "term_high_error_rows": "top_error_count_term_high_error_rows",
            "term_high_error_rate_pct": "top_error_count_term_error_rate_pct",
            "term_median_ape_pct": "top_error_count_term_median_ape_pct",
            "term_bias_pct": "top_error_count_term_bias_pct",
        }
    )
    by_count = by_count[
        [
            "id",
            "top_error_count_feature",
            "top_error_count_term_high_error_rows",
            "top_error_count_term_error_rate_pct",
            "top_error_count_term_median_ape_pct",
            "top_error_count_term_bias_pct",
        ]
    ]

    by_rate = by_rate.rename(
        columns={
            "term_feature": "top_error_rate_feature",
            "term_high_error_rows": "top_error_rate_term_high_error_rows",
            "term_high_error_rate_pct": "top_error_rate_term_error_rate_pct",
            "term_median_ape_pct": "top_error_rate_term_median_ape_pct",
            "term_bias_pct": "top_error_rate_term_bias_pct",
        }
    )
    by_rate = by_rate[
        [
            "id",
            "top_error_rate_feature",
            "top_error_rate_term_high_error_rows",
            "top_error_rate_term_error_rate_pct",
            "top_error_rate_term_median_ape_pct",
            "top_error_rate_term_bias_pct",
        ]
    ]

    out = out.merge(by_count, on="id", how="left").merge(by_rate, on="id", how="left")
    return out


def add_review_priority(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["review_priority_score"] = (
        out["ape_pct"].fillna(0)
        + 8 * out["needs_address_fix_flag"]
        + 6 * out["needs_structural_fix_flag"]
        + 5 * out["global_model_flag"]
        + 4 * out["broad_subtype_flag"]
        + 4 * out["local_tail_flag"]
        + 0.08 * out["top_error_rate_term_error_rate_pct"].fillna(0)
    )

    reasons = []
    for row in out.itertuples(index=False):
        row_reasons = []
        if getattr(row, "is_high_error_20pct"):
            row_reasons.append("high_error_gt20pct")
        if getattr(row, "needs_address_fix_flag"):
            row_reasons.append("address_missing_or_weak")
        if getattr(row, "needs_structural_fix_flag"):
            row_reasons.append("structural_missing")
        if getattr(row, "global_model_flag"):
            row_reasons.append("global_fallback")
        if getattr(row, "broad_subtype_flag"):
            row_reasons.append("broad_subtype")
        if getattr(row, "local_tail_flag"):
            row_reasons.append("local_ppm2_tail")
        reasons.append(";".join(row_reasons) if row_reasons else "low_priority")
    out["review_reason"] = reasons
    return out.sort_values(["is_high_error_20pct", "review_priority_score"], ascending=[False, False])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(ERROR_CASES)
    term_summary = pd.read_csv(TERM_SUMMARY)
    feature_rank = pd.read_csv(FEATURE_RANK)
    numeric_profile = pd.read_csv(NUMERIC_PROFILE)

    dataset = add_issue_flags(cases)
    dataset = attach_worst_term_stats(dataset, build_term_lookup(term_summary))
    dataset = add_review_priority(dataset)

    dataset.to_csv(OUT_DIR / "price_band_trained_model_review_dataset.csv", index=False)
    dataset[dataset["is_high_error_20pct"] == 1].to_csv(OUT_DIR / "price_band_trained_model_high_error_dataset.csv", index=False)
    dataset.groupby("price_band_label").head(100).to_csv(OUT_DIR / "price_band_trained_model_review_sample.csv", index=False)

    # Compact report-ready tables.
    feature_rank.to_csv(OUT_DIR / "report_feature_rank_by_band.csv", index=False)
    term_summary.sort_values(["price_band_label", "high_error_rows"], ascending=[True, False]).groupby("price_band_label").head(30).to_csv(
        OUT_DIR / "report_top_terms_by_count_by_band.csv",
        index=False,
    )
    term_summary[(term_summary["rows"] >= 100) & (term_summary["high_error_rows"] >= 30)].sort_values(
        ["price_band_label", "high_error_rate_pct", "high_error_rows"], ascending=[True, False, False]
    ).groupby("price_band_label").head(30).to_csv(OUT_DIR / "report_top_terms_by_rate_by_band.csv", index=False)
    numeric_profile.to_csv(OUT_DIR / "report_numeric_feature_profile_by_band.csv", index=False)

    summary = dataset.groupby("price_band_label").agg(
        rows=("id", "count"),
        high_error_rows=("is_high_error_20pct", "sum"),
        high_error_rate_pct=("is_high_error_20pct", lambda s: float(s.mean() * 100)),
        address_fix_rows=("needs_address_fix_flag", "sum"),
        structural_fix_rows=("needs_structural_fix_flag", "sum"),
        broad_subtype_rows=("broad_subtype_flag", "sum"),
        global_model_rows=("global_model_flag", "sum"),
        local_tail_rows=("local_tail_flag", "sum"),
        median_ape_pct=("ape_pct", "median"),
        mean_ape_pct=("ape_pct", "mean"),
        bias_pct=("signed_error_pct", "mean"),
    ).reset_index()
    summary.to_csv(OUT_DIR / "review_dataset_summary_by_band.csv", index=False)

    print(f"Saved review dataset to {OUT_DIR}")
    print("Dataset rows:", len(dataset))
    print(summary.to_string(index=False))
    print("\nSample high priority rows:")
    cols = [
        "id",
        "price_band_label",
        "ape_pct",
        "signed_error_pct",
        "review_priority_score",
        "review_reason",
        "addr_last_1",
        "addr_last_2",
        "addr_last_3",
        "street_proxy",
        "property_subtype",
        "source",
        "area_m2",
        "num_floors",
        "floor_area_m2",
        "frontage_m",
        "road_width_m",
        "top_error_count_feature",
        "top_error_rate_feature",
    ]
    print(dataset[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
