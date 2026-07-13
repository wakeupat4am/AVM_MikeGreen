from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from common import (  # noqa: E402
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    add_missing_quality_features,
    add_refined_subtype_features,
    add_source_features,
    build_modeling_dataset,
    create_interaction,
    fill_missing_location_from_text,
    load_raw_data,
    preprocess_raw_data,
)
from extract_from_text import extract_features_from_text  # noqa: E402


OUT_DIR = ROOT / "draft" / "pre_retrain_feature_fixes"
RAW_CSV = RAW_DATA_DIR / "mick_df_full_raw.csv"


IMPORTANT_COLUMNS = [
    "id",
    "title",
    "address",
    "price_vnd",
    "area_m2",
    "property_subtype",
    "property_subtype_refined",
    "subtype_price_segment",
    "source",
    "source_family",
    "source_quality_tier",
    "addr_last_1",
    "addr_last_2",
    "addr_last_3",
    "street_proxy",
    "location_quality_score",
    "address_missing_count",
    "street_proxy_missing",
    "addr_last_2_missing",
    "num_floors",
    "floor_area_m2",
    "frontage_m",
    "road_width_m",
    "width_m",
    "length_m",
    "structural_missing_count",
    "num_floors_missing",
    "floor_area_m2_missing",
    "frontage_m_missing",
    "road_width_m_missing",
    "width_m_missing",
    "length_m_missing",
    "broad_subtype_flag",
    "source_thocu_flag",
    "source_verified_flag",
    "source_sm_flag",
]


def missing_rates(df: pd.DataFrame, label: str) -> pd.DataFrame:
    cols = [
        "addr_last_2",
        "addr_last_3",
        "street_proxy",
        "num_floors",
        "floor_area_m2",
        "frontage_m",
        "road_width_m",
        "width_m",
        "length_m",
    ]
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        if df[col].dtype == "object":
            missing = df[col].fillna("missing").astype(str).str.lower().isin(["missing", "", "nan", "none", "_"])
        else:
            missing = df[col].isna()
        rows.append({"dataset": label, "column": col, "missing_rows": int(missing.sum()), "missing_pct": float(missing.mean() * 100)})
    return pd.DataFrame(rows)


def apply_feature_fixes_to_processed(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Apply newly added non-destructive fixes to an already processed dataset."""
    fixed = df_clean.copy()
    if "floor_area_m2" not in fixed.columns:
        fixed["floor_area_m2"] = pd.NA
    fixed, _ = extract_features_from_text(fixed)
    fixed = fill_missing_location_from_text(fixed)
    fixed = add_source_features(fixed)

    create_interaction(fixed, "floor_area_m2", "area_m2", "num_floors")
    create_interaction(fixed, "floors_x_area", "num_floors", "area_m2")
    create_interaction(fixed, "frontage_x_road_width", "frontage_m", "road_width_m")
    create_interaction(fixed, "floors_x_frontage", "num_floors", "frontage_m")
    create_interaction(fixed, "floors_x_business", "num_floors", "is_business_text")
    create_interaction(fixed, "car_access_x_road_width", "is_car_access_text", "road_width_m")
    create_interaction(fixed, "area_x_frontage", "area_m2", "frontage_m")
    create_interaction(fixed, "corner_x_frontage", "is_corner_text", "frontage_m")
    create_interaction(fixed, "business_x_road_width", "is_business_text", "road_width_m")
    create_interaction(fixed, "project_x_area", "is_project_text", "area_m2")

    fixed = add_refined_subtype_features(fixed)
    fixed = add_missing_quality_features(fixed)
    return fixed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-processed",
        action="store_true",
        help="Use current full_cleaned_dataset.parquet as input and apply non-destructive feature fixes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.from_processed:
        processed_path = PROCESSED_DATA_DIR / "full_cleaned_dataset.parquet"
        if not processed_path.exists():
            raise FileNotFoundError(processed_path)
        print(f"Loading current processed clean data from {processed_path}")
        raw = pd.read_parquet(processed_path)
        print(f"Processed clean rows: {len(raw):,}")
        print("Applying non-destructive feature fixes to processed data")
        fixed_clean = apply_feature_fixes_to_processed(raw)
    else:
        if not RAW_CSV.exists():
            raise FileNotFoundError(RAW_CSV)
        print(f"Loading raw data from {RAW_CSV}")
        raw = load_raw_data(RAW_CSV)
        print(f"Raw rows: {len(raw):,}")
        print("Preprocessing with feature fixes")
        fixed_clean = preprocess_raw_data(raw)

    fixed_model = build_modeling_dataset(fixed_clean)

    fixed_clean.to_parquet(OUT_DIR / "fixed_full_cleaned_dataset.parquet", index=False)
    fixed_model.to_parquet(OUT_DIR / "fixed_landed_residential_house_modeling_dataset.parquet", index=False)

    sample_cols = [col for col in IMPORTANT_COLUMNS if col in fixed_model.columns]
    fixed_model[sample_cols].head(5000).to_csv(OUT_DIR / "fixed_modeling_dataset_sample_5000.csv", index=False)
    fixed_model[sample_cols].to_csv(OUT_DIR / "fixed_modeling_dataset_review_columns.csv.gz", index=False, compression="gzip")

    old_model_path = PROCESSED_DATA_DIR / "landed_residential_house_modeling_dataset.parquet"
    comparison = []
    if old_model_path.exists():
        old_model = pd.read_parquet(old_model_path)
        comparison.append(missing_rates(old_model, "current_processed_model"))
    comparison.append(missing_rates(fixed_model, "fixed_candidate_model"))
    missing_comparison = pd.concat(comparison, ignore_index=True)
    missing_comparison.to_csv(OUT_DIR / "missing_rate_comparison.csv", index=False)

    subtype_counts = (
        fixed_model.groupby(["property_subtype", "property_subtype_refined"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    subtype_counts.to_csv(OUT_DIR / "subtype_refinement_counts.csv", index=False)

    source_counts = (
        fixed_model.groupby(["source_family", "source_quality_tier"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    source_counts.to_csv(OUT_DIR / "source_bias_feature_counts.csv", index=False)

    metadata = {
        "input_mode": "from_processed" if args.from_processed else "from_raw",
        "input_rows": int(len(raw)),
        "fixed_clean_rows": int(len(fixed_clean)),
        "fixed_model_rows": int(len(fixed_model)),
        "output_dir": str(OUT_DIR),
        "new_columns_present": [col for col in IMPORTANT_COLUMNS if col in fixed_model.columns],
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved fixed candidate dataset to {OUT_DIR}")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print("\nMissing comparison:")
    print(missing_comparison.to_string(index=False))
    print("\nSubtype refinements:")
    print(subtype_counts.head(20).to_string(index=False))
    print("\nSource features:")
    print(source_counts.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
