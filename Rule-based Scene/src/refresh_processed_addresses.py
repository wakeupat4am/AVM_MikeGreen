from __future__ import annotations

from common import PROCESSED_DATA_DIR, build_modeling_dataset, canonical_district_name, canonical_province_name, canonical_street_proxy, canonical_ward_name, canonicalize_series_with_cache, make_split, parse_address_components, save_processed_data
import pandas as pd


def main():
    cleaned_path = PROCESSED_DATA_DIR / "full_cleaned_dataset.parquet"
    if not cleaned_path.exists():
        raise FileNotFoundError(f"Missing cleaned dataset at {cleaned_path}")

    df_clean = pd.read_parquet(cleaned_path)
    if "address" not in df_clean.columns:
        raise KeyError("Expected 'address' column in cleaned dataset.")

    raw_cols = ["addr_last_1_raw", "addr_last_2_raw", "addr_last_3_raw", "street_proxy_raw"]
    if not all(col in df_clean.columns for col in raw_cols):
        parsed_address = df_clean["address"].fillna("").astype(str).apply(parse_address_components).apply(pd.Series)
        df_clean["addr_last_1_raw"] = parsed_address["addr_last_1_raw"]
        df_clean["addr_last_2_raw"] = parsed_address["addr_last_2_raw"]
        df_clean["addr_last_3_raw"] = parsed_address["addr_last_3_raw"]
        df_clean["street_proxy_raw"] = parsed_address["street_proxy_raw"]

    df_clean["addr_last_1"] = canonicalize_series_with_cache(df_clean["addr_last_1_raw"], canonical_province_name)
    df_clean["addr_last_2"] = canonicalize_series_with_cache(df_clean["addr_last_2_raw"], canonical_district_name)
    df_clean["addr_last_3"] = canonicalize_series_with_cache(df_clean["addr_last_3_raw"], canonical_ward_name)
    df_clean["street_proxy"] = canonicalize_series_with_cache(df_clean["street_proxy_raw"], canonical_street_proxy)

    df_model = build_modeling_dataset(df_clean)
    split_data = make_split(df_model)
    save_processed_data(df_clean, df_model, split_data)

    print("Refreshed address normalization outputs at:", PROCESSED_DATA_DIR)
    print("Clean rows:", len(df_clean))
    print("Model rows:", len(df_model))
    print("Train rows:", len(split_data.train_df))
    print("Test rows:", len(split_data.test_df))
    print("Split method:", split_data.split_method)


if __name__ == "__main__":
    main()
