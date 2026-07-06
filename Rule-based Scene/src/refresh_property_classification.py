from __future__ import annotations

import pandas as pd

from common import PROCESSED_DATA_DIR, apply_property_classification, build_modeling_dataset, make_split, save_processed_data


def main():
    cleaned_path = PROCESSED_DATA_DIR / "full_cleaned_dataset.parquet"
    if not cleaned_path.exists():
        raise FileNotFoundError(f"Missing cleaned dataset at {cleaned_path}")

    df_clean = pd.read_parquet(cleaned_path)
    df_clean = apply_property_classification(df_clean)
    df_model = build_modeling_dataset(df_clean)
    split_data = make_split(df_model)
    save_processed_data(df_clean, df_model, split_data)

    print("Refreshed property classification outputs at:", PROCESSED_DATA_DIR)
    print("Clean rows:", len(df_clean))
    print("Model rows:", len(df_model))
    print("Train rows:", len(split_data.train_df))
    print("Test rows:", len(split_data.test_df))
    print("Split method:", split_data.split_method)


if __name__ == "__main__":
    main()
