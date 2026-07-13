from __future__ import annotations

import os
from pathlib import Path

from common import PROCESSED_DATA_DIR, RANDOM_STATE, RAW_DATA_DIR, build_modeling_dataset, load_raw_data, make_split, preprocess_raw_data, save_processed_data


def main():
    raw_csv = RAW_DATA_DIR / "mick_df_full_raw.csv"
    if not raw_csv.exists():
        raise FileNotFoundError(f"Missing raw dataset at {raw_csv}")

    df_raw = load_raw_data(raw_csv)
    max_raw_rows = int(os.environ.get("MAX_RAW_ROWS", "250000"))
    if max_raw_rows > 0 and len(df_raw) > max_raw_rows:
        df_raw = df_raw.sample(max_raw_rows, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"Sampled raw dataset down to {max_raw_rows:,} rows for reproducible scene execution.")

    df_clean = preprocess_raw_data(df_raw)
    df_model = build_modeling_dataset(df_clean)
    split_data = make_split(df_model)
    save_processed_data(df_clean, df_model, split_data)

    print("Saved processed outputs to:", PROCESSED_DATA_DIR)
    print("Raw rows:", len(df_raw))
    print("Clean rows:", len(df_clean))
    print("Model rows:", len(df_model))
    print("Train rows:", len(split_data.train_df))
    print("Test rows:", len(split_data.test_df))
    print("Split method:", split_data.split_method)


if __name__ == "__main__":
    main()
