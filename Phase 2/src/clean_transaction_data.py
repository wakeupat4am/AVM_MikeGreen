#!/usr/bin/env python3
"""Phase 2 — clean the Hà Nội transaction/inventory dataset.

Strategy (risk-removal, leakage-safe):
  1. Prune dead / constant / redundant / near-empty columns.
  2. Deduplicate by property_id  ->  keep SOLD first, then the latest last_modified_date.
  3. Drop impossible targets (non-positive price / area).
  4. Clip garbage numerics to NULL (don't fabricate values).
  5. Null invalid geocoordinates (outside the Hà Nội box) + add has_geo flag.
  6. Two-sided price/m2 plausibility -> FLAG (not delete), keep an audit set.
  7. Parse dates, derive recency.

Outputs (to Phase 2/data/):
  transaction_cleaned.parquet / .csv   the cleaned table
  transaction_cleaned_sample_100.csv   a 100-row preview
  filter_summary.csv                   rows removed per step
"""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
RAW = DATA / "transaction_data_26_07_10.parquet"

DROP_EMPTY = ["import_price", "original_price", "minimum_selling_price", "seller_expected_net_income",
              "UCHI_is_checked", "UCHI_info", "UCHI_other_info", "UCHI_transaction_info"]
# Kept on request: availability/sold tracking needs `available_for_sale_status`
# (AVAILABLE / NOT_AVAILABLE) alongside `status` (SOLD / REVOKED / ON_STOCK / LOCKED).
# `classify` and `inventory_type` are single-valued here (no modeling signal) but kept for record.
DROP_CONSTANT = []
DROP_REDUNDANT = ["address_number0", "reference_selling_price_gross"]
DROP_NEAR_EMPTY = ["direction_name", "land_number_new", "map_number_new",
                   "original_latitude", "original_longitude"]
DROP_INTERNAL = ["warehouse_import_id", "inventory_code"]

# Clip-to-NULL ranges for garbage numerics.
CLIP = {
    "construction_year": (1900, 2026),
    "number_of_floors": (1, 15),
    "facade_width": (0.5, 30),
    "min_waist_width_to_main_road": (0, 50),
    "max_waist_width_to_main_road": (0, 50),
    "main_road_dist": (0, 5000),
}
# Hà Nội metropolitan bounding box.
LAT_LO, LAT_HI, LON_LO, LON_HI = 20.5, 21.5, 105.3, 106.2
# Price/m2 plausibility band (VND); median ~155M, p1 ~54M, p99 ~692M.
PPM2_LOW, PPM2_HIGH = 20e6, 800e6


def main():
    df = pd.read_parquet(RAW)
    log = [("raw", len(df), 0)]

    # 1. Prune columns.
    drop = [c for c in DROP_EMPTY + DROP_CONSTANT + DROP_REDUNDANT + DROP_NEAR_EMPTY + DROP_INTERNAL if c in df.columns]
    df = df.drop(columns=drop)
    print(f"dropped {len(drop)} columns")

    # 2. Deduplicate by property_id: SOLD first, then latest last_modified_date.
    df["last_modified_date"] = pd.to_datetime(df["last_modified_date"], errors="coerce")
    df["_is_sold"] = (df["status"] == "SOLD").astype(int)
    n0 = len(df)
    df = (df.sort_values(["_is_sold", "last_modified_date"], ascending=[False, False])
            .drop_duplicates("property_id", keep="first")
            .drop(columns="_is_sold")
            .reset_index(drop=True))
    log.append(("dedup by property_id (SOLD>latest)", len(df), n0 - len(df)))
    print(f"dedup: {n0} -> {len(df)}")

    # 3. Drop impossible targets.
    n0 = len(df)
    df = df[(df["reference_selling_price"] > 0) & (df["total_area"] > 10)].reset_index(drop=True)
    log.append(("drop non-positive price / area<=10", len(df), n0 - len(df)))

    # 4. Clip garbage numerics -> NULL.
    for col, (lo, hi) in CLIP.items():
        if col in df.columns:
            bad = ~df[col].between(lo, hi) & df[col].notna()
            df.loc[bad, col] = np.nan
            if bad.sum():
                print(f"  nulled {int(bad.sum())} garbage values in {col}")

    # 5. Geocoordinates: null invalid, add flag.
    df["has_geo"] = df["latitude"].between(LAT_LO, LAT_HI) & df["longitude"].between(LON_LO, LON_HI)
    df.loc[~df["has_geo"], ["latitude", "longitude"]] = np.nan
    print(f"  valid geo: {int(df['has_geo'].sum())} / {len(df)}")

    # 6. Derived fields + two-sided price plausibility flag (keep rows).
    df["price_per_m2"] = df["reference_selling_price"] / df["total_area"]
    df["price_bil"] = df["reference_selling_price"] / 1e9
    df["suspicious_price"] = (df["price_per_m2"] < PPM2_LOW) | (df["price_per_m2"] > PPM2_HIGH)
    print(f"  flagged suspicious price/m2: {int(df['suspicious_price'].sum())} "
          f"({df['suspicious_price'].mean()*100:.1f}%)")

    # 7. Dates & recency.
    df["created"] = pd.to_datetime(df["created_date"], format="%m/%d/%Y", errors="coerce")
    df["listing_year"] = df["created"].dt.year
    rec_map = {2025: 1.0, 2024: 0.9, 2023: 0.7, 2022: 0.5}
    df["recency_weight"] = df["listing_year"].map(rec_map).fillna(0.5)

    # missingness flags that are themselves signal
    df["has_construction_year"] = df["construction_year"].notna()

    # ---- write outputs ----
    df.to_parquet(DATA / "transaction_cleaned.parquet", index=False)
    df.to_csv(DATA / "transaction_cleaned.csv", index=False)
    df.head(100).to_csv(DATA / "transaction_cleaned_sample_100.csv", index=False)
    summary = pd.DataFrame(log, columns=["step", "rows_after", "rows_removed"])
    summary.to_csv(DATA / "filter_summary.csv", index=False)

    print("\n=== filter summary ===")
    print(summary.to_string(index=False))
    print(f"\nFINAL cleaned rows: {len(df)}  |  columns: {df.shape[1]}")
    print("SOLD in cleaned set:", int((df["status"] == "SOLD").sum()))
    print("training-usable (not suspicious):", int((~df["suspicious_price"]).sum()))
    print("\nsaved -> data/transaction_cleaned.parquet / .csv / _sample_100.csv / filter_summary.csv")
    return df


if __name__ == "__main__":
    main()
