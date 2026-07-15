#!/usr/bin/env python3
"""Phase 3 — merge Phase 1 Hà Nội scraped listings into the Phase 2 cleaned schema.

Output keeps the Phase 2 *cleaned* column format. Phase 1 Hà Nội listings are mapped onto that
schema and appended. For the appended listings the availability rule requested is:

    status / available_for_sale_status =
        AVAILABLE   if listing_date is in October 2025 or later, or any 2026 date
        UNKNOWN     if older than that (this includes undated listings)

A `dataset_source` column marks each row's origin ("phase2_transaction" vs "phase1_listing").
"""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
P2 = ROOT / "Phase 2" / "data" / "transaction_cleaned.parquet"
P1 = ROOT / "Phase 1" / "Rule-based Scene" / "data" / "processed" / "full_cleaned_dataset.parquet"
OUT = pathlib.Path(__file__).resolve().parents[1] / "data"

# availability cutoff: October 2025 (inclusive) and everything after
CUTOFF = pd.Timestamp("2025-10-01")


def norm_admin(val: str, prefix_map: dict[str, str]) -> str | None:
    """Normalise a lowercase admin token like 'quận đống đa' -> 'Q. Đống Đa' (Phase 2 format)."""
    if not isinstance(val, str) or val.strip() in ("", "missing"):
        return None
    s = val.strip().lower()
    for src, short in prefix_map.items():
        if s.startswith(src + " "):
            return short + " " + s[len(src) + 1:].title()
    return s.title()  # no known prefix -> just title-case


DIST_PREFIX = {"quận": "Q.", "huyện": "H.", "thị xã": "TX.", "thành phố": "TP."}
WARD_PREFIX = {"phường": "P.", "xã": "X.", "thị trấn": "TT."}


def load_phase1_hanoi() -> pd.DataFrame:
    cols = ["id", "addr_last_2", "addr_last_3", "street_proxy", "address",
            "listing_date", "listing_year", "area_m2", "price_vnd", "num_floors",
            "frontage_m", "built_year", "recency_weight", "is_landed_residential_house"]
    df = pd.read_parquet(P1, columns=cols)
    df = df[df["addr_last_2"].isna() | ~df["addr_last_2"].isna()]  # keep all, filter below
    df = df[df["is_landed_residential_house"] == 1].copy()          # match Phase 2 scope (landed houses)
    # Hà Nội only: province lives one token above district; detect via address tail
    prov = pd.read_parquet(P1, columns=["addr_last_1"]).loc[df.index, "addr_last_1"]
    df = df[prov.astype(str).str.contains("nội", case=False, na=False)].copy()
    df = df[(df["price_vnd"] > 0) & (df["area_m2"] > 0)].copy()      # need a usable price + area
    return df


def map_phase1(df: pd.DataFrame) -> pd.DataFrame:
    lm = pd.to_datetime(df["listing_date"], errors="coerce")
    avail = lm >= CUTOFF                                            # Oct-2025 or later (incl. 2026)
    state = np.where(avail, "AVAILABLE", "UNKNOWN")                 # undated -> UNKNOWN

    ppm2 = df["price_vnd"] / df["area_m2"]
    out = pd.DataFrame({
        "status": state,
        "classify": "LISTING",
        "property_id": "P1_" + df["id"].astype(str),
        "created_date": lm.dt.strftime("%m/%d/%Y"),
        "last_modified_date": lm,
        "available_for_sale_status": state,
        "reference_selling_price": df["price_vnd"].astype(float),
        "selling_furniture": None,
        "inventory_type": None,
        "commission_rate": np.nan,
        "service_contract_status": None,
        "address_number": None,
        "ref_property_code": None,
        "latitude": np.nan,                                         # Phase 1 has no geocoordinates
        "longitude": np.nan,
        "total_area": df["area_m2"].astype(float),
        "min_waist_width_to_main_road": np.nan,
        "max_waist_width_to_main_road": np.nan,
        "main_road_dist": np.nan,
        "shape_name": None,
        "facade_width": df["frontage_m"].astype(float),
        "number_of_floors": df["num_floors"].astype(float),
        "construction_year": df["built_year"].astype(float),
        "main_street_name": df["street_proxy"],
        "ward_name": df["addr_last_3"].map(lambda v: norm_admin(v, WARD_PREFIX)),
        "district_name": df["addr_last_2"].map(lambda v: norm_admin(v, DIST_PREFIX)),
        "short_name": None,
        "long_name": df["address"],
        "has_geo": False,
        "price_per_m2": ppm2.astype(float),
        "price_bil": (df["price_vnd"] / 1e9).astype(float),
        "suspicious_price": (ppm2 < 20e6) | (ppm2 > 800e6),
        "created": lm,
        "listing_year": df["listing_year"],
        "recency_weight": df["recency_weight"],
        "has_construction_year": df["built_year"].notna(),
        "dataset_source": "phase1_listing",
    })
    return out


def main():
    p2 = pd.read_parquet(P2)
    p2 = p2.copy()
    p2["dataset_source"] = "phase2_transaction"

    p1 = map_phase1(load_phase1_hanoi())
    # align to Phase 2 column order (+ dataset_source at the end)
    cols = list(p2.columns)
    p1 = p1.reindex(columns=cols)

    merged = pd.concat([p2, p1], ignore_index=True)

    OUT.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT / "phase3_merged_dataset.parquet", index=False)
    merged.head(100).to_csv(OUT / "phase3_merged_preview_100.csv", index=False)

    # ---- summary ----
    print(f"Phase 2 transaction rows : {len(p2):,}")
    print(f"Phase 1 Hà Nội listings  : {len(p1):,}")
    print(f"Phase 3 merged total     : {len(merged):,}  cols={merged.shape[1]}")
    print()
    print("=== dataset_source ==="); print(merged["dataset_source"].value_counts().to_string())
    print()
    print("=== status (merged) ==="); print(merged["status"].value_counts(dropna=False).to_string())
    print()
    print("=== phase1_listing availability (the requested rule) ===")
    print(p1["available_for_sale_status"].value_counts(dropna=False).to_string())
    print(f"  (AVAILABLE = listing_date >= {CUTOFF.date()}; UNKNOWN = older or undated)")
    print("\nsaved:")
    print("  data/phase3_merged_dataset.parquet")
    print("  data/phase3_merged_preview_100.csv")


if __name__ == "__main__":
    main()
