from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


QUANTILE_LEVELS = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
QUANTILE_COLUMNS = ["Q01", "Q05", "Q25", "Q50", "Q75", "Q95", "Q99"]
HIERARCHY = [
    ("district_subtype", ["addr_last_2", "property_subtype"]),
    ("city_subtype", ["addr_last_1", "property_subtype"]),
    ("subtype", ["property_subtype"]),
    ("global", []),
]


@dataclass
class OutlierArtifacts:
    train_df: pd.DataFrame
    other_dfs: Dict[str, pd.DataFrame]
    quantile_tables: Dict[str, pd.DataFrame]


def impossible_row_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    mask |= df["price_vnd"].fillna(0) <= 0
    mask |= df["area_m2"].fillna(0) <= 0
    mask |= df["price_per_m2"].fillna(0) <= 0

    numeric_bounds = {
        "num_floors": (0, 100),
        "bedrooms": (0, 100),
        "bathrooms": (0, 100),
        "frontage_m": (0.5, 200),
        "road_width_m": (0.5, 200),
    }
    for col, (lo, hi) in numeric_bounds.items():
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        valid = values.isna() | ((values >= lo) & (values <= hi))
        mask |= ~valid
    return mask


def remove_impossible_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[~impossible_row_mask(df)].copy()


def compute_group_quantiles(df: pd.DataFrame, group_cols: List[str], ppm2_col: str = "price_per_m2") -> pd.DataFrame:
    valid = df.loc[df[ppm2_col].notna() & (df[ppm2_col] > 0)].copy()
    if group_cols:
        quantile_table = (
            valid.groupby(group_cols)[ppm2_col]
            .quantile(QUANTILE_LEVELS)
            .unstack()
            .reset_index()
        )
    else:
        global_vals = valid[ppm2_col].quantile(QUANTILE_LEVELS)
        quantile_table = pd.DataFrame([global_vals.values], columns=global_vals.index)
    quantile_table.columns = list(group_cols) + QUANTILE_COLUMNS
    if not group_cols:
        quantile_table["__global_key"] = "global"
    return quantile_table


def fit_local_quantile_tables(train_df: pd.DataFrame, ppm2_col: str = "price_per_m2") -> Dict[str, pd.DataFrame]:
    return {level_name: compute_group_quantiles(train_df, group_cols, ppm2_col=ppm2_col) for level_name, group_cols in HIERARCHY}


def _fill_quantiles_by_hierarchy(df: pd.DataFrame, quantile_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in QUANTILE_COLUMNS:
        out[col] = np.nan
    out["quantile_source"] = "missing"

    working = df.copy()
    working["__global_key"] = "global"
    for level_name, group_cols in HIERARCHY:
        join_cols = group_cols if group_cols else ["__global_key"]
        table = quantile_tables[level_name]
        merged = working[join_cols].merge(table, how="left", on=join_cols)
        merged.index = df.index
        mask = out["Q01"].isna() & merged["Q01"].notna()
        if mask.any():
            for col in QUANTILE_COLUMNS:
                out.loc[mask, col] = merged.loc[mask, col]
            out.loc[mask, "quantile_source"] = level_name
    return out


def assign_local_bands(df: pd.DataFrame, ppm2_col: str = "price_per_m2") -> pd.DataFrame:
    ppm2 = df[ppm2_col]
    df["is_low_local_outlier"] = ppm2 < df["Q05"]
    df["is_high_local_outlier"] = ppm2 > df["Q95"]
    df["is_extreme_local_outlier"] = (ppm2 < df["Q01"]) | (ppm2 > df["Q99"])

    conditions = [
        ppm2 < df["Q01"],
        (ppm2 >= df["Q01"]) & (ppm2 < df["Q05"]),
        (ppm2 >= df["Q05"]) & (ppm2 < df["Q25"]),
        (ppm2 >= df["Q25"]) & (ppm2 <= df["Q75"]),
        (ppm2 > df["Q75"]) & (ppm2 <= df["Q95"]),
        (ppm2 > df["Q95"]) & (ppm2 <= df["Q99"]),
        ppm2 > df["Q99"],
    ]
    labels = [
        "below_q01",
        "q01_q05",
        "q05_q25",
        "q25_q75",
        "q75_q95",
        "q95_q99",
        "above_q99",
    ]
    df["local_ppm2_band"] = np.select(conditions, labels, default="unknown")

    weight_map = {
        "q25_q75": 1.0,
        "q05_q25": 0.9,
        "q75_q95": 0.9,
        "q01_q05": 0.5,
        "q95_q99": 0.5,
        "below_q01": 0.2,
        "above_q99": 0.2,
        "unknown": 0.2,
    }
    df["sample_weight"] = df["local_ppm2_band"].map(weight_map).fillna(0.2)
    return df


def apply_local_quantile_tables(df: pd.DataFrame, quantile_tables: Dict[str, pd.DataFrame], ppm2_col: str = "price_per_m2") -> pd.DataFrame:
    enriched = df.copy()
    quantiles = _fill_quantiles_by_hierarchy(enriched, quantile_tables)
    for col in QUANTILE_COLUMNS + ["quantile_source"]:
        enriched[col] = quantiles[col]
    enriched = assign_local_bands(enriched, ppm2_col=ppm2_col)
    return enriched


def prepare_outlier_artifacts(train_df: pd.DataFrame, other_dfs: Dict[str, pd.DataFrame] | None = None, ppm2_col: str = "price_per_m2") -> OutlierArtifacts:
    other_dfs = other_dfs or {}
    train_clean = remove_impossible_rows(train_df)
    quantile_tables = fit_local_quantile_tables(train_clean, ppm2_col=ppm2_col)
    train_enriched = apply_local_quantile_tables(train_clean, quantile_tables, ppm2_col=ppm2_col)

    enriched_others: Dict[str, pd.DataFrame] = {}
    for name, df in other_dfs.items():
        df_clean = remove_impossible_rows(df)
        enriched_others[name] = apply_local_quantile_tables(df_clean, quantile_tables, ppm2_col=ppm2_col)

    return OutlierArtifacts(train_df=train_enriched, other_dfs=enriched_others, quantile_tables=quantile_tables)


def quantile_diagnostics(quantile_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for level_name, table in quantile_tables.items():
        rows.append({"level": level_name, "groups": len(table)})
    return pd.DataFrame(rows)
