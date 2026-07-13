from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from common import RANDOM_STATE, ROOT


DEFAULT_INPUT = ROOT / "draft" / "pre_retrain_feature_fixes" / "fixed_landed_residential_house_modeling_dataset.parquet"
DEFAULT_OUTPUT_DIR = ROOT / "draft" / "deduplication"

STOPWORDS = {
    "ban",
    "nha",
    "dat",
    "can",
    "gap",
    "chinh",
    "chu",
    "gia",
    "ty",
    "trieu",
    "mat",
    "tien",
    "hem",
    "ngo",
    "duong",
    "pho",
    "phuong",
    "quan",
    "huyen",
    "thanh",
    "pho",
    "dien",
    "tich",
    "shr",
    "so",
    "hong",
    "do",
    "lh",
    "lien",
    "he",
    "zalo",
}


def normalize_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value).lower()
    text = text.replace("đ", "d")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\b(tp|tp\.|q|q\.|h|h\.|p|p\.|x|x\.|tt|tt\.)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def significant_tokens(text: str, max_tokens: int = 14) -> list[str]:
    tokens = []
    for token in text.split():
        if token in STOPWORDS:
            continue
        if token.isdigit():
            continue
        if len(token) <= 2:
            continue
        tokens.append(token)
    return tokens[:max_tokens]


def fingerprint(text: str) -> str:
    tokens = significant_tokens(text)
    if not tokens:
        return "missing"
    return " ".join(sorted(set(tokens))[:12])


def token_set(text: str) -> frozenset[str]:
    return frozenset(significant_tokens(text, max_tokens=20))


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    intersection = len(left & right)
    union = len(left | right)
    return intersection / union if union else 0.0


def safe_ratio(a: float, b: float) -> float:
    if pd.isna(a) or pd.isna(b) or a <= 0 or b <= 0:
        return np.inf
    return max(a, b) / min(a, b)


class UnionFind:
    def __init__(self, values: Iterable[int]):
        self.parent = {int(v): int(v) for v in values}
        self.rank = {int(v): 0 for v in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            self.parent[root_left] = root_right
        elif self.rank[root_left] > self.rank[root_right]:
            self.parent[root_right] = root_left
        else:
            self.parent[root_right] = root_left
            self.rank[root_left] += 1


def prepare_blocking_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["title", "address", "addr_last_1", "addr_last_2", "addr_last_3", "street_proxy", "property_subtype"]:
        if col not in out.columns:
            out[col] = "missing"

    out["title_norm"] = out["title"].map(normalize_text)
    out["address_norm"] = out["address"].map(normalize_text)
    out["title_tokens"] = out["title_norm"].map(token_set)
    out["address_tokens"] = out["address_norm"].map(token_set)
    out["street_norm"] = out["street_proxy"].map(normalize_text).replace("", "missing")
    out["district_norm"] = out["addr_last_2"].map(normalize_text).replace("", "missing")
    out["ward_norm"] = out["addr_last_3"].map(normalize_text).replace("", "missing")
    out["province_norm"] = out["addr_last_1"].map(normalize_text).replace("", "missing")
    out["subtype_norm"] = out["property_subtype"].map(normalize_text).replace("", "missing")
    out["title_fp"] = out["title_norm"].map(fingerprint)

    area = pd.to_numeric(out["area_m2"], errors="coerce")
    out["area_bin_2m"] = np.where(area.notna(), np.round(area / 2.0).astype("Int64").astype(str), "missing")
    out["area_bin_5m"] = np.where(area.notna(), np.round(area / 5.0).astype("Int64").astype(str), "missing")
    out["location_key"] = (
        out["province_norm"].astype(str)
        + "|"
        + out["district_norm"].astype(str)
        + "|"
        + out["ward_norm"].astype(str)
        + "|"
        + out["street_norm"].astype(str)
    )
    out["strict_block_key"] = (
        out["location_key"]
        + "|"
        + out["subtype_norm"].astype(str)
        + "|"
        + out["area_bin_2m"].astype(str)
        + "|"
        + out["title_fp"].astype(str)
    )
    out["loose_block_key"] = (
        out["location_key"]
        + "|"
        + out["subtype_norm"].astype(str)
        + "|"
        + out["area_bin_5m"].astype(str)
    )
    return out


def should_link(left: pd.Series, right: pd.Series, area_tolerance: float, price_ratio_limit: float) -> bool:
    area_ratio = safe_ratio(float(left["area_m2"]), float(right["area_m2"]))
    if area_ratio > (1.0 + area_tolerance):
        return False

    title_sim = jaccard(left["title_tokens"], right["title_tokens"])
    address_sim = jaccard(left["address_tokens"], right["address_tokens"])
    price_ratio = safe_ratio(float(left["price_vnd"]), float(right["price_vnd"]))

    same_street = left["location_key"] == right["location_key"]
    strong_text_match = title_sim >= 0.78 or address_sim >= 0.85
    acceptable_price = price_ratio <= price_ratio_limit

    # If text/address is nearly identical, tolerate wider price spread because price
    # disagreement is exactly the duplicated-label noise this audit is trying to expose.
    if same_street and strong_text_match and price_ratio <= max(price_ratio_limit, 3.0):
        return True
    if same_street and title_sim >= 0.62 and address_sim >= 0.70 and acceptable_price:
        return True
    return False


def cluster_duplicates(
    df: pd.DataFrame,
    area_tolerance: float,
    price_ratio_limit: float,
    max_pairwise_block: int,
) -> pd.Series:
    uf = UnionFind(df.index)

    # Exact-ish pass: same canonical location, subtype, tight area bin, and title fingerprint.
    strict_sizes = df.groupby("strict_block_key", observed=True).size()
    for key, size in strict_sizes[strict_sizes > 1].items():
        idx = df.index[df["strict_block_key"] == key].tolist()
        first = idx[0]
        for other in idx[1:]:
            uf.union(first, other)

    # Conservative fuzzy pass inside bounded loose + fingerprint-prefix blocks.
    # This keeps candidate counts small enough for full-dataset execution.
    fuzzy_key = df["loose_block_key"].astype(str) + "|" + df["title_fp"].astype(str).str[:24]
    for _, block in df.groupby(fuzzy_key, observed=True):
        if len(block) < 2:
            continue
        if len(block) > max_pairwise_block:
            # Avoid quadratic explosion in huge market buckets. The strict pass above
            # already catches the high-confidence duplicates in these blocks.
            continue
        rows = list(block.iterrows())
        for i in range(len(rows)):
            left_idx, left = rows[i]
            for j in range(i + 1, len(rows)):
                right_idx, right = rows[j]
                if should_link(left, right, area_tolerance, price_ratio_limit):
                    uf.union(int(left_idx), int(right_idx))

    roots = pd.Series({idx: uf.find(int(idx)) for idx in df.index}, name="cluster_root")
    root_to_number = {root: number for number, root in enumerate(sorted(roots.unique()), start=1)}
    return roots.map(root_to_number).astype("int64")


def choose_canonical_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    source_quality = out.get("source_quality_tier", pd.Series("other_source", index=out.index)).fillna("other_source").astype(str)
    source_score = source_quality.map({"major_portal": 3, "thocu_source": 2, "other_source": 1}).fillna(1)
    completeness_cols = [
        "area_m2",
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
    completeness = pd.Series(0, index=out.index, dtype="float64")
    for col in completeness_cols:
        if col not in out.columns:
            continue
        s = out[col]
        if pd.api.types.is_numeric_dtype(s):
            completeness += s.notna().astype(int)
        else:
            completeness += (~s.fillna("missing").astype(str).str.lower().isin(["missing", "_", "", "nan"])).astype(int)

    out["_source_score"] = source_score
    out["_completeness_score"] = completeness
    out["_listing_date_sort"] = pd.to_datetime(out.get("listing_date", pd.Series(pd.NaT, index=out.index)), errors="coerce")
    out = out.sort_values(
        ["dedup_cluster_id", "_completeness_score", "_source_score", "_listing_date_sort", "text_len"],
        ascending=[True, False, False, False, False],
        kind="mergesort",
    )
    canonical = out.groupby("dedup_cluster_id", observed=True).head(1).copy()
    return canonical.drop(columns=["_source_score", "_completeness_score", "_listing_date_sort"], errors="ignore")


def add_cluster_stats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    group = out.groupby("dedup_cluster_id", observed=True, sort=False)
    stats = group.agg(
        dedup_cluster_size=("id", "size"),
        cluster_price_vnd_median=("price_vnd", "median"),
        cluster_price_vnd_min=("price_vnd", "min"),
        cluster_price_vnd_max=("price_vnd", "max"),
        cluster_area_m2_median=("area_m2", "median"),
        cluster_area_m2_min=("area_m2", "min"),
        cluster_area_m2_max=("area_m2", "max"),
    )
    if "source" in out.columns:
        source_counts = out[["dedup_cluster_id", "source"]].drop_duplicates().groupby("dedup_cluster_id", sort=False).size()
        stats["cluster_source_count"] = source_counts.reindex(stats.index).fillna(0).astype("int64")
    else:
        stats["cluster_source_count"] = 0
    stats["cluster_price_spread_ratio"] = stats["cluster_price_vnd_max"] / stats["cluster_price_vnd_min"].replace(0, np.nan)
    stats["cluster_area_spread_ratio"] = stats["cluster_area_m2_max"] / stats["cluster_area_m2_min"].replace(0, np.nan)
    out = out.merge(stats.reset_index(), on="dedup_cluster_id", how="left")
    out["dedup_cluster_weight"] = 1.0 / np.sqrt(out["dedup_cluster_size"].clip(lower=1))
    out["is_duplicate_cluster"] = out["dedup_cluster_size"] > 1
    out["is_conflicting_duplicate_cluster"] = (
        out["is_duplicate_cluster"]
        & (out["cluster_price_spread_ratio"].fillna(1) >= 1.25)
    )
    return out


def refresh_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["price_vnd"] = out["cluster_price_vnd_median"]
    out["price_per_m2"] = out["price_vnd"] / out["area_m2"].replace(0, np.nan)
    out["price_per_m2_million"] = out["price_per_m2"] / 1_000_000
    out["log_price"] = np.log1p(out["price_vnd"])
    out["log_price_per_m2"] = np.log1p(out["price_per_m2"])
    return out


def drop_internal_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(
        columns=[
            "title_tokens",
            "address_tokens",
            "title_norm",
            "address_norm",
            "street_norm",
            "district_norm",
            "ward_norm",
            "province_norm",
            "subtype_norm",
            "title_fp",
            "area_bin_2m",
            "area_bin_5m",
            "location_key",
            "strict_block_key",
            "loose_block_key",
        ],
        errors="ignore",
    )


def write_report(df: pd.DataFrame, canonical: pd.DataFrame, median_price: pd.DataFrame, output_dir: Path, config: dict) -> None:
    duplicate_rows = int(df["is_duplicate_cluster"].sum())
    conflicting_rows = int(df["is_conflicting_duplicate_cluster"].sum())
    duplicate_clusters = int((canonical["dedup_cluster_size"] > 1).sum())
    conflicting_clusters = int((canonical["cluster_price_spread_ratio"].fillna(1) >= 1.25).sum())
    lines = [
        "# Deduplication Audit",
        "",
        f"- Input rows: `{len(df):,}`",
        f"- Canonical deduplicated rows: `{len(canonical):,}`",
        f"- Removed duplicate rows if using canonical dataset: `{len(df) - len(canonical):,}`",
        f"- Duplicate clusters: `{duplicate_clusters:,}`",
        f"- Rows inside duplicate clusters: `{duplicate_rows:,}`",
        f"- Conflicting duplicate clusters with price spread >= 1.25x: `{conflicting_clusters:,}`",
        f"- Rows inside conflicting duplicate clusters: `{conflicting_rows:,}`",
        "",
        "## Method",
        "",
        "- Normalize Vietnamese title/address/location text by lowercasing, removing accents/punctuation, and removing common listing words.",
        "- Block candidates by canonical province, district, ward, street, property subtype, area bin, and title fingerprint.",
        "- Link rows only when area is close and title/address similarity is high.",
        "- Save both canonical-observed and median-price deduplicated datasets.",
        "",
        "## Recommended Training Input",
        "",
        "- Use `deduplicated_median_price_modeling_dataset.parquet` for the next robust retrain because it keeps one row per cluster and uses cluster median price to reduce duplicated label noise.",
        "- Keep `clustered_all_rows.parquet` for audit and optional sample weighting experiments.",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(config, ensure_ascii=False, indent=2),
        "```",
    ]
    (output_dir / "DEDUPLICATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--area-tolerance", type=float, default=0.08)
    parser.add_argument("--price-ratio-limit", type=float, default=2.0)
    parser.add_argument("--max-pairwise-block", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path).reset_index(drop=True)
    prepared = prepare_blocking_features(df)
    prepared["dedup_cluster_id"] = cluster_duplicates(
        prepared,
        area_tolerance=args.area_tolerance,
        price_ratio_limit=args.price_ratio_limit,
        max_pairwise_block=args.max_pairwise_block,
    )
    clustered = add_cluster_stats(prepared)
    canonical = choose_canonical_rows(clustered)
    median_price = refresh_target_columns(canonical)

    clustered_export = drop_internal_columns(clustered)
    canonical_export = drop_internal_columns(canonical)
    median_price_export = drop_internal_columns(median_price)

    clustered_export.to_parquet(output_dir / "clustered_all_rows.parquet", index=False)
    canonical_export.to_parquet(output_dir / "deduplicated_canonical_modeling_dataset.parquet", index=False)
    median_price_export.to_parquet(output_dir / "deduplicated_median_price_modeling_dataset.parquet", index=False)

    review_cols = [
        "dedup_cluster_id",
        "dedup_cluster_size",
        "is_conflicting_duplicate_cluster",
        "cluster_price_spread_ratio",
        "cluster_area_spread_ratio",
        "id",
        "title",
        "address",
        "source",
        "price_vnd",
        "area_m2",
        "price_per_m2",
        "property_subtype",
        "property_subtype_refined",
        "addr_last_1",
        "addr_last_2",
        "addr_last_3",
        "street_proxy",
        "url",
    ]
    review_cols = [col for col in review_cols if col in clustered.columns]
    clustered[clustered["is_duplicate_cluster"]].sort_values(
        ["dedup_cluster_size", "dedup_cluster_id", "price_vnd"],
        ascending=[False, True, True],
    )[review_cols].head(5000).to_csv(output_dir / "duplicate_cluster_review_top5000.csv", index=False)

    clustered[clustered["is_conflicting_duplicate_cluster"]].sort_values(
        ["cluster_price_spread_ratio", "dedup_cluster_id"],
        ascending=[False, True],
    )[review_cols].head(5000).to_csv(output_dir / "conflicting_duplicate_review_top5000.csv", index=False)

    cluster_summary = canonical[
        [
            "dedup_cluster_id",
            "dedup_cluster_size",
            "cluster_price_spread_ratio",
            "cluster_area_spread_ratio",
            "cluster_source_count",
            "price_vnd",
            "cluster_price_vnd_median",
            "area_m2",
            "cluster_area_m2_median",
            "property_subtype",
            "property_subtype_refined",
            "addr_last_1",
            "addr_last_2",
            "addr_last_3",
            "street_proxy",
            "title",
        ]
    ].sort_values(["dedup_cluster_size", "cluster_price_spread_ratio"], ascending=[False, False])
    cluster_summary.to_csv(output_dir / "cluster_summary.csv", index=False)

    metrics = {
        "input_path": str(input_path),
        "input_rows": int(len(df)),
        "clustered_rows": int(len(clustered)),
        "deduplicated_rows": int(len(canonical)),
        "removed_duplicate_rows": int(len(df) - len(canonical)),
        "duplicate_clusters": int((canonical["dedup_cluster_size"] > 1).sum()),
        "rows_in_duplicate_clusters": int(clustered["is_duplicate_cluster"].sum()),
        "conflicting_duplicate_clusters": int((canonical["cluster_price_spread_ratio"].fillna(1) >= 1.25).sum()),
        "rows_in_conflicting_duplicate_clusters": int(clustered["is_conflicting_duplicate_cluster"].sum()),
        "area_tolerance": args.area_tolerance,
        "price_ratio_limit": args.price_ratio_limit,
        "max_pairwise_block": args.max_pairwise_block,
        "random_state": RANDOM_STATE,
    }
    (output_dir / "deduplication_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(clustered, canonical, median_price, output_dir, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("Saved deduplication outputs to", output_dir)


if __name__ == "__main__":
    main()
