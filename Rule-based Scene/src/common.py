from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


RANDOM_STATE = 42
MIN_SUBTYPE_ROWS = 200
MIN_ROWS_FOR_SUBTYPE_MODEL = 500

ROOT = Path("/Users/dienmayhaituyet/Documents/AVM/Rule-based Scene")
RAW_DATA_DIR = ROOT / "data" / "raw"
PROCESSED_DATA_DIR = ROOT / "data" / "processed"
NOTEBOOK_DIR = ROOT / "notebooks"
SRC_DIR = ROOT / "src"
VIS_DIR = ROOT / "Visualizations"
VIS_OUTPUT_DIR = VIS_DIR / "visualizations"
EVAL_DIR = ROOT / "Evaluation"

for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, NOTEBOOK_DIR, SRC_DIR, VIS_DIR, VIS_OUTPUT_DIR, EVAL_DIR]:
    path.mkdir(parents=True, exist_ok=True)


EXCLUDE_APARTMENT = ["chung cư", "chung cu", "căn hộ", "can ho", "apartment", "condotel", "tập thể", "tap the"]
EXCLUDE_LAND_ONLY = [
    "đất nền",
    "dat nen",
    "bán đất",
    "ban dat",
    "đất thổ cư",
    "dat tho cu",
    "đất ở",
    "dat o",
    "đất vườn",
    "dat vuon",
    "đất nông nghiệp",
    "dat nong nghiep",
]
EXCLUDE_COMMERCIAL_ONLY = ["văn phòng", "van phong", "mặt bằng", "mat bang", "kho xưởng", "kho xuong"]
VILLA = ["biệt thự", "biet thu", "villa"]
SHOPHOUSE = ["shophouse", "shop house", "nhà phố thương mại", "nha pho thuong mai"]
TOWNHOUSE_PROJECT = ["liền kề", "lien ke", "khu đô thị", "khu do thi", "dự án", "du an", "compound"]
FRONTAGE = ["mặt tiền", "mat tien", "mặt phố", "mat pho", "mặt đường", "mat duong", "mt ", "mt."]
ALLEY = ["hẻm", "hem", "ngõ", "ngo", "kiệt", "kiet"]
NORMAL_HOUSE = ["nhà thổ cư", "nha tho cu", "thổ cư", "tho cu", "nhà riêng", "nha rieng", "nhà phố", "nha pho"]
BUSINESS = ["kinh doanh", "kd", "cho thuê", "cho thue", "dòng tiền", "dong tien", "buôn bán", "mở shop", "mo shop"]
CAR_ACCESS = ["ô tô", "oto", "xe hơi", "xe hoi", "ôtô", "xe tải", "xe tai"]
CORNER = ["góc", "goc", "2 mặt tiền", "hai mặt tiền", "lô góc", "lo goc"]
LEGAL = ["sổ đỏ", "so do", "sổ hồng", "so hong", "pháp lý", "phap ly", "hoàn công", "hoan cong"]
NEW_HOUSE = ["nhà mới", "nha moi", "mới xây", "moi xay", "xây mới", "xay moi"]
RENOVATED = ["mới sửa", "moi sua", "đã sửa", "da sua", "renovated", "nâng cấp", "nang cap"]
PROJECT = ["dự án", "du an", "compound", "khu dân cư", "khu dan cu", "phân khu", "phan khu"]
URBAN = ["khu đô thị", "khu do thi", "đô thị", "do thi", "cityland", "vinhomes", "phú mỹ hưng", "phu my hung"]


GLOBAL_NUMERIC_FEATURES = [
    "area_m2",
    "num_floors",
    "floor_area_m2",
    "frontage_m",
    "road_width_m",
    "width_m",
    "length_m",
    "bedrooms",
    "bathrooms",
    "is_frontage_text",
    "is_alley_text",
    "is_business_text",
    "is_car_access_text",
    "is_corner_text",
    "is_villa_text",
    "has_legal_text",
    "is_new_house_text",
    "is_renovated_text",
    "is_project_text",
    "is_urban_area_text",
    "text_len",
    "word_count",
    "listing_year",
    "listing_month",
    "floors_x_area",
    "frontage_x_road_width",
    "floors_x_frontage",
    "floors_x_business",
    "car_access_x_road_width",
    "area_x_frontage",
]
GLOBAL_CATEGORICAL_FEATURES = [
    "property_subtype",
    "addr_last_1",
    "addr_last_2",
    "addr_last_3",
    "street_proxy",
    "legal",
    "door_direction",
    "source",
    "project_name",
]

SUBTYPE_FEATURE_MAP = {
    "normal_house": {
        "numeric": [
            "area_m2",
            "num_floors",
            "floor_area_m2",
            "bedrooms",
            "bathrooms",
            "width_m",
            "length_m",
            "is_car_access_text",
            "has_legal_text",
            "is_new_house_text",
            "is_renovated_text",
            "text_len",
            "word_count",
            "listing_year",
            "listing_month",
        ],
        "categorical": ["addr_last_1", "addr_last_2", "addr_last_3", "street_proxy", "legal", "door_direction", "source"],
    },
    "alley_house": {
        "numeric": [
            "area_m2",
            "num_floors",
            "floor_area_m2",
            "road_width_m",
            "is_alley_text",
            "is_frontage_text",
            "is_car_access_text",
            "has_legal_text",
            "bedrooms",
            "bathrooms",
            "listing_year",
            "listing_month",
            "text_len",
            "word_count",
        ],
        "categorical": ["addr_last_1", "addr_last_2", "addr_last_3", "street_proxy", "legal", "source"],
    },
    "alley_frontage_house": {
        "numeric": [
            "area_m2",
            "num_floors",
            "floor_area_m2",
            "frontage_m",
            "road_width_m",
            "is_alley_text",
            "is_frontage_text",
            "is_business_text",
            "is_car_access_text",
            "is_corner_text",
            "frontage_x_road_width",
            "floors_x_frontage",
            "floors_x_business",
            "area_x_frontage",
            "listing_year",
            "listing_month",
        ],
        "categorical": ["addr_last_1", "addr_last_2", "addr_last_3", "street_proxy", "legal", "source"],
    },
    "frontage_house": {
        "numeric": [
            "area_m2",
            "num_floors",
            "floor_area_m2",
            "frontage_m",
            "road_width_m",
            "is_business_text",
            "is_corner_text",
            "is_car_access_text",
            "has_legal_text",
            "frontage_x_road_width",
            "floors_x_frontage",
            "floors_x_business",
            "car_access_x_road_width",
            "area_x_frontage",
            "listing_year",
            "listing_month",
        ],
        "categorical": ["addr_last_1", "addr_last_2", "addr_last_3", "street_proxy", "legal", "source"],
    },
    "villa": {
        "numeric": [
            "area_m2",
            "num_floors",
            "floor_area_m2",
            "width_m",
            "length_m",
            "frontage_m",
            "bedrooms",
            "bathrooms",
            "is_villa_text",
            "is_corner_text",
            "is_car_access_text",
            "has_legal_text",
            "is_project_text",
            "is_urban_area_text",
            "area_x_frontage",
            "listing_year",
            "listing_month",
        ],
        "categorical": ["addr_last_1", "addr_last_2", "addr_last_3", "street_proxy", "legal", "source", "project_name"],
    },
    "townhouse_project": {
        "numeric": [
            "area_m2",
            "num_floors",
            "floor_area_m2",
            "frontage_m",
            "road_width_m",
            "is_project_text",
            "is_urban_area_text",
            "has_legal_text",
            "project_x_area",
            "frontage_x_road_width",
            "listing_year",
            "listing_month",
        ],
        "categorical": ["addr_last_1", "addr_last_2", "addr_last_3", "street_proxy", "legal", "source", "project_name"],
    },
    "shophouse": {
        "numeric": [
            "area_m2",
            "num_floors",
            "floor_area_m2",
            "frontage_m",
            "road_width_m",
            "is_business_text",
            "is_corner_text",
            "is_project_text",
            "is_car_access_text",
            "frontage_x_road_width",
            "floors_x_frontage",
            "floors_x_business",
            "business_x_road_width",
            "area_x_frontage",
            "listing_year",
            "listing_month",
        ],
        "categorical": ["addr_last_1", "addr_last_2", "addr_last_3", "street_proxy", "legal", "source", "project_name"],
    },
}


TARGET_DEFS = {
    "price": {
        "target_col": "log_price",
        "actual_col": "price_vnd",
        "prediction_col": "pred_price_vnd",
        "human_name": "Total price",
    },
    "price_per_m2": {
        "target_col": "log_price_per_m2",
        "actual_col": "price_per_m2",
        "prediction_col": "pred_price_per_m2",
        "human_name": "Price per m2",
    },
}


@dataclass
class SplitData:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    split_method: str


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    return re.sub(r"\s+", " ", text)


def vn_float_from_text(value):
    if pd.isna(value):
        return np.nan
    text = normalize_text(value)
    if text in {"", "none", "nan", "---", "_", "thỏa thuận", "thoả thuận", "liên hệ"}:
        return np.nan
    text = text.replace(",", ".")
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    if not nums:
        return np.nan
    try:
        return float(nums[0])
    except Exception:
        return np.nan


def clean_measurement(value, min_val=None, max_val=None):
    val = vn_float_from_text(value)
    if pd.isna(val):
        return np.nan
    if min_val is not None and val < min_val:
        return np.nan
    if max_val is not None and val > max_val:
        return np.nan
    return val


def parse_price_vnd(price_text, area_m2=np.nan):
    if pd.isna(price_text):
        return np.nan
    text = normalize_text(price_text)
    text = text.replace(",", ".")
    text = text.replace("tỉ", "tỷ").replace("ty", "tỷ").replace("ti", "tỷ")
    text = text.replace("trieu", "triệu").replace(" tr/", " triệu/")
    if text in {"", "none", "nan", "---", "_", "thỏa thuận", "thoả thuận", "liên hệ"}:
        return np.nan

    ppm2_million = re.search(r"(\d+(?:\.\d+)?)\s*(triệu|tr)\s*/\s*m\s*2?", text)
    ppm2_billion = re.search(r"(\d+(?:\.\d+)?)\s*tỷ\s*/\s*m\s*2?", text)
    if ppm2_million:
        return float(ppm2_million.group(1)) * 1_000_000 * area_m2 if pd.notna(area_m2) else np.nan
    if ppm2_billion:
        return float(ppm2_billion.group(1)) * 1_000_000_000 * area_m2 if pd.notna(area_m2) else np.nan

    m = re.search(r"(\d+(?:\.\d+)?)\s*tỷ\s*(\d+)?", text)
    if m:
        main = float(m.group(1))
        decimal_part = m.group(2)
        if decimal_part is not None and len(decimal_part) <= 2:
            main += float("0." + decimal_part)
        return main * 1_000_000_000

    m = re.search(r"(\d+(?:\.\d+)?)\s*(triệu|tr)\b", text)
    if m:
        return float(m.group(1)) * 1_000_000

    value = vn_float_from_text(text)
    if pd.isna(value):
        return np.nan
    if value >= 1_000_000:
        return value
    if 100 <= value < 1_000:
        return value * 1_000_000
    if 1 <= value < 100:
        return value * 1_000_000_000
    return np.nan


def safe_divide(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full(len(a), np.nan, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b) & (b != 0)
    out[mask] = a[mask] / b[mask]
    return out


def has_any(text: str, keywords: Iterable[str]) -> bool:
    normalized = normalize_text(text)
    return any(keyword in normalized for keyword in keywords)


def series_has_any(series: pd.Series, keywords: Iterable[str]) -> pd.Series:
    escaped = [re.escape(keyword) for keyword in keywords if keyword]
    if not escaped:
        return pd.Series(False, index=series.index)
    pattern = "|".join(escaped)
    return series.str.contains(pattern, regex=True, na=False)


def classify_property_subtype_v2(text: str) -> str:
    t = normalize_text(text)
    if has_any(t, EXCLUDE_APARTMENT):
        return "exclude_apartment"
    if has_any(t, EXCLUDE_LAND_ONLY):
        return "exclude_land_only"
    if has_any(t, EXCLUDE_COMMERCIAL_ONLY) and not has_any(t, NORMAL_HOUSE + VILLA + SHOPHOUSE + TOWNHOUSE_PROJECT):
        return "exclude_commercial_only"
    if has_any(t, VILLA):
        return "villa"
    if has_any(t, SHOPHOUSE):
        return "shophouse"
    if has_any(t, TOWNHOUSE_PROJECT):
        return "townhouse_project"
    alley_flag = has_any(t, ALLEY)
    frontage_flag = has_any(t, FRONTAGE)
    if alley_flag and frontage_flag:
        return "alley_frontage_house"
    if frontage_flag:
        return "frontage_house"
    if alley_flag:
        return "alley_house"
    if has_any(t, NORMAL_HOUSE) or "thocu" in t or "tho_cu" in t:
        return "normal_house"
    return "unknown"


def address_part(address, idx_from_right):
    text = normalize_text(address)
    if not text:
        return "missing"
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) >= idx_from_right:
        return parts[-idx_from_right]
    return "missing"


def first_existing(df: pd.DataFrame, candidates: List[str]):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def create_interaction(df: pd.DataFrame, target_col: str, left_col: str, right_col: str):
    if left_col in df.columns and right_col in df.columns:
        left = pd.to_numeric(df[left_col], errors="coerce")
        right = pd.to_numeric(df[right_col], errors="coerce")
        df[target_col] = left * right
    else:
        df[target_col] = np.nan


def load_raw_data(raw_csv: Path | None = None) -> pd.DataFrame:
    raw_csv = raw_csv or RAW_DATA_DIR / "mick_df_full_raw.csv"
    return pd.read_csv(raw_csv)


def preprocess_raw_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    text_fields = [c for c in ["title", "address", "description", "source", "legal", "project_name"] if c in df.columns]
    df["all_text"] = ""
    for col in text_fields:
        df["all_text"] = df["all_text"] + " " + df[col].fillna("").astype(str)
    df["all_text_lower"] = df["all_text"].map(normalize_text)

    df["area_m2"] = df["area"].apply(lambda x: clean_measurement(x, min_val=5, max_val=10000)) if "area" in df.columns else np.nan
    df["price_vnd"] = [parse_price_vnd(p, a) for p, a in zip(df.get("price", pd.Series(dtype=object)), df["area_m2"])] if "price" in df.columns else np.nan

    numeric_map = {
        "number_of_floors": ("num_floors", 0, 100),
        "house_front": ("frontage_m", 0.5, 200),
        "road_wide": ("road_width_m", 0.5, 200),
        "bedroom": ("bedrooms", 0, 100),
        "bathroom": ("bathrooms", 0, 100),
        "width": ("width_m", 0.5, 200),
        "length": ("length_m", 0.5, 1000),
    }
    for raw_col, (new_col, lo, hi) in numeric_map.items():
        if raw_col in df.columns:
            df[new_col] = df[raw_col].apply(lambda x: clean_measurement(x, min_val=lo, max_val=hi))
        else:
            df[new_col] = np.nan

    legal_col = first_existing(df, ["legal", "legal_status", "juridical"])
    df["legal"] = df[legal_col].fillna("missing").astype(str) if legal_col else "missing"
    direction_col = first_existing(df, ["door_direction", "direction", "house_direction"])
    df["door_direction"] = df[direction_col].fillna("missing").astype(str) if direction_col else "missing"
    project_col = first_existing(df, ["project_name", "project", "project_title"])
    df["project_name"] = df[project_col].fillna("missing").astype(str) if project_col else "missing"
    address_col = first_existing(df, ["address", "full_address", "location", "street"])
    df["address"] = df[address_col].fillna("").astype(str) if address_col else ""

    df["addr_last_1"] = df["address"].apply(lambda x: address_part(x, 1))
    df["addr_last_2"] = df["address"].apply(lambda x: address_part(x, 2))
    df["addr_last_3"] = df["address"].apply(lambda x: address_part(x, 3))
    df["street_proxy"] = df["address"].apply(lambda x: address_part(x, 4))

    for date_col in ["updated_at", "post_date", "created_at", "date"]:
        if date_col in df.columns:
            df[date_col + "_parsed"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    parsed_candidates = [c for c in df.columns if c.endswith("_parsed")]
    main_date_col = "updated_at_parsed" if "updated_at_parsed" in parsed_candidates else (parsed_candidates[0] if parsed_candidates else None)
    if main_date_col:
        df["listing_date"] = df[main_date_col]
        df["listing_year"] = df["listing_date"].dt.year
        df["listing_month"] = df["listing_date"].dt.month
    else:
        df["listing_date"] = pd.NaT
        df["listing_year"] = np.nan
        df["listing_month"] = np.nan

    if "url" in df.columns:
        df = df.drop_duplicates(subset=["url"], keep="first")
    elif "id" in df.columns:
        df = df.drop_duplicates(subset=["id"], keep="first")
    else:
        df = df.drop_duplicates()

    text_series = df["all_text_lower"].fillna("")
    exclude_apartment = series_has_any(text_series, EXCLUDE_APARTMENT)
    exclude_land_only = series_has_any(text_series, EXCLUDE_LAND_ONLY)
    exclude_commercial_only = series_has_any(text_series, EXCLUDE_COMMERCIAL_ONLY)
    villa_flag = series_has_any(text_series, VILLA)
    shophouse_flag = series_has_any(text_series, SHOPHOUSE)
    townhouse_project_flag = series_has_any(text_series, TOWNHOUSE_PROJECT)
    frontage_flag = series_has_any(text_series, FRONTAGE)
    alley_flag = series_has_any(text_series, ALLEY)
    normal_house_flag = series_has_any(text_series, NORMAL_HOUSE) | text_series.str.contains("thocu|tho_cu", regex=True, na=False)

    commercial_only_flag = exclude_commercial_only & ~(normal_house_flag | villa_flag | shophouse_flag | townhouse_project_flag)
    df["property_subtype"] = np.select(
        [
            exclude_apartment,
            exclude_land_only,
            commercial_only_flag,
            villa_flag,
            shophouse_flag,
            townhouse_project_flag,
            alley_flag & frontage_flag,
            frontage_flag,
            alley_flag,
            normal_house_flag,
        ],
        [
            "exclude_apartment",
            "exclude_land_only",
            "exclude_commercial_only",
            "villa",
            "shophouse",
            "townhouse_project",
            "alley_frontage_house",
            "frontage_house",
            "alley_house",
            "normal_house",
        ],
        default="unknown",
    )
    df["property_group"] = np.where(
        df["property_subtype"].isin(list(SUBTYPE_FEATURE_MAP.keys())),
        "landed_residential_house",
        "out_of_scope",
    )

    signal_map = {
        "is_frontage_text": FRONTAGE,
        "is_alley_text": ALLEY,
        "is_business_text": BUSINESS,
        "is_car_access_text": CAR_ACCESS,
        "is_corner_text": CORNER,
        "is_villa_text": VILLA,
        "has_legal_text": LEGAL,
        "is_new_house_text": NEW_HOUSE,
        "is_renovated_text": RENOVATED,
        "is_project_text": PROJECT,
        "is_urban_area_text": URBAN,
    }
    for feature_name, keywords in signal_map.items():
        df[feature_name] = series_has_any(text_series, keywords).astype(int)

    df["text_len"] = df["all_text_lower"].str.len()
    df["word_count"] = df["all_text_lower"].str.split().apply(len)
    df["price_per_m2"] = safe_divide(df["price_vnd"], df["area_m2"])
    df["price_per_m2_million"] = df["price_per_m2"] / 1_000_000

    create_interaction(df, "floor_area_m2", "area_m2", "num_floors")
    create_interaction(df, "floors_x_area", "num_floors", "area_m2")
    create_interaction(df, "frontage_x_road_width", "frontage_m", "road_width_m")
    create_interaction(df, "floors_x_frontage", "num_floors", "frontage_m")
    create_interaction(df, "floors_x_business", "num_floors", "is_business_text")
    create_interaction(df, "car_access_x_road_width", "is_car_access_text", "road_width_m")
    create_interaction(df, "area_x_frontage", "area_m2", "frontage_m")
    create_interaction(df, "corner_x_frontage", "is_corner_text", "frontage_m")
    create_interaction(df, "business_x_road_width", "is_business_text", "road_width_m")
    create_interaction(df, "project_x_area", "is_project_text", "area_m2")

    return df


def build_modeling_dataset(df_clean: pd.DataFrame) -> pd.DataFrame:
    avm = df_clean[df_clean["property_group"] == "landed_residential_house"].copy()
    avm = avm[
        avm["price_vnd"].between(100_000_000, 1_000_000_000_000)
        & avm["area_m2"].between(10, 5000)
        & avm["price_per_m2_million"].between(1, 2000)
    ].copy()
    avm["log_price"] = np.log1p(avm["price_vnd"])
    avm["log_price_per_m2"] = np.log1p(avm["price_per_m2"])
    valid_subtypes = avm["property_subtype"].value_counts()
    valid_subtypes = valid_subtypes[valid_subtypes >= MIN_SUBTYPE_ROWS].index
    avm = avm[avm["property_subtype"].isin(valid_subtypes)].copy()
    return avm


def make_split(df_model: pd.DataFrame) -> SplitData:
    use_time_split = df_model["listing_date"].notna().mean() >= 0.7
    if use_time_split:
        ordered = df_model.sort_values("listing_date").reset_index(drop=True)
        split_idx = int(len(ordered) * 0.8)
        train_df = ordered.iloc[:split_idx].copy()
        test_df = ordered.iloc[split_idx:].copy()
        return SplitData(train_df=train_df, test_df=test_df, split_method="time_based")

    train_df, test_df = train_test_split(
        df_model,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df_model["property_subtype"],
    )
    return SplitData(train_df=train_df.copy(), test_df=test_df.copy(), split_method="stratified_random")


def save_processed_data(df_clean: pd.DataFrame, df_model: pd.DataFrame, split_data: SplitData):
    df_clean.to_parquet(PROCESSED_DATA_DIR / "full_cleaned_dataset.parquet", index=False)
    df_model.to_parquet(PROCESSED_DATA_DIR / "landed_residential_house_modeling_dataset.parquet", index=False)
    split_data.train_df.to_parquet(PROCESSED_DATA_DIR / "train_dataset.parquet", index=False)
    split_data.test_df.to_parquet(PROCESSED_DATA_DIR / "test_dataset.parquet", index=False)
    metadata = {
        "split_method": split_data.split_method,
        "clean_rows": int(len(df_clean)),
        "model_rows": int(len(df_model)),
        "train_rows": int(len(split_data.train_df)),
        "test_rows": int(len(split_data.test_df)),
        "subtypes": split_data.train_df["property_subtype"].value_counts().to_dict(),
    }
    (PROCESSED_DATA_DIR / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def load_processed_train_test() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_parquet(PROCESSED_DATA_DIR / "train_dataset.parquet")
    test_df = pd.read_parquet(PROCESSED_DATA_DIR / "test_dataset.parquet")
    return train_df, test_df


def filter_existing_features(df: pd.DataFrame, numeric_features: List[str], categorical_features: List[str]) -> Tuple[List[str], List[str]]:
    return [c for c in numeric_features if c in df.columns], [c for c in categorical_features if c in df.columns]


def global_features(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    return filter_existing_features(df, GLOBAL_NUMERIC_FEATURES, GLOBAL_CATEGORICAL_FEATURES)


def subtype_features(df: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
    out: Dict[str, Dict[str, List[str]]] = {}
    for subtype, feature_dict in SUBTYPE_FEATURE_MAP.items():
        num, cat = filter_existing_features(df, feature_dict["numeric"], feature_dict["categorical"])
        out[subtype] = {"numeric": num, "categorical": cat}
    return out


def build_ridge_pipeline(numeric_features: List[str], categorical_features: List[str], alpha: float) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            ("cat", Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20))]), categorical_features),
        ],
        remainder="drop",
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", Ridge(alpha=alpha, random_state=RANDOM_STATE))])


def build_hgbr_pipeline(numeric_features: List[str], categorical_features: List[str]) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            ("cat", Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), categorical_features),
        ],
        remainder="drop",
    )
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.06,
        max_iter=350,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=RANDOM_STATE,
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


def build_lgbm_pipeline(numeric_features: List[str], categorical_features: List[str], params: Dict) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            ("cat", Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), categorical_features),
        ],
        remainder="drop",
    )
    base_params = {
        "objective": "regression",
        "random_state": RANDOM_STATE,
        "verbosity": -1,
    }
    base_params.update(params)
    model = LGBMRegressor(**base_params)
    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


def metric_row(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {
            "count": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "MAPE": np.nan,
            "Median_APE": np.nan,
            "HitRate_10_pct": np.nan,
            "HitRate_20_pct": np.nan,
            "Bias_pct": np.nan,
            "OvervaluationRate_pct": np.nan,
            "UndervaluationRate_pct": np.nan,
        }

    pe = (y_pred - y_true) / y_true
    ape = np.abs(pe)
    return {
        "count": len(y_true),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": math.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": np.mean(ape) * 100,
        "Median_APE": np.median(ape) * 100,
        "HitRate_10_pct": np.mean(ape <= 0.10) * 100,
        "HitRate_20_pct": np.mean(ape <= 0.20) * 100,
        "Bias_pct": np.mean(pe) * 100,
        "OvervaluationRate_pct": np.mean(pe > 0) * 100,
        "UndervaluationRate_pct": np.mean(pe < 0) * 100,
    }


def build_range_labels(series: pd.Series, q: int = 5, prefix: str = "") -> pd.Series:
    if series.nunique(dropna=True) < 2:
        return pd.Series(["all"] * len(series), index=series.index)
    return pd.qcut(series, q=min(q, series.nunique(dropna=True)), duplicates="drop").astype(str).rename(prefix)


def evaluate_levels(df_eval: pd.DataFrame, actual_col: str, pred_col: str, model_name: str, target_name: str) -> Dict[str, pd.DataFrame]:
    outputs: Dict[str, pd.DataFrame] = {}
    rows = [{"level": "overall_test_set", "group": "ALL", "model": model_name, "target": target_name, **metric_row(df_eval[actual_col], df_eval[pred_col])}]
    outputs["overall_test_set"] = pd.DataFrame(rows)

    def by_group(level_name: str, series: pd.Series) -> pd.DataFrame:
        level_rows = []
        for value, idx in series.groupby(series).groups.items():
            subset = df_eval.loc[idx]
            level_rows.append({"level": level_name, "group": value, "model": model_name, "target": target_name, **metric_row(subset[actual_col], subset[pred_col])})
        return pd.DataFrame(level_rows)

    outputs["by_subtype"] = by_group("by_subtype", df_eval["property_subtype"])
    outputs["by_city_province"] = by_group("by_city_province", df_eval["addr_last_1"].fillna("missing"))
    outputs["by_source"] = by_group("by_source", df_eval["source"].fillna("missing"))
    outputs["by_price_range"] = by_group("by_price_range", build_range_labels(df_eval["price_vnd"], prefix="price_range"))
    outputs["by_area_range"] = by_group("by_area_range", build_range_labels(df_eval["area_m2"], prefix="area_range"))
    return outputs


def evaluate_custom_levels(
    df_eval: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    model_name: str,
    target_name: str,
    group_definitions: Dict[str, pd.Series],
) -> Dict[str, pd.DataFrame]:
    outputs: Dict[str, pd.DataFrame] = {}
    outputs["overall_test_set"] = pd.DataFrame(
        [{"level": "overall_test_set", "group": "ALL", "model": model_name, "target": target_name, **metric_row(df_eval[actual_col], df_eval[pred_col])}]
    )

    for level_name, series in group_definitions.items():
        level_rows = []
        filled = series.fillna("missing")
        for value, idx in filled.groupby(filled).groups.items():
            subset = df_eval.loc[idx]
            level_rows.append({"level": level_name, "group": value, "model": model_name, "target": target_name, **metric_row(subset[actual_col], subset[pred_col])})
        outputs[level_name] = pd.DataFrame(level_rows)
    return outputs


def standard_group_definitions(df_eval: pd.DataFrame) -> Dict[str, pd.Series]:
    groups = {
        "by_subtype": df_eval["property_subtype"],
        "by_region": df_eval["addr_last_1"],
        "by_price_range": build_range_labels(df_eval["price_vnd"], prefix="price_range"),
        "by_area_range": build_range_labels(df_eval["area_m2"], prefix="area_range"),
        "by_source": df_eval["source"],
    }
    if "local_ppm2_band" in df_eval.columns:
        groups["by_outlier_band"] = df_eval["local_ppm2_band"]
    return groups


def pick_best_metric(metrics_df: pd.DataFrame) -> Tuple[float, float]:
    row = metrics_df.iloc[0]
    return float(row["Median_APE"]), float(row["HitRate_20_pct"])


def optimize_ridge(train_df: pd.DataFrame, numeric_features: List[str], categorical_features: List[str], target_col: str, actual_col: str) -> Tuple[float, pd.DataFrame]:
    tr_df, val_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=train_df["property_subtype"],
    )
    candidate_alphas = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    rows = []
    for alpha in candidate_alphas:
        model = build_ridge_pipeline(numeric_features, categorical_features, alpha=alpha)
        feature_cols = numeric_features + categorical_features
        model.fit(tr_df[feature_cols], tr_df[target_col])
        preds = np.expm1(model.predict(val_df[feature_cols]))
        metrics = metric_row(val_df[actual_col], preds)
        rows.append({"alpha": alpha, **metrics})
    results = pd.DataFrame(rows).sort_values(["Median_APE", "HitRate_20_pct", "RMSE"], ascending=[True, False, True]).reset_index(drop=True)
    return float(results.iloc[0]["alpha"]), results


def optimize_lgbm(train_df: pd.DataFrame, numeric_features: List[str], categorical_features: List[str], target_col: str, actual_col: str) -> Tuple[Dict, pd.DataFrame]:
    tr_df, val_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=train_df["property_subtype"],
    )
    grid = list(
        ParameterGrid(
            {
                "n_estimators": [300, 600],
                "learning_rate": [0.03, 0.05],
                "num_leaves": [31, 63],
                "subsample": [0.8],
                "colsample_bytree": [0.8],
                "min_child_samples": [20, 50],
            }
        )
    )
    rows = []
    feature_cols = numeric_features + categorical_features
    for params in grid:
        model = build_lgbm_pipeline(numeric_features, categorical_features, params=params)
        model.fit(tr_df[feature_cols], tr_df[target_col])
        preds = np.expm1(model.predict(val_df[feature_cols]))
        metrics = metric_row(val_df[actual_col], preds)
        rows.append({"params": json.dumps(params, sort_keys=True), **metrics})
    results = pd.DataFrame(rows).sort_values(["Median_APE", "HitRate_20_pct", "RMSE"], ascending=[True, False, True]).reset_index(drop=True)
    best_params = json.loads(results.iloc[0]["params"])
    return best_params, results


def save_level_outputs(level_outputs: Dict[str, pd.DataFrame], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for level_name, df_level in level_outputs.items():
        df_level.to_csv(out_dir / f"{level_name}.csv", index=False)


def save_predictions(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def summarize_model_selection(global_metrics: pd.DataFrame, subtype_metrics: pd.DataFrame, model_name: str, target_name: str) -> pd.DataFrame:
    rows = []
    for subtype in sorted(subtype_metrics["group"].unique()):
        if subtype == "ALL":
            continue
        global_row = subtype_metrics[(subtype_metrics["group"] == subtype) & (subtype_metrics["model"] == f"{model_name} global") & (subtype_metrics["target"] == target_name)]
        subtype_row = subtype_metrics[(subtype_metrics["group"] == subtype) & (subtype_metrics["model"] == f"{model_name} subtype") & (subtype_metrics["target"] == target_name)]
        if global_row.empty or subtype_row.empty:
            continue
        g = global_row.iloc[0]
        s = subtype_row.iloc[0]
        use_subtype = (s["Median_APE"] < g["Median_APE"]) and (s["HitRate_20_pct"] >= g["HitRate_20_pct"])
        rows.append(
            {
                "target": target_name,
                "property_subtype": subtype,
                "recommended_model": f"{model_name} subtype" if use_subtype else f"{model_name} global",
                "global_Median_APE": g["Median_APE"],
                "subtype_Median_APE": s["Median_APE"],
                "global_HitRate_20_pct": g["HitRate_20_pct"],
                "subtype_HitRate_20_pct": s["HitRate_20_pct"],
                "global_Bias_pct": g["Bias_pct"],
                "subtype_Bias_pct": s["Bias_pct"],
            }
        )
    return pd.DataFrame(rows)


def list_metric_columns() -> List[str]:
    return ["MAE", "RMSE", "MAPE", "Median_APE", "HitRate_10_pct", "HitRate_20_pct", "Bias_pct", "OvervaluationRate_pct", "UndervaluationRate_pct"]
