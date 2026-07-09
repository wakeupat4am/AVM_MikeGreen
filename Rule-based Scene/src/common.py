from __future__ import annotations

import json
import math
import re

from extract_from_text import extract_features_from_text
from difflib import get_close_matches
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
HOUSE_TEXT_SIGNAL = [
    "bán nhà",
    "ban nha",
    "nhà ",
    "nha ",
    "tấm",
    "tam",
    "lầu",
    "lau",
    "trệt",
    "tret",
    "phòng ngủ",
    "phong ngu",
    "pn",
    "wc",
    "sổ hồng hoàn công",
    "so hong hoan cong",
    "hoàn công",
    "hoan cong",
    "đang cho thuê",
    "dang cho thue",
]
BUSINESS = ["kinh doanh", "kd", "cho thuê", "cho thue", "dòng tiền", "dong tien", "buôn bán", "mở shop", "mo shop"]
CAR_ACCESS = ["ô tô", "oto", "xe hơi", "xe hoi", "ôtô", "xe tải", "xe tai"]
CORNER = ["góc", "goc", "2 mặt tiền", "hai mặt tiền", "lô góc", "lo goc"]
GENERAL_LANDED_HOUSE = [
    "căn góc",
    "can goc",
    "thổ cư",
    "tho cu",
    "nhà",
    "nha",
    "bán nhà",
    "ban nha",
]
LEGAL = ["sổ đỏ", "so do", "sổ hồng", "so hong", "pháp lý", "phap ly", "hoàn công", "hoan cong"]
NEW_HOUSE = ["nhà mới", "nha moi", "mới xây", "moi xay", "xây mới", "xay moi"]
RENOVATED = ["mới sửa", "moi sua", "đã sửa", "da sua", "renovated", "nâng cấp", "nang cap"]
PROJECT = ["dự án", "du an", "compound", "khu dân cư", "khu dan cu", "phân khu", "phan khu"]
URBAN = ["khu đô thị", "khu do thi", "đô thị", "do thi", "cityland", "vinhomes", "phú mỹ hưng", "phu my hung"]

PROVINCE_ALIAS_MAP = {
    "hồ chí minh": ["hồ chí minh", "ho chi minh", "tp.hcm", "tphcm", "tp hồ chí minh", "tp. hồ chí minh", "sài gòn", "sai gon", "hcm"],
    "hà nội": ["hà nội", "ha noi", "hn", "hà nội"],
    "đà nẵng": ["đà nẵng", "da nang", "tp. đà nẵng", "tp.đà nẵng", "thành phố đà nẵng"],
    "hải phòng": ["hải phòng", "hai phong", "tp. hải phòng", "tp.hải phòng", "thành phố hải phòng"],
    "cần thơ": ["cần thơ", "can tho", "tp. cần thơ", "tp.cần thơ", "thành phố cần thơ"],
    "bà rịa vũng tàu": ["bà rịa vũng tàu", "ba ria vung tau", "bà rịa - vũng tàu"],
    "đồng nai": ["đồng nai", "dong nai", "đồng nai."],
    "bình dương": ["bình dương", "binh duong", "bình dương."],
    "khánh hòa": ["khánh hòa", "khanh hoa", "khánh hòa."],
    "long an": ["long an"],
    "đắk lắk": ["đắk lắk", "dak lak", "đăk lăk"],
    "huế": ["huế", "hue", "thừa thiên huế", "thua thien hue"],
    "lâm đồng": ["lâm đồng", "lam dong"],
    "bắc ninh": ["bắc ninh", "bac ninh"],
    "quảng nam": ["quảng nam", "quang nam"],
    "bình thuận": ["bình thuận", "binh thuan"],
    "quảng ninh": ["quảng ninh", "quang ninh"],
    "tây ninh": ["tây ninh", "tay ninh"],
    "thanh hóa": ["thanh hóa", "thanh hoa", "thanh hóa."],
    "bình định": ["bình định", "binh dinh"],
    "nam định": ["nam định", "nam dinh"],
    "thái bình": ["thái bình", "thai binh"],
    "nghệ an": ["nghệ an", "nghe an"],
    "hải dương": ["hải dương", "hai duong"],
    "vĩnh phúc": ["vĩnh phúc", "vinh phuc"],
    "hòa bình": ["hòa bình", "hoa binh"],
    "tiền giang": ["tiền giang", "tien giang"],
    "phú yên": ["phú yên", "phu yen"],
    "bình phước": ["bình phước", "binh phuoc"],
    "kiên giang": ["kiên giang", "kien giang"],
    "trà vinh": ["trà vinh", "tra vinh"],
    "hưng yên": ["hưng yên", "hung yen"],
    "hà nam": ["hà nam", "ha nam"],
    "bắc giang": ["bắc giang", "bac giang"],
    "ninh thuận": ["ninh thuận", "ninh thuan"],
    "quảng ngãi": ["quảng ngãi", "quang ngai"],
    "bến tre": ["bến tre", "ben tre"],
    "phú thọ": ["phú thọ", "phu tho"],
    "gia lai": ["gia lai"],
    "an giang": ["an giang"],
    "thái nguyên": ["thái nguyên", "thai nguyen"],
    "lào cai": ["lào cai", "lao cai"],
    "quảng trị": ["quảng trị", "quang tri"],
    "đồng tháp": ["đồng tháp", "dong thap"],
    "hậu giang": ["hậu giang", "hau giang"],
    "vĩnh long": ["vĩnh long", "vinh long"],
    "ninh bình": ["ninh bình", "ninh binh"],
    "quảng bình": ["quảng bình", "quang binh"],
    "kon tum": ["kon tum", "kontum"],
    "sóc trăng": ["sóc trăng", "soc trang"],
    "sơn la": ["sơn la", "son la"],
    "yên bái": ["yên bái", "yen bai"],
    "hà tĩnh": ["hà tĩnh", "ha tinh"],
    "đắk nông": ["đắk nông", "dak nong", "đăk nông"],
    "lạng sơn": ["lạng sơn", "lang son"],
    "cà mau": ["cà mau", "ca mau"],
    "tuyên quang": ["tuyên quang", "tuyen quang"],
    "bạc liêu": ["bạc liêu", "bac lieu"],
}

DISTRICT_EXACT_ALIAS_MAP = {
    "biên hòa": "thành phố biên hòa",
    "nha trang": "thành phố nha trang",
    "vũng tàu": "thành phố vũng tàu",
    "dĩ an": "thành phố dĩ an",
    "thuận an": "thành phố thuận an",
    "thủ dầu một": "thành phố thủ dầu một",
    "hạ long": "thành phố hạ long",
    "hội an": "thành phố hội an",
    "hải dương": "thành phố hải dương",
    "nam định": "thành phố nam định",
    "quy nhơn": "thành phố quy nhơn",
    "đà lạt": "thành phố đà lạt",
    "vinh": "thành phố vinh",
    "hải châu": "quận hải châu",
    "thanh khê": "quận thanh khê",
    "sơn trà": "quận sơn trà",
    "lê chân": "quận lê chân",
    "hải an": "quận hải an",
    "ngô quyền": "quận ngô quyền",
    "ninh kiều": "quận ninh kiều",
    "thủy nguyên": "huyện thủy nguyên",
    "đức hòa": "huyện đức hòa",
    "an dương": "huyện an dương",
    "nhơn trạch": "huyện nhơn trạch",
    "trảng bom": "huyện trảng bom",
    "vĩnh cửu": "huyện vĩnh cửu",
    "bến lức": "huyện bến lức",
    # Tone variants: hoà → hòa (modern Vietnamese standard; confirmed same district, same province)
    "huyện đức hoà": "huyện đức hòa",
    "thành phố thanh hoá": "thành phố thanh hóa",
    "thành phố tuy hoà": "thành phố tuy hòa",
    "huyện hoà vang": "huyện hòa vang",
    "thành phố hoà bình": "thành phố hòa bình",
    "thị xã ninh hoà": "thị xã ninh hòa",
    "huyện đông hoà": "huyện đông hòa",
    "huyện hiệp hoà": "huyện hiệp hòa",
    "huyện hoà thành": "huyện hòa thành",
    # Tone variants: thuỷ → thủy
    "quận bình thuỷ": "quận bình thủy",
    "thị xã hương thuỷ": "thị xã hương thủy",
    "huyện lạc thuỷ": "huyện lạc thủy",
    # Tone variants: thuỵ → thụy
    "huyện kiến thuỵ": "huyện kiến thụy",
    "huyện thái thuỵ": "huyện thái thụy",
    # đăk → đắk (Tây Nguyên spelling variant; confirmed same district, same province)
    "huyện đăk hà": "huyện đắk hà",
    "huyện đăk đoa": "huyện đắk đoa",
}

MALFORMED_LOCATION_HINTS = [
    "bán nhà",
    "giá ",
    "liên hệ",
    "địa chỉ",
    "liền kề",
    "đường ",
    "đ.",
    "phố ",
    "ngõ ",
    "khu đô thị",
    "quận ",
    "huyện ",
    "phường ",
    "xã ",
]

MISSING_LIKE_TOKENS = {
    "",
    "-",
    "--",
    "---",
    "_",
    "n/a",
    "na",
    "nan",
    "none",
    "null",
    "unknown",
    "missing",
    "không rõ",
    "chưa rõ",
    "chua ro",
}

PROJECT_LIKE_PATTERN = re.compile(
    r"\b(vinhomes|novaworld|waterpoint|aqua city|centa|sun casa|khu đô thị|khu do thi|kđt|kdt|dự án|du an|compound)\b",
    flags=re.IGNORECASE,
)
LISTING_NOISE_PATTERN = re.compile(
    r"\b(bán nhà|ban nha|bán đất|ban dat|liên hệ|lien he|giá|gia re|giá rẻ|vị trí đẹp|vi tri dep)\b|^địa chỉ\s*:",
    flags=re.IGNORECASE,
)
ADMIN_PREFIX_PATTERN = re.compile(r"^(phường|xã|quận|huyện|thành phố|thị xã|thị trấn)\s", flags=re.IGNORECASE)
STREET_PREFIX_PATTERN = re.compile(
    r"^(đường|phố|ngõ|hẻm|kiệt|ấp|thôn|khu phố|khu dân cư|quốc lộ|tỉnh lộ|hương lộ|đại lộ)\s",
    flags=re.IGNORECASE,
)


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
    "location_quality_score",
    "address_missing_count",
    "structural_missing_count",
    "num_floors_missing",
    "floor_area_m2_missing",
    "frontage_m_missing",
    "road_width_m_missing",
    "width_m_missing",
    "length_m_missing",
    "addr_last_2_missing",
    "addr_last_3_missing",
    "street_proxy_missing",
    "broad_subtype_flag",
    "source_thocu_flag",
    "source_verified_flag",
    "source_sm_flag",
]
GLOBAL_CATEGORICAL_FEATURES = [
    "property_subtype",
    "property_subtype_refined",
    "subtype_price_segment",
    "addr_last_1",
    "addr_last_2",
    "addr_last_3",
    "street_proxy",
    "legal",
    "door_direction",
    "source",
    "source_family",
    "source_quality_tier",
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

COMMON_SUBTYPE_NUMERIC_FEATURES = [
    "location_quality_score",
    "address_missing_count",
    "structural_missing_count",
    "num_floors_missing",
    "floor_area_m2_missing",
    "frontage_m_missing",
    "road_width_m_missing",
    "width_m_missing",
    "length_m_missing",
    "addr_last_2_missing",
    "addr_last_3_missing",
    "street_proxy_missing",
    "broad_subtype_flag",
    "source_thocu_flag",
    "source_verified_flag",
    "source_sm_flag",
]
COMMON_SUBTYPE_CATEGORICAL_FEATURES = [
    "property_subtype_refined",
    "subtype_price_segment",
    "source_family",
    "source_quality_tier",
]
for _feature_dict in SUBTYPE_FEATURE_MAP.values():
    _feature_dict["numeric"] = list(dict.fromkeys(_feature_dict["numeric"] + COMMON_SUBTYPE_NUMERIC_FEATURES))
    _feature_dict["categorical"] = list(dict.fromkeys(_feature_dict["categorical"] + COMMON_SUBTYPE_CATEGORICAL_FEATURES))


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


def normalize_address_string(address: str) -> str:
    text = normalize_text(address)
    if not text:
        return ""
    text = re.sub(r"[;|]+", ",", text)
    text = re.sub(r"\s+-\s+", ", ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r",\s*,+", ", ", text)
    text = re.sub(r"\s+", " ", text).strip(" ,")
    return text


def split_address_segments(address: str) -> List[str]:
    text = normalize_address_string(address)
    if not text:
        return []
    segments = [compact_admin_text(part) for part in text.split(",") if compact_admin_text(part)]
    deduped: List[str] = []
    seen = set()
    for segment in segments:
        if segment not in seen:
            deduped.append(segment)
            seen.add(segment)
    return deduped


def extract_admin_phrase(segment: str, prefixes: Tuple[str, ...]) -> str:
    if not segment:
        return ""
    prefix_pattern = "|".join(re.escape(prefix) for prefix in prefixes)
    matches = re.findall(rf"\b(?:{prefix_pattern})\s+[^,]+", segment)
    return compact_admin_text(matches[-1]) if matches else ""


def is_likely_province_segment(segment: str) -> bool:
    canonical = canonical_province_name(segment)
    if canonical == "missing":
        return False
    expanded = expand_admin_abbreviations(segment)
    return canonical == compact_admin_text(canonical) or canonical != expanded


def is_likely_district_segment(segment: str) -> bool:
    expanded = expand_admin_abbreviations(segment)
    if re.search(r"^(quận|huyện|thành phố|thị xã)\s", expanded):
        return True
    return compact_admin_text(expanded) in DISTRICT_EXACT_ALIAS_MAP


def sanitize_street_candidate(text: str) -> str:
    if not text:
        return "missing"
    candidate = expand_street_abbreviations(text)
    candidate = re.sub(
        r"^(bán nhà riêng|bán nhà mặt phố|bán nhà mặt tiền|bán nhà biệt thự|bán nhà|bán đất|liền kề tại|liền kề|nhà riêng tại|nhà mặt phố tại)\s+",
        "",
        candidate,
    )
    candidate = compact_admin_text(candidate)
    if not candidate or candidate == "-":
        return "missing"
    if candidate in {"riêng", "bán nhà", "bán đất", "mặt phố", "mặt tiền", "liền kề", "biệt thự"}:
        return "missing"
    has_street_prefix = re.search(r"^(đường|phố|ngõ|hẻm|kiệt|ấp|thôn|khu phố|khu dân cư)\s", candidate) is not None
    if not has_street_prefix and re.search(r"\b(bán|nhà|riêng|đất|liền kề|biệt thự|mặt phố|mặt tiền)\b", candidate):
        return "missing"
    if re.search(r"^(phường|xã|quận|huyện|thành phố|thị xã|thị trấn)\s", candidate):
        return "missing"
    if len(candidate) > 80:
        return "missing"
    return candidate


def extract_street_candidate(segment: str) -> str:
    if not segment:
        return "missing"
    segment = expand_street_abbreviations(segment)
    street_match = re.search(r"\b(đường|phố|ngõ|hẻm|kiệt|ấp|thôn|khu phố|khu dân cư)\s+[^,]+", segment)
    if street_match:
        candidate = street_match.group(0)
    else:
        split_match = re.split(r"\b(phường|xã|quận|huyện|thành phố|thị xã|thị trấn)\b", segment, maxsplit=1)
        candidate = split_match[0] if split_match else segment
    return sanitize_street_candidate(candidate)


def parse_address_components(address: str) -> Dict[str, str]:
    segments = split_address_segments(address)
    if not segments:
        return {
            "street_proxy_raw": "missing",
            "addr_last_3_raw": "missing",
            "addr_last_2_raw": "missing",
            "addr_last_1_raw": "missing",
        }

    province_raw = "missing"
    province_idx = None
    for idx in range(len(segments) - 1, -1, -1):
        if is_likely_province_segment(segments[idx]):
            province_raw = segments[idx]
            province_idx = idx
            break

    search_segments = segments if province_idx is None else segments[:province_idx]

    ward_raw = "missing"
    district_raw = "missing"
    street_raw = "missing"

    district_candidates: List[str] = []
    for segment in search_segments:
        expanded = expand_admin_abbreviations(segment)
        if ward_raw == "missing":
            ward_phrase = extract_admin_phrase(expanded, ("phường", "xã", "thị trấn"))
            if ward_phrase:
                ward_raw = ward_phrase

        if is_likely_district_segment(segment):
            district_phrase = extract_admin_phrase(expanded, ("quận", "huyện", "thành phố", "thị xã"))
            district_candidates.append(district_phrase or compact_admin_text(expanded))

        if street_raw == "missing":
            street_candidate = extract_street_candidate(segment)
            if street_candidate != "missing":
                street_raw = street_candidate

    if district_candidates:
        district_raw = district_candidates[0]

    if street_raw != "missing":
        expanded_street = expand_street_abbreviations(street_raw)
        if ward_raw != "missing" and ward_raw in expanded_street:
            street_raw = sanitize_street_candidate(expanded_street.split(ward_raw, 1)[0])
        if district_raw != "missing" and district_raw in expanded_street:
            street_raw = sanitize_street_candidate(expanded_street.split(district_raw, 1)[0])

    return {
        "street_proxy_raw": street_raw if street_raw else "missing",
        "addr_last_3_raw": ward_raw if ward_raw else "missing",
        "addr_last_2_raw": district_raw if district_raw else "missing",
        "addr_last_1_raw": province_raw if province_raw else "missing",
    }


def extract_province_from_text(text: str) -> str:
    normalized = normalize_text(text)
    for canonical, aliases in PROVINCE_ALIAS_MAP.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(normalize_text(alias))}\b", normalized):
                return canonical
    return "missing"


def extract_district_from_text(text: str) -> str:
    normalized = expand_admin_abbreviations(normalize_text(text))
    candidates = re.findall(r"\b(?:quận|huyện|thành phố|thị xã)\s+[^,.;|\n]{1,45}", normalized)
    for candidate in reversed(candidates):
        district = canonical_district_name(candidate)
        if district != "missing":
            return district
    return "missing"


def extract_ward_from_text(text: str) -> str:
    normalized = expand_admin_abbreviations(normalize_text(text))
    candidates = re.findall(r"\b(?:phường|xã|thị trấn)\s+[^,.;|\n]{1,45}", normalized)
    for candidate in reversed(candidates):
        ward = canonical_ward_name(candidate)
        if ward != "missing":
            return ward
    return "missing"


def extract_street_from_text(text: str) -> str:
    normalized = expand_street_abbreviations(normalize_text(text))
    candidates = re.findall(
        r"\b(?:đường|phố|ngõ|hẻm|kiệt|quốc lộ|tỉnh lộ|hương lộ|đại lộ)\s+[^,.;|\n]{1,55}",
        normalized,
    )
    for candidate in candidates:
        street = canonical_street_proxy(candidate)
        if street != "missing":
            return street
    return "missing"


def fill_missing_location_from_text(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing canonical address parts from title/description/address text without overriding clean values."""
    df = df.copy()
    source_text = df.get("all_text_lower", pd.Series("", index=df.index)).fillna("").astype(str)

    if "addr_last_1" in df.columns:
        mask = df["addr_last_1"].eq("missing")
        if mask.any():
            extracted = source_text[mask].map(extract_province_from_text)
            df.loc[mask & extracted.ne("missing"), "addr_last_1"] = extracted[extracted.ne("missing")]

    if "addr_last_2" in df.columns:
        mask = df["addr_last_2"].eq("missing")
        if mask.any():
            extracted = source_text[mask].map(extract_district_from_text)
            df.loc[mask & extracted.ne("missing"), "addr_last_2"] = extracted[extracted.ne("missing")]

    if "addr_last_3" in df.columns:
        mask = df["addr_last_3"].eq("missing")
        if mask.any():
            extracted = source_text[mask].map(extract_ward_from_text)
            df.loc[mask & extracted.ne("missing"), "addr_last_3"] = extracted[extracted.ne("missing")]

    if "street_proxy" in df.columns:
        mask = df["street_proxy"].eq("missing")
        if mask.any():
            extracted = source_text[mask].map(extract_street_from_text)
            df.loc[mask & extracted.ne("missing"), "street_proxy"] = extracted[extracted.ne("missing")]

    return df


def compact_admin_text(value: str) -> str:
    text = normalize_text(value)
    text = re.sub(r"[;|]+", ",", text)
    text = re.sub(r"\s*\.\s*", ".", text)
    text = re.sub(r"\s+", " ", text).strip(" -_,.")
    return text


def normalize_missing_token(value: str) -> str:
    text = compact_admin_text(value)
    if text in MISSING_LIKE_TOKENS:
        return "missing"
    return text


def strip_known_noise_prefixes(text: str) -> str:
    out = compact_admin_text(text)
    out = re.sub(r"^địa chỉ\s*:\s*", "", out)
    out = re.sub(r"^(xã|phường)\s+(phường|xã)\s+", r"\2 ", out)
    out = re.sub(r"^(?:tại|tp)\s+", "", out)
    return compact_admin_text(out)


def truncate_at_secondary_location_noise(text: str) -> str:
    out = compact_admin_text(text)
    out = re.split(r"\s*[,:;]\s*", out, maxsplit=1)[0]
    out = re.sub(r"\.(?=(đường|ngõ|hẻm|kiệt|ấp|thôn|khu phố|quốc lộ|tỉnh lộ|hương lộ|đại lộ|khu đô thị|khu do thi|dự án|du an)\b)", " ", out)
    out = re.sub(
        r"\s+(?:đường|ngõ|hẻm|kiệt|ấp|thôn|khu phố|quốc lộ|tỉnh lộ|hương lộ|đại lộ|khu đô thị|khu do thi|dự án|du an)\s+.*$",
        "",
        out,
        flags=re.IGNORECASE,
    )
    return compact_admin_text(out)


def has_malformed_location_shape(text: str) -> bool:
    if not text or text == "missing":
        return False
    if len(text) > 50:
        return True
    if ":" in text:
        return True
    return any(hint in text for hint in MALFORMED_LOCATION_HINTS)


def expand_admin_abbreviations(text: str) -> str:
    if not text:
        return text
    text = normalize_missing_token(text)
    if text == "missing":
        return text
    substitutions = [
        (r"\btp\.\s*", "thành phố "),
        (r"\btp\s+", "thành phố "),
        (r"\bq\.\s*", "quận "),
        (r"\bq\s+", "quận "),
        (r"\bh\.\s*", "huyện "),
        (r"\bh\s+", "huyện "),
        (r"\bp\.\s*", "phường "),
        (r"\bp\s+", "phường "),
        (r"\bx\.\s*", "xã "),
        (r"\bx\s+", "xã "),
        (r"\btx\.\s*", "thị xã "),
        (r"\btt\.\s*", "thị trấn "),
    ]
    out = text
    for pattern, repl in substitutions:
        out = re.sub(pattern, repl, out)
    return compact_admin_text(out)


def expand_street_abbreviations(text: str) -> str:
    if not text:
        return text
    out = normalize_missing_token(text)
    if out == "missing":
        return out
    substitutions = [
        (r"^đ\.\s*", "đường "),
        (r"^duong\s+", "đường "),
        (r"^dg\.\s*", "đường "),
        (r"^ng\.\s*", "ngõ "),
        (r"^kp\.\s*", "khu phố "),
        (r"^kdc\.\s*", "khu dân cư "),
        (r"^kđt\.\s*", "khu đô thị "),
        (r"^kdt\.\s*", "khu đô thị "),
        (r"^tt\.\s*", "thị trấn "),
    ]
    for pattern, repl in substitutions:
        out = re.sub(pattern, repl, out)
    return compact_admin_text(out)


def canonical_province_name(value: str) -> str:
    text = strip_known_noise_prefixes(value)
    text = normalize_missing_token(text)
    if text == "missing":
        return "missing"
    text_no_punct = text.replace(".", " ")
    text_no_punct = re.sub(r"\s+", " ", text_no_punct).strip()

    alias_lookup = {}
    for canonical, aliases in PROVINCE_ALIAS_MAP.items():
        for alias in aliases:
            alias_lookup[compact_admin_text(alias).replace(".", " ")] = canonical

    if text_no_punct in alias_lookup:
        return alias_lookup[text_no_punct]

    for alias, canonical in alias_lookup.items():
        if alias and alias in text_no_punct:
            return canonical

    matches = get_close_matches(text_no_punct, list(alias_lookup.keys()), n=1, cutoff=0.92)
    if matches:
        return alias_lookup[matches[0]]

    if has_malformed_location_shape(text_no_punct):
        return "missing"
    if ADMIN_PREFIX_PATTERN.search(text_no_punct):
        return "missing"
    return text_no_punct


def canonical_district_name(value: str) -> str:
    text = strip_known_noise_prefixes(value)
    text = expand_admin_abbreviations(text)
    text = truncate_at_secondary_location_noise(text)
    text = normalize_missing_token(text)
    if text == "missing":
        return "missing"
    if canonical_province_name(text) != "missing":
        return "missing"
    if has_malformed_location_shape(text) and not re.match(r"^(quận|huyện|thành phố|thị xã|thị trấn)\s", text):
        return "missing"
    text = re.sub(r"\s*-\s*(quận|huyện|thành phố|thị xã)\s+.*$", "", text).strip(" -_,.")
    # Truncate if a street keyword appears after the district name (e.g. "quận hải châu đường X").
    # "phố" is excluded: "thành phố X" would be corrupted to "thành" if bare "phố" matched.
    text = re.sub(r"\s+(?:đường|ngõ|hẻm|kiệt|khu phố)\s+.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bbiên hoà\b", "biên hòa", text)
    if text in DISTRICT_EXACT_ALIAS_MAP:
        text = DISTRICT_EXACT_ALIAS_MAP[text]
    else:
        alias_matches = get_close_matches(text, list(DISTRICT_EXACT_ALIAS_MAP.keys()), n=1, cutoff=0.94)
        if alias_matches:
            text = DISTRICT_EXACT_ALIAS_MAP[alias_matches[0]]
    return compact_admin_text(text)


def canonical_ward_name(value: str) -> str:
    text = strip_known_noise_prefixes(value)
    text = expand_admin_abbreviations(text)
    text = truncate_at_secondary_location_noise(text)
    text = normalize_missing_token(text)
    if text == "missing":
        return "missing"
    if has_malformed_location_shape(text) and not re.match(r"^(phường|xã|thị trấn)\s", text):
        return "missing"
    if canonical_province_name(text) != "missing":
        return "missing"
    if text == "-":
        return "missing"
    return compact_admin_text(text)


def canonical_street_proxy(value: str) -> str:
    text = strip_known_noise_prefixes(value)
    text = expand_admin_abbreviations(text)
    text = sanitize_street_candidate(text)
    if not text or text == "missing":
        return "missing"
    if ADMIN_PREFIX_PATTERN.search(text):
        return "missing"
    if LISTING_NOISE_PATTERN.search(text):
        return "missing"
    if PROJECT_LIKE_PATTERN.search(text):
        return "missing"
    if canonical_province_name(text) != "missing":
        return "missing"
    if len(text) > 80:
        return "missing"
    return text


def first_existing(df: pd.DataFrame, candidates: List[str]):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def canonicalize_series_with_cache(series: pd.Series, fn) -> pd.Series:
    filled = series.fillna("missing").astype(str)
    unique_values = pd.Index(filled.unique())
    mapping = {value: fn(value) for value in unique_values}
    return filled.map(mapping)


def create_interaction(df: pd.DataFrame, target_col: str, left_col: str, right_col: str):
    if left_col in df.columns and right_col in df.columns:
        left = pd.to_numeric(df[left_col], errors="coerce")
        right = pd.to_numeric(df[right_col], errors="coerce")
        product = left * right
        if target_col in df.columns:
            df[target_col] = pd.to_numeric(df[target_col], errors="coerce").combine_first(product)
        else:
            df[target_col] = product
    else:
        if target_col not in df.columns:
            df[target_col] = np.nan


def canonical_source_family(value: str) -> str:
    text = normalize_text(value)
    text = re.sub(r"https?://", "", text)
    text = re.sub(r"www\.", "", text)
    text = text.split("/")[0].strip()
    text = text.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    if not text or text in MISSING_LIKE_TOKENS:
        return "missing"
    for suffix in ["_thocu_sm", "_verified"]:
        text = text.replace(suffix, "")
    known = [
        "batdongsan.com.vn",
        "bds68.com.vn",
        "alonhadat.com.vn",
        "mogi.vn",
        "muaban.net",
        "123nhadatviet.com",
        "dothi.net",
        "bds123.vn",
        "homedy.com",
        "nhadat.cafeland.vn",
        "chotot.com",
    ]
    for domain in known:
        if domain in text:
            return domain
    return text[:60]


def add_source_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    source = df.get("source", pd.Series("missing", index=df.index)).fillna("missing").astype(str)
    normalized = source.map(normalize_text)
    df["source_family"] = source.map(canonical_source_family)
    df["source_thocu_flag"] = normalized.str.contains("thocu", na=False).astype(int)
    df["source_sm_flag"] = normalized.str.contains("_sm| thocu sm|sm$", regex=True, na=False).astype(int)
    df["source_verified_flag"] = normalized.str.contains("verified", na=False).astype(int)
    df["source_quality_tier"] = np.select(
        [
            df["source_verified_flag"].eq(1),
            df["source_family"].isin(["batdongsan.com.vn", "alonhadat.com.vn", "bds68.com.vn"]),
            df["source_thocu_flag"].eq(1),
        ],
        ["verified", "major_portal", "thocu_source"],
        default="other_source",
    )
    return df


def add_missing_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_missing_cols = ["num_floors", "floor_area_m2", "frontage_m", "road_width_m", "width_m", "length_m"]
    address_missing_cols = ["addr_last_2", "addr_last_3", "street_proxy"]
    for col in numeric_missing_cols:
        if col in df.columns:
            df[f"{col}_missing"] = pd.to_numeric(df[col], errors="coerce").isna().astype(int)
    for col in address_missing_cols:
        if col in df.columns:
            normalized = df[col].fillna("missing").astype(str).map(normalize_text)
            df[f"{col}_missing"] = normalized.isin(MISSING_LIKE_TOKENS).astype(int)

    numeric_flags = [f"{col}_missing" for col in numeric_missing_cols if f"{col}_missing" in df.columns]
    address_flags = [f"{col}_missing" for col in address_missing_cols if f"{col}_missing" in df.columns]
    df["structural_missing_count"] = df[numeric_flags].sum(axis=1) if numeric_flags else 0
    df["address_missing_count"] = df[address_flags].sum(axis=1) if address_flags else 0
    df["location_quality_score"] = 3 - df["address_missing_count"]
    return df


def add_refined_subtype_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    subtype = df.get("property_subtype", pd.Series("unknown", index=df.index)).fillna("unknown").astype(str)
    refined = subtype.copy()

    road_width = pd.to_numeric(df.get("road_width_m", pd.Series(np.nan, index=df.index)), errors="coerce")
    frontage = pd.to_numeric(df.get("frontage_m", pd.Series(np.nan, index=df.index)), errors="coerce")
    floors = pd.to_numeric(df.get("num_floors", pd.Series(np.nan, index=df.index)), errors="coerce")
    area = pd.to_numeric(df.get("area_m2", pd.Series(np.nan, index=df.index)), errors="coerce")
    is_business = pd.to_numeric(df.get("is_business_text", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    is_car = pd.to_numeric(df.get("is_car_access_text", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    is_project = pd.to_numeric(df.get("is_project_text", pd.Series(0, index=df.index)), errors="coerce").fillna(0)

    refined.loc[subtype.eq("frontage_house") & ((road_width >= 10) | (frontage >= 7))] = "frontage_house_wide_road"
    refined.loc[subtype.eq("frontage_house") & is_business.eq(1)] = "frontage_house_business"
    refined.loc[subtype.eq("normal_house") & is_car.eq(1)] = "normal_house_car_access"
    refined.loc[subtype.eq("normal_house") & floors.ge(4)] = "normal_house_many_floors"
    refined.loc[subtype.eq("townhouse_project") & is_project.eq(1)] = "townhouse_project_confirmed"
    refined.loc[subtype.eq("townhouse_project") & area.ge(120)] = "townhouse_project_large_area"

    df["property_subtype_refined"] = refined
    df["broad_subtype_flag"] = subtype.isin(["frontage_house", "normal_house", "townhouse_project"]).astype(int)
    df["subtype_price_segment"] = np.select(
        [
            subtype.isin(["villa", "shophouse"]),
            subtype.isin(["frontage_house", "alley_frontage_house"]) | refined.str.contains("business|wide_road", regex=True, na=False),
            subtype.isin(["normal_house", "alley_house"]),
        ],
        ["premium_like", "street_commercial_like", "standard_residential_like"],
        default="project_or_other_like",
    )
    return df


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
    df["floor_area_m2"] = np.nan

    df, _text_fill_counts = extract_features_from_text(df)

    legal_col = first_existing(df, ["legal", "legal_status", "juridical"])
    df["legal"] = df[legal_col].fillna("missing").astype(str) if legal_col else "missing"
    direction_col = first_existing(df, ["door_direction", "direction", "house_direction"])
    df["door_direction"] = df[direction_col].fillna("missing").astype(str) if direction_col else "missing"
    project_col = first_existing(df, ["project_name", "project", "project_title"])
    df["project_name"] = df[project_col].fillna("missing").astype(str) if project_col else "missing"
    address_col = first_existing(df, ["address", "full_address", "location", "street"])
    df["address"] = df[address_col].fillna("").astype(str) if address_col else ""
    df = add_source_features(df)

    parsed_address = df["address"].apply(parse_address_components).apply(pd.Series)
    df["addr_last_1_raw"] = parsed_address["addr_last_1_raw"]
    df["addr_last_2_raw"] = parsed_address["addr_last_2_raw"]
    df["addr_last_3_raw"] = parsed_address["addr_last_3_raw"]
    df["street_proxy_raw"] = parsed_address["street_proxy_raw"]

    df["addr_last_1"] = canonicalize_series_with_cache(df["addr_last_1_raw"], canonical_province_name)
    df["addr_last_2"] = canonicalize_series_with_cache(df["addr_last_2_raw"], canonical_district_name)
    df["addr_last_3"] = canonicalize_series_with_cache(df["addr_last_3_raw"], canonical_ward_name)
    df["street_proxy"] = canonicalize_series_with_cache(df["street_proxy_raw"], canonical_street_proxy)
    df = fill_missing_location_from_text(df)

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

    df = add_recency_weight(df)

    df = apply_property_classification(df)
    text_series = df["all_text_lower"].fillna("")

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
    df = add_refined_subtype_features(df)

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
    df = add_missing_quality_features(df)

    return df


def apply_property_classification(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text_series = df["all_text_lower"].fillna("")
    apartment_signal = series_has_any(text_series, EXCLUDE_APARTMENT)
    land_only_signal = series_has_any(text_series, EXCLUDE_LAND_ONLY)
    exclude_commercial_only = series_has_any(text_series, EXCLUDE_COMMERCIAL_ONLY)
    villa_flag = series_has_any(text_series, VILLA)
    shophouse_flag = series_has_any(text_series, SHOPHOUSE)
    townhouse_project_flag = series_has_any(text_series, TOWNHOUSE_PROJECT)
    frontage_flag = series_has_any(text_series, FRONTAGE)
    alley_flag = series_has_any(text_series, ALLEY)
    normal_house_flag = series_has_any(text_series, NORMAL_HOUSE) | text_series.str.contains("thocu|tho_cu", regex=True, na=False)
    corner_house_flag = series_has_any(text_series, CORNER)
    general_landed_house_flag = series_has_any(text_series, GENERAL_LANDED_HOUSE)
    house_text_signal = series_has_any(text_series, HOUSE_TEXT_SIGNAL)
    structural_house_signal = pd.Series(False, index=df.index)
    for col in ["num_floors", "bedrooms", "bathrooms"]:
        if col in df.columns:
            structural_house_signal = structural_house_signal | pd.to_numeric(df[col], errors="coerce").fillna(0).gt(0)
    if "floor_area_m2" in df.columns:
        structural_house_signal = structural_house_signal | pd.to_numeric(df["floor_area_m2"], errors="coerce").fillna(0).gt(0)

    strong_house_signal = (
        normal_house_flag
        | general_landed_house_flag
        | house_text_signal
        | frontage_flag
        | alley_flag
        | corner_house_flag
        | villa_flag
        | shophouse_flag
        | townhouse_project_flag
        | structural_house_signal
    )
    exclude_apartment = apartment_signal & ~strong_house_signal
    exclude_land_only = land_only_signal & ~strong_house_signal
    base_house_flag = strong_house_signal & ~(exclude_apartment | exclude_land_only)

    commercial_only_flag = exclude_commercial_only & ~(base_house_flag | villa_flag | shophouse_flag | townhouse_project_flag)
    landed_residential_general_flag = base_house_flag & ~commercial_only_flag

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
            base_house_flag,
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
    df["general_property_type"] = np.select(
        [
            exclude_apartment,
            exclude_land_only,
            commercial_only_flag,
            landed_residential_general_flag,
        ],
        [
            "apartment",
            "land_only",
            "commercial_only",
            "landed_residential_house",
        ],
        default="unknown",
    )
    df["is_landed_residential_house"] = landed_residential_general_flag.astype(int)
    df["property_group"] = np.where(
        df["property_subtype"].isin(list(SUBTYPE_FEATURE_MAP.keys())),
        "landed_residential_house",
        "out_of_scope",
    )
    return df


RECENCY_WEIGHT_MAP: Dict[int, float] = {2025: 1.0, 2024: 0.8, 2023: 0.5, 2022: 0.4, 2021: 0.3}


def _best_listing_date(df: pd.DataFrame) -> pd.Series:
    """Return the best available date per row: updated_at > post_date > NaT."""
    best = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for col in ["updated_at_parsed", "post_date_parsed"]:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            best = best.combine_first(parsed)
    return best


def remove_near_duplicates(
    df: pd.DataFrame,
    area_bucket_size: float = 3.0,
    price_bucket_size: float = 200_000_000,
) -> Tuple[pd.DataFrame, int]:
    """
    Remove re-listed properties: same normalized address + similar area + similar price.
    Only applied where district, area_m2, and price_vnd are all known.
    Within each duplicate group keeps the most recent listing; ties broken by row position.
    """
    required = ["addr_last_2", "area_m2", "price_vnd", "addr_last_1", "addr_last_3", "street_proxy"]
    if not all(c in df.columns for c in required):
        return df, 0

    can_dedup = (
        df["addr_last_2"].notna()
        & (df["addr_last_2"] != "missing")
        & df["area_m2"].notna()
        & df["price_vnd"].notna()
    )
    candidates = df.loc[can_dedup].copy()
    no_dedup_rows = df.loc[~can_dedup]

    if candidates.empty:
        return df, 0

    area_bucket = (candidates["area_m2"] / area_bucket_size).round() * area_bucket_size
    price_bucket = (candidates["price_vnd"] / price_bucket_size).round() * price_bucket_size
    addr_key = (
        candidates["addr_last_1"].fillna("") + "|"
        + candidates["addr_last_2"].fillna("") + "|"
        + candidates["addr_last_3"].fillna("") + "|"
        + candidates["street_proxy"].fillna("")
    )
    candidates["_dedup_key"] = addr_key + "|" + area_bucket.astype(str) + "|" + price_bucket.astype(str)
    candidates["_best_date"] = _best_listing_date(candidates)
    candidates["_pos"] = range(len(candidates))  # positional tiebreaker

    candidates_sorted = candidates.sort_values(
        ["_dedup_key", "_best_date", "_pos"],
        ascending=[True, False, False],
        na_position="last",
    )
    kept = candidates_sorted.drop_duplicates(subset=["_dedup_key"], keep="first")
    kept = kept.drop(columns=["_dedup_key", "_best_date", "_pos"])

    n_removed = len(candidates) - len(kept)
    result = pd.concat([kept, no_dedup_rows]).sort_index()
    return result, int(n_removed)


def add_recency_weight(df: pd.DataFrame) -> pd.DataFrame:
    """Add recency_weight column based on listing year: 1.0 for 2025, 0.8 for 2024, 0.5 for undated."""
    best_date = _best_listing_date(df)
    df = df.copy()
    df["recency_weight"] = best_date.dt.year.map(RECENCY_WEIGHT_MAP).fillna(0.5)
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
    df_clean.to_csv(PROCESSED_DATA_DIR / "full_cleaned_dataset.csv.gz", index=False, compression="gzip")
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


def load_processed_train_test(data_path: str | Path | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if data_path is None:
        train_df = pd.read_parquet(PROCESSED_DATA_DIR / "train_dataset.parquet")
        test_df = pd.read_parquet(PROCESSED_DATA_DIR / "test_dataset.parquet")
        return train_df, test_df

    df_model = pd.read_parquet(data_path)
    split_data = make_split(df_model)
    return split_data.train_df, split_data.test_df


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


def is_commercial_listing(df: pd.DataFrame) -> pd.Series:
    """Flag hotel / office / commercial listings that carry residential subtypes but follow a
    different pricing logic (revenue, room count, star rating — none of which are in the feature
    set). Requires a commercial keyword AND commercial scale, so an ordinary house that merely
    mentions a nearby hotel is not caught. These are out of scope for the residential AVM and are
    excluded from training / routed to review at inference. Price is used for the training-time
    filter; an inference-time variant would drop the price clause (price is the unknown)."""
    if "all_text_lower" in df.columns:
        txt = df["all_text_lower"].fillna("")
    else:
        txt = (df.get("title", pd.Series("", index=df.index)).fillna("").astype(str) + " "
               + df.get("description", pd.Series("", index=df.index)).fillna("").astype(str)).str.lower()
    hotel = txt.str.contains(r"khách sạn|khach san| hotel|resort", regex=True, na=False)
    office = txt.str.contains(
        r"tòa nhà văn phòng|toà nhà văn phòng|văn phòng cho thuê|cao ốc|office building|to[àa] building",
        regex=True, na=False)
    floors = pd.to_numeric(df.get("num_floors"), errors="coerce")
    area = pd.to_numeric(df.get("area_m2"), errors="coerce")
    price = pd.to_numeric(df.get("price_vnd"), errors="coerce")
    scale = (floors >= 6) | (area >= 250) | (price >= 20e9)
    return (hotel & scale.fillna(False)) | office


def trimmed_mean(values: np.ndarray, proportion: float) -> float:
    """Mean after dropping `proportion` of the mass from each tail. Leakage-free,
    no scipy dependency. Falls back to the plain mean when trimming empties the array."""
    v = np.sort(np.asarray(values, dtype=float))
    n = len(v)
    if n == 0:
        return float("nan")
    k = int(np.floor(n * proportion))
    core = v[k : n - k] if n - 2 * k > 0 else v
    return float(np.mean(core))


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
            "Median_Bias_pct": np.nan,
            "Bias_pct_trimmed10": np.nan,
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
        # Robust bias views: the raw mean Bias_pct is a mean-of-ratios dominated by a
        # right tail of underpriced/mislabeled listings. The median and 10%-trimmed
        # mean show whether the model itself is centered (they should sit near 0).
        "Median_Bias_pct": np.median(pe) * 100,
        "Bias_pct_trimmed10": trimmed_mean(pe, 0.10) * 100,
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
    return ["MAE", "RMSE", "MAPE", "Median_APE", "HitRate_10_pct", "HitRate_20_pct", "Bias_pct", "Median_Bias_pct", "Bias_pct_trimmed10", "OvervaluationRate_pct", "UndervaluationRate_pct"]
