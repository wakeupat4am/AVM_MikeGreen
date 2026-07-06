"""
Text-based feature extraction for missing structured fields.

Reads title + description text (already lowercased in all_text_lower) and fills
NaN slots in num_floors, bedrooms, bathrooms, frontage_m, road_width_m, width_m,
length_m using regex patterns common in Vietnamese real estate listings.

Rules:
- Never overrides an already-parsed (non-NaN) value.
- For dimension combos (e.g. "4x15m"), assigns smaller value → width_m,
  larger value → length_m (user-confirmed convention).
- "nhà cấp 4" is a house type, not a floor count — intentionally excluded.
"""

from __future__ import annotations

import re
from typing import Dict, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _normalize_measurements(text: str) -> str:
    """
    Normalize Vietnamese shorthand measurement notation before pattern matching.
    '4m5' → '4.5m'  (digit-m-digit means decimal, e.g. 4 metres 50 cm)
    '4m50' → '4.50m'
    """
    return re.sub(r'(\d)m(\d{1,2})(?=[^\d]|$)', r'\1.\2m', text)


def _vn_num(s: str) -> float:
    """Parse a Vietnamese measurement string like '4,5' or '4.5' to float."""
    s = s.strip().replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def _first_match_float(text: str, pattern: str, group: int = 1) -> float:
    """Return the first regex match converted to float, or NaN."""
    m = re.search(pattern, text)
    if m:
        return _vn_num(m.group(group))
    return np.nan


# ---------------------------------------------------------------------------
# per-feature extractors (operate on a single lowercased text string)
# ---------------------------------------------------------------------------

def _extract_num_floors(text: str) -> float:
    """
    Priority order:
      1. "X trệt Y lầu" / "Xt Yl" shorthand  → X + Y
      2. "X tầng"
      3. "X lầu"  (standalone)
    """
    # "1 trệt 2 lầu" or "1 tret 2 lau"
    m = re.search(r"(\d+)\s*tr[eệ]t\W+(\d+)\s*l[aầ]u", text)
    if m:
        val = int(m.group(1)) + int(m.group(2))
        if 1 <= val <= 30:
            return float(val)

    # shorthand "1t2l"
    m = re.search(r"\b(\d+)t(\d+)l\b", text)
    if m:
        val = int(m.group(1)) + int(m.group(2))
        if 1 <= val <= 30:
            return float(val)

    # "3 tầng"
    m = re.search(r"(\d+)\s*tầng", text)
    if m:
        val = int(m.group(1))
        if 1 <= val <= 30:
            return float(val)

    # "3 tấm" / "2 mê" are common southern/central floor-count wording.
    for pat in [r"(\d+)\s*tấm\b", r"(\d+)\s*mê\b"]:
        m = re.search(pat, text)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 30:
                return float(val)

    # "2 lầu" (standalone — avoid matching "2 lầu X phòng" dim text)
    m = re.search(r"(\d+)\s*lầu", text)
    if m:
        val = int(m.group(1))
        if 1 <= val <= 30:
            return float(val)

    return np.nan


def _extract_bedrooms(text: str) -> float:
    """
    Patterns: "X phòng ngủ", "X PN", "X pn"
    """
    for pat in [
        r"(\d+)\s*ph[oò]ng\s*ng[uủ]",
        r"(\d+)\s*pn\b",
    ]:
        m = re.search(pat, text)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 20:
                return float(val)
    return np.nan


def _extract_bathrooms(text: str) -> float:
    """
    Patterns: "X WC", "X toilet", "X vệ sinh", "X phòng tắm"
    """
    for pat in [
        r"(\d+)\s*wc\b",
        r"(\d+)\s*toilet",
        r"(\d+)\s*v[eệ]\s*sinh",
        r"(\d+)\s*ph[oò]ng\s*t[aắ]m",
    ]:
        m = re.search(pat, text)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 20:
                return float(val)
    return np.nan


def _extract_frontage(text: str) -> float:
    """
    Patterns: "ngang Xm", "mặt tiền Xm", "MT Xm"
    """
    for pat in [
        r"ngang(?:\s+lớn|\s+rộng)?\s*([\d,\.]+)\s*m",
        r"m[aặ]t\s*ti[eề]n(?:\s+rộng)?\s*([\d,\.]+)\s*m",
        r"\bmt\s*([\d,\.]+)(?:\s*/\s*[\d,\.]+)?\s*m",
        r"nở\s*hậu\s*([\d,\.]+)\s*m",
    ]:
        val = _first_match_float(text, pat)
        if not np.isnan(val) and 0.5 <= val <= 100:
            return val
    return np.nan


def _extract_road_width(text: str) -> float:
    """
    Patterns: "đường Xm", "lộ Xm"
    Looks for explicit road-width phrases only (avoids road names like "đường 22/12").
    """
    for pat in [
        r"đ[uư][oờ]ng\s*(?:rộng\s*)?([\d,\.]+)\s*m(?!\d)",
        r"(?:hẻm|hem|ngõ|ngo|kiệt|kiet)\s*(?:rộng\s*)?([\d,\.]+)\s*m(?!\d)",
        r"l[oộ](?:\s*giới)?\s*(?:rộng\s*)?([\d,\.]+)\s*m(?!\d)",
        r"(?:hẻm|hem)\s*xe\s*hơi\s*([\d,\.]+)\s*m(?!\d)",
    ]:
        val = _first_match_float(text, pat)
        if not np.isnan(val) and 0.5 <= val <= 100:
            return val
    return np.nan


def _extract_width_length(text: str) -> Tuple[float, float]:
    """
    Extract width_m and length_m.

    Sources (tried in priority order):
      1. Explicit "ngang Xm" / "rộng Xm" → width
         Explicit "dài Xm" / "sâu Xm"    → length
      2. Dimension combo "AxB m" or "AxBm"
         → smaller value = width, larger = length (user decision)
    """
    width = np.nan
    length = np.nan

    # explicit width signals
    for pat in [
        r"ngang(?:\s+lớn|\s+rộng)?\s*([\d,\.]+)\s*m",
        r"r[oộ]ng\s*([\d,\.]+)\s*m",
        r"m[aặ]t\s*ti[eề]n(?:\s+rộng)?\s*([\d,\.]+)\s*m",
        r"\bmt\s*([\d,\.]+)(?:\s*/\s*[\d,\.]+)?\s*m",
    ]:
        val = _first_match_float(text, pat)
        if not np.isnan(val) and 0.5 <= val <= 200:
            width = val
            break

    # explicit length signals
    for pat in [r"dài\s*([\d,\.]+)\s*m", r"sâu\s*([\d,\.]+)\s*m"]:
        val = _first_match_float(text, pat)
        if not np.isnan(val) and 0.5 <= val <= 1000:
            length = val
            break

    # dimension combo fills whichever is still missing
    # pattern handles both "4x15m" and "4.5mx11.6m" (post-normalization form)
    if np.isnan(width) or np.isnan(length):
        m = re.search(
            r"([\d,\.]+)\s*m?\s*[xX×]\s*([\d,\.]+)\s*m?(?!²|2)",
            text,
        )
        if m:
            a, b = _vn_num(m.group(1)), _vn_num(m.group(2))
            if not (np.isnan(a) or np.isnan(b)):
                w_combo = min(a, b)
                l_combo = max(a, b)
                if np.isnan(width) and 0.5 <= w_combo <= 200:
                    width = w_combo
                if np.isnan(length) and 0.5 <= l_combo <= 1000:
                    length = l_combo

    return width, length


# ---------------------------------------------------------------------------
# main public function
# ---------------------------------------------------------------------------

TARGET_COLS = ["num_floors", "bedrooms", "bathrooms", "frontage_m", "road_width_m", "width_m", "length_m", "floor_area_m2"]


def _extract_floor_area(text: str) -> float:
    """Extract explicit usable/gross floor area; fallback to NaN if it is only land area."""
    patterns = [
        r"(?:diện\s*tích\s*sàn|dt\s*sàn|dts|tổng\s*dt\s*sàn|tổng\s*diện\s*tích\s*sàn)\s*([\d,\.]+)\s*m(?:2|²)?",
        r"(?:sàn\s*xây\s*dựng|diện\s*tích\s*xây\s*dựng)\s*([\d,\.]+)\s*m(?:2|²)?",
    ]
    for pat in patterns:
        val = _first_match_float(text, pat)
        if not np.isnan(val) and 10 <= val <= 5000:
            return val
    return np.nan


def extract_features_from_text(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Fill NaN values in structured feature columns using text extracted from
    the 'all_text_lower' column (title + description, already lowercased).

    Returns:
        df       : dataframe with NaN slots filled where text evidence exists
        summary  : dict mapping feature name → number of rows filled
    """
    df = df.copy()

    # normalize shorthand before all pattern matching (e.g. "4m5" → "4.5m")
    text = df["all_text_lower"].fillna("").astype(str).map(_normalize_measurements)

    # track how many rows were filled per feature
    fill_counts: Dict[str, int] = {col: 0 for col in TARGET_COLS}

    # --- num_floors ---
    mask = df["num_floors"].isna()
    if mask.any():
        extracted = text[mask].map(_extract_num_floors)
        filled = extracted.notna()
        fill_idx = extracted[filled].index
        df.loc[fill_idx, "num_floors"] = extracted.loc[fill_idx]
        fill_counts["num_floors"] = int(filled.sum())

    # --- bedrooms ---
    mask = df["bedrooms"].isna()
    if mask.any():
        extracted = text[mask].map(_extract_bedrooms)
        filled = extracted.notna()
        fill_idx = extracted[filled].index
        df.loc[fill_idx, "bedrooms"] = extracted.loc[fill_idx]
        fill_counts["bedrooms"] = int(filled.sum())

    # --- bathrooms ---
    mask = df["bathrooms"].isna()
    if mask.any():
        extracted = text[mask].map(_extract_bathrooms)
        filled = extracted.notna()
        fill_idx = extracted[filled].index
        df.loc[fill_idx, "bathrooms"] = extracted.loc[fill_idx]
        fill_counts["bathrooms"] = int(filled.sum())

    # --- frontage_m ---
    mask = df["frontage_m"].isna()
    if mask.any():
        extracted = text[mask].map(_extract_frontage)
        filled = extracted.notna()
        fill_idx = extracted[filled].index
        df.loc[fill_idx, "frontage_m"] = extracted.loc[fill_idx]
        fill_counts["frontage_m"] = int(filled.sum())

    # --- road_width_m ---
    mask = df["road_width_m"].isna()
    if mask.any():
        extracted = text[mask].map(_extract_road_width)
        filled = extracted.notna()
        fill_idx = extracted[filled].index
        df.loc[fill_idx, "road_width_m"] = extracted.loc[fill_idx]
        fill_counts["road_width_m"] = int(filled.sum())

    # --- width_m and length_m (share the same extractor) ---
    mask_w = df["width_m"].isna()
    mask_l = df["length_m"].isna()
    need_mask = mask_w | mask_l
    if need_mask.any():
        results = text[need_mask].map(_extract_width_length)
        widths = results.map(lambda t: t[0])
        lengths = results.map(lambda t: t[1])

        w_filled = mask_w[need_mask] & widths.notna()
        l_filled = mask_l[need_mask] & lengths.notna()

        width_idx = widths[w_filled].index
        length_idx = lengths[l_filled].index
        df.loc[width_idx, "width_m"] = widths.loc[width_idx]
        df.loc[length_idx, "length_m"] = lengths.loc[length_idx]

        fill_counts["width_m"] = int(w_filled.sum())
        fill_counts["length_m"] = int(l_filled.sum())

    # --- floor_area_m2 ---
    # If text provides a direct floor-area value, prefer it over area*num_floors.
    if "floor_area_m2" in df.columns:
        mask = df["floor_area_m2"].isna()
        if mask.any():
            extracted = text[mask].map(_extract_floor_area)
            filled = extracted.notna()
            fill_idx = extracted[filled].index
            df.loc[fill_idx, "floor_area_m2"] = extracted.loc[fill_idx]
            fill_counts["floor_area_m2"] = int(filled.sum())

    return df, fill_counts
