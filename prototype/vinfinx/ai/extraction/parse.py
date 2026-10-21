"""Strict parsers for printed fields, per language (MACRO_DOCUMENT_SPEC.md, "Number and date formats"). A parser
returns None when the text does not match the document's own conventions exactly; it never guesses. "143.8" is not a
Vietnamese number (the thousands group needs three digits), so a swapped separator is rejected, not misread."""
from __future__ import annotations

import re
import unicodedata
from datetime import date
from decimal import Decimal

from ..schemas.common import Frequency, Transformation, Unit

DASHES = str.maketrans({c: "-" for c in "‒–—―−﹣－"})
_NUMBER = {
    "en": re.compile(r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$"),
    "vi": re.compile(r"^[+-]?(?:\d{1,3}(?:\.\d{3})+|\d+)(?:,\d+)?$"),
}
_MONTHS_EN = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}

UNIT_WORDS = {  # printed unit -> catalogue unit; anything else is a unit the document should not use
    "%": Unit.PCT, "listings": Unit.COUNT, "tin": Unit.COUNT, "transactions": Unit.COUNT, "giao dịch": Unit.COUNT,
    "pts": Unit.INDEX_POINTS, "điểm": Unit.INDEX_POINTS, "bps": Unit.BPS, "điểm cơ bản": Unit.BPS,
}
CHANGE_SUFFIX = {Unit.PCT: {"pp", "đpt"}, Unit.COUNT: {""}, Unit.INDEX_POINTS: {"pts", "điểm"}, Unit.BPS: {"bps", "đcb"}}
MEASURES = {
    "level": Transformation.LEVEL, "mức": Transformation.LEVEL, "y/y": Transformation.YOY,
    "so với cùng kỳ": Transformation.YOY, "m/m": Transformation.MOM, "so với tháng trước": Transformation.MOM,
    "ytd": Transformation.YTD, "từ đầu năm": Transformation.YTD, "index (base 100)": Transformation.INDEX,
    "chỉ số (gốc 100)": Transformation.INDEX,
}
SEGMENTS = {"all": None, "tất cả": None, "landed houses": "Landed houses", "nhà riêng": "Landed houses"}
REVISION = {"final": "final", "chính thức": "final", "revised": "revised", "đã điều chỉnh": "revised",
            "preliminary": "preliminary", "sơ bộ": "preliminary"}
QUALITY = {"ok": "ok", "đạt": "ok", "low": "low", "thấp": "low"}


def clean(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).translate(DASHES).split())


def number(text: str, lang: str) -> Decimal | None:
    t = clean(text).replace(" ", "")
    if not _NUMBER[lang].match(t):
        return None
    t = t.replace(",", "") if lang == "en" else t.replace(".", "").replace(",", ".")
    return Decimal(t)


def change(text: str, lang: str, unit: Unit) -> Decimal | None:
    """A signed change with the suffix its unit requires ("−0.5 pp", "+38", "−1,5 điểm")."""
    t = clean(text)
    m = re.match(r"^([+-]?[\d.,]+)\s*(.*)$", t)
    if not m or m.group(2).lower() not in CHANGE_SUFFIX.get(unit, set()):
        return None
    value = number(m.group(1), lang)
    if value is None or (value != 0 and m.group(1)[0] not in "+-"):
        return None  # a non-zero change is always printed with its sign
    return value


def unit(text: str) -> Unit | None:
    return UNIT_WORDS.get(clean(text))


def measure(text: str) -> Transformation | None:
    return MEASURES.get(clean(text).lower())


def segment(text: str) -> tuple[bool, str | None]:
    t = clean(text).lower()
    return (t in SEGMENTS), SEGMENTS.get(t)


def calendar_date(text: str, lang: str) -> date | None:
    t = clean(text)
    try:
        if lang == "en":
            m = re.match(r"^(\d{1,2}) ([A-Z][a-z]{2}) (\d{4})$", t)
            return date(int(m.group(3)), _MONTHS_EN[m.group(2)], int(m.group(1))) if m else None
        m = re.match(r"^(\d{2})/(\d{2})/(\d{4})$", t)
        return date(int(m.group(3)), int(m.group(2)), int(m.group(1))) if m else None
    except (KeyError, ValueError):
        return None


def period(text: str, lang: str, frequency: Frequency) -> str | None:
    """The catalogue period label (2026-09, 2026-Q3, 2026-10-01), or None."""
    t = clean(text)
    if frequency == Frequency.SNAPSHOT:
        d = calendar_date(t, lang)
        return d.isoformat() if d else None
    if frequency == Frequency.QUARTERLY:
        m = re.match(r"^Q([1-4]) (\d{4})$" if lang == "en" else r"^Quý ([1-4])/(\d{4})$", t)
        return f"{m.group(2)}-Q{m.group(1)}" if m else None
    if lang == "en":
        m = re.match(r"^([A-Z][a-z]{2}) (\d{4})$", t)
        return f"{m.group(2)}-{_MONTHS_EN[m.group(1)]:02d}" if m and m.group(1) in _MONTHS_EN else None
    m = re.match(r"^(0[1-9]|1[0-2])/(\d{4})$", t)
    return f"{m.group(2)}-{m.group(1)}" if m else None


def uncertainty(text: str, lang: str, value: Decimal, unit_: Unit) -> tuple[Decimal, Decimal, Decimal] | None:
    """"±0.1 pp (90%)" around `value` -> (lower, upper, level)."""
    t = clean(text)
    m = re.match(r"^±([\d.,]+)\s*(\S*)\s*\((\d{1,2})%\)$", t)
    if not m or m.group(2).lower() not in CHANGE_SUFFIX.get(unit_, set()):
        return None
    half = number(m.group(1), lang)
    if half is None:
        return None
    return value - half, value + half, Decimal(m.group(3)) / 100


def label_matches(text: str, expected: str, others: list[str], cutoff: float = 0.9) -> bool:
    """A printed label (a district or segment name), tolerating a misread letter: at least `cutoff` similar to the
    expected label and closer to it than to any other label. Used for labels only, never for values."""
    import difflib

    t = clean(text).lower()
    score = difflib.SequenceMatcher(None, t, expected.lower()).ratio()
    return score >= cutoff and all(difflib.SequenceMatcher(None, t, o.lower()).ratio() < score for o in others)
