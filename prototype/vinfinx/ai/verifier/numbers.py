"""Find numbers in free text and say which evidence values they could be.

English and Vietnamese write numbers differently ("1,250.5" vs "1.250,5"), and "1.250" means 1.25 in one
and 1250 in the other. So each number gets every plausible reading, and it counts as supported if any
reading matches an allowed value. Spelled-out numbers ("four percent") are not detected in v0.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

NUMBER = re.compile(r"(?<![\w.,])([-−]?)(\d+(?:[.,]\d+)*)")
UNIT_HINTS = [
    (re.compile(r"\s*(?:%|percent\b|per cent\b|phần trăm)", re.IGNORECASE), "pct"),
    (re.compile(r"\s*(?:bps\b|basis points?\b|điểm cơ bản)", re.IGNORECASE), "bps"),
    (re.compile(r"\s*(?:pp\b|percentage points?\b|điểm phần trăm|đpt\b)", re.IGNORECASE), "pp"),
]


@dataclass(frozen=True)
class NumberMention:
    raw: str
    readings: frozenset[Decimal]
    unit_hint: str | None  # "pct", "bps", "pp" or None
    start: int


def normalize(value: Decimal) -> Decimal:
    """4.0, 4 and 4.00 compare equal."""
    return value.normalize() if value != 0 else Decimal(0)


def readings(token: str) -> set[Decimal]:
    parts = re.split(r"[.,]", token)
    out: set[str] = {"".join(parts)}  # every separator is a thousands separator ("1.250" -> 1250)
    if len(parts) > 1:
        # the last separator is the decimal point, the others are thousands separators
        out.add("".join(parts[:-1]) + "." + parts[-1])
    values = set()
    for s in out:
        try:
            values.add(normalize(Decimal(s)))
        except InvalidOperation:
            pass
    return values


def find_numbers(text: str) -> list[NumberMention]:
    mentions = []
    for m in NUMBER.finditer(text):
        sign, token = m.group(1), m.group(2)
        values = readings(token)
        if sign:
            values = {-v for v in values}
        hint = None
        for pattern, unit in UNIT_HINTS:
            if pattern.match(text, m.end()):
                hint = unit
                break
        mentions.append(NumberMention(raw=m.group(0), readings=frozenset(values), unit_hint=hint, start=m.start()))
    return mentions
