"""Scores OCR output against a generated document's ground truth: for each field, the words drawn inside its box are
compared with what was printed. Engine-independent; needs only pydantic."""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from ..documents.ground_truth import FieldTruth, GroundTruth
from ..schemas.evidence import Presence
from . import OcrPage

MARGIN_X, MARGIN_Y = 0.003, 0.004  # normalized; tolerates boxes a few pixels off
VALUE_ATTRIBUTES = {"value", "baseline_value", "change_vs_baseline", "previous_value", "observation_period",
                    "baseline_period", "publication_date", "metric_id", "unit"}


# Dash-like characters a number parser treats as a minus or hyphen: figure dash, en dash, em dash, horizontal bar,
# minus sign, small and full-width hyphen-minus. OCR engines return any of them for a printed minus.
DASHES = str.maketrans({c: "-" for c in "‒–—―−﹣－"})


def normalize(text: str) -> str:
    """What reading the field must preserve: Unicode NFC, dashes as a hyphen, single spaces."""
    text = unicodedata.normalize("NFC", text).translate(DASHES)
    return " ".join(text.split())


@dataclass
class FieldResult:
    field_id: str
    attribute: str
    presence: str  # ground truth
    expected: str | None  # normalized printed text
    read: str | None  # normalized OCR text inside the box; None if nothing was read there
    confidence: float | None  # lowest word confidence inside the box
    outcome: str  # correct / wrong / missing (present fields); read / not_read (absent or unreadable fields)


def words_in(page: OcrPage, f: FieldTruth) -> list:
    b = f.bbox
    hits = [w for w in page.words
            if b.x0 - MARGIN_X <= (w.bbox.x0 + w.bbox.x1) / 2 <= b.x1 + MARGIN_X
            and b.y0 - MARGIN_Y <= (w.bbox.y0 + w.bbox.y1) / 2 <= b.y1 + MARGIN_Y]
    return sorted(hits, key=lambda w: (w.line_id, w.bbox.x0))


def score_field(f: FieldTruth, pages: dict[int, OcrPage]) -> FieldResult:
    expected = normalize(f.printed_text) if f.printed_text else None
    hits = words_in(pages[f.page], f) if f.bbox is not None and f.page in pages else []
    read = normalize(" ".join(w.text for w in hits)) if hits else None
    conf = min(w.confidence for w in hits) if hits else None
    if f.presence == Presence.PRESENT:
        outcome = "missing" if read is None else ("correct" if read == expected else "wrong")
    else:
        outcome = "not_read" if read is None else "read"
    return FieldResult(f.field_id, str(f.attribute), f.presence.value, expected, read, conf, outcome)


def score_document(truth: GroundTruth, pages: dict[int, OcrPage]) -> list[FieldResult]:
    return [score_field(f, pages) for f in truth.fields]


def summarize(results: list[FieldResult], thresholds=(0.8, 0.9, 0.95)) -> dict:
    present = [r for r in results if r.presence == "present"]
    values = [r for r in present if r.attribute in VALUE_ATTRIBUTES]
    other = [r for r in results if r.presence != "present"]

    def counts(rs):
        n = len(rs)
        c = {o: sum(r.outcome == o for r in rs) for o in ("correct", "wrong", "missing")}
        out = {"fields": n, **c, "accuracy": round(c["correct"] / n, 3) if n else None}
        for t in thresholds:  # the dangerous case: wrong, and confident enough to be trusted
            out[f"wrong_conf_ge_{t}"] = sum(r.outcome == "wrong" and r.confidence is not None
                                            and r.confidence >= t for r in rs)
        return out

    return {"all_fields": counts(present), "value_fields": counts(values),
            "absent_or_unreadable": {"fields": len(other), "read_anyway": sum(r.outcome == "read" for r in other),
                                     **{f"read_conf_ge_{t}": sum(r.outcome == "read" and r.confidence is not None
                                                                 and r.confidence >= t for r in other)
                                        for t in thresholds}}}
