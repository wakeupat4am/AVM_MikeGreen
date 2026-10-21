"""Field extraction and validation for the macro-status bulletin (handoff §5; MACRO_DOCUMENT_SPEC.md): OCR pages ->
EvidenceItems. An item is trusted only if every check passes; any failure makes it untrusted with a reason, so a
misreading should end up untrusted, never wrong and trusted.

Checks per row: metric ID in the catalogue; period, value, baseline and change parse strictly for the document's
language; Change = Value − Baseline; printed unit, measure, geography and segment agree with the catalogue; values in
range, whole numbers for counts; observation before publication; OCR confidence at or above `min_confidence`; no
figure for the metric in the commentary that contradicts the table. The commentary itself is never evidence.
The Qwen second reading is not wired in yet (`second_reading` stays empty)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal

from ..catalog import Catalogue
from ..evaluation.cases import period_bounds
from ..ocr import OcrPage
from ..schemas.common import Unit
from ..schemas.evidence import (
    EvidenceItem, ExtractorVersion, FieldAttribute, FieldExtraction, OcrProvenance, Presence, ValidationCheck,
    catalogue_violations,
)
from . import parse
from .layout import Cell, Column, full_text, line_after, read_table

EXTRACTOR_VERSION = "macro_doc_extractor/0.1"
PUBLISHER = "Lotus Bay Economic Observatory"
# The bulletin's template: its printed column labels, alignment and relative widths. The layout reader uses the
# labels it can read to place the template on the page; positions are never taken from the generator's output.
TABLE1 = [Column(None, "Metric", "Chỉ tiêu", "left", 540), Column("metric_id", "Metric ID", "Mã chỉ tiêu", "left", 680),
          Column("geography", "Geography", "Địa bàn", "left", 430),
          Column("segment", "Segment", "Phân khúc", "left", 230),
          Column("transformation", "Measure", "Thước đo", "left", 260),
          Column("observation_period", "Period", "Kỳ", "left", 200), Column("value", "Value", "Giá trị", "right", 140),
          Column("unit", "Unit", "Đơn vị", "left", 170), Column("baseline_value", "Baseline", "Kỳ gốc", "right", 140),
          Column("baseline_period", "Baseline period", "Kỳ so sánh", "left", 218),
          Column("change_vs_baseline", "Change", "Thay đổi", "right", 200)]
TABLE2 = [Column("metric_id", "Metric ID", "Mã chỉ tiêu", "left", 800), Column("source", "Source", "Nguồn", "left", 600),
          Column("revision_status", "Revision status", "Tình trạng điều chỉnh", "left", 350),
          Column("previous_value", "Previous value", "Giá trị trước", "right", 330),
          Column("data_quality", "Data quality", "Chất lượng dữ liệu", "left", 300),
          Column("uncertainty", "Uncertainty", "Độ bất định", "left", 500),
          Column("next_update", "Next update", "Kỳ cập nhật tiếp", "left", 328)]
CONFIDENCE_ATTRIBUTES = ("value", "baseline_value", "change_vs_baseline", "observation_period", "baseline_period")


@dataclass
class Extraction:
    items: list[EvidenceItem] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)  # rows that could not become an item, and why
    publication_date: datetime | None = None
    notes: dict = field(default_factory=dict)


def _contradicted(commentary: str, names: list[str], value: Decimal, lang: str) -> bool:
    """Does a commentary sentence naming the metric state a different figure for it?"""
    for sentence in re.split(r"(?<=[.!?])\s+", commentary):
        low = sentence.lower()
        if not any(n.lower() in low for n in names):
            continue
        for raw in re.findall(r"[+-]?\d[\d.,]*\d|\d", sentence):
            n = parse.number(raw.rstrip(".,"), lang)
            if n is not None and n != value and not (1900 <= n <= 2100):  # years are dates, not figures
                return True
    return False


def extract(pages: dict[int, OcrPage], *, lang: str, catalogue: Catalogue, document_id: str, document_hash: str,
            retrieved_at: datetime, synthetic: bool, min_confidence: float = 0.0) -> Extraction:
    known = set(catalogue.instances)
    out = Extraction()
    if 1 not in pages:
        out.failures.append({"metric_id": None, "reason": "PAGE_1_MISSING"})
        return out
    extractor = ExtractorVersion(engine=pages[1].engine, engine_version=pages[1].engine_version,
                                 model=f"{pages[1].model}|{EXTRACTOR_VERSION}")
    t1 = read_table(pages[1], TABLE1, lang, known)
    t2 = read_table(pages[2], TABLE2, lang, known) if 2 in pages else None
    out.notes = {"headers_page1": t1.headers_found, "calibrated_page1": t1.calibrated, "skew_page1": t1.skew_degrees,
                 "unmatched_ids": t1.unmatched_ids, "headers_page2": t2.headers_found if t2 else None,
                 "calibrated_page2": t2.calibrated if t2 else None}
    published = line_after(pages[1], ["Published:", "Ngày phát hành:"])
    pub_date = parse.calendar_date(published.text, lang) if published else None
    out.publication_date = datetime(pub_date.year, pub_date.month, pub_date.day, tzinfo=timezone.utc) if pub_date else None
    printed_id = line_after(pages[1], ["Document ID:", "Mã tài liệu:"])
    commentary = full_text(pages[3]) if 3 in pages else ""
    page_hashes = {n: p.page_sha256 for n, p in pages.items()}

    for metric_id in sorted(known):
        cells = t1.rows.get(metric_id)
        if cells is None:
            out.failures.append({"metric_id": metric_id, "reason": "ROW_NOT_FOUND"})
            continue
        inst = catalogue.get(metric_id)
        d, g = inst.definition, inst.geography

        def text(attribute: str) -> str | None:
            c = cells.get(attribute)
            return c.text if c else None

        label = parse.period(text("observation_period") or "", lang, d.frequency)
        if label is None:
            out.failures.append({"metric_id": metric_id, "reason": "PERIOD_UNREADABLE",
                                 "read": text("observation_period")})
            continue
        start, end = period_bounds(label, d.frequency)
        reasons: list[str] = []
        checks: dict[str, list[ValidationCheck]] = {}

        def check(attribute: str, name: str, ok: bool, detail: str | None = None) -> None:
            checks.setdefault(attribute, []).append(ValidationCheck(check=name, passed=ok, detail=detail))
            if not ok:
                reasons.append(f"{name.upper()}:{attribute}" if detail is None else f"{name.upper()}:{detail}")

        # value: absent (blank, n/a), unreadable (text that does not parse) or present
        vt = text("value")
        if vt is None or parse.clean(vt).lower() in ("n/a", "", "-"):
            presence, value = Presence.ABSENT, None
            reasons.append("VALUE_ABSENT")
        else:
            value = parse.number(vt, lang)
            presence = Presence.PRESENT if value is not None else Presence.UNREADABLE
            if value is None:
                reasons.append("VALUE_UNREADABLE")

        printed_unit = parse.unit(text("unit") or "")
        check("unit", "unit_recognized", printed_unit is not None, f"unit:{text('unit')}")
        unit = printed_unit or d.unit
        transformation = parse.measure(text("transformation") or "")
        check("transformation", "measure_recognized", transformation is not None)
        names = [x.name_en if lang == "en" else x.name_vi for x in catalogue.geographies]
        own = g.name_en if lang == "en" else g.name_vi
        geo_ok = parse.label_matches(text("geography") or "", own, [n for n in names if n != own])
        check("geography", "geography_matches_metric_id", geo_ok)
        seg_known, segment = parse.segment(text("segment") or "")
        check("segment", "segment_matches_scope", seg_known and (segment is not None) == (d.scope == "market"))

        baseline = parse.number(text("baseline_value") or "", lang)
        check("baseline_value", "baseline_parses", baseline is not None)
        baseline_label = parse.period(text("baseline_period") or "", lang, d.frequency)
        check("baseline_period", "baseline_period_parses", baseline_label is not None)
        change = parse.change(text("change_vs_baseline") or "", lang, unit)
        check("change_vs_baseline", "change_parses", change is not None)
        if value is not None and baseline is not None and change is not None:
            check("value", "change_matches_baseline", value - baseline == change,
                  f"{value} - {baseline} != {change}" if value - baseline != change else None)
        if unit == Unit.COUNT:
            for name, v in (("value", value), ("baseline_value", baseline)):
                if v is not None:
                    check(name, "whole_number", v == v.to_integral_value())
        if out.publication_date is not None:
            check("observation_period", "period_before_publication", end < out.publication_date)
        low_conf = [a for a in CONFIDENCE_ATTRIBUTES if a in cells and cells[a].conf < min_confidence]
        if low_conf:
            reasons.append("LOW_CONFIDENCE:" + ",".join(low_conf))
        if value is not None and _contradicted(commentary, [d.name_en, d.name_vi], value, lang):
            reasons.append("CONFLICTING_SOURCES:commentary")

        # page 2: metadata for the same metric
        meta = (t2.rows.get(metric_id) if t2 else None) or {}
        source = meta["source"].text if "source" in meta else PUBLISHER
        caveats = [] if "source" in meta else ["SOURCE_DESK_NOT_READ: source is the publisher, not the desk printed on page 2"]
        revision = parse.REVISION.get(parse.clean(meta["revision_status"].text).lower()) if "revision_status" in meta else None
        quality = parse.QUALITY.get(parse.clean(meta["data_quality"].text).lower(), "unknown") if "data_quality" in meta else "unknown"
        previous = parse.number(meta["previous_value"].text, lang) if "previous_value" in meta else None
        unc, flags_extra = None, []
        if "uncertainty" in meta and value is not None:
            parsed = parse.uncertainty(meta["uncertainty"].text, lang, value, unit)
            if parsed is None:
                flags_extra = ["UNCERTAINTY_UNREADABLE"]  # dropped; an unreadable interval does not taint the value
            else:
                unc = {"kind": "interval", "lower": parsed[0], "upper": parsed[1], "level": parsed[2]}
        flags = (["LOW_QUALITY"] if quality == "low" else []) + flags_extra

        def extraction(attribute: str, normalized) -> tuple[FieldAttribute, FieldExtraction] | None:
            c: Cell | None = cells.get(attribute) or meta.get(attribute)
            fa = FieldAttribute(attribute)
            page = 2 if attribute in ("source", "revision_status", "previous_value", "data_quality", "uncertainty") else 1
            if c is None:
                return fa, FieldExtraction(attribute=fa, presence=Presence.ABSENT, page=page, bbox=None, ocr_text=None,
                                           normalized_value=None, confidence=None, extractor=extractor,
                                           validations=checks.get(attribute, []))
            pres = Presence.PRESENT if normalized is not None else Presence.UNREADABLE
            if attribute == "value":
                pres = presence if presence != Presence.ABSENT else Presence.UNREADABLE
            box = dict(zip(("x0", "y0", "x1", "y1"), c.box))
            return fa, FieldExtraction(attribute=fa, presence=pres, page=page, bbox=box, ocr_text=c.text,
                                       normalized_value=None if pres != Presence.PRESENT else str(normalized),
                                       confidence=c.conf, extractor=extractor, validations=checks.get(attribute, []))

        fields = dict(e for e in (
            extraction("metric_id", metric_id), extraction("observation_period", label),
            extraction("value", value), extraction("unit", printed_unit.value if printed_unit else None),
            extraction("baseline_value", baseline), extraction("baseline_period", baseline_label),
            extraction("change_vs_baseline", change), extraction("geography", g.geo_id if geo_ok else None),
            extraction("transformation", transformation.value if transformation else None),
        ) if e is not None)
        item_data = {
            "evidence_id": f"{metric_id}.{label}", "metric_id": metric_id, "geography": g.geo_id, "segment": segment,
            "source": source, "source_document_id": printed_id.text if printed_id else None,
            "observation_period": {"label": label, "start": start, "end": end},
            "publication_date": out.publication_date if out.publication_date and out.publication_date <= retrieved_at else None,
            "retrieved_at": retrieved_at, "value": value if presence == Presence.PRESENT else None, "unit": unit,
            "transformation": transformation or d.transformation,
            "baseline": {"value": baseline, "unit": unit, "period_label": baseline_label or ""} if baseline is not None else None,
            "revision_status": revision, "revision_of": str(previous) if revision == "revised" and previous is not None else None,
            "data_quality": quality, "quality_flags": flags, "uncertainty": unc,
            "freshness": {"deadline": catalogue.freshness_deadline(metric_id, end),
                          "catalogue_version": catalogue.version, "catalogue_hash": catalogue.catalogue_hash},
            "presence": presence, "trust": "trusted", "trust_reasons": [], "synthetic": synthetic, "caveats": caveats,
            "provenance": OcrProvenance(document_id=document_id, document_hash=document_hash, page_hashes=page_hashes,
                                        fields=fields),
        }
        # the catalogue checks run on the item itself (range, unit, measure, freshness source)
        provisional = EvidenceItem.model_validate({**item_data, "trust": "untrusted", "trust_reasons": ["CHECKING"]})
        reasons += catalogue_violations(provisional, catalogue)
        reasons = list(dict.fromkeys(reasons))
        item_data.update(trust="untrusted" if reasons else "trusted", trust_reasons=reasons)
        out.items.append(EvidenceItem.model_validate(item_data))
    return out
