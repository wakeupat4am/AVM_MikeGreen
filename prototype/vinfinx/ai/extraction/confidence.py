"""Evidence for choosing an OCR confidence threshold (handoff §5 Q1), from OCR runs already scored on generated sets.

1. Calibration: for each confidence band, how often is a field read exactly right? Split by value fields vs labels
   and by clean vs degraded pages. A calibrated engine is right about as often as its confidence says.
2. Residual errors: fields that are wrong inside items that end up trusted, i.e. errors no other check caught.
   Only these can a threshold still stop. For each field, the confidences of its residual errors and of its correct
   readings show what a threshold would catch and what it would cost.

    python -m ai.extraction.confidence --run docs-v0.1-draft=<ocr_run> --run docs-v0.1-holdout-s7=<ocr_run>

Needs only the cached OCR output, so it runs in either environment.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from ..catalog import load_catalogue
from ..documents.ground_truth import GroundTruth
from ..ocr import OcrPage
from ..ocr.match import VALUE_ATTRIBUTES, score_document
from .layout import read_table
from .macro_document import TABLE2, extract

from ..paths import ROOT  # noqa: E402  (one place for server paths: ai/paths.py)
BANDS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.01)]
THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95)
CONTENT_VARIANTS_PREFIX = "degraded_"


def _band(c: float) -> str:
    for lo, hi in BANDS:
        if lo <= c < hi:
            return f"{lo:.2f}-{min(hi, 1):.2f}"
    return "?"


def _documents(set_name: str, ocr_run: str):
    set_dir = ROOT / "fixtures" / set_name
    pages_dir = ROOT / "evaluation" / "ocr" / ocr_run / "pages"
    for entry in json.loads((set_dir / "MANIFEST.json").read_text(encoding="utf-8"))["documents"]:
        truth = GroundTruth.model_validate_json((set_dir / entry["document_id"] / "ground_truth.json").read_text("utf-8"))
        pages = {p.page: OcrPage.model_validate_json(
            (pages_dir / f"{truth.document_id}-p{p.page}.json").read_text("utf-8")) for p in truth.pages}
        yield truth, pages


def calibration(docs) -> dict:
    """band -> {field class / page condition -> [correct, total]}"""
    table: dict = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for truth, pages in docs:
        condition = "degraded" if truth.variant.startswith(CONTENT_VARIANTS_PREFIX) else "clean"
        for r in score_document(truth, pages):
            if r.presence != "present" or r.outcome == "missing" or r.confidence is None:
                continue
            kind = "value" if r.attribute in VALUE_ATTRIBUTES else "label"
            for key in (f"{kind}/{condition}", f"{kind}/all", "all"):
                cell = table[_band(r.confidence)][key]
                cell[0] += r.outcome == "correct"
                cell[1] += 1
    return {b: {k: {"correct": v[0], "fields": v[1], "accuracy": round(v[0] / v[1], 3)} for k, v in sorted(d.items())}
            for b, d in sorted(table.items())}


def _truth_uncertainty(true_value: str | None):
    if not true_value:
        return None
    _, lo, hi, level = true_value.split(":")
    return Decimal(lo), Decimal(hi), Decimal(level)


def residuals(docs, catalogue) -> dict:
    """Per field: confidences of correct readings and of wrong ones inside trusted items (min confidence 0)."""
    out: dict = defaultdict(lambda: {"correct": [], "wrong": [], "defaulted": 0, "not_taken": 0, "examples": []})
    known = set(catalogue.instances)
    for truth, pages in docs:
        fields = {f.field_id: f for f in truth.fields}
        ex = extract(pages, lang=truth.language, catalogue=catalogue, document_id=truth.document_id,
                     document_hash=truth.pdf.sha256, retrieved_at=truth.evaluate_at, synthetic=True, min_confidence=0)
        meta = read_table(pages[2], TABLE2, truth.language, known).rows if 2 in pages else {}
        for item in ex.items:
            if item.trust != "trusted":
                continue
            m = item.metric_id
            # page-1 fields, from the item's provenance
            for attribute, fx in item.provenance.fields.items():
                gt = fields.get(f"{m}/{attribute.value}")
                if gt is None or fx.confidence is None or fx.normalized_value is None:
                    continue
                expected = gt.true_value
                ok = fx.normalized_value == expected or (attribute.value == "unit" and expected == fx.normalized_value)
                if attribute.value == "geography":
                    ok = fx.normalized_value == expected
                out[attribute.value]["correct" if ok else "wrong"].append(round(fx.confidence, 3))
                if not ok:
                    out[attribute.value]["examples"].append([truth.document_id, m, expected, fx.normalized_value])
            # page-2 fields: what the item says vs the ground truth, with the cell's confidence
            row = meta.get(m, {})
            checks = {
                "source": (item.source, fields[f"{m}/source"].true_value),
                "revision_status": (item.revision_status, fields[f"{m}/revision_status"].true_value),
                "data_quality": (item.data_quality, fields[f"{m}/data_quality"].true_value),
                "previous_value": (item.revision_of, fields[f"{m}/previous_value"].true_value),
                "uncertainty": ((item.uncertainty.lower, item.uncertainty.upper, item.uncertainty.level)
                                if item.uncertainty else None,
                                _truth_uncertainty(fields[f"{m}/uncertainty"].true_value)),
            }
            for attribute, (got, expected) in checks.items():
                cell = row.get(attribute)
                if cell is None:
                    if got is not None and got != "unknown":
                        out[attribute]["defaulted"] += 1  # not read, but the item still carries a value
                        if got != expected:
                            out[attribute]["examples"].append([truth.document_id, m, str(expected), f"default:{got}"])
                    continue
                if got is None or got == "unknown":
                    out[attribute]["not_taken"] += 1  # read but rejected by its parser: dropped, not wrong
                    continue
                ok = got == expected or (attribute == "previous_value" and expected is not None
                                         and Decimal(got) == Decimal(expected))
                out[attribute]["correct" if ok else "wrong"].append(round(cell.conf, 3))
                if not ok:
                    out[attribute]["examples"].append([truth.document_id, m, str(expected), str(got)])
    summary = {}
    for attribute, d in sorted(out.items()):
        c, w = d["correct"], d["wrong"]
        summary[attribute] = {
            "correct": len(c), "wrong_in_trusted_items": len(w), "not_read_but_defaulted": d["defaulted"],
            "read_but_dropped": d["not_taken"],
            "wrong_confidences": sorted(w),
            "by_threshold": {str(t): {"wrong_stopped": sum(x < t for x in w), "correct_lost": sum(x < t for x in c)}
                             for t in THRESHOLDS},
            "examples": d["examples"][:8],
        }
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, help="<set>=<ocr run id>, repeatable")
    args = p.parse_args()
    catalogue = load_catalogue()
    runs = [r.split("=", 1) for r in args.run]
    report = {"runs": dict(runs),
              "calibration": calibration(d for s, r in runs for d in _documents(s, r)),
              "residuals": residuals((d for s, r in runs for d in _documents(s, r)), catalogue)}
    out = ROOT / "evaluation" / "extraction" / f"confidence-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("calibration: accuracy (fields) per confidence band")
    keys = ["value/clean", "value/degraded", "label/clean", "label/degraded"]
    print(f"{'band':10s} " + " ".join(f"{k:>20s}" for k in keys))
    for band, d in report["calibration"].items():
        print(f"{band:10s} " + " ".join(f"{d[k]['accuracy']:>12.3f} ({d[k]['fields']:5d})" if k in d else f"{'-':>20s}"
                                        for k in keys))
    print("\nresidual errors (wrong fields inside trusted items):")
    for a, d in report["residuals"].items():
        stops = ", ".join(f"{t}: stops {v['wrong_stopped']}/{d['wrong_in_trusted_items']}, loses {v['correct_lost']}"
                          for t, v in d["by_threshold"].items())
        print(f"  {a:20s} correct {d['correct']:5d} wrong {d['wrong_in_trusted_items']:3d} defaulted "
              f"{d['not_read_but_defaulted']:3d} dropped {d['read_but_dropped']:3d} | {stops}")
    print(f"report: {out}")


if __name__ == "__main__":
    main()
