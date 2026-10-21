"""End-to-end measure of the document chain (OCR -> extraction -> validation) on a generated set: how many evidence
items come out **wrong and trusted**. That is the number the controls exist to keep at zero.

    $VINFINX_ROOT/env-ocr/bin/python -m ai.extraction.evaluate --ocr-run <run_id>     # reuse OCR output
    $VINFINX_ROOT/env-ocr/bin/python -m ai.extraction.evaluate                        # run OCR first

For each document and each catalogue metric the item is one of:
- ok:            trusted, and metric, period, value, baseline and baseline period equal the ground truth;
- WRONG_TRUSTED: trusted, but one of those is wrong, or the value was not printed or not readable;
- MISSED_FLAG:   trusted and read correctly, but the document says it must not be trusted (an inconsistent
                 tampering, a unit the catalogue does not use, a contradiction in the commentary);
- caught:        untrusted, and it was wrong or had to be flagged;
- lost:          untrusted although it was right (coverage lost to caution);
- not_extracted: no item (row or period not found).
`tampered_consistent` items that pass are counted as ok and reported separately as accepted alterations: the spec
expects them to be undetectable at this stage.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from ..catalog import load_catalogue
from ..documents.ground_truth import GroundTruth
from ..ocr import OcrPage
from .macro_document import extract

ROOT = Path(os.environ.get("VINFINX_ROOT", "/mnt/disk1/backup_user/24chuong.ta/vinfinx"))
THRESHOLDS = (0.0, 0.5, 0.7, 0.8, 0.9)
CLASSES = ("ok", "WRONG_TRUSTED", "MISSED_FLAG", "caught", "lost", "not_extracted")


def must_flag(truth: GroundTruth) -> set[str]:
    """Metric IDs the ground truth says must not end up trusted."""
    out = set()
    for e in truth.expected:
        if e.field_id and e.trust == "untrusted" and e.field_id != "*":
            out.add(e.field_id.split("/")[0])
        if e.evidence_id and e.decision_effect in ("must_not_cite_as_trusted", "must_abstain_on_metric"):
            out.add(e.evidence_id.rsplit(".", 1)[0])
    return out


def classify(truth: GroundTruth, items, catalogue) -> dict[str, str]:
    fields = {f.field_id: f for f in truth.fields}
    flagged = must_flag(truth)
    by_metric = {i.metric_id: i for i in items}
    out = {}
    for metric_id in catalogue.instances:
        item = by_metric.get(metric_id)
        if item is None:
            out[metric_id] = "not_extracted"
            continue
        tv = fields[f"{metric_id}/value"]
        correct = (tv.true_value is not None and item.value is not None and str(item.value) == tv.true_value
                   and item.observation_period.label == fields[f"{metric_id}/observation_period"].true_value
                   and item.baseline is not None
                   and str(item.baseline.value) == fields[f"{metric_id}/baseline_value"].true_value
                   and item.baseline.period_label == fields[f"{metric_id}/baseline_period"].true_value)
        if item.trust == "trusted":
            out[metric_id] = ("MISSED_FLAG" if metric_id in flagged else "ok") if correct else "WRONG_TRUSTED"
        else:
            out[metric_id] = "lost" if correct and metric_id not in flagged else "caught"
    return out


def run(set_dir: Path, ocr_dir: Path, thresholds=THRESHOLDS) -> dict:
    catalogue = load_catalogue()
    manifest = json.loads((set_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    per_doc, groups = [], defaultdict(lambda: defaultdict(Counter))
    wrong_examples = []
    for entry in manifest["documents"]:
        doc_dir = set_dir / entry["document_id"]
        truth = GroundTruth.model_validate_json((doc_dir / "ground_truth.json").read_text(encoding="utf-8"))
        pages = {p.page: OcrPage.model_validate_json(
            (ocr_dir / "pages" / f"{truth.document_id}-p{p.page}.json").read_text(encoding="utf-8")) for p in truth.pages}
        doc = {"document_id": truth.document_id, "variant": truth.variant, "language": truth.language, "by_threshold": {}}
        for th in thresholds:
            ex = extract(pages, lang=truth.language, catalogue=catalogue, document_id=truth.document_id,
                         document_hash=truth.pdf.sha256, retrieved_at=truth.evaluate_at, synthetic=True,
                         min_confidence=th)
            classes = classify(truth, ex.items, catalogue)
            counts = Counter(classes.values())
            if truth.variant == "tampered_consistent":
                counts["altered_accepted"] = sum(1 for m, c in classes.items() if c == "ok"
                                                 and f"{m}/value" in truth.changed_fields)
            doc["by_threshold"][str(th)] = {"counts": dict(counts), "notes": ex.notes,
                                            "failures": ex.failures[:5]}
            groups[th][f"{truth.variant}/{truth.language}"].update(counts)
            for m, c in classes.items():
                if c in ("WRONG_TRUSTED", "MISSED_FLAG"):
                    item = next(i for i in ex.items if i.metric_id == m)
                    wrong_examples.append({"threshold": th, "document_id": truth.document_id, "metric_id": m,
                                           "class": c, "value": str(item.value),
                                           "baseline": str(item.baseline.value) if item.baseline else None,
                                           "period": item.observation_period.label})
        per_doc.append(doc)
    return {"set": set_dir.name, "ocr_run": ocr_dir.name, "thresholds": list(thresholds),
            "by_variant": {str(th): {k: dict(v) for k, v in sorted(g.items())} for th, g in groups.items()},
            "totals": {str(th): dict(sum(g.values(), Counter())) for th, g in groups.items()},
            "wrong_or_missed": wrong_examples, "per_document": per_doc}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", default="docs-v0.1-draft")
    p.add_argument("--ocr-run", help="an existing run under $VINFINX_ROOT/evaluation/ocr/; default: run OCR now")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    set_dir = ROOT / "fixtures" / args.set
    if args.ocr_run:
        ocr_dir = ROOT / "evaluation" / "ocr" / args.ocr_run
    else:
        from ..ocr.evaluate import evaluate as run_ocr

        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-tesseract-psm3"
        ocr_dir = ROOT / "evaluation" / "ocr" / run_id
        run_ocr(set_dir, ocr_dir, "tesseract", 3, args.workers, None)
    report = run(set_dir, ocr_dir)
    out = ROOT / "evaluation" / "extraction" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    for th, t in report["totals"].items():
        print(f"min confidence {th}: " + ", ".join(f"{c} {t.get(c, 0)}" for c in CLASSES)
              + (f", altered_accepted {t['altered_accepted']}" if "altered_accepted" in t else ""))
    th = "0.0"
    print(f"\nby variant at min confidence {th}:")
    print(f"{'variant/lang':34s} " + " ".join(f"{c[:8]:>8s}" for c in CLASSES))
    for k, c in report["by_variant"][th].items():
        print(f"{k:34s} " + " ".join(f"{c.get(x, 0):8d}" for x in CLASSES))
    print(f"report: {out}")


if __name__ == "__main__":
    main()
