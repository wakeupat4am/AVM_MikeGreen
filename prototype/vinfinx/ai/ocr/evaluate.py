"""Run an OCR engine over a generated document set and score it against the ground truth (PIPELINE.md, "Choosing
the engine": field accuracy on clean and degraded variants, and how often a wrong reading is confident).

    $VINFINX_ROOT/env-ocr/bin/python -m ai.ocr.evaluate --set docs-v0.1-draft --workers 8

Writes OCR output per page (so scoring can be redone without re-reading) and a report under
$VINFINX_ROOT/evaluation/ocr/<run_id>/. Generated data: never commit it.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from ..documents.ground_truth import GroundTruth
from . import OcrPage
from .match import score_document, summarize

from ..paths import ROOT  # noqa: E402  (one place for server paths: ai/paths.py)
LANGUAGES = {"en": "eng", "vi": "vie+eng"}  # the document's language is known for the synthetic set


def _engine(name: str, psm: int):
    if name == "tesseract":
        from .tesseract import TesseractEngine

        return TesseractEngine(psm=psm)
    raise ValueError(f"unknown engine {name}")


def _read_document(args) -> tuple[str, list[dict]]:
    doc_dir, out_dir, engine_name, psm = args
    os.environ.setdefault("OMP_THREAD_LIMIT", "1")  # one core per worker; parallelism comes from the pool
    engine = _engine(engine_name, psm)
    truth = GroundTruth.model_validate_json((doc_dir / "ground_truth.json").read_text(encoding="utf-8"))
    pages = []
    for p in truth.pages:
        cache = out_dir / "pages" / f"{truth.document_id}-p{p.page}.json"
        if cache.exists():
            pages.append(json.loads(cache.read_text(encoding="utf-8")))
            continue
        result = engine.read(str(doc_dir / p.file), p.page, LANGUAGES[truth.language])
        cache.write_text(result.model_dump_json(), encoding="utf-8")
        pages.append(result.model_dump(mode="json"))
    return truth.document_id, pages


def evaluate(set_dir: Path, out_dir: Path, engine: str, psm: int, workers: int, only: list[str] | None) -> dict:
    (out_dir / "pages").mkdir(parents=True, exist_ok=True)
    manifest = json.loads((set_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    docs = sorted(set_dir / d["document_id"] for d in manifest["documents"])  # only what the set's manifest lists
    if only:
        docs = [d for d in docs if any(f"-{v}-" in d.name for v in only)]
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        read = dict(pool.map(_read_document, [(d, out_dir, engine, psm) for d in docs]))
    groups: dict[tuple[str, str], list] = defaultdict(list)
    seconds: dict[tuple[str, str], list[float]] = defaultdict(list)
    documents = []
    for d in docs:
        truth = GroundTruth.model_validate_json((d / "ground_truth.json").read_text(encoding="utf-8"))
        pages = {p["page"]: OcrPage.model_validate(p) for p in read[truth.document_id]}
        results = score_document(truth, pages)
        key = (truth.variant, truth.language)
        groups[key] += results
        seconds[key] += [p.seconds for p in pages.values()]
        documents.append({"document_id": truth.document_id, "variant": truth.variant, "params": truth.variant_params,
                          "language": truth.language, "summary": summarize(results),
                          "misses": [vars(r) for r in results if r.outcome in ("wrong", "missing", "read")]})
    by_variant = {f"{v}/{lang}": {**summarize(rs), "seconds_per_page": round(sum(seconds[(v, lang)]) /
                                                                            len(seconds[(v, lang)]), 1)}
                  for (v, lang), rs in sorted(groups.items())}
    first = next(iter(read.values()))[0]
    report = {"run_id": out_dir.name, "engine": first["engine"], "engine_version": first["engine_version"],
              "model": first["model"], "set": set_dir.name, "documents": len(docs), "workers": workers,
              "wall_seconds": round(time.time() - started), "by_variant": by_variant, "per_document": documents}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", default="docs-v0.1-draft")
    p.add_argument("--engine", default="tesseract")
    p.add_argument("--psm", type=int, default=3)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--only", help="comma-separated variants, e.g. clean,degraded_blur")
    args = p.parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{args.engine}-psm{args.psm}"
    report = evaluate(ROOT / "fixtures" / args.set, ROOT / "evaluation" / "ocr" / run_id, args.engine, args.psm,
                      args.workers, args.only.split(",") if args.only else None)
    print(f"{report['engine']} {report['engine_version']} {report['model']}: {report['documents']} documents, "
          f"{report['wall_seconds']} s")
    print(f"{'variant/lang':42s} {'acc':>6s} {'vals':>6s} {'wrong':>5s} {'w>=.9':>5s} {'miss':>5s} {'s/page':>6s}")
    for k, v in report["by_variant"].items():
        a, val = v["all_fields"], v["value_fields"]
        print(f"{k:42s} {a['accuracy']:6.3f} {val['accuracy']:6.3f} {a['wrong']:5d} {a['wrong_conf_ge_0.9']:5d} "
              f"{a['missing']:5d} {v['seconds_per_page']:6.1f}")
    print(f"report: {ROOT / 'evaluation' / 'ocr' / run_id / 'report.json'}")


if __name__ == "__main__":
    main()
