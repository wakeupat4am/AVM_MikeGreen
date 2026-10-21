"""Generate synthetic macro-status documents with ground truth (MACRO_DOCUMENT_SPEC.md).

    python -m ai.documents.generate --set-version docs-v0.1-draft                 # every variant, en and vi
    python -m ai.documents.generate --variants clean,degraded_blur --languages en --out /some/dir

Writes, per document, page-<n>.png (or .jpg), document.pdf (image-only) and ground_truth.json, plus a
MANIFEST.json for the set, under $VINFINX_ROOT/fixtures/<set-version>/ by default. Generated documents are
test data: never commit them (commit only the manifest of a frozen set).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import PIL
from PIL import Image

from ..catalog import Catalogue, load_catalogue
from . import GENERATOR_NAME, GENERATOR_VERSION
from .degrade import IMAGE_DEFAULTS, PageImage, apply, unchanged
from .ground_truth import Expected, FieldTruth, GeneratorInfo, GroundTruth, PageTruth
from .render import render, font_hashes
from .variants import CONTENT_DEFAULTS, build_content
from ..schemas.evidence import BBox, Presence

ALL_VARIANTS = {**CONTENT_DEFAULTS, **IMAGE_DEFAULTS}


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def document_id(issue: str, variant: str, params: dict[str, Any], lang: str, seed: int) -> str:
    # a minus becomes "neg": "-" separates the ID's parts, and dropping it made -1 and 1 collide
    parts = [re.sub(r"[^A-Za-z0-9_.]", "", (str(v).split(".")[-1] if k == "metric_id" else str(v)).replace("-", "neg"))
             for k, v in params.items()]
    suffix = "".join(f"-{p}" for p in parts)
    return f"LBEO-BUL-{issue}-{variant}{suffix}-{lang}-s{seed}"


def _png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def generate_document(variant: str, params: dict[str, Any], *, catalogue: Catalogue, as_of: datetime, seed: int,
                      lang: str, out_root: Path) -> GroundTruth:
    if variant not in ALL_VARIANTS:
        raise ValueError(f"unknown variant {variant}")
    built = build_content(variant if variant in CONTENT_DEFAULTS else "clean", params, catalogue, as_of, seed, lang)
    doc = built.doc
    images, placed = render(doc)
    if variant in IMAGE_DEFAULTS:
        pages, states, changed = apply(variant, params, images, placed, seed)
    else:
        pages, states, changed = unchanged(images, placed)
        changed = list(doc.changed_fields)

    doc_id = document_id(doc.issue, variant, params, lang, seed)
    out = out_root / doc_id
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    page_truths = []
    for n, pg in enumerate(pages, start=1):
        data = pg.encoded if pg.encoded is not None else _png(pg.image)
        name = f"page-{n}.{pg.ext}"
        (out / name).write_bytes(data)
        page_truths.append(PageTruth(page=n, file=name, sha256=_sha(data), dpi=pg.dpi,
                                     width_px=pg.image.width, height_px=pg.image.height))
    pdf = _write_pdf(out / "document.pdf", pages, doc.as_of, doc_id)

    sizes = {n: pg.image.size for n, pg in enumerate(pages, start=1)}
    fields = []
    for s in states:
        w, h = sizes[s.placed.page]
        bbox = None
        if s.box is not None:
            x0, y0, x1, y1 = s.box
            bbox = BBox(x0=max(x0 / w, 0), y0=max(y0 / h, 0), x1=min(x1 / w, 1), y1=min(y1 / h, 1))
        cell = s.placed.cell
        fields.append(FieldTruth(
            field_id=s.placed.field_id, metric_id=s.placed.metric_id, attribute=s.placed.attribute,
            page=s.placed.page, bbox=bbox,
            printed_text=cell.printed if s.box is not None else None,
            true_value=cell.true_value if s.presence == Presence.PRESENT else None, presence=s.presence))

    expected = list(built.expected)
    if variant in IMAGE_DEFAULTS:
        expected = [Expected(field_id="*", reason="CORRECT_OR_UNTRUSTED",
                             note="each field is read correctly or marked untrusted; never wrong and trusted")]
        expected += [Expected(field_id=f.field_id, presence=f.presence) for f in fields if f.field_id in changed]

    base_params: dict[str, Any] = {}
    truth = GroundTruth(
        document_id=doc_id, base_document_id=document_id(doc.issue, "clean", base_params, lang, seed),
        variant=variant, variant_params=params, language=lang, as_of=doc.as_of, evaluate_at=built.evaluate_at,
        generator=GeneratorInfo(name=GENERATOR_NAME, version=GENERATOR_VERSION, seed=seed, pillow=PIL.__version__,
                                fonts=font_hashes()),
        catalogue={"version": catalogue.version, "hash": catalogue.catalogue_hash},
        pdf=pdf, pages=page_truths, fields=fields, changed_fields=changed, expected=expected)
    (out / "ground_truth.json").write_text(truth.model_dump_json(indent=2), encoding="utf-8")
    return truth


def _write_pdf(path: Path, pages: list[PageImage], as_of: datetime, doc_id: str) -> PageTruth:
    first, rest = pages[0].image, [p.image for p in pages[1:]]
    first.save(path, "PDF", save_all=True, append_images=rest, resolution=float(pages[0].dpi),
               title="Macro and Housing Market Status Bulletin (SYNTHETIC TEST DATA)",
               subject=f"Synthetic test data from a fictional publisher; not real economic data. {doc_id}",
               keywords="synthetic, test data, fictional publisher, vinfinx",
               producer=f"{GENERATOR_NAME} {GENERATOR_VERSION}", creator=GENERATOR_NAME,
               creationDate=as_of.utctimetuple(), modDate=as_of.utctimetuple())
    data = path.read_bytes()
    return PageTruth(page=1, file=path.name, sha256=_sha(data), dpi=pages[0].dpi, width_px=first.width,
                     height_px=first.height)


def generate_set(out_root: Path, *, as_of: datetime, seed: int, languages: list[str],
                 variants: list[str] | None = None, catalogue: Catalogue | None = None) -> dict[str, Any]:
    catalogue = catalogue or load_catalogue()
    out_root.mkdir(parents=True, exist_ok=True)
    documents = []
    for variant in variants or list(ALL_VARIANTS):
        for params in ALL_VARIANTS[variant]:
            for lang in languages:
                truth = generate_document(variant, params, catalogue=catalogue, as_of=as_of, seed=seed, lang=lang,
                                          out_root=out_root)
                files = {p.file: p.sha256 for p in truth.pages}
                files["document.pdf"] = truth.pdf.sha256
                files["ground_truth.json"] = _sha((out_root / truth.document_id / "ground_truth.json").read_bytes())
                if any(d["document_id"] == truth.document_id for d in documents):
                    raise ValueError(f"two settings give the same document ID {truth.document_id}")
                documents.append({"document_id": truth.document_id, "variant": variant, "params": params,
                                  "language": lang, "files": files})
                print(truth.document_id, flush=True)
    manifest = {"synthetic": True, "generator": {"name": GENERATOR_NAME, "version": GENERATOR_VERSION},
                "as_of": as_of.isoformat(), "seed": seed, "languages": languages,
                "catalogue": {"version": catalogue.version, "hash": catalogue.catalogue_hash},
                "documents": documents}
    (out_root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    root = Path(os.environ.get("VINFINX_ROOT", "/mnt/disk1/backup_user/24chuong.ta/vinfinx"))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set-version", default="docs-v0.1-draft")
    p.add_argument("--out", type=Path, help="default: $VINFINX_ROOT/fixtures/<set-version>")
    p.add_argument("--as-of", default="2026-10-05", help="publication date (UTC)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--languages", default="en,vi")
    p.add_argument("--variants", help=f"comma-separated; default all: {', '.join(ALL_VARIANTS)}")
    args = p.parse_args()
    as_of = datetime.fromisoformat(args.as_of).replace(tzinfo=timezone.utc)
    out = args.out or root / "fixtures" / args.set_version
    manifest = generate_set(out, as_of=as_of, seed=args.seed, languages=args.languages.split(","),
                            variants=args.variants.split(",") if args.variants else None)
    print(f"{len(manifest['documents'])} documents in {out}")


if __name__ == "__main__":
    main()
