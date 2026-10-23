"""Page text (CORPUS_INDEX_PLAN.md, I1): the PDF's text layer with pypdfium2; a page without one but with an image is
rendered at 300 dpi and read by Tesseract in env-ocr. The stored page text is what chunk offsets point into."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from . import OCR_PYTHON

MIN_CHARS = 50  # below this a page has no usable text layer
OCR_DPI = 300
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f￾￿]")
# A symbol between two letters ("t!c", "t\"ng"): a legacy Vietnamese font without a Unicode map. Measured on the
# dev corpus 2026-10-22: one such document scored 21 per 1,000 characters (max 41); every other page 0-6.
SYMBOL_IN_WORD = re.compile(r"[A-Za-zÀ-ỹ][!\"#$%&()*+<=>@\[\]^_`{|}~\\][A-Za-zÀ-ỹ]")
GARBLED_PER_1000 = 10.0


def garbled(text: str) -> bool:
    """True when the text layer looks broken (see SYMBOL_IN_WORD); such a page is read by OCR instead."""
    return len(SYMBOL_IN_WORD.findall(text)) / max(len(text), 1) * 1000 > GARBLED_PER_1000


@dataclass
class PageText:
    page: int
    text: str
    extractor: str
    ocr_mean_confidence: float | None = None

    @property
    def text_hash(self) -> str:
        return "sha256:" + hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def clean(text: str) -> str:
    """NFC, Unix line ends, no control characters. Applied once, before storing, so offsets stay valid."""
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    return CONTROL.sub("", text)


def pdfium_version() -> str:
    import pypdfium2 as pdfium

    return f"pypdfium2 {pdfium.PYPDFIUM_INFO} / pdfium {pdfium.PDFIUM_INFO}"


def extract_pages(pdf_path: Path, ocr: bool = True, lang: str = "vie+eng") -> list[PageText]:
    import pypdfium2 as pdfium
    import pypdfium2.raw as raw

    doc = pdfium.PdfDocument(str(pdf_path))
    out: list[PageText] = []
    scanned: list[int] = []
    try:
        for i in range(len(doc)):
            page = doc[i]
            tp = page.get_textpage()
            text = clean(tp.get_text_range())
            tp.close()
            if len(text.strip()) >= MIN_CHARS and not garbled(text):
                out.append(PageText(i + 1, text, f"pdfium:{pdfium.PDFIUM_INFO}"))
            elif len(text.strip()) >= MIN_CHARS or any(
                    True for _ in page.get_objects(filter=[raw.FPDF_PAGEOBJ_IMAGE], max_depth=2)):
                scanned.append(i)  # broken text layer, or an image with no text: read by OCR
            page.close()
        if ocr and scanned:
            with tempfile.TemporaryDirectory() as tmp:
                images = []
                for i in scanned:
                    path = Path(tmp) / f"p{i + 1}.png"
                    doc[i].render(scale=OCR_DPI / 72).to_pil().convert("L").save(path)
                    images.append(str(path))
                for row in ocr_images(images, lang):
                    page_no = int(Path(row["image"]).stem[1:])
                    text = clean(row["text"])
                    if len(text.strip()) >= MIN_CHARS:
                        out.append(PageText(page_no, text, f"tesseract:{row['model']}", row["mean_confidence"]))
    finally:
        doc.close()
    return sorted(out, key=lambda p: p.page)


def ocr_images(images: list[str], lang: str) -> list[dict]:
    """Runs ai.corpus.ocr_images in env-ocr (Tesseract lives there) and returns one row per image."""
    here = Path(__file__).resolve().parents[2]  # prototype/vinfinx, so `ai` is importable
    result = subprocess.run([str(OCR_PYTHON), "-m", "ai.corpus.ocr_images", "--lang", lang, *images],
                            capture_output=True, text=True, cwd=here, check=True, timeout=60 * len(images) + 60)
    return json.loads(result.stdout)
