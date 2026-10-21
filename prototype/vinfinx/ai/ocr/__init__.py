"""OCR service (PIPELINE.md, "OCR service (pluggable)"): every engine returns the same OcrPage structure, words with
boxes and confidence, so each extracted field links back to a visible span. OCR is an evidence extractor, not a truth
oracle (handoff §5).

Engines run in the OCR environment ($VINFINX_ROOT/env-ocr); this package's matching and scoring code needs only
pydantic, so it also runs (and is tested) in the main environment."""
from __future__ import annotations

from typing import Protocol

from pydantic import Field

from ..schemas.common import StrictModel
from ..schemas.evidence import BBox


class OcrWord(StrictModel):
    text: str
    bbox: BBox  # normalized to the page, origin top-left
    confidence: float = Field(ge=0, le=1)
    line_id: int


class OcrPage(StrictModel):
    schema_version: str = "ocr_page/0.2"
    engine: str
    engine_version: str
    model: str  # e.g. "tessdata_best@e12c65a:vie+eng"
    page_sha256: str
    page: int = Field(ge=1)
    width_px: int = Field(gt=0)  # of the page image as received; word boxes are normalized to it
    height_px: int = Field(gt=0)
    rotation_applied: int = 0  # degrees clockwise the image was turned before reading (0, 90, 180, 270)
    orientation_confidence: float | None = None
    words: list[OcrWord]
    seconds: float


class OcrEngine(Protocol):
    name: str

    def read(self, image_path: str, page: int, languages: str) -> OcrPage: ...
