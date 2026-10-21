"""Ground truth for one generated document instance (MACRO_DOCUMENT_SPEC.md, "Ground truth")."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import AwareDatetime, Field, model_validator

from ..schemas.common import Sha256, StrictModel
from ..schemas.evidence import BBox, FieldAttribute, Presence

HeaderAttribute = Literal["publisher", "title", "issue", "document_id", "publication_date"]
Language = Literal["en", "vi"]


class PageTruth(StrictModel):
    page: int = Field(ge=1)
    file: str
    sha256: Sha256
    dpi: int = Field(gt=0)
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)


class FieldTruth(StrictModel):
    field_id: str  # <metric_id>/<attribute>, or document/<attribute> for header fields
    metric_id: str | None  # None for header fields
    attribute: FieldAttribute | HeaderAttribute
    page: int = Field(ge=1)
    bbox: BBox | None  # where it was drawn, after any rotation or crop; None if not on the page
    printed_text: str | None  # exactly what was rendered; None if nothing was printed
    true_value: str | None  # normalized ("6.0", "1250", "2026-09"); None if absent or unreadable
    presence: Presence

    @model_validator(mode="after")
    def _consistent(self) -> FieldTruth:
        expected_prefix = "document" if self.metric_id is None else self.metric_id
        if self.field_id != f"{expected_prefix}/{self.attribute}":
            raise ValueError(f"field_id {self.field_id} does not match its metric and attribute")
        if self.presence == Presence.PRESENT and (self.bbox is None or self.printed_text is None
                                                  or self.true_value is None):
            raise ValueError(f"{self.field_id}: a present field needs bbox, printed_text and true_value")
        if self.presence == Presence.UNREADABLE and (self.bbox is None or self.true_value is not None):
            raise ValueError(f"{self.field_id}: an unreadable field has a bbox and no true_value")
        if self.presence == Presence.ABSENT and self.true_value is not None:
            raise ValueError(f"{self.field_id}: an absent field has no true_value")
        return self


class Expected(StrictModel):
    """What the pipeline should conclude. `field_id` "*" means every field of the document."""

    field_id: str | None = None
    evidence_id: str | None = None
    presence: Presence | None = None
    trust: Literal["trusted", "untrusted"] | None = None
    reason: str | None = None
    decision_effect: str | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _subject(self) -> Expected:
        if (self.field_id is None) == (self.evidence_id is None):
            raise ValueError("an expectation is about exactly one field or one evidence item")
        return self


class GeneratorInfo(StrictModel):
    name: str
    version: str
    seed: int
    pillow: str
    fonts: dict[str, Sha256]  # font file name -> hash (rendering depends on the exact font file)


class GroundTruth(StrictModel):
    schema_version: Literal["macro_doc_ground_truth/0.1"] = "macro_doc_ground_truth/0.1"
    synthetic: Literal[True] = True
    document_id: str
    base_document_id: str
    variant: str
    variant_params: dict[str, Any]
    language: Language
    as_of: AwareDatetime  # the publication date printed on the document
    evaluate_at: AwareDatetime  # when the pipeline should evaluate it (later than as_of for `stale`)
    generator: GeneratorInfo
    catalogue: dict[str, str]
    pdf: PageTruth | None = None  # the image-only PDF (page 0 = the whole file)
    pages: list[PageTruth] = Field(min_length=1)
    fields: list[FieldTruth]
    changed_fields: list[str]
    expected: list[Expected]

    @model_validator(mode="after")
    def _consistent(self) -> GroundTruth:
        ids = [f.field_id for f in self.fields]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate field_id")
        unknown = set(self.changed_fields) - set(ids)
        if unknown:
            raise ValueError(f"changed_fields not in fields: {sorted(unknown)}")
        pages = {p.page for p in self.pages}
        if any(f.page not in pages for f in self.fields):
            raise ValueError("a field is on a page that does not exist")
        if self.evaluate_at < self.as_of:
            raise ValueError("evaluate_at is before as_of")
        return self
