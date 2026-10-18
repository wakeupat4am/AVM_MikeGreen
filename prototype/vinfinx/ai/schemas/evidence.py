"""EvidenceItem: one observation, with the handoff §3 input contract and §5 per-field OCR provenance."""
from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import AwareDatetime, Field, model_validator

from .common import Sha256, StrictModel, Transformation, Unit, content_hash

if TYPE_CHECKING:
    from ..catalog import Catalogue

Confidence = Annotated[float, Field(ge=0, le=1)]


class Presence(StrEnum):
    PRESENT = "present"
    ABSENT = "absent"  # not on the document
    UNREADABLE = "unreadable"  # printed, but cannot be read (handoff §5 Q5)


class FieldAttribute(StrEnum):
    """Each printed field of an indicator row on the macro-status document."""

    METRIC_ID = "metric_id"
    GEOGRAPHY = "geography"
    SEGMENT = "segment"
    TRANSFORMATION = "transformation"
    OBSERVATION_PERIOD = "observation_period"
    VALUE = "value"
    UNIT = "unit"
    BASELINE_VALUE = "baseline_value"
    BASELINE_PERIOD = "baseline_period"
    CHANGE_VS_BASELINE = "change_vs_baseline"  # kept for cross-field validation only
    SOURCE = "source"
    REVISION_STATUS = "revision_status"
    PREVIOUS_VALUE = "previous_value"  # printed only for revised figures
    DATA_QUALITY = "data_quality"
    UNCERTAINTY = "uncertainty"
    NEXT_UPDATE = "next_update"  # the document's own claim; never used for freshness
    PUBLICATION_DATE = "publication_date"  # document header, copied to every item


class BBox(StrictModel):
    """Normalized page coordinates in [0, 1], origin top-left, so they do not depend on resolution."""

    x0: float = Field(ge=0, le=1)
    y0: float = Field(ge=0, le=1)
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _ordered(self) -> BBox:
        if not (self.x0 < self.x1 and self.y0 < self.y1):
            raise ValueError("bbox must have x0 < x1 and y0 < y1")
        return self


class ExtractorVersion(StrictModel):
    engine: str  # e.g. "paddleocr", "qwen"
    engine_version: str
    model: str | None = None


class SecondReading(StrictModel):
    extractor: ExtractorVersion
    text: str | None
    normalized_value: str | None
    agrees: bool


class ValidationCheck(StrictModel):
    check: str  # e.g. "valid_range", "change_matches_baseline", "unit_matches_catalogue"
    passed: bool
    detail: str | None = None


class ReviewerAction(StrictModel):
    reviewer: str
    action: Literal["accept", "correct", "reject", "mark_unreadable"]
    at: AwareDatetime
    corrected_value: str | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _correction_has_value(self) -> ReviewerAction:
        if (self.action == "correct") != (self.corrected_value is not None):
            raise ValueError("corrected_value is required for, and only for, action 'correct'")
        return self


class FieldExtraction(StrictModel):
    """One printed field as read from the page (handoff §5: never keep only the normalized value)."""

    attribute: FieldAttribute
    presence: Presence
    page: int = Field(ge=1)
    bbox: BBox | None
    ocr_text: str | None
    normalized_value: str | None
    confidence: Confidence | None
    extractor: ExtractorVersion
    second_reading: SecondReading | None = None
    validations: list[ValidationCheck] = []
    reviewer_action: ReviewerAction | None = None

    @model_validator(mode="after")
    def _presence_consistent(self) -> FieldExtraction:
        if self.presence == Presence.PRESENT and (self.bbox is None or self.ocr_text is None):
            raise ValueError(f"{self.attribute}: a present field needs bbox and ocr_text")
        if self.presence == Presence.UNREADABLE and (self.bbox is None or self.normalized_value is not None):
            raise ValueError(f"{self.attribute}: an unreadable field has a bbox but no normalized_value")
        if self.presence == Presence.ABSENT and (self.bbox, self.ocr_text, self.normalized_value) != (None,) * 3:
            raise ValueError(f"{self.attribute}: an absent field has no bbox, ocr_text or normalized_value")
        return self


class OcrProvenance(StrictModel):
    kind: Literal["ocr"] = "ocr"
    document_id: str
    document_hash: Sha256  # hash of the original file as received
    page_hashes: dict[int, Sha256]
    fields: dict[FieldAttribute, FieldExtraction]

    @model_validator(mode="after")
    def _keys_match(self) -> OcrProvenance:
        for attribute, extraction in self.fields.items():
            if extraction.attribute != attribute:
                raise ValueError(f"fields[{attribute}] holds an extraction of {extraction.attribute}")
            if extraction.page not in self.page_hashes:
                raise ValueError(f"{attribute}: page {extraction.page} has no page hash")
        if FieldAttribute.VALUE not in self.fields:
            raise ValueError("an OCR evidence item needs a 'value' field extraction, even if absent")
        return self


class SystemProvenance(StrictModel):
    """Evidence exported from an internal system (AVM output, authorized customer records)."""

    kind: Literal["system"] = "system"
    system: str  # e.g. "avm.phase3"
    record_id: str
    export_hash: Sha256
    exported_at: AwareDatetime


Provenance = Annotated[OcrProvenance | SystemProvenance, Field(discriminator="kind")]


class Period(StrictModel):
    label: str  # catalogue period format: YYYY-MM-DD, YYYY-MM or YYYY-Qn
    start: AwareDatetime
    end: AwareDatetime

    @model_validator(mode="after")
    def _ordered(self) -> Period:
        if self.start > self.end:
            raise ValueError("period start is after its end")
        return self


class Baseline(StrictModel):
    value: Decimal
    unit: Unit
    period_label: str


class Uncertainty(StrictModel):
    kind: Literal["interval", "stderr"]
    lower: Decimal | None = None
    upper: Decimal | None = None
    stderr: Decimal | None = Field(default=None, ge=0)
    level: Decimal | None = Field(default=None, gt=0, lt=1)  # e.g. 0.9 for a 90% interval

    @model_validator(mode="after")
    def _shape(self) -> Uncertainty:
        if self.kind == "interval":
            if self.lower is None or self.upper is None or self.level is None or self.lower > self.upper:
                raise ValueError("an interval needs lower <= upper and a level")
        elif self.stderr is None:
            raise ValueError("kind 'stderr' needs stderr")
        return self


class Freshness(StrictModel):
    """Computed by code from the catalogue; a document cannot declare its own expiry."""

    deadline: AwareDatetime
    catalogue_version: str
    catalogue_hash: Sha256


class EvidenceItem(StrictModel):
    schema_version: Literal["evidence_item/0.1"] = "evidence_item/0.1"
    evidence_id: str
    metric_id: str
    geography: str
    segment: str | None = None

    source: str
    source_document_id: str | None = None
    observation_period: Period
    publication_date: AwareDatetime | None
    retrieved_at: AwareDatetime

    value: Decimal | None
    unit: Unit
    transformation: Transformation
    baseline: Baseline | None = None

    revision_status: Literal["preliminary", "revised", "final"] | None = None
    revision_of: str | None = None
    data_quality: Literal["ok", "low", "unknown"]
    quality_flags: list[str] = []
    uncertainty: Uncertainty | None = None

    freshness: Freshness
    presence: Presence
    trust: Literal["trusted", "untrusted"]
    trust_reasons: list[str] = []

    synthetic: bool
    caveats: list[str] = []
    provenance: Provenance

    @model_validator(mode="after")
    def _consistent(self) -> EvidenceItem:
        if self.evidence_id != f"{self.metric_id}.{self.observation_period.label}":
            raise ValueError("evidence_id must be <metric_id>.<observation_period.label>")
        parts = self.metric_id.split(".")
        if len(parts) != 3 or parts[1] != self.geography:
            raise ValueError("metric_id must be <scope>.<geography>.<key> with the item's geography")
        if (self.presence == Presence.PRESENT) != (self.value is not None):
            raise ValueError("value is set if and only if presence is 'present'")
        if self.presence != Presence.PRESENT and self.trust == "trusted":
            raise ValueError("an absent or unreadable value cannot be trusted")
        if (self.trust == "untrusted") != bool(self.trust_reasons):
            raise ValueError("trust_reasons are required for, and only for, untrusted items")
        if (self.revision_of is not None) and self.revision_status != "revised":
            raise ValueError("revision_of is only allowed on revised items")
        if self.publication_date is not None and self.publication_date > self.retrieved_at:
            raise ValueError("publication_date is after retrieved_at")
        return self

    @property
    def item_hash(self) -> str:
        return content_hash(self)


def catalogue_violations(item: EvidenceItem, catalogue: Catalogue) -> list[str]:
    """Checks that need the catalogue. A violation makes the item untrusted; it is not a schema error,
    because an extraction can be wrong without the record being malformed."""
    try:
        metric = catalogue.get(item.metric_id).definition
    except KeyError:
        return [f"UNKNOWN_METRIC:{item.metric_id}"]
    out = []
    try:
        catalogue.evidence_id(item.metric_id, item.observation_period.label)
    except ValueError:
        out.append(f"PERIOD_FORMAT:{item.observation_period.label}")
    if item.unit != metric.unit:
        out.append(f"UNIT_MISMATCH:{item.unit.value}!={metric.unit.value}")
    if item.transformation != metric.transformation:
        out.append(f"TRANSFORMATION_MISMATCH:{item.transformation.value}!={metric.transformation.value}")
    low, high = metric.valid_range
    if item.value is not None and not (low <= item.value <= high):
        out.append(f"OUT_OF_RANGE:{item.value}")
    expected = catalogue.freshness_deadline(item.metric_id, item.observation_period.end)
    if (item.freshness.deadline, item.freshness.catalogue_hash) != (expected, catalogue.catalogue_hash):
        out.append("FRESHNESS_NOT_FROM_CATALOGUE")
    return out
