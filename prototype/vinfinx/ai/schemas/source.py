"""Research-corpus source records and source lists (docs/CORPUS_SOURCE_CRITERIA.md, docs/SOURCE_AGENT_PLAN.md)."""
from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from .common import Sha256, StrictModel, Target, content_hash


class PublisherType(StrEnum):
    """C1 accepted publisher types."""

    CENTRAL_BANK = "central_bank"
    STATISTICS_OFFICE = "statistics_office"
    INTERNATIONAL_ORGANISATION = "international_organisation"  # IMF, World Bank, ADB, BIS, OECD
    PEER_REVIEWED_JOURNAL = "peer_reviewed_journal"
    WORKING_PAPER = "working_paper"  # a university's or one of the institutions above


class Criterion(StrEnum):
    C1_PUBLISHER = "C1"
    C2_LICENCE = "C2"
    C3_LANGUAGE = "C3"
    C4_RELEVANCE = "C4"
    C5_GEOGRAPHY = "C5"
    C6_RETRIEVABLE = "C6"
    C7_DATE = "C7"


class Outcome(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNCLEAR = "unclear"  # a person decides


class Check(StrictModel):
    criterion: Criterion
    outcome: Outcome
    detail: str
    by: Literal["code", "source_agent"]


class Geography(StrEnum):
    VN = "vn"
    CROSS_COUNTRY_INCL_VN = "cross_country_incl_vn"
    COMPARABLE_EMERGING = "comparable_emerging"
    OTHER = "other"


ACCEPTED_GEOGRAPHIES = frozenset({Geography.VN, Geography.CROSS_COUNTRY_INCL_VN, Geography.COMPARABLE_EMERGING})


class VerifiedQuote(StrictModel):
    """A quote the agent gave, found word for word (whitespace aside) in the text it was shown."""

    quote: str = Field(min_length=20)
    char_start: int = Field(ge=0)  # offsets into the screened text (see Screening.text_hash)
    char_end: int

    @model_validator(mode="after")
    def _span(self) -> VerifiedQuote:
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be after char_start")
        return self


class Screening(StrictModel):
    """The source agent's judgement of C4 and C5, as checked by code. The model never writes provenance."""

    relevant: Literal["yes", "no", "unclear"]
    metrics: list[str]
    targets: list[Target]
    geography: Geography
    qualifies_because: str
    quotes: list[VerifiedQuote]
    unverified_quotes: list[str]  # given by the model but not found in the text: never evidence
    text_hash: Sha256  # the exact text shown to the model
    text_chars: int
    model: str
    prompt_version: str
    trace_id: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    seconds: float | None = None
    raw_output_hash: Sha256


class ProposedBy(StrictModel):
    agent: Literal["source_agent"] = "source_agent"
    agent_version: str
    model: str
    prompt_version: str
    run_id: str
    query_file_hash: Sha256
    publisher_file_hash: Sha256


class FoundBy(StrictModel):
    adapter: Literal["openalex", "worldbank", "bis", "sbv", "nso"]
    query_id: str
    raw_sha256: Sha256  # the stored API response or page the candidate came from


class Decision(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNCLEAR = "unclear"
    BAD_DOCUMENT = "bad_document"


class Review(StrictModel):
    decision: Literal["accept", "reject"]
    note: str | None = None
    reviewer: str
    reviewed_at: AwareDatetime


class SourceRecord(StrictModel):
    schema_version: Literal["source_record/0.1"] = "source_record/0.1"
    source_id: str = Field(pattern=r"^src-\d{4}$")
    title: str
    publisher: str | None
    publisher_type: PublisherType | None
    country: str | None
    language: Literal["vi", "en"] | None
    publication_date: date | None
    url: str  # landing page or DOI link
    doi: str | None
    pdf_url: str | None
    licence: str | None
    licence_url: str | None
    retrieved_at: AwareDatetime | None
    retrieved_by: Literal["code", "person"] | None
    sha256: Sha256 | None  # of the stored original file
    found_by: list[FoundBy] = Field(min_length=1)
    checks: list[Check]
    screening: Screening | None
    qualifies_because: str | None
    proposed_by: ProposedBy
    decision: Decision
    failed_criterion: Criterion | None
    decision_reason: str
    review: Review | None = None

    @model_validator(mode="after")
    def _consistent(self) -> SourceRecord:
        if self.decision is Decision.REJECTED and self.failed_criterion is None:
            raise ValueError("a rejected source must name the failed criterion")
        if self.decision is Decision.ACCEPTED:
            missing = [c for c in Criterion if not any(k.criterion is c and k.outcome is Outcome.PASS
                                                         for k in self.checks)]
            if missing:
                raise ValueError(f"an accepted source must pass every criterion; not passed: {missing}")
            if self.screening is None or not self.screening.quotes:
                raise ValueError("an accepted source needs at least one verified quote")
            if self.sha256 is None:
                raise ValueError("an accepted source must have a stored, hashed original")
        return self

    @property
    def signed_decision(self) -> str | None:
        return self.review.decision if self.review else None


class SourceList(StrictModel):
    schema_version: Literal["source_list/0.1"] = "source_list/0.1"
    version: str  # e.g. "source_list/v0.1" when signed off, "<run_id>-draft" before
    status: Literal["draft", "signed_off"]
    run_id: str
    created_at: AwareDatetime
    records: list[SourceRecord]
    signed_off_by: str | None = None
    signed_off_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def _signed(self) -> SourceList:
        ids = [r.source_id for r in self.records]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate source_id")
        if self.status == "signed_off":
            if not (self.signed_off_by and self.signed_off_at):
                raise ValueError("a signed-off list needs signed_off_by and signed_off_at")
            unreviewed = [r.source_id for r in self.records if r.review is None]
            if unreviewed:
                raise ValueError(f"every record must be reviewed before sign-off: {unreviewed[:5]}")
        return self

    @property
    def list_hash(self) -> str:
        return content_hash(self)

    def accepted(self) -> list[SourceRecord]:
        """What ingestion may use: the reviewer's decision, never the agent's alone."""
        return [r for r in self.records if r.review is not None and r.review.decision == "accept"]
