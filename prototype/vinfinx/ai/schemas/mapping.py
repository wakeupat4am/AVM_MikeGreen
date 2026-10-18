"""Offline research contracts: quotes from the corpus, mapping proposals, their verification, mapping sets."""
from __future__ import annotations

import hashlib
from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from .common import GenerationMeta, Sha256, StrictModel, Target, canonical_json, content_hash
from .evidence import ValidationCheck

GEO_ID = r"^[a-z]+(?:-[a-z0-9]+)*$"
METRIC_KEY = r"^[a-z][a-z0-9_]*$"


class CorpusChunk(StrictModel):
    """One retrieved chunk of a frozen corpus document: what the mapping agent saw and may quote."""

    chunk_id: str
    source_id: str
    document_hash: Sha256
    page: int = Field(ge=1)
    text: str = Field(min_length=1)
    publisher: str
    publication_date: date
    geography: str = Field(pattern=GEO_ID)
    language: Literal["vi", "en"]
    corpus_version: str
    synthetic: bool = False


class RelationshipEvidence(StrictModel):
    """One verbatim quote from a frozen research-corpus chunk."""

    schema_version: Literal["relationship_evidence/0.1"] = "relationship_evidence/0.1"
    evidence_id: str
    source_id: str
    document_hash: Sha256
    chunk_id: str
    page: int = Field(ge=1)
    char_start: int = Field(ge=0)  # offsets into the chunk text
    char_end: int
    quote: str = Field(min_length=1)
    publisher: str
    publication_date: date
    geography: str = Field(pattern=GEO_ID)  # e.g. "vn", or another country the source is about
    language: Literal["vi", "en"]
    corpus_version: str

    @model_validator(mode="after")
    def _span(self) -> RelationshipEvidence:
        if self.char_end - self.char_start != len(self.quote):
            raise ValueError("quote length must equal char_end - char_start")
        return self


class Direction(StrEnum):
    POSITIVE = "positive"  # the target moves with the metric
    NEGATIVE = "negative"  # the target moves against the metric (a rate cut raising borrowing capacity)


class SizeRange(StrEnum):
    QUALITATIVE = "qualitative"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class QuoteRef(StrictModel):
    chunk_id: str
    quote: str = Field(min_length=1)


class MappingProposal(StrictModel):
    """What the mapping agent (Qwen) generates. Keyed by catalogue metric key, not by district."""

    metric_key: str = Field(pattern=METRIC_KEY)
    target: Target
    direction: Direction
    lag_months: tuple[int, int]
    size_range: SizeRange
    geography: str = Field(pattern=GEO_ID)  # where the evidence applies, e.g. "vn"
    evidence: list[QuoteRef] = Field(min_length=1)

    @model_validator(mode="after")
    def _lag(self) -> MappingProposal:
        low, high = self.lag_months
        if not (0 <= low <= high <= 60):
            raise ValueError("lag_months must be [low, high] with 0 <= low <= high <= 60")
        return self


def proposal_generation_schema(metric_keys: list[str], chunk_ids: list[str], geographies: list[str]) -> dict:
    """JSON Schema for one mapping-agent request: it can only cite chunks that were retrieved for it.
    Whether each quote is verbatim is checked afterwards by the mapping verifier."""
    return {
        "type": "object",
        "properties": {
            "metric_key": {"enum": metric_keys},
            "target": {"enum": [t.value for t in Target]},
            "direction": {"enum": [d.value for d in Direction]},
            "lag_months": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0, "maximum": 60},
                "minItems": 2,
                "maxItems": 2,
            },
            "size_range": {"enum": [s.value for s in SizeRange]},
            "geography": {"enum": geographies},
            "evidence": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {"chunk_id": {"enum": chunk_ids}, "quote": {"type": "string"}},
                    "required": ["chunk_id", "quote"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["metric_key", "target", "direction", "lag_months", "size_range", "geography", "evidence"],
        "additionalProperties": False,
    }


class ProposalRecord(StrictModel):
    schema_version: Literal["mapping_proposal_record/0.1"] = "mapping_proposal_record/0.1"
    proposal: MappingProposal
    meta: GenerationMeta
    corpus_version: str
    index_version: str


class SupportCheck(StrictModel):
    """Does the quote back the stated direction? A different model from the mapping agent (handoff §6)."""

    model: str
    verdict: Literal["supports", "contradicts", "insufficient"]
    rationale: str
    checked_at: AwareDatetime


class MappingVerification(StrictModel):
    proposal_hash: Sha256
    rule_checks: list[ValidationCheck] = Field(min_length=1)
    support_check: SupportCheck | None = None  # placeholder until the support check is built
    status: Literal["accepted", "rejected"]
    reasons: list[str] = []

    @model_validator(mode="after")
    def _status(self) -> MappingVerification:
        failed = [c.check for c in self.rule_checks if not c.passed]
        support_ok = self.support_check is None or self.support_check.verdict == "supports"
        if self.status == "accepted" and (failed or not support_ok):
            raise ValueError("a proposal with a failed rule check or unsupported quote cannot be accepted")
        if self.status == "rejected" and not self.reasons:
            raise ValueError("a rejected proposal needs reasons")
        return self


class MappingEntry(StrictModel):
    proposal: MappingProposal
    verification: MappingVerification

    @model_validator(mode="after")
    def _matches(self) -> MappingEntry:
        if self.verification.proposal_hash != content_hash(self.proposal):
            raise ValueError("verification does not belong to this proposal")
        if self.verification.status != "accepted":
            raise ValueError("only accepted proposals enter a mapping set")
        return self


class MappingSet(StrictModel):
    """A registry entry. mapping_id is derived from the content, so a changed mapping gets a new ID;
    validating or rejecting it later changes the status, not the ID."""

    schema_version: Literal["mapping_set/0.1"] = "mapping_set/0.1"
    mapping_id: str = Field(pattern=r"^mapping\.v\d+\.[0-9a-f]{12}$")
    version: int = Field(ge=1)
    entries: list[MappingEntry] = Field(min_length=1)
    corpus_version: str
    index_version: str
    created_at: AwareDatetime
    status: Literal["unvalidated", "validated", "rejected"]
    reviewed_by: str | None = None
    reviewed_at: AwareDatetime | None = None

    @staticmethod
    def compute_id(version: int, entries: list[MappingEntry], corpus_version: str, index_version: str) -> str:
        payload = canonical_json(
            {
                "version": version,
                "entries": [e.model_dump(mode="json") for e in entries],
                "corpus_version": corpus_version,
                "index_version": index_version,
            }
        )
        return f"mapping.v{version}.{hashlib.sha256(payload).hexdigest()[:12]}"

    @model_validator(mode="after")
    def _consistent(self) -> MappingSet:
        if self.mapping_id != self.compute_id(self.version, self.entries, self.corpus_version, self.index_version):
            raise ValueError("mapping_id does not match the content")
        keys = [(e.proposal.metric_key, e.proposal.target, e.proposal.geography) for e in self.entries]
        if len(set(keys)) != len(keys):
            raise ValueError("one entry per (metric_key, target, geography)")
        if (self.status != "unvalidated") != (self.reviewed_by is not None and self.reviewed_at is not None):
            raise ValueError("reviewed_by and reviewed_at are required once validated or rejected, and only then")
        return self

    def lookup(self, metric_key: str, target: Target, geography: str) -> MappingEntry | None:
        for e in self.entries:
            if (e.proposal.metric_key, e.proposal.target, e.proposal.geography) == (metric_key, target, geography):
                return e
        return None
