"""VerifierResult: the outcome of a rules check on a recommendation or a mapping proposal."""
from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import AwareDatetime, model_validator

from .common import Sha256, StrictModel
from .evidence import ValidationCheck


class VerifierCode(StrEnum):
    # BLOCK: the output cannot be used
    SCHEMA_INVALID = "SCHEMA_INVALID"
    NO_CITATIONS = "NO_CITATIONS"
    UNKNOWN_EVIDENCE = "UNKNOWN_EVIDENCE"
    STALE_EVIDENCE = "STALE_EVIDENCE"
    UNTRUSTED_EVIDENCE = "UNTRUSTED_EVIDENCE"
    UNIT_MISMATCH = "UNIT_MISMATCH"
    UNSUPPORTED_NUMBER = "UNSUPPORTED_NUMBER"
    CONFLICTING_SIGNALS_MUST_ABSTAIN = "CONFLICTING_SIGNALS_MUST_ABSTAIN"
    ACTION_NOT_ALLOWED = "ACTION_NOT_ALLOWED"
    MAPPING_MISMATCH = "MAPPING_MISMATCH"
    BUNDLE_SIGNATURE_INVALID = "BUNDLE_SIGNATURE_INVALID"
    BUNDLE_EXPIRED = "BUNDLE_EXPIRED"
    QUOTE_NOT_VERBATIM = "QUOTE_NOT_VERBATIM"
    # ROUTE_TO_HUMAN: usable only after review
    MAGNITUDE_ABOVE_CAP = "MAGNITUDE_ABOVE_CAP"
    HUMAN_REVIEW_REQUESTED = "HUMAN_REVIEW_REQUESTED"
    ABSTAINED = "ABSTAINED"


ROUTE_CODES = frozenset(
    {VerifierCode.MAGNITUDE_ABOVE_CAP, VerifierCode.HUMAN_REVIEW_REQUESTED, VerifierCode.ABSTAINED}
)


class Finding(StrictModel):
    code: VerifierCode
    detail: str | None = None


class VerifierResult(StrictModel):
    schema_version: Literal["verifier_result/0.1"] = "verifier_result/0.1"
    decision: Literal["PASS", "ROUTE_TO_HUMAN", "BLOCK"]
    findings: list[Finding] = []
    checks: list[ValidationCheck] = []
    verifier_version: str
    subject: Literal["recommendation", "mapping_proposal"]
    subject_hash: Sha256
    checked_at: AwareDatetime

    @model_validator(mode="after")
    def _decision_matches_findings(self) -> VerifierResult:
        codes = {f.code for f in self.findings}
        if codes - ROUTE_CODES:
            expected = "BLOCK"
        elif codes:
            expected = "ROUTE_TO_HUMAN"
        else:
            expected = "PASS"
        if self.decision != expected:
            raise ValueError(f"findings {sorted(codes)} require decision {expected}, not {self.decision}")
        return self
