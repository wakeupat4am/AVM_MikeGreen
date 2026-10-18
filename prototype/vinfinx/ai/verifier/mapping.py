"""Mapping verifier v0: rule checks on a mapping agent's proposal against the chunks it was given.

Rules confirm that each quote exists word for word, comes from a retrieved chunk about the right geography,
and is not too old. Whether the quote actually supports the stated direction is a judgment: that is the
support check (a different model, handoff §6), passed in by the caller. Without it, a proposal can pass the
rules with the direction reversed, as the 2026-10-18 live check showed.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime

from pydantic import ValidationError

from ..catalog import Catalogue
from ..schemas.common import content_hash
from ..schemas.evidence import ValidationCheck
from ..schemas.mapping import CorpusChunk, MappingProposal, MappingVerification, RelationshipEvidence, SupportCheck
from ..schemas.verifier import ROUTE_CODES, Finding, VerifierCode, VerifierResult
from . import VERIFIER_VERSION

C = VerifierCode
MIN_QUOTE_CHARS = 20  # provisional: shorter quotes ("rates fell") cannot carry a relationship
MAX_SOURCE_AGE_YEARS = 10  # provisional: owned by whoever signs off the source criteria


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _locate(quote: str, text: str) -> tuple[int, int] | None:
    """Character span of the quote in the chunk text. Only runs of whitespace may differ (PDF line breaks)."""
    start = text.find(quote)
    if start >= 0:
        return start, start + len(quote)
    pattern = r"\s+".join(re.escape(w) for w in _squash(quote).split(" "))
    m = re.search(pattern, text)
    return (m.start(), m.end()) if m else None


def verify_proposal(
    raw_output: str,
    chunks: list[CorpusChunk],
    *,
    now: datetime,
    catalogue: Catalogue,
    support_check: SupportCheck | None = None,
) -> tuple[VerifierResult, MappingProposal | None, list[RelationshipEvidence]]:
    """Returns the verifier result, the parsed proposal (if it parsed) and one RelationshipEvidence per
    verified quote, ready for the mapping registry."""
    findings: list[Finding] = []
    checks: list[ValidationCheck] = []

    def check(name: str, problems: list[tuple[VerifierCode, str]], skipped: bool = False) -> None:
        detail = "; ".join(d for _, d in problems) or ("skipped: no support check available" if skipped else None)
        checks.append(ValidationCheck(check=name, passed=not problems, detail=detail))
        findings.extend(Finding(code=code, detail=d) for code, d in problems)

    def result() -> VerifierResult:
        codes = {f.code for f in findings}
        return VerifierResult(
            decision="BLOCK" if codes - ROUTE_CODES else ("ROUTE_TO_HUMAN" if codes else "PASS"),
            findings=findings,
            checks=checks,
            verifier_version=VERIFIER_VERSION,
            subject="mapping_proposal",
            subject_hash="sha256:" + hashlib.sha256(raw_output.encode("utf-8")).hexdigest(),
            checked_at=now,
        )

    try:
        proposal = MappingProposal.model_validate_json(raw_output)
    except ValidationError as exc:
        check("schema", [(C.SCHEMA_INVALID, f"{exc.error_count()} error(s): {exc.errors()[0]['msg']}")])
        return result(), None, []
    check("schema", [])

    keys = {m.key for m in catalogue.metrics}
    check("metric_known", [] if proposal.metric_key in keys else [(C.UNKNOWN_METRIC, proposal.metric_key)])

    by_id = {c.chunk_id: c for c in chunks}
    quote_problems, geo_problems, age_problems = [], [], []
    verified: list[RelationshipEvidence] = []
    for i, ref in enumerate(proposal.evidence):
        chunk = by_id.get(ref.chunk_id)
        if chunk is None:
            quote_problems.append((C.UNKNOWN_EVIDENCE, f"{ref.chunk_id}: not among the retrieved chunks"))
            continue
        if len(_squash(ref.quote)) < MIN_QUOTE_CHARS:
            quote_problems.append((C.QUOTE_TOO_SHORT, f"{ref.chunk_id}: {len(_squash(ref.quote))} characters"))
            continue
        span = _locate(ref.quote, chunk.text)
        if span is None:
            quote_problems.append((C.QUOTE_NOT_VERBATIM, f"{ref.chunk_id}: quote not found in the chunk"))
            continue
        if chunk.geography != proposal.geography:
            geo_problems.append((C.GEOGRAPHY_MISMATCH, f"{ref.chunk_id} is about {chunk.geography}, proposal is for {proposal.geography}"))
        age = now.date().year - chunk.publication_date.year
        if age > MAX_SOURCE_AGE_YEARS:
            age_problems.append((C.STALE_EVIDENCE, f"{ref.chunk_id}: published {chunk.publication_date}"))
        start, end = span
        verified.append(RelationshipEvidence(
            evidence_id=f"rel.{chunk.chunk_id}.{start}-{end}",
            source_id=chunk.source_id,
            document_hash=chunk.document_hash,
            chunk_id=chunk.chunk_id,
            page=chunk.page,
            char_start=start,
            char_end=end,
            quote=chunk.text[start:end],  # store the chunk's own text, not the model's copy
            publisher=chunk.publisher,
            publication_date=chunk.publication_date,
            geography=chunk.geography,
            language=chunk.language,
            corpus_version=chunk.corpus_version,
        ))
    check("quotes_verbatim", quote_problems)
    check("geography", geo_problems)
    check("source_age", age_problems)

    if support_check is None:
        check("support", [], skipped=True)
    else:
        check("support", [] if support_check.verdict == "supports"
              else [(C.SUPPORT_NOT_ESTABLISHED, f"{support_check.model}: {support_check.verdict}")])
    return result(), proposal, verified


def find_conflicts(proposals: list[MappingProposal]) -> list[Finding]:
    """Accepted proposals for the same (metric, target, geography) that point in opposite directions."""
    seen: dict[tuple, set[str]] = {}
    for p in proposals:
        seen.setdefault((p.metric_key, p.target.value, p.geography), set()).add(p.direction.value)
    return [
        Finding(code=C.CONFLICTING_SOURCES, detail=f"{key[0]} -> {key[1]} ({key[2]}): both directions")
        for key, directions in seen.items()
        if len(directions) > 1
    ]


def to_verification(
    result: VerifierResult, proposal: MappingProposal, support_check: SupportCheck | None = None
) -> MappingVerification:
    """The registry record of this verification. Only a PASS is accepted into a mapping set."""
    accepted = result.decision == "PASS"
    return MappingVerification(
        proposal_hash=content_hash(proposal),
        rule_checks=result.checks,
        support_check=support_check,
        status="accepted" if accepted else "rejected",
        reasons=[] if accepted else [f.code.value for f in result.findings],
    )
