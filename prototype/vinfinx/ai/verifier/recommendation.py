"""Recommendation verifier v0: checks the decision engine's raw output against its signed evidence bundle.

Fails closed: anything it cannot confirm blocks. Abstaining is always safe, so an abstention routes to a
human instead of being pushed to bluff. Checks form, not reasoning: a causal story that uses only cited
numbers passes here, and the adversarial set measures how often that happens.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime
from decimal import Decimal

from pydantic import ValidationError

from ..catalog import Catalogue
from ..schemas.bundle import EvidenceBundle, verify_bundle
from ..schemas.common import Action
from ..schemas.evidence import EvidenceItem, ValidationCheck
from ..schemas.recommendation import Recommendation
from ..schemas.verifier import ROUTE_CODES, Finding, VerifierCode, VerifierResult
from . import VERIFIER_VERSION
from .numbers import find_numbers, normalize

C = VerifierCode
UNIT_OF_HINT = {"pct": "pct", "bps": "bps", "pp": "pp"}


class _Collector:
    def __init__(self) -> None:
        self.findings: list[Finding] = []
        self.checks: list[ValidationCheck] = []

    def check(self, name: str, problems: list[tuple[VerifierCode, str]]) -> None:
        self.checks.append(ValidationCheck(check=name, passed=not problems,
                                           detail="; ".join(d for _, d in problems) or None))
        self.findings.extend(Finding(code=code, detail=detail) for code, detail in problems)

    def result(self, raw_output: str, now: datetime) -> VerifierResult:
        codes = {f.code for f in self.findings}
        decision = "BLOCK" if codes - ROUTE_CODES else ("ROUTE_TO_HUMAN" if codes else "PASS")
        return VerifierResult(
            decision=decision,
            findings=self.findings,
            checks=self.checks,
            verifier_version=VERIFIER_VERSION,
            subject="recommendation",
            subject_hash="sha256:" + hashlib.sha256(raw_output.encode("utf-8")).hexdigest(),
            checked_at=now,
        )


def _allowed_numbers(item: EvidenceItem) -> dict[Decimal, set[str]]:
    """Numbers an explanation may state about a cited item, with the unit each one is in."""
    unit = item.unit.value
    out: dict[Decimal, set[str]] = {}

    def add(value: Decimal, units: set[str]) -> None:
        out.setdefault(normalize(value), set()).update(units)

    if item.value is not None:
        add(item.value, {unit})
    if item.baseline is not None:
        add(item.baseline.value, {item.baseline.unit.value})
        if item.value is not None:
            change = item.value - item.baseline.value
            change_units = {"pp", "pct"} if unit == "pct" else {unit}
            add(change, change_units)
            add(abs(change), change_units)
    if item.uncertainty is not None:
        for v in (item.uncertainty.lower, item.uncertainty.upper, item.uncertainty.stderr):
            if v is not None:
                add(v, {unit})
    # dates of the observation and baseline periods ("September 2026", "2026-09", "Q3")
    labels = [item.observation_period.label] + ([item.baseline.period_label] if item.baseline else [])
    for label in labels:
        for part in re.findall(r"\d+", label):
            add(Decimal(part), {"date"})
    for dt in (item.observation_period.start, item.observation_period.end):
        add(Decimal(dt.day), {"date"})
    return out


def _check_numbers(explanation: str, cited: list[EvidenceItem], strip: list[str]) -> list[tuple[VerifierCode, str]]:
    text = explanation
    for s in sorted(strip, key=len, reverse=True):  # IDs contain digits ("...2026-09"); remove them first
        text = text.replace(s, " ")
    allowed: dict[Decimal, set[str]] = {}
    for item in cited:
        for value, units in _allowed_numbers(item).items():
            allowed.setdefault(value, set()).update(units)
    problems = []
    for mention in find_numbers(text):
        matches = [allowed[v] for v in mention.readings if v in allowed]
        if not matches:
            problems.append((C.UNSUPPORTED_NUMBER, mention.raw))
            continue
        if mention.unit_hint is not None:
            units = set().union(*matches)
            if UNIT_OF_HINT[mention.unit_hint] not in units:
                problems.append((C.UNIT_MISMATCH, f"{mention.raw} stated in {mention.unit_hint}, evidence in {sorted(units - {'date'})}"))
    return problems


def _check_reason_codes(rec: Recommendation, cited: list[EvidenceItem], catalogue: Catalogue) -> list[tuple[VerifierCode, str]]:
    known = set(catalogue.reason_codes())
    stems = {m.reason_code_stem: m.key for m in catalogue.metrics}
    cited_by_key: dict[str, list[EvidenceItem]] = {}
    for item in cited:
        cited_by_key.setdefault(item.metric_id.split(".")[2], []).append(item)
    problems = []
    for code in rec.reason_codes:
        if code not in known:
            problems.append((C.UNSUPPORTED_REASON_CODE, f"{code}: not in the catalogue"))
            continue
        stem, _, direction = code.rpartition("_")
        if direction not in ("RISE", "DROP") or stem not in stems:
            continue  # CONFLICTING_SIGNALS, STALE_EVIDENCE, INSUFFICIENT_EVIDENCE
        items = cited_by_key.get(stems[stem], [])
        if not items:
            problems.append((C.UNSUPPORTED_REASON_CODE, f"{code}: no cited evidence for {stems[stem]}"))
            continue
        for item in items:
            if item.value is not None and item.baseline is not None and item.value != item.baseline.value:
                actual = "RISE" if item.value > item.baseline.value else "DROP"
                if actual != direction:
                    problems.append((C.UNSUPPORTED_REASON_CODE,
                                     f"{code}: {item.evidence_id} went from {item.baseline.value} to {item.value}"))
    return problems


def verify_recommendation(
    raw_output: str, bundle: EvidenceBundle, *, now: datetime, keys: dict[str, bytes], catalogue: Catalogue
) -> VerifierResult:
    out = _Collector()

    # 1. the bundle itself: signed with a known key, unchanged, not expired
    out.check("bundle_signature", [] if verify_bundle(bundle, keys) else [(C.BUNDLE_SIGNATURE_INVALID, bundle.bundle_id)])
    out.check("bundle_not_expired", [(C.BUNDLE_EXPIRED, str(bundle.expires_at))] if bundle.is_expired(now) else [])

    # 2. the output parses into the contract (unknown actions fail here)
    try:
        rec = Recommendation.model_validate_json(raw_output)
    except ValidationError as exc:
        out.check("schema", [(C.SCHEMA_INVALID, f"{exc.error_count()} error(s): {exc.errors()[0]['msg']}")])
        return out.result(raw_output, now)
    out.check("schema", [])

    policy = bundle.policy
    items = {i.evidence_id: i for i in bundle.items}

    # 3. the mapping, the action and the impacts are the policy engine's, not the model's
    out.check("mapping_id", [] if rec.mapping_id == bundle.mapping_id
              else [(C.MAPPING_MISMATCH, f"{rec.mapping_id} != {bundle.mapping_id}")])
    out.check("action_allowed", [] if rec.recommended_action in policy.allowed_actions
              else [(C.ACTION_NOT_ALLOWED, rec.recommended_action.value)])
    targets = {t.target for t in policy.targets}
    out.check("impacts_in_policy", [(C.IMPACT_NOT_IN_POLICY, t.value) for t in rec.customer_impacts if t not in targets])
    must_abstain = [t.target.value for t in policy.targets if t.must_abstain]
    out.check("abstains_on_conflict", [(C.CONFLICTING_SIGNALS_MUST_ABSTAIN, ", ".join(must_abstain))]
              if must_abstain and rec.recommended_action != Action.ABSTAIN else [])

    # 4. citations: present, in the bundle, fresh, trusted
    problems = []
    if rec.recommended_action != Action.ABSTAIN and not rec.evidence_ids:
        problems.append((C.NO_CITATIONS, "no evidence_ids"))
    cited = []
    for eid in rec.evidence_ids:
        item = items.get(eid)
        if item is None:
            problems.append((C.UNKNOWN_EVIDENCE, eid))
        elif item.freshness.deadline < now:
            problems.append((C.STALE_EVIDENCE, f"{eid} expired {item.freshness.deadline:%Y-%m-%d}"))
        elif item.trust != "trusted":
            problems.append((C.UNTRUSTED_EVIDENCE, f"{eid}: {', '.join(item.trust_reasons)}"))
        else:
            cited.append(item)
    out.check("citations", problems)

    # 5. reason codes name metrics that were cited, in the direction the evidence shows
    out.check("reason_codes", _check_reason_codes(rec, cited, catalogue))

    # 6. every number in the explanation comes from cited, usable evidence, in the right unit
    out.check("explanation_numbers", _check_numbers(rec.explanation, cited, list(rec.evidence_ids) + [rec.mapping_id]))

    # 7. what needs a human
    route = []
    if rec.magnitude.rank > policy.magnitude_cap.rank:
        route.append((C.MAGNITUDE_ABOVE_CAP, f"{rec.magnitude.value} > {policy.magnitude_cap.value}"))
    if rec.requires_human_review:
        route.append((C.HUMAN_REVIEW_REQUESTED, "requested by the output"))
    if rec.recommended_action == Action.ABSTAIN:
        route.append((C.ABSTAINED, "abstained"))
    out.check("routing", route)
    return out.result(raw_output, now)
