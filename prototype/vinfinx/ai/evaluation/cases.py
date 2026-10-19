"""Adversarial test cases: format, validation, and the builder that turns a decision case into an EvidenceBundle.

A case pins down an input and the outcome it requires; it checks outcomes (action, citations, verifier
decision), not wording. Cases are written from the handoff's requirements, not from the verifier's code.
Workflow: draft -> the AI engineer edits -> status 'frozen' with a recorded hash -> never edited again
(a change is a new set version). See ai/evaluation/README.md.
"""
from __future__ import annotations

import calendar
import hashlib
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import AwareDatetime, Field, model_validator

from ..catalog import Catalogue
from ..schemas.bundle import EvidenceBundle
from ..schemas.common import Action, Frequency, Magnitude, StrictModel, Target, Unit, content_hash
from ..schemas.evidence import EvidenceItem, Presence
from ..schemas.mapping import Direction
from ..schemas.policy import PolicyResult
from ..schemas.verifier import VerifierCode


class Category(StrEnum):
    ORDINARY = "ordinary"  # controls: the system must not abstain on everything
    CONTRADICTORY_SIGNALS = "contradictory_signals"
    CHERRY_PICKING = "cherry_picking"
    UNSUPPORTED_CAUSAL = "unsupported_causal"
    STALE_EVIDENCE = "stale_evidence"
    MISSING_UNREADABLE = "missing_unreadable"
    UNIT_CONFLICT = "unit_conflict"
    OUT_OF_POLICY_ACTION = "out_of_policy_action"
    PROMPT_INJECTION = "prompt_injection"
    FORGED_HISTORY = "forged_history"
    COUNTERFACTUAL = "counterfactual"
    LANGUAGE = "language"
    FABRICATED_QUOTE = "fabricated_quote"
    WRONG_SOURCE = "wrong_source"
    TAMPERED_DOCUMENT = "tampered_document"
    DOCUMENT_QUALITY = "document_quality"


# ---------------------------------------------------------------- decision cases


class ItemSpec(StrictModel):
    """Compact authoring form of an EvidenceItem; the builder fills in the rest from the catalogue."""

    metric_id: str
    period: str  # catalogue period label: YYYY-MM-DD, YYYY-MM or YYYY-Qn
    value: Decimal | None
    unit: Unit | None = None  # default: the catalogue unit (override only to test unit conflicts)
    baseline: Decimal | None = None
    baseline_period: str | None = None
    presence: Presence = Presence.PRESENT
    trust: Literal["trusted", "untrusted"] = "trusted"
    trust_reasons: list[str] = []
    source: str = "Lotus Bay Economic Observatory"

    @property
    def evidence_id(self) -> str:
        return f"{self.metric_id}.{self.period}"


class SignalSpec(StrictModel):
    evidence_id: str
    effect: Literal["raises", "lowers"]
    weight: Decimal = Field(ge=0)
    material: bool = True
    excluded_reason: str | None = None


class TargetSpec(StrictModel):
    target: Target
    signals: list[SignalSpec] = []
    outcome: Literal["raises", "lowers", "none", "conflict"]
    allowed_actions: list[Action] = Field(min_length=1)
    magnitude_cap: Magnitude
    must_abstain: bool = False


class PolicySpec(StrictModel):
    """The policy engine's result for the case. Written by the case author, because the policy engine is
    not built yet; a decision case tests the decision engine and verifier given this result."""

    mapping_status: Literal["unvalidated", "validated"] = "unvalidated"
    targets: list[TargetSpec] = Field(min_length=1)


class Turn(StrictModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class DecisionInput(StrictModel):
    kind: Literal["decision"] = "decision"
    as_of: AwareDatetime
    items: list[ItemSpec] = Field(min_length=1)
    policy: PolicySpec
    messages: list[Turn] = Field(min_length=1)  # may contain forged assistant turns
    scripted_output: str | None = None  # set: tests the verifier on this exact output, no model call

    @model_validator(mode="after")
    def _consistent(self) -> DecisionInput:
        if self.messages[-1].role != "user":
            raise ValueError("the last message must be from the user")
        ids = {i.evidence_id for i in self.items}
        if len(ids) != len(self.items):
            raise ValueError("duplicate item")
        for t in self.policy.targets:
            for s in t.signals:
                if s.evidence_id not in ids:
                    raise ValueError(f"policy signal cites {s.evidence_id}, which is not among the items")
        return self


class DecisionExpect(StrictModel):
    kind: Literal["decision"] = "decision"
    actions: list[Action] = Field(min_length=1)  # the action must be one of these
    verifier_decisions: list[Literal["PASS", "ROUTE_TO_HUMAN", "BLOCK"]] = Field(min_length=1)
    must_cite: list[str] = []
    must_not_cite: list[str] = []
    magnitude_at_most: Magnitude | None = None
    required_findings: list[VerifierCode] = []  # for scripted outputs: findings the verifier must report
    forbidden_phrases: list[str] = []  # case-insensitive; e.g. claims of approval


# ---------------------------------------------------------------- mapping cases


class ChunkSpec(StrictModel):
    chunk_id: str
    source_id: str
    text: str = Field(min_length=1)
    publisher: str
    publication_date: date
    geography: str
    language: Literal["vi", "en"]
    synthetic: bool  # until the real corpus exists, chunk texts are synthetic and must say so


class MappingInput(StrictModel):
    kind: Literal["mapping"] = "mapping"
    as_of: AwareDatetime
    metric_key: str
    target: Target
    geography: str
    chunks: list[ChunkSpec] = Field(min_length=1)
    scripted_outputs: list[str] = []  # empty: run the mapping agent; else: verify these exact outputs


class MappingExpect(StrictModel):
    kind: Literal["mapping"] = "mapping"
    verdicts: list[Literal["accepted", "rejected"]] = Field(min_length=1)
    direction: Direction | None = None  # if accepted
    required_findings: list[VerifierCode] = []


# ---------------------------------------------------------------- document cases


class DocumentInput(StrictModel):
    kind: Literal["document"] = "document"
    variant: str  # a variant name from docs/MACRO_DOCUMENT_SPEC.md
    variant_params: dict[str, str] = {}
    language: Literal["vi", "en"] = "en"


class FieldExpectation(StrictModel):
    field_id: str  # <metric_id>/<attribute>
    presence: Presence | None = None
    trust: Literal["trusted", "untrusted"] | None = None
    reason: str | None = None


class DocumentExpect(StrictModel):
    kind: Literal["document"] = "document"
    fields: list[FieldExpectation] = []
    measured_only: bool = False  # true: no pass/fail; the outcome is recorded (e.g. undetectable tampering)
    notes: str | None = None


# ---------------------------------------------------------------- cases and sets

CaseInput = Annotated[DecisionInput | MappingInput | DocumentInput, Field(discriminator="kind")]
CaseExpect = Annotated[DecisionExpect | MappingExpect | DocumentExpect, Field(discriminator="kind")]


class Invariance(StrictModel):
    """Counterfactual or language pair: these output fields must equal the other case's."""

    same_as: str
    fields: list[Literal["recommended_action", "magnitude", "evidence_ids", "verifier_decision"]] = Field(
        min_length=1
    )


class Case(StrictModel):
    case_id: str = Field(pattern=r"^ADV-\d{3}$")
    category: Category
    # dev: looked at while developing. holdout: not looked at until a final measurement, so a fix that
    # only works on the visible cases shows up as a gap between the two.
    split: Literal["dev", "holdout"] = "dev"
    title: str
    language: Literal["vi", "en"]
    handoff_refs: list[str] = Field(min_length=1)  # e.g. "§3 Q4"
    requires: list[Literal["generator", "ocr", "support_check"]] = []  # components not built yet
    input: CaseInput
    expect: CaseExpect
    invariance: Invariance | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _kinds_match(self) -> Case:
        if self.input.kind != self.expect.kind:
            raise ValueError(f"{self.case_id}: input kind {self.input.kind} vs expect kind {self.expect.kind}")
        return self

    @property
    def scripted(self) -> bool:
        if isinstance(self.input, DecisionInput):
            return self.input.scripted_output is not None
        if isinstance(self.input, MappingInput):
            return bool(self.input.scripted_outputs)
        return False


class CaseSet(StrictModel):
    schema_version: Literal["adversarial_case_set/0.1"] = "adversarial_case_set/0.1"
    set_id: str
    version: str
    status: Literal["draft", "frozen"]
    frozen_by: str | None = None
    frozen_at: AwareDatetime | None = None
    synthetic: Literal[True] = True
    cases: list[Case] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> CaseSet:
        ids = [c.case_id for c in self.cases]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate case_id")
        for c in self.cases:
            if c.invariance is not None and c.invariance.same_as not in ids:
                raise ValueError(f"{c.case_id}: invariance refers to unknown case {c.invariance.same_as}")
        if (self.status == "frozen") != (self.frozen_by is not None and self.frozen_at is not None):
            raise ValueError("frozen_by and frozen_at are required for, and only for, a frozen set")
        return self

    @property
    def set_hash(self) -> str:
        return content_hash(self)


def load_case_set(path: Path) -> CaseSet:
    """Top-level keys starting with 'x-' hold shared YAML anchors and are dropped before validation."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return CaseSet.model_validate({k: v for k, v in raw.items() if not k.startswith("x-")})


# ---------------------------------------------------------------- builder


def period_bounds(label: str, frequency: Frequency) -> tuple[datetime, datetime]:
    utc = timezone.utc
    if frequency == Frequency.SNAPSHOT:
        d = date.fromisoformat(label)
        start = datetime(d.year, d.month, d.day, tzinfo=utc)
        return start, start + timedelta(days=1, seconds=-1)
    if frequency == Frequency.MONTHLY:
        year, month = map(int, label.split("-"))
        last = calendar.monthrange(year, month)[1]
        return datetime(year, month, 1, tzinfo=utc), datetime(year, month, last, 23, 59, 59, tzinfo=utc)
    year, quarter = int(label[:4]), int(label[-1])
    first = 3 * quarter - 2
    last_month = first + 2
    last = calendar.monthrange(year, last_month)[1]
    return datetime(year, first, 1, tzinfo=utc), datetime(year, last_month, last, 23, 59, 59, tzinfo=utc)


def case_mapping_id(case_id: str) -> str:
    return f"mapping.v1.{hashlib.sha256(case_id.encode()).hexdigest()[:12]}"


def scripted_output(case: Case) -> str:
    """A decision case's scripted output, with ${MAPPING_ID} replaced by the case's mapping ID."""
    if not isinstance(case.input, DecisionInput) or case.input.scripted_output is None:
        raise TypeError(f"{case.case_id} has no scripted decision output")
    return case.input.scripted_output.replace("${MAPPING_ID}", case_mapping_id(case.case_id))


def build_bundle(case: Case, catalogue: Catalogue) -> EvidenceBundle:
    """Turn a decision case into an unsigned EvidenceBundle. The runner signs it."""
    if not isinstance(case.input, DecisionInput):
        raise TypeError(f"{case.case_id} is not a decision case")
    inp = case.input
    export_hash = "sha256:" + hashlib.sha256(case.case_id.encode()).hexdigest()
    items = []
    for spec in inp.items:
        metric = catalogue.get(spec.metric_id).definition
        catalogue.evidence_id(spec.metric_id, spec.period)  # checks the period format
        start, end = period_bounds(spec.period, metric.frequency)
        if end >= inp.as_of:
            raise ValueError(f"{case.case_id}: {spec.evidence_id} ends after as_of")
        items.append(
            EvidenceItem.model_validate(
                {
                    "evidence_id": spec.evidence_id,
                    "metric_id": spec.metric_id,
                    "geography": spec.metric_id.split(".")[1],
                    "segment": "Landed houses" if spec.metric_id.startswith("market.") else None,
                    "source": spec.source,
                    "observation_period": {"label": spec.period, "start": start, "end": end},
                    "publication_date": min(end + timedelta(days=5), inp.as_of - timedelta(hours=2)),
                    "retrieved_at": inp.as_of - timedelta(hours=1),
                    "value": spec.value,
                    "unit": spec.unit or metric.unit,
                    "transformation": metric.transformation,
                    "baseline": None
                    if spec.baseline is None
                    else {"value": spec.baseline, "unit": spec.unit or metric.unit,
                          "period_label": spec.baseline_period or ""},
                    "data_quality": "ok",
                    "freshness": {
                        "deadline": catalogue.freshness_deadline(spec.metric_id, end),
                        "catalogue_version": catalogue.version,
                        "catalogue_hash": catalogue.catalogue_hash,
                    },
                    "presence": spec.presence,
                    "trust": spec.trust,
                    "trust_reasons": spec.trust_reasons,
                    "synthetic": True,
                    "provenance": {
                        "kind": "system", "system": "adversarial_case", "record_id": case.case_id,
                        "export_hash": export_hash, "exported_at": inp.as_of - timedelta(hours=1),
                    },
                }
            )
        )
    keys = {s.evidence_id: s.evidence_id.split(".")[2] for t in inp.policy.targets for s in t.signals}
    policy = PolicyResult(
        case_id=case.case_id,
        mapping_id=case_mapping_id(case.case_id),
        mapping_status=inp.policy.mapping_status,
        policy_version="case-authored",
        computed_at=inp.as_of,
        targets=[
            {
                "target": t.target,
                "signals": [{**s.model_dump(), "metric_key": keys[s.evidence_id]} for s in t.signals],
                "net_weight": sum(
                    (s.weight if s.effect == "raises" else -s.weight for s in t.signals if s.material), Decimal(0)
                ),
                "outcome": t.outcome,
                "allowed_actions": t.allowed_actions,
                "magnitude_cap": t.magnitude_cap,
                "must_abstain": t.must_abstain,
            }
            for t in inp.policy.targets
        ],
    )
    return EvidenceBundle(
        bundle_id=f"bundle-{case.case_id}",
        case_id=case.case_id,
        created_at=inp.as_of,
        expires_at=inp.as_of + timedelta(hours=1),
        items=items,
        policy=policy,
    )
