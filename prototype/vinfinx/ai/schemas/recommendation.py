"""Recommendation: the decision engine's output (handoff §3 plus mapping_id), and its per-request JSON Schema."""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, model_validator

from .common import Action, GenerationMeta, Magnitude, Sha256, StrictModel, Target

if TYPE_CHECKING:
    from ..catalog import Catalogue
    from .bundle import EvidenceBundle


class Recommendation(StrictModel):
    """Exactly the fields Qwen generates. Provenance is added by code in RecommendationRecord."""

    recommended_action: Action
    magnitude: Magnitude
    reason_codes: list[str]
    evidence_ids: list[str]
    mapping_id: str
    customer_impacts: list[Target]
    uncertainty: float = Field(ge=0, le=1)
    requires_human_review: bool
    explanation: str = Field(min_length=1)

    @model_validator(mode="after")
    def _cited(self) -> Recommendation:
        if self.recommended_action != Action.ABSTAIN and not self.evidence_ids:
            raise ValueError("every action other than abstain must cite evidence")
        return self


# Caps the reason-code array. xgrammar enforces maxItems but ignores uniqueItems, so this bound is what stops
# the model repeating one code until max_tokens (baseline 2026-10-18: 4 of 20 cases).
MAX_REASON_CODES = 3
# xgrammar enforces maxItems but not uniqueItems: without a cap Qwen once repeated evidence IDs until max_tokens and never
# closed the JSON (8 of 62 runner outputs, 2026-10-23). Lists are capped at the number of distinct values, at most 8.
MAX_EVIDENCE_IDS = 8


class RecommendationRecord(StrictModel):
    schema_version: Literal["recommendation_record/0.1"] = "recommendation_record/0.1"
    recommendation: Recommendation
    meta: GenerationMeta
    bundle_id: str
    bundle_hash: Sha256


def citable_evidence_ids(bundle: EvidenceBundle, at: datetime | None = None) -> list[str]:
    """Items the decision may cite: trusted, fresh at `at` (default: when the bundle was created) and not
    excluded by the policy result (baseline 2026-10-18 and the first-fix run: excluded items were cited)."""
    at = at or bundle.created_at
    excluded = {s.evidence_id for t in bundle.policy.targets for s in t.signals if s.excluded_reason}
    return [i.evidence_id for i in bundle.items
            if i.trust == "trusted" and i.freshness.deadline >= at and i.evidence_id not in excluded]


def supported_general_codes(bundle: EvidenceBundle, at: datetime) -> list[str]:
    """The general reason codes this bundle supports at `at`. Shared by the generation schema and the verifier
    (v0.1), so the schema never offers a code the verifier would reject. Codes with no rule here are never
    supported."""
    targets, items = bundle.policy.targets, bundle.items
    excluded = [s.excluded_reason for t in targets for s in t.signals if s.excluded_reason]
    rules = {
        "CONFLICTING_SIGNALS": any(t.outcome == "conflict" for t in targets),
        "STALE_EVIDENCE": "STALE" in excluded or any(i.freshness.deadline < at for i in items),
        "INSUFFICIENT_EVIDENCE": (any(t.outcome == "none" for t in targets) or bool(excluded)
                                  or any(i.presence != "present" or i.trust != "trusted" for i in items)),
    }
    return [code for code, supported in rules.items() if supported]


def supported_reason_codes(bundle: EvidenceBundle, catalogue: Catalogue, at: datetime | None = None) -> list[str]:
    """Reason codes the bundle can support: <stem>_RISE or _DROP for each citable item that moved from its
    baseline, in the direction it moved (the verifier's rule), plus the supported general codes."""
    at = at or bundle.created_at
    citable = set(citable_evidence_ids(bundle, at))
    codes = []
    for item in bundle.items:
        if item.evidence_id not in citable:
            continue
        if item.value is None or item.baseline is None or item.value == item.baseline.value:
            continue
        stem = catalogue.get(item.metric_id).definition.reason_code_stem
        code = f"{stem}_{'RISE' if item.value > item.baseline.value else 'DROP'}"
        if code not in codes:
            codes.append(code)
    general = supported_general_codes(bundle, at)
    return codes + [c for c in catalogue.extra_reason_codes if c in general]


def _array(values: list[str], *, min_items: int = 0, max_items: int | None = None) -> dict[str, Any]:
    if not values:
        return {"type": "array", "maxItems": 0}  # an empty enum is not a valid schema
    out: dict[str, Any] = {"type": "array", "items": {"enum": values}}
    if min_items:
        out["minItems"] = min_items
    if max_items is not None:
        out["maxItems"] = max_items
    return out


def generation_schema(bundle: EvidenceBundle, catalogue: Catalogue) -> dict[str, Any]:
    """JSON Schema for one request, built from its bundle. Qwen cannot generate another mapping ID, an action
    the policy does not allow, a magnitude above the cap, a citation of an item it may not cite (unknown,
    stale, untrusted or excluded), a reason code the bundle does not support, or a non-abstain action without
    a citation. If nothing is citable, only abstain is possible. The verifier still checks everything."""
    cap = bundle.policy.magnitude_cap
    citable = citable_evidence_ids(bundle)
    targets = [t.target.value for t in bundle.policy.targets]

    def branch(actions: list[str], min_citations: int) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {  # key order is the generation order; keep it the same in every branch
                "recommended_action": {"enum": actions},
                "magnitude": {"enum": [m.value for m in Magnitude if m.rank <= cap.rank]},
                "reason_codes": _array(supported_reason_codes(bundle, catalogue), max_items=MAX_REASON_CODES),
                "evidence_ids": _array(citable, min_items=min_citations,
                                       max_items=min(len(citable), MAX_EVIDENCE_IDS)),
                "mapping_id": {"enum": [bundle.mapping_id]},
                "customer_impacts": _array(targets, max_items=len(targets)),
                "uncertainty": {"type": "number", "minimum": 0, "maximum": 1},
                "requires_human_review": {"type": "boolean"},
                "explanation": {"type": "string"},  # no maxLength: it cuts text mid-word; bounded by max_tokens
            },
            "required": [
                "recommended_action", "magnitude", "reason_codes", "evidence_ids", "mapping_id",
                "customer_impacts", "uncertainty", "requires_human_review", "explanation",
            ],
            "additionalProperties": False,
        }

    acting = [a.value for a in bundle.policy.allowed_actions if a != Action.ABSTAIN]
    abstain = branch([Action.ABSTAIN.value], 0)
    if not citable or not acting:
        return abstain
    return {"anyOf": [branch(acting, 1), abstain]}
