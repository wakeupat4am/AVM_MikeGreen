"""Recommendation: the decision engine's output (handoff §3 plus mapping_id), and its per-request JSON Schema."""
from __future__ import annotations

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


class RecommendationRecord(StrictModel):
    schema_version: Literal["recommendation_record/0.1"] = "recommendation_record/0.1"
    recommendation: Recommendation
    meta: GenerationMeta
    bundle_id: str
    bundle_hash: Sha256


def supported_reason_codes(bundle: EvidenceBundle, catalogue: Catalogue) -> list[str]:
    """Reason codes the bundle can support: <stem>_RISE or _DROP for each metric whose item moved from its
    baseline, in the direction it moved (the verifier's rule), plus the catalogue's extra codes."""
    codes = []
    for item in bundle.items:
        if item.value is None or item.baseline is None or item.value == item.baseline.value:
            continue
        stem = catalogue.get(item.metric_id).definition.reason_code_stem
        code = f"{stem}_{'RISE' if item.value > item.baseline.value else 'DROP'}"
        if code not in codes:
            codes.append(code)
    return codes + list(catalogue.extra_reason_codes)


def generation_schema(bundle: EvidenceBundle, catalogue: Catalogue) -> dict[str, Any]:
    """JSON Schema for one request, built from its bundle. Qwen cannot generate an evidence ID that is not
    in the bundle, another mapping ID, an action the policy does not allow, or a magnitude above the cap.
    or a reason code for a metric the bundle does not show moving that way. The verifier still checks everything."""
    cap = bundle.policy.magnitude_cap
    return {
        "type": "object",
        "properties": {
            "recommended_action": {"enum": [a.value for a in bundle.policy.allowed_actions]},
            "magnitude": {"enum": [m.value for m in Magnitude if m.rank <= cap.rank]},
            "reason_codes": {"type": "array", "items": {"enum": supported_reason_codes(bundle, catalogue)},
                             "maxItems": MAX_REASON_CODES},
            "evidence_ids": {"type": "array", "items": {"enum": bundle.evidence_ids}},
            "mapping_id": {"enum": [bundle.mapping_id]},
            "customer_impacts": {"type": "array", "items": {"enum": [t.target.value for t in bundle.policy.targets]}},
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
