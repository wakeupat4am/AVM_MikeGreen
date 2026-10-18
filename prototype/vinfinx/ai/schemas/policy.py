"""PolicyResult: what the deterministic policy engine decided, per target. The decision engine can only cite it."""
from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from .common import LOW_IMPACT_ACTIONS, Action, Magnitude, StrictModel, Target


class SignalUse(StrictModel):
    evidence_id: str
    metric_key: str
    effect: Literal["raises", "lowers"]  # from the mapping, applied by code; never by the LLM
    weight: Decimal = Field(ge=0)
    material: bool
    excluded_reason: str | None = None  # e.g. "STALE", "UNTRUSTED", "BELOW_MATERIALITY"

    @model_validator(mode="after")
    def _excluded(self) -> SignalUse:
        if self.material == (self.excluded_reason is not None):
            raise ValueError("a signal is either material or excluded with a reason")
        return self


class TargetOutcome(StrictModel):
    target: Target
    signals: list[SignalUse]
    net_weight: Decimal  # signed: raises positive, lowers negative, material signals only
    outcome: Literal["raises", "lowers", "none", "conflict"]
    allowed_actions: list[Action] = Field(min_length=1)
    magnitude_cap: Magnitude
    must_abstain: bool

    @model_validator(mode="after")
    def _consistent(self) -> TargetOutcome:
        expected = sum(
            (s.weight if s.effect == "raises" else -s.weight for s in self.signals if s.material), Decimal(0)
        )
        if self.net_weight != expected:
            raise ValueError("net_weight must equal the signed sum of material signal weights")
        if self.outcome == "conflict" and not self.must_abstain:
            raise ValueError("a conflict must abstain")
        if self.must_abstain and Action.ABSTAIN not in self.allowed_actions:
            raise ValueError("must_abstain requires 'abstain' among the allowed actions")
        return self


class PolicyResult(StrictModel):
    schema_version: Literal["policy_result/0.1"] = "policy_result/0.1"
    case_id: str
    mapping_id: str
    mapping_status: Literal["unvalidated", "validated"]  # a rejected mapping is never applied
    policy_version: str
    computed_at: AwareDatetime
    targets: list[TargetOutcome] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> PolicyResult:
        names = [t.target for t in self.targets]
        if len(set(names)) != len(names):
            raise ValueError("one outcome per target")
        if self.mapping_status == "unvalidated":
            for t in self.targets:
                if not set(t.allowed_actions) <= LOW_IMPACT_ACTIONS:
                    raise ValueError(f"{t.target}: an unvalidated mapping allows low-impact actions only")
                if t.magnitude_cap.rank > Magnitude.LOW.rank:
                    raise ValueError(f"{t.target}: an unvalidated mapping caps magnitude at 'low'")
        return self

    @property
    def allowed_actions(self) -> list[Action]:
        out = {a for t in self.targets for a in t.allowed_actions} | {Action.ABSTAIN}
        return [a for a in Action if a in out]

    @property
    def magnitude_cap(self) -> Magnitude:
        return max((t.magnitude_cap for t in self.targets), key=lambda m: m.rank)

    @property
    def cited_evidence_ids(self) -> set[str]:
        return {s.evidence_id for t in self.targets for s in t.signals}
