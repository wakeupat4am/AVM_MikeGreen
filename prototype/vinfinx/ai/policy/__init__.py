"""Deterministic policy engine (PIPELINE.md, online decision step 3): applies a mapping version and the policy
config to the evidence and produces a PolicyResult. No model is involved."""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from ..catalog import Catalogue
from ..schemas.common import Action, StrictModel, Target, content_hash

POLICY_PATH = Path(__file__).with_name("policy_v0.yaml")
Outcome = Literal["raises", "lowers", "none", "conflict"]


class Materiality(StrictModel):
    absolute: Decimal | None = Field(default=None, gt=0)  # in the metric's unit
    relative: Decimal | None = Field(default=None, gt=0)  # share of |baseline|

    @model_validator(mode="after")
    def _one(self) -> Materiality:
        if (self.absolute is None) == (self.relative is None):
            raise ValueError("give exactly one of absolute or relative")
        return self

    def met(self, change: Decimal, baseline: Decimal) -> bool:
        limit = self.absolute if self.absolute is not None else self.relative * abs(baseline)
        return abs(change) >= limit


class PolicyConfig(StrictModel):
    policy_version: str
    status: Literal["provisional", "approved"]
    approved: str
    signed_off_by: str | None = None
    weights: dict[Target, dict[str, Decimal]]
    materiality: dict[str, Materiality]
    conflict_threshold: Decimal = Field(gt=0)
    medium_cap_from_net: Decimal = Field(gt=0)
    actions: dict[Literal["unvalidated", "validated"], dict[Outcome, list[Action]]]

    @model_validator(mode="after")
    def _consistent(self) -> PolicyConfig:
        for target, weights in self.weights.items():
            for key, w in weights.items():
                if w < 0:
                    raise ValueError(f"{target}/{key}: negative weight")
                if key not in self.materiality:
                    raise ValueError(f"{key}: weighted but has no materiality threshold")
        for status in ("unvalidated", "validated"):
            if set(self.actions.get(status, {})) != {"raises", "lowers", "none", "conflict"}:
                raise ValueError(f"actions.{status} must cover every outcome")
        if Action.ABSTAIN not in self.actions["unvalidated"]["conflict"] or \
                Action.ABSTAIN not in self.actions["validated"]["conflict"]:
            raise ValueError("a conflict must allow abstain")
        if self.status == "approved" and not self.signed_off_by:
            raise ValueError("an approved policy needs the domain owner's sign-off")
        return self

    @property
    def policy_hash(self) -> str:
        return content_hash(self)

    def check_against(self, catalogue: Catalogue) -> None:
        keys = {m.key for m in catalogue.metrics}
        unknown = ({k for w in self.weights.values() for k in w} | set(self.materiality)) - keys
        if unknown:
            raise ValueError(f"metrics not in the catalogue: {sorted(unknown)}")


def load_policy_config(path: Path = POLICY_PATH, catalogue: Catalogue | None = None) -> PolicyConfig:
    with open(path, encoding="utf-8") as f:
        config = PolicyConfig.model_validate(yaml.safe_load(f))
    if catalogue is not None:
        config.check_against(catalogue)
    return config


from .engine import compute_policy  # noqa: E402  (engine imports PolicyConfig from here)

__all__ = ["PolicyConfig", "Materiality", "load_policy_config", "compute_policy", "POLICY_PATH"]
