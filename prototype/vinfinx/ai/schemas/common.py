"""Conventions shared by every VinFinx schema (docs/SPECS_AND_SCHEMAS_PLAN.md, Part D)."""
from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Annotated, Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

Sha256 = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]


class StrictModel(BaseModel):
    """Unknown fields are rejected and objects cannot be changed after creation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Unit(StrEnum):
    PCT = "pct"
    PP = "pp"  # percentage points
    BPS = "bps"
    VND = "vnd"
    VND_BN = "vnd_bn"
    VND_PER_M2 = "vnd_per_m2"
    COUNT = "count"
    INDEX_POINTS = "index_points"
    RATIO = "ratio"


class Transformation(StrEnum):
    LEVEL = "level"
    YOY = "yoy"
    MOM = "mom"
    YTD = "ytd"
    INDEX = "index"


class Frequency(StrEnum):
    SNAPSHOT = "snapshot"  # period label YYYY-MM-DD
    MONTHLY = "monthly"  # YYYY-MM
    QUARTERLY = "quarterly"  # YYYY-Qn


class GeographyLevel(StrEnum):
    NATIONAL = "national"
    CITY = "city"
    DISTRICT = "district"


class Action(StrEnum):
    """Handoff §3 recommended actions."""

    HOLD = "hold"
    OFFER_REVIEW = "offer_review"
    REQUEST_EVIDENCE = "request_evidence"
    NO_CHANGE = "no_change"
    ABSTAIN = "abstain"


# What an unvalidated mapping may drive (PIPELINE.md, policy engine step 4).
LOW_IMPACT_ACTIONS = frozenset({Action.OFFER_REVIEW, Action.REQUEST_EVIDENCE, Action.NO_CHANGE, Action.ABSTAIN})


class Magnitude(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def rank(self) -> int:
        return list(Magnitude).index(self)


class Target(StrEnum):
    """What a macro signal can affect; mappings are keyed by target."""

    BORROWING_CAPACITY = "borrowing_capacity"
    VALUATION_RANGE = "valuation_range"


class GenerationMeta(StrictModel):
    """Added by code to every LLM output. The LLM never writes its own provenance."""

    model: str
    model_revision: str
    precision: str
    engine: str  # e.g. "transformers==5.19.0"
    prompt_version: str
    trace_id: str
    generated_at: AwareDatetime
    raw_output_hash: Sha256


def canonical_json(obj: BaseModel | Any) -> bytes:
    """Sorted keys, no whitespace, UTF-8. Decimals and datetimes are serialized as strings."""
    data = obj.model_dump(mode="json") if isinstance(obj, BaseModel) else obj
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def content_hash(obj: BaseModel | Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(obj)).hexdigest()
