"""Metric catalogue: the one place metric IDs, units and freshness defaults are defined."""
from __future__ import annotations

import functools
import re
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from ..schemas.common import Frequency, GeographyLevel, StrictModel, Transformation, Unit, content_hash

CATALOGUE_PATH = Path(__file__).with_name("metrics.yaml")

GEO_ID = r"[a-z]+(?:-[a-z0-9]+)*"
METRIC_KEY = r"[a-z][a-z0-9_]*"
METRIC_ID_RE = re.compile(rf"^(macro|market)\.({GEO_ID})\.({METRIC_KEY})$")
PERIOD_RE = {
    Frequency.SNAPSHOT: re.compile(r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$"),
    Frequency.MONTHLY: re.compile(r"^\d{4}-(0[1-9]|1[0-2])$"),
    Frequency.QUARTERLY: re.compile(r"^\d{4}-Q[1-4]$"),
}


class Geography(StrictModel):
    geo_id: str = Field(pattern=rf"^{GEO_ID}$")
    level: GeographyLevel
    name_en: str
    name_vi: str
    parent: str | None = None
    phase3_district_name: str | None = None


class MetricDefinition(StrictModel):
    key: str = Field(pattern=rf"^{METRIC_KEY}$")
    scope: Literal["macro", "market"]
    geography_level: GeographyLevel
    name_en: str
    name_vi: str
    unit: Unit
    transformation: Transformation
    frequency: Frequency
    freshness_days: int = Field(gt=0)
    valid_range: tuple[Decimal, Decimal]
    reason_code_stem: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")

    @model_validator(mode="after")
    def _range_ordered(self) -> MetricDefinition:
        if self.valid_range[0] >= self.valid_range[1]:
            raise ValueError(f"{self.key}: valid_range must be [low, high] with low < high")
        return self


class MetricInstance(StrictModel):
    """A metric definition applied to one geography: what a metric ID points to."""

    metric_id: str
    definition: MetricDefinition
    geography: Geography


class Catalogue(StrictModel):
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    status: Literal["provisional", "approved"]
    geographies: list[Geography] = Field(min_length=1)
    metrics: list[MetricDefinition] = Field(min_length=1)
    extra_reason_codes: list[str] = []

    @model_validator(mode="after")
    def _consistent(self) -> Catalogue:
        geo_ids = [g.geo_id for g in self.geographies]
        if len(set(geo_ids)) != len(geo_ids):
            raise ValueError("duplicate geo_id")
        for g in self.geographies:
            if g.parent is not None and g.parent not in geo_ids:
                raise ValueError(f"{g.geo_id}: unknown parent {g.parent}")
        keys = [m.key for m in self.metrics]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate metric key")
        stems = [m.reason_code_stem for m in self.metrics]
        if len(set(stems)) != len(stems):
            raise ValueError("duplicate reason_code_stem")
        levels = {g.level for g in self.geographies}
        for m in self.metrics:
            if m.geography_level not in levels:
                raise ValueError(f"{m.key}: no geography at level {m.geography_level}")
        return self

    @functools.cached_property
    def instances(self) -> dict[str, MetricInstance]:
        """Every metric ID in the catalogue: each metric expanded over the geographies at its level."""
        out = {}
        for m in self.metrics:
            for g in self.geographies:
                if g.level == m.geography_level:
                    metric_id = f"{m.scope}.{g.geo_id}.{m.key}"
                    out[metric_id] = MetricInstance(metric_id=metric_id, definition=m, geography=g)
        return out

    def get(self, metric_id: str) -> MetricInstance:
        try:
            return self.instances[metric_id]
        except KeyError:
            raise KeyError(f"unknown metric_id: {metric_id}") from None

    def evidence_id(self, metric_id: str, period: str) -> str:
        frequency = self.get(metric_id).definition.frequency
        if not PERIOD_RE[frequency].match(period):
            raise ValueError(f"period {period!r} does not match {frequency.value} format for {metric_id}")
        return f"{metric_id}.{period}"

    def reason_codes(self) -> list[str]:
        codes = [f"{m.reason_code_stem}_{d}" for m in self.metrics for d in ("RISE", "DROP")]
        return codes + list(self.extra_reason_codes)

    def freshness_deadline(self, metric_id: str, observation_end: datetime) -> datetime:
        """End of the observation period + the metric's freshness_days. Never taken from a document."""
        if observation_end.tzinfo is None:
            raise ValueError("observation_end must be timezone-aware")
        return observation_end + timedelta(days=self.get(metric_id).definition.freshness_days)

    @property
    def catalogue_hash(self) -> str:
        return content_hash(self)


def load_catalogue(path: Path = CATALOGUE_PATH) -> Catalogue:
    with open(path, encoding="utf-8") as f:
        return Catalogue.model_validate(yaml.safe_load(f))
