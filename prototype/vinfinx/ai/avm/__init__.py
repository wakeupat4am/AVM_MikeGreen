"""Phase 3 AVM output as evidence for one property (RUNNER_PLAN.md, R3).

The Phase 3 row for the customer's property becomes four EvidenceItems: the band's lower edge, central estimate and upper
edge (P10, P50, P90) and the asking price, in billion VND. They are context metrics: no reason codes and no policy weight,
so they never move a decision; a recommendation may cite them for the valuation range, and the verifier checks any number
quoted from them. The file is read, never changed. Phase 3 is model output from asking prices, Hà Nội only: a triage band,
not an appraisal, and every item says so.
"""
from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from ..catalog import Catalogue
from ..evaluation.cases import period_bounds
from ..schemas.common import Transformation, Unit
from ..schemas.evidence import (EvidenceItem, Freshness, Period, Presence, SystemProvenance, Uncertainty,
                                catalogue_violations)

REPO = Path(__file__).resolve().parents[4]
PHASE3_INTERVALS = REPO / "Phase 3" / "data" / "phase3_available_intervals_v2.csv"
SYSTEM = "avm.phase3"
SOURCE = "VinFinx AVM Phase 3 (model output)"
BAND_LEVEL = Decimal("0.8")  # P10-P90
CAVEATS = ["Model output from listing asking prices, not an appraisal", "Hà Nội landed houses only",
           "Triage band: use the P10-P90 range, not the central estimate alone (Phase 3 MODEL_CARD.md)"]
COLUMNS = {"value_p10": "P10_B", "value_p50": "P50_B", "value_p90": "P90_B", "asking_price": "asking_B"}


class AvmError(Exception):
    pass


def file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def load_row(property_id: str, path: Path = PHASE3_INTERVALS) -> dict:
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["property_id"] == property_id:
                return row
    raise AvmError(f"property {property_id} is not in {path.name}")


def district_of(row: dict, catalogue: Catalogue) -> str:
    for g in catalogue.geographies:
        if g.phase3_district_name and g.phase3_district_name == row["district_name"]:
            return g.geo_id
    raise AvmError(f"district {row['district_name']!r} is not a catalogue geography")


def avm_items(property_id: str, catalogue: Catalogue, retrieved_at: datetime,
              path: Path = PHASE3_INTERVALS) -> list[EvidenceItem]:
    row = load_row(property_id, path)
    geo = district_of(row, catalogue)
    day = row["last_modified_date"][:10]
    start, end = period_bounds(day, catalogue.get(f"avm.{geo}.value_p50").definition.frequency)
    exported_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    provenance = SystemProvenance(system=SYSTEM, record_id=property_id, export_hash=file_sha256(path),
                                  exported_at=exported_at)
    flags = [f"AVM_FLAG:{row['flag']}", f"OVER_P90_PCT:{row['over_P90_pct']}"]
    if row.get("time_adjusted") != "True":
        flags.append("NOT_TIME_ADJUSTED")
    band = Uncertainty(kind="interval", lower=Decimal(row["P10_B"]), upper=Decimal(row["P90_B"]), level=BAND_LEVEL)
    items = []
    for key, column in COLUMNS.items():
        metric_id = f"avm.{geo}.{key}"
        item = EvidenceItem(
            evidence_id=f"{metric_id}.{day}", metric_id=metric_id, geography=geo, source=SOURCE,
            source_document_id=property_id, observation_period=Period(label=day, start=start, end=end),
            publication_date=None, retrieved_at=retrieved_at, value=Decimal(row[column]), unit=Unit.VND_BN,
            transformation=Transformation.LEVEL, data_quality="ok", quality_flags=flags,
            uncertainty=band if key == "value_p50" else None,
            freshness=Freshness(deadline=catalogue.freshness_deadline(metric_id, end),
                                catalogue_version=catalogue.version, catalogue_hash=catalogue.catalogue_hash),
            presence=Presence.PRESENT, trust="trusted", synthetic=False, caveats=CAVEATS, provenance=provenance)
        problems = catalogue_violations(item, catalogue)
        if problems:
            item = item.model_copy(update={"trust": "untrusted", "trust_reasons": problems})
        items.append(item)
    return items
