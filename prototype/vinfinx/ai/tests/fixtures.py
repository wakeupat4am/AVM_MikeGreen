"""Shared test objects, built around the worked cases in AI_ENGINEER_FINDINGS.md (2026-10-11).

Bundle X: refinancing rate 4.0% (raises borrowing capacity).
Bundle Y: X plus Dong Da listing inventory rising (lowers borrowing capacity). The two signals oppose
and the net is small, so the policy engine reports a conflict and the decision must abstain.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from ai.catalog import load_catalogue
from ai.schemas.bundle import EvidenceBundle
from ai.schemas.common import content_hash
from ai.schemas.evidence import EvidenceItem
from ai.schemas.mapping import MappingEntry, MappingProposal, MappingSet, MappingVerification
from ai.schemas.policy import PolicyResult

CAT = load_catalogue()
H = "sha256:" + "a" * 64
NOW = datetime(2026, 10, 10, 9, 0, tzinfo=timezone.utc)
SYSTEM = {"kind": "system", "system": "macro.fixture", "record_id": "r", "export_hash": H, "exported_at": NOW}


def item(metric_id: str, label: str, start: datetime, end: datetime, value: str, unit: str, transformation: str):
    return EvidenceItem.model_validate(
        {
            "evidence_id": f"{metric_id}.{label}",
            "metric_id": metric_id,
            "geography": metric_id.split(".")[1],
            "source": "Lotus Bay Economic Observatory",
            "observation_period": {"label": label, "start": start, "end": end},
            "publication_date": NOW - timedelta(days=5),
            "retrieved_at": NOW - timedelta(days=4),
            "value": value,
            "unit": unit,
            "transformation": transformation,
            "data_quality": "ok",
            "freshness": {
                "deadline": CAT.freshness_deadline(metric_id, end),
                "catalogue_version": CAT.version,
                "catalogue_hash": CAT.catalogue_hash,
            },
            "presence": "present",
            "trust": "trusted",
            "synthetic": True,
            "provenance": SYSTEM,
        }
    )


REFI = item(
    "macro.vn.refi_rate", "2026-09",
    datetime(2026, 9, 1, tzinfo=timezone.utc), datetime(2026, 9, 30, 23, 59, 59, tzinfo=timezone.utc),
    "4.0", "pct", "level",
)
INVENTORY = item(
    "market.hn-dong-da.listing_inventory", "2026-10-01",
    datetime(2026, 10, 1, tzinfo=timezone.utc), datetime(2026, 10, 1, 23, 59, 59, tzinfo=timezone.utc),
    "18", "count", "level",
)


def mapping_set(status: str = "unvalidated") -> MappingSet:
    proposals = [
        MappingProposal(
            metric_key="refi_rate", target="borrowing_capacity", direction="negative", lag_months=(1, 6),
            size_range="medium", geography="vn",
            evidence=[{"chunk_id": "src1:c4", "quote": "Lower policy rates ease mortgage affordability."}],
        ),
        MappingProposal(
            metric_key="listing_inventory", target="borrowing_capacity", direction="negative", lag_months=(0, 3),
            size_range="low", geography="vn",
            evidence=[{"chunk_id": "src2:c9", "quote": "Rising unsold inventory weighs on collateral values."}],
        ),
    ]
    entries = [
        MappingEntry(
            proposal=p,
            verification=MappingVerification(
                proposal_hash=content_hash(p),
                rule_checks=[{"check": "quote_verbatim", "passed": True}],
                status="accepted",
            ),
        )
        for p in proposals
    ]
    mid = MappingSet.compute_id(1, entries, "corpus.v1", "index.v1")
    reviewed = {} if status == "unvalidated" else {"reviewed_by": "reviewer", "reviewed_at": NOW}
    return MappingSet(
        mapping_id=mid, version=1, entries=entries, corpus_version="corpus.v1", index_version="index.v1",
        created_at=NOW, status=status, **reviewed,
    )


MAPPING = mapping_set()


def policy_x() -> PolicyResult:
    """Refinancing rate fell: raises borrowing capacity. Unvalidated mapping, so low-impact actions only."""
    return PolicyResult(
        case_id="case-1", mapping_id=MAPPING.mapping_id, mapping_status="unvalidated", policy_version="policy.v0",
        computed_at=NOW,
        targets=[{
            "target": "borrowing_capacity",
            "signals": [{"evidence_id": REFI.evidence_id, "metric_key": "refi_rate", "effect": "raises",
                         "weight": Decimal("1.0"), "material": True}],
            "net_weight": Decimal("1.0"),
            "outcome": "raises",
            "allowed_actions": ["offer_review", "request_evidence", "no_change"],
            "magnitude_cap": "low",
            "must_abstain": False,
        }],
    )


def policy_y() -> PolicyResult:
    """Rate signal raises, inventory signal lowers, net small: conflict, must abstain."""
    return PolicyResult(
        case_id="case-1", mapping_id=MAPPING.mapping_id, mapping_status="unvalidated", policy_version="policy.v0",
        computed_at=NOW,
        targets=[{
            "target": "borrowing_capacity",
            "signals": [
                {"evidence_id": REFI.evidence_id, "metric_key": "refi_rate", "effect": "raises",
                 "weight": Decimal("1.0"), "material": True},
                {"evidence_id": INVENTORY.evidence_id, "metric_key": "listing_inventory", "effect": "lowers",
                 "weight": Decimal("0.9"), "material": True},
            ],
            "net_weight": Decimal("0.1"),
            "outcome": "conflict",
            "allowed_actions": ["abstain", "request_evidence"],
            "magnitude_cap": "none",
            "must_abstain": True,
        }],
    )


def bundle(policy: PolicyResult, items: list[EvidenceItem]) -> EvidenceBundle:
    return EvidenceBundle(
        bundle_id="bundle-1", case_id="case-1", created_at=NOW, expires_at=NOW + timedelta(hours=1),
        items=items, policy=policy,
    )


BUNDLE_X = bundle(policy_x(), [REFI])
BUNDLE_Y = bundle(policy_y(), [REFI, INVENTORY])
