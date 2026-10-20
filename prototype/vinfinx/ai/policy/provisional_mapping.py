"""A hand-built, SYNTHETIC mapping set for testing the policy engine until the mapping agent exists
(docs/POLICY_ENGINE_PLAN.md, "Not in this step"). Its directions are the ones the frozen adversarial cases were
written with. Its quotes are synthetic test text, not research; it must never be used for a real decision."""
from __future__ import annotations

from datetime import datetime, timezone

from ..schemas.common import content_hash
from ..schemas.mapping import MappingEntry, MappingProposal, MappingSet, MappingVerification

CORPUS_VERSION = "synthetic-provisional"
CREATED_AT = datetime(2026, 10, 20, tzinfo=timezone.utc)

_PROPOSALS = [
    # A rate cut raises borrowing capacity: the target moves against the metric.
    ("refi_rate", "negative", (1, 6), "medium", "syn-src1:c4",
     "a reduction in the policy rate was followed within one to six months by lower mortgage lending rates"),
    ("mortgage_rate_avg", "negative", (0, 3), "medium", "syn-prov:c1",
     "SYNTHETIC: lower mortgage lending rates increase the loan amounts households can service"),
    # Rising unsold inventory lowers borrowing capacity (collateral values soften).
    ("listing_inventory", "negative", (0, 3), "low", "syn-prov:c2",
     "SYNTHETIC: rising unsold listing inventory weighs on collateral values and secured lending"),
]


def provisional_mapping_set() -> MappingSet:
    entries = []
    for key, direction, lag, size, chunk_id, quote in _PROPOSALS:
        proposal = MappingProposal(metric_key=key, target="borrowing_capacity", direction=direction, lag_months=lag,
                                   size_range=size, geography="vn",
                                   evidence=[{"chunk_id": chunk_id, "quote": quote}])
        entries.append(MappingEntry(proposal=proposal, verification=MappingVerification(
            proposal_hash=content_hash(proposal),
            rule_checks=[{"check": "hand_built_synthetic", "passed": True,
                          "detail": "provisional test mapping, not produced by the mapping agent"}],
            status="accepted")))
    mapping_id = MappingSet.compute_id(1, entries, CORPUS_VERSION, CORPUS_VERSION)
    return MappingSet(mapping_id=mapping_id, version=1, entries=entries, corpus_version=CORPUS_VERSION,
                      index_version=CORPUS_VERSION, created_at=CREATED_AT, status="unvalidated")
