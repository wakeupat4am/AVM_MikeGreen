"""The policy engine: evidence + mapping + policy config -> PolicyResult (docs/POLICY_ENGINE_PLAN.md).

For each target the config weights and the mapping covers:
1. signals: each item whose metric the mapping links to the target, with an effect from the mapping's
   direction and the observed change, a weight from the config, and an exclusion reason if it cannot count;
2. net weight of the material signals;
3. outcome: none, conflict (opposing signals, small net), or the net's direction;
4. allowed actions and magnitude cap from the outcome and the mapping's status.
"""
from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from ..catalog import Catalogue
from ..schemas.common import GeographyLevel, Magnitude, Target
from ..schemas.evidence import EvidenceItem, Presence
from ..schemas.mapping import Direction, MappingEntry, MappingSet
from ..schemas.policy import PolicyResult, SignalUse, TargetOutcome

if TYPE_CHECKING:
    from . import PolicyConfig


def _mapping_entry(mapping: MappingSet, catalogue: Catalogue, metric_key: str, target: Target,
                   geo_id: str) -> MappingEntry | None:
    """The mapping for this metric and target at the item's geography, else at its nearest ancestor (research on
    Viet Nam applies to its districts)."""
    parents = {g.geo_id: g.parent for g in catalogue.geographies}
    while geo_id is not None:
        entry = mapping.lookup(metric_key, target, geo_id)
        if entry is not None:
            return entry
        geo_id = parents.get(geo_id)
    return None


def _signal(item: EvidenceItem, metric_key: str, direction: Direction, weight: Decimal, config: PolicyConfig,
            now: datetime) -> SignalUse:
    change = None
    if item.value is not None and item.baseline is not None and item.baseline.unit == item.unit:
        change = item.value - item.baseline.value
    effect = None
    if change:  # None or zero: no direction
        effect = "raises" if (change > 0) == (direction == Direction.POSITIVE) else "lowers"

    if item.freshness.deadline < now:
        reason = "STALE"
    elif item.presence != Presence.PRESENT or item.trust != "trusted":
        reason = "UNTRUSTED"
    elif change is None:
        reason = "NO_BASELINE"
    elif not config.materiality[metric_key].met(change, item.baseline.value):
        reason = "BELOW_MATERIALITY"
    else:
        reason = None
    return SignalUse(evidence_id=item.evidence_id, metric_key=metric_key, effect=effect, weight=weight,
                     material=reason is None, excluded_reason=reason)


def _outcome(target: Target, signals: list[SignalUse], mapping_status: str, config: PolicyConfig) -> TargetOutcome:
    material = [s for s in signals if s.material]
    net = sum((s.weight if s.effect == "raises" else -s.weight for s in material), Decimal(0))
    both_ways = {s.effect for s in material} == {"raises", "lowers"}
    if not material:
        outcome = "none"
    elif both_ways and abs(net) < config.conflict_threshold:
        outcome = "conflict"
    elif net == 0:
        outcome = "none"  # only zero-weight signals
    else:
        outcome = "raises" if net > 0 else "lowers"

    if outcome in ("none", "conflict"):
        cap = Magnitude.NONE
    elif mapping_status == "validated" and abs(net) >= config.medium_cap_from_net:
        cap = Magnitude.MEDIUM
    else:
        cap = Magnitude.LOW
    return TargetOutcome(target=target, signals=signals, net_weight=net, outcome=outcome,
                         allowed_actions=config.actions[mapping_status][outcome], magnitude_cap=cap,
                         must_abstain=outcome == "conflict")


def compute_policy(items: Iterable[EvidenceItem], mapping: MappingSet, config: PolicyConfig, *,
                   catalogue: Catalogue, customer_district: str, case_id: str, now: datetime) -> PolicyResult:
    """Apply the mapping and the policy config to the evidence. `customer_district` is the geo_id of the customer's
    property: district-level metrics count only for that district. `now` decides staleness."""
    if mapping.status == "rejected":
        raise ValueError(f"{mapping.mapping_id} is rejected and must not be applied")
    items = list(items)
    targets = []
    for target, weights in config.weights.items():
        if not any(e.proposal.target == target for e in mapping.entries):
            continue
        signals = []
        for item in items:
            inst = catalogue.get(item.metric_id)
            key = inst.definition.key
            if key not in weights:
                continue
            if inst.geography.level == GeographyLevel.DISTRICT and inst.geography.geo_id != customer_district:
                continue  # another district's market says nothing about this customer's property
            entry = _mapping_entry(mapping, catalogue, key, target, inst.geography.geo_id)
            if entry is None:
                continue
            signals.append(_signal(item, key, entry.proposal.direction, weights[key], config, now))
        targets.append(_outcome(target, signals, mapping.status, config))
    if not targets:
        raise ValueError("the mapping covers no target the policy config weights")
    return PolicyResult(case_id=case_id, mapping_id=mapping.mapping_id, mapping_status=mapping.status,
                        policy_version=config.policy_version, policy_hash=config.policy_hash, computed_at=now,
                        targets=targets)
