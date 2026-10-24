"""Mapping registry for the site (MAPPING_AGENT_PLAN.md, M6-M7): runs, sets, the runner's choice, and review."""
from __future__ import annotations

import json
from pathlib import Path

from .. import mapping as mp
from ..catalog import load_catalogue
from ..policy import load_policy_config
from .service import ApiError, Store, _note, check_reviewer


def _runs(registry: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((registry / "runs").glob("*.json"), reverse=True)]


def overview(registry: Path = mp.REGISTRY) -> dict:
    runs = _runs(registry) if (registry / "runs").exists() else []
    sets = mp.registry_sets(registry) if registry.exists() else []
    mset, source, why = mp.select_mapping(load_policy_config(catalogue=load_catalogue()), registry)
    return {"runner": {"mapping_id": mset.mapping_id, "source": source, "reason": why, "status": mset.status},
            "sets": [{"version": s.version, "mapping_id": s.mapping_id, "status": s.status, "entries": len(s.entries),
                      "corpus_version": s.corpus_version, "created_at": s.created_at, "reviewed_by": s.reviewed_by}
                     for s in sets],
            "runs": [{"run_id": r["run_id"], "corpus_version": r["corpus_version"], "counts": r["counts"],
                      "mapping_set": r.get("mapping_set"), "support_cost_usd": r["support_cost_usd"],
                      "started_at": r["started_at"]} for r in runs]}


def get_run(run_id: str, registry: Path = mp.REGISTRY) -> dict:
    path = registry / "runs" / f"{run_id}.json"
    if "/" in run_id or not path.exists():
        raise ApiError(404, f"no mapping run {run_id}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("mapping_set"):
        s = mp.load_set(record["mapping_set"]["version"], registry)
        record["mapping_set"].update(status=s.status, reviewed_by=s.reviewed_by, reviewed_at=s.reviewed_at)
    return record


def review(store: Store, version: int, reviewer: str, action: str, note: str | None,
           registry: Path = mp.REGISTRY) -> dict:
    check_reviewer(reviewer)
    if not (registry / f"mapping-v{version}.json").exists():
        raise ApiError(404, f"no mapping set v{version}")
    try:
        s = mp.review_set(version, reviewer, {"validate": "validated", "reject": "rejected"}.get(action, action),
                          _note(note), store.now(), registry)
    except (ValueError, FileExistsError) as exc:
        raise ApiError(409 if "already" in str(exc) or isinstance(exc, FileExistsError) else 400, str(exc)) from None
    store.log("mapping_set_reviewed", reviewer, mapping_id=s.mapping_id, status=s.status)
    return {"mapping_id": s.mapping_id, "status": s.status, "reviewed_by": s.reviewed_by}
