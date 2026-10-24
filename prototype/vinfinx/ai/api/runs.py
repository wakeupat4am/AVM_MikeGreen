"""Pipeline runs for the site (RUNNER_PLAN.md, R5-R8): start a run, follow it, review a routed one, and give the
customer the latest released recommendation.

A run starts in a background thread, one at a time (Qwen serves one request at a time anyway). The run record is the
runner's; a reviewer's action is written beside it in review.json, once, and logged in review_actions.jsonl. A
recommendation is released to the customer only if the verifier passed it or a reviewer approved it.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from .. import runner as rn
from .service import ApiError, Store, _name, _note, _write_new, check_reviewer

REVIEW_ACTIONS = ("approve", "reject", "request_evidence")
_RUN_LOCK = threading.Lock()  # one run at a time
_current: dict = {}


def pipeline_inputs(fixtures: Path = rn.FIXTURES, set_name: str = rn.DEMO_SET) -> dict:
    customers = [{k: c[k] for k in ("customer_id", "display_name", "property_id", "district", "synthetic")}
                 for c in rn.load_customers().values()]
    return {"set": set_name, "documents": rn.documents(set_name, fixtures), "customers": customers,
            "running": _current.get("run_id")}


def start_run(store: Store, reviewer: str, document_id: str, customer_id: str, runs_dir: Path = rn.RUNS,
              run_fn=rn.run_pipeline, **run_kwargs) -> dict:
    """Starts the run in a thread and returns once its record exists (or the runner refused the inputs)."""
    check_reviewer(reviewer)
    if not _RUN_LOCK.acquire(blocking=False):
        raise ApiError(409, f"a run is in progress ({_current.get('run_id')}); wait for it to finish")
    started, failure, mine = threading.Event(), {}, {}

    def on_update(record):
        _current["run_id"] = mine["run_id"] = record["run_id"]
        started.set()

    def work():
        try:
            run_fn(document_id, customer_id, runs_dir=runs_dir, on_update=on_update, **run_kwargs)
        except Exception as exc:  # refused before the record existed: unknown document, held-out set, no key
            failure["error"] = f"{type(exc).__name__}: {exc}"
            started.set()
        finally:
            _current.clear()
            _RUN_LOCK.release()

    threading.Thread(target=work, daemon=True).start()
    started.wait(timeout=60)
    if failure:
        raise ApiError(400, failure["error"])
    run_id = mine.get("run_id")
    store.log("pipeline_run_started", reviewer, run_id=run_id, document_id=document_id, customer_id=customer_id)
    return {"run_id": run_id}


def _run_dir(runs_dir: Path, run_id: str) -> Path:
    d = runs_dir / _name(run_id)
    if not (d / "run.json").exists():
        raise ApiError(404, f"no run {run_id}")
    return d


def _review(d: Path) -> dict | None:
    return json.loads((d / "review.json").read_text(encoding="utf-8")) if (d / "review.json").exists() else None


def get_run(runs_dir: Path, run_id: str) -> dict:
    d = _run_dir(runs_dir, run_id)
    record = json.loads((d / "run.json").read_text(encoding="utf-8"))
    return {**record, "review": _review(d), "released": _released(record, _review(d))}


def _released(record: dict, review: dict | None) -> bool:
    return record.get("verdict") == "PASS" or (record.get("verdict") == "ROUTE_TO_HUMAN" and
                                               (review or {}).get("action") == "approve")


def _stage(record: dict, name: str) -> dict:
    return next((s for s in record["stages"] if s["name"] == name), {})


def list_runs(runs_dir: Path) -> list[dict]:
    out = []
    for p in sorted(runs_dir.glob("*/run.json"), reverse=True):
        r = json.loads(p.read_text(encoding="utf-8"))
        review = _review(p.parent)
        doc, ex = _stage(r, "document"), _stage(r, "extraction")
        rec = _stage(r, "decision").get("recommendation") or {}
        out.append({"run_id": r["run_id"], "started_at": r["started_at"], "status": r["status"],
                    "verdict": r.get("verdict"), "document_id": doc.get("document_id"), "variant": doc.get("variant"),
                    "customer_id": r["customer"]["customer_id"], "trusted": ex.get("trusted"),
                    "untrusted": ex.get("untrusted"), "action": rec.get("recommended_action"),
                    "error": next((s.get("error") for s in r["stages"] if s["status"] == "error"), None),
                    "review": review, "released": _released(r, review)})
    return out


def page_image(runs_dir: Path, run_id: str, page: int, fixtures: Path = rn.FIXTURES) -> Path:
    """A page image of the run's bulletin: only documents of a runnable set, only pages the run recorded."""
    doc = _stage(get_run(runs_dir, run_id), "document")
    name = f"page-{page}.png"
    if name not in (doc.get("pages") or {}):
        raise ApiError(404, "no such page")
    path = rn.set_dir(doc["set"], fixtures) / doc["document_id"] / name
    if not path.exists():
        raise ApiError(404, "no such page")
    return path


def review_run(store: Store, runs_dir: Path, run_id: str, reviewer: str, action: str, note: str | None) -> dict:
    check_reviewer(reviewer)
    if action not in REVIEW_ACTIONS:
        raise ApiError(400, f"action must be one of {REVIEW_ACTIONS}")
    record = get_run(runs_dir, run_id)
    if record.get("verdict") != "ROUTE_TO_HUMAN":
        raise ApiError(409, f"only runs routed to a person can be reviewed (this one: {record.get('verdict')})")
    note = _note(note)
    if action != "approve" and not note:
        raise ApiError(400, "rejecting or requesting evidence needs a note")
    review = {"run_id": run_id, "action": action, "note": note, "reviewer": reviewer, "at": store.now().isoformat(),
              "bundle_hash": record["versions"].get("bundle_hash")}
    _write_new(_run_dir(runs_dir, run_id) / "review.json", json.dumps(review, ensure_ascii=False, indent=1))
    store.log("pipeline_run_reviewed", reviewer, run_id=run_id, review_action=action,
              bundle_hash=review["bundle_hash"])
    return review


def customer_latest(runs_dir: Path, customer_id: str) -> dict | None:
    """The newest released recommendation for the customer, with the evidence it cites and the property's band."""
    for summary in list_runs(runs_dir):
        if summary["customer_id"] != customer_id or not summary["released"]:
            continue
        r = get_run(runs_dir, summary["run_id"])
        rec = _stage(r, "decision").get("recommendation") or {}
        items = {i["evidence_id"]: i for i in _stage(r, "bundle")["bundle"]["items"]}
        cited = [items[e] for e in dict.fromkeys(rec.get("evidence_ids", [])) if e in items]
        avm = {i["metric_id"].split(".")[-1]: i for i in _stage(r, "avm").get("items", [])}
        return {"run_id": r["run_id"], "verdict": r["verdict"], "review": r["review"], "finished_at": r["finished_at"],
                "recommendation": rec, "property": _stage(r, "avm").get("property"),
                "band": {k: {"value": v["value"], "date": v["observation_period"]["label"], "source": v["source"],
                             "caveats": v["caveats"]} for k, v in avm.items()},
                "sources": [{"evidence_id": i["evidence_id"], "source": i["source"], "value": i["value"],
                             "unit": i["unit"], "period": i["observation_period"]["label"],
                             "published": i.get("publication_date"), "synthetic": i["synthetic"]} for i in cited],
                "mapping_status": r["versions"].get("mapping_status"), "policy_status": r["versions"].get("policy_status")}
    return None


def audit(store: Store, runs_dir: Path, limit: int = 200) -> dict:
    runs = []
    for s in list_runs(runs_dir)[:limit]:
        r = get_run(runs_dir, s["run_id"])
        runs.append({**s, "versions": r["versions"], "pdf_sha256": _stage(r, "document").get("pdf_sha256"),
                     "avm_file_sha256": _stage(r, "avm").get("file_sha256")})
    log = store.logs / "review_actions.jsonl"
    actions = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()][-limit:] if log.exists() else []
    return {"runs": runs, "actions": list(reversed(actions))}
