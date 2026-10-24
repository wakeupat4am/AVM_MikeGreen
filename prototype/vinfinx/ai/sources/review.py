"""Import the reviewer's decisions and freeze the signed-off source list (SOURCE_AGENT_PLAN.md, "Review and
sign-off").

    python -m ai.sources.review --run-id <id> --reviewer jimmytrn154 --version source_list/v0.1
    python -m ai.sources.review --run-id <run 2> --reviewer jimmytrn154 --version source_list/v0.2 --base source_list/v0.1

Every row needs `accept` or `reject`; overruling the agent needs a note; a source can only be accepted if its
original file is stored and hashed. A version is never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..schemas.source import Decision, Review, SourceList, SourceRecord
from . import CORPUS


class ReviewError(Exception):
    pass


def overruled(rec: SourceRecord) -> bool:
    d = rec.signed_decision
    return (d == "accept" and rec.decision is Decision.REJECTED) or (d == "reject" and rec.decision is
                                                                       Decision.ACCEPTED)


def apply_review(draft: SourceList, rows: list[dict[str, str]], reviewer: str, version: str,
                 at: datetime, base: SourceList | None = None) -> SourceList:
    """The signed list: the draft's records with the reviewer's decisions; with `base` (a signed list), every base
    record first, unchanged, then the draft's (SOURCE_RUN_2_PLAN.md, V5): nothing signed before is dropped or
    re-decided."""
    by_id = {r["source_id"]: r for r in rows}
    problems, records = [], []
    for rec in draft.records:
        row = by_id.get(rec.source_id)
        if row is None:
            problems.append(f"{rec.source_id}: missing from the review table")
            continue
        decision = (row.get("reviewer_decision") or "").strip().lower()
        note = (row.get("reviewer_note") or "").strip() or None
        if decision not in ("accept", "reject"):
            problems.append(f"{rec.source_id}: reviewer_decision must be accept or reject, got {decision!r}")
            continue
        overrules = (decision == "accept" and rec.decision is Decision.REJECTED) or (
            decision == "reject" and rec.decision is Decision.ACCEPTED)
        if overrules and not note:
            problems.append(f"{rec.source_id}: overruling the agent ({rec.decision.value}) needs a reviewer_note")
        if decision == "accept" and rec.sha256 is None:
            problems.append(f"{rec.source_id}: cannot accept a source whose original is not stored "
                            "(download it into corpus/inbox and re-run the source agent first)")
        records.append(rec.model_copy(update={"review": Review(decision=decision, note=note, reviewer=reviewer,
                                                               reviewed_at=at)}))
    extra = set(by_id) - {r.source_id for r in draft.records}
    problems += [f"{sid}: not in the draft list" for sid in sorted(extra)]
    kept = []
    if base is not None:
        if base.status != "signed_off":
            problems.append(f"{base.version}: a base list must be signed off")
        if base.version == version:
            problems.append(f"{version}: the new version must differ from its base")
        clash = {r.source_id for r in base.records} & {r.source_id for r in draft.records}
        problems += [f"{sid}: already in {base.version}" for sid in sorted(clash)]
        kept = list(base.records)
    if problems:
        raise ReviewError("\n".join(problems))
    return SourceList(version=version, status="signed_off", run_id=draft.run_id, created_at=draft.created_at,
                      records=kept + [SourceRecord.model_validate(r.model_dump()) for r in records],
                      signed_off_by=reviewer, signed_off_at=at)


VERSION_PATTERN = r"^source_list/v\d+\.\d+$"


def signed_path(lists: Path, version: str) -> Path:
    return lists / (version.replace("/", "-") + ".json")


def summary(signed: SourceList, out: Path) -> dict:
    return {"version": signed.version, "list_hash": signed.list_hash, "records": len(signed.records),
            "accepted": len(signed.accepted()),
            "agent_overruled": sum(1 for r in signed.records if overruled(r)),
            "unclear_decided": sum(1 for r in signed.records if r.decision is Decision.UNCLEAR),
            "signed_off_by": signed.signed_off_by, "file": str(out)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", required=True)
    p.add_argument("--reviewer", required=True)
    p.add_argument("--version", required=True, help="e.g. source_list/v0.1")
    p.add_argument("--csv", type=Path, help="default: corpus/runs/<run-id>/review.csv")
    p.add_argument("--base", help="a signed list to build on, e.g. source_list/v0.1 (its records are kept unchanged)")
    a = p.parse_args()
    if not re.match(VERSION_PATTERN, a.version):
        sys.exit(f"--version must look like source_list/v0.1, got {a.version!r}")
    lists = CORPUS / "source_lists"
    out = signed_path(lists, a.version)
    if out.exists():
        sys.exit(f"{out} exists; a signed-off version is never overwritten")
    draft = SourceList.model_validate_json((lists / f"{a.run_id}-draft.json").read_text(encoding="utf-8"))
    with open(a.csv or CORPUS / "runs" / a.run_id / "review.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    base = None
    if a.base:
        base = SourceList.model_validate_json(signed_path(lists, a.base).read_text(encoding="utf-8"))
    try:
        signed = apply_review(draft, rows, a.reviewer, a.version, datetime.now(timezone.utc), base)
    except ReviewError as exc:
        sys.exit(f"review not applied:\n{exc}")
    out.write_text(signed.model_dump_json(indent=1), encoding="utf-8")
    print(json.dumps(summary(signed, out), indent=1))


if __name__ == "__main__":
    main()
