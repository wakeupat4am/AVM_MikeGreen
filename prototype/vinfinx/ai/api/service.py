"""What the Evidence sources view reads and writes (docs/PIPELINE_API_PLAN.md, A3-A7).

Writes go through the same functions as the command line (`ai.sources.review.apply_review`,
`ai.corpus.evaluate.apply_review`), so the site cannot sign or freeze anything the CLI would refuse. Row decisions
are saved as they are made (A4); nothing is signed or frozen until the reviewer asks and every check passes. Every
write is appended to `logs/review_actions.jsonl` with the reviewer, the time and the hashes it applied to.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ..corpus import CORPUS, LOGS
from ..corpus.evaluate import apply_review as apply_gold_review
from ..corpus.evaluate import measure, review_sample
from ..schemas.common import content_hash
from ..schemas.source import SourceList
from ..sources.review import VERSION_PATTERN, ReviewError, apply_review, signed_path, summary

REVIEWERS = [r.strip() for r in os.environ.get("VINFINX_REVIEWERS", "jimmytrn154").split(",") if r.strip()]
NOTE_CHARS = 500
QUERY_CHARS = 200
SEARCH_MODES = ("hybrid", "vector", "keyword")
GOLD_LABELS = ("supports", "related", "irrelevant")
NAME = re.compile(r"^[A-Za-z0-9._-]+$")  # run IDs and corpus versions: no path separators


class ApiError(Exception):
    def __init__(self, status: int, message: str, problems: list[str] | None = None) -> None:
        super().__init__(message)
        self.status, self.message, self.problems = status, message, problems or []


@dataclass
class Store:
    corpus: Path = CORPUS
    logs: Path = LOGS
    now: Callable[[], datetime] = field(default=lambda: datetime.now(timezone.utc))

    @property
    def lists(self) -> Path:
        return self.corpus / "source_lists"

    def run_dir(self, run_id: str) -> Path:
        return self.corpus / "runs" / _name(run_id)

    def version_dir(self, version: str) -> Path:
        d = self.corpus / "versions" / _name(version)
        if not (d / "manifest.json").exists():
            raise ApiError(404, f"no corpus version {version}")
        return d

    def log(self, action: str, reviewer: str | None, **details) -> None:
        self.logs.mkdir(parents=True, exist_ok=True)
        with open(self.logs / "review_actions.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"at": self.now().isoformat(), "action": action, "reviewer": reviewer, **details},
                               ensure_ascii=False) + "\n")


def _name(value: str) -> str:
    if not NAME.match(value or ""):
        raise ApiError(400, f"invalid name {value!r}")
    return value


def check_reviewer(reviewer: str | None) -> str:
    """Guards against typos, not attackers: the SSH tunnel is the access control (A2)."""
    if reviewer not in REVIEWERS:
        raise ApiError(403, f"reviewer must be one of {REVIEWERS}")
    return reviewer


def _note(note: str | None) -> str | None:
    note = (note or "").strip() or None
    if note and len(note) > NOTE_CHARS:
        raise ApiError(400, f"a note is at most {NOTE_CHARS} characters")
    return note


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _write_new(path: Path, text: str) -> None:
    """A signed list or a reviewed gold set is never overwritten."""
    try:
        with open(path, "x", encoding="utf-8") as f:
            f.write(text)
    except FileExistsError:
        raise ApiError(409, f"{path.name} exists; it is never overwritten") from None


# ---------------------------------------------------------------- source lists


def _draft(store: Store, run_id: str) -> SourceList:
    path = store.lists / f"{_name(run_id)}-draft.json"
    if not path.exists():
        raise ApiError(404, f"no draft list for run {run_id}")
    return SourceList.model_validate_json(path.read_text(encoding="utf-8"))


def _signed_lists(store: Store) -> list[SourceList]:
    return [SourceList.model_validate_json(p.read_text(encoding="utf-8"))
            for p in sorted(store.lists.glob("source_list-v*.json"))]


def _source_progress(store: Store, run_id: str) -> dict:
    """Saved row decisions; before the first save, any decisions already typed into the CLI's review.csv. The source
    agent pre-fills `reject` on the rows it rejected: that default is not a person's decision, so it is skipped and
    the reviewer confirms those rows in the site (one bulk action)."""
    path = store.run_dir(run_id) / "review_progress.json"
    if path.exists():
        return _read_json(path)["rows"]
    rows, csv_path = {}, store.run_dir(run_id) / "review.csv"
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                d = (r.get("reviewer_decision") or "").strip().lower()
                note = (r.get("reviewer_note") or "").strip()
                if r.get("agent_decision") == "rejected" and d == "reject" and not note:
                    continue
                if d in ("accept", "reject"):
                    rows[r["source_id"]] = {"decision": d, "note": (r.get("reviewer_note") or "").strip() or None,
                                            "reviewer": None, "at": None, "from": "review.csv"}
    return rows


def source_runs(store: Store) -> list[dict]:
    signed = _signed_lists(store)
    out = []
    for path in sorted(store.lists.glob("*-draft.json")):
        draft = SourceList.model_validate_json(path.read_text(encoding="utf-8"))
        report_path = store.run_dir(draft.run_id) / "report.json"
        report = _read_json(report_path) if report_path.exists() else {}
        progress = _source_progress(store, draft.run_id)
        out.append({"run_id": draft.run_id, "created_at": draft.created_at, "list_hash": draft.list_hash,
                    "records": len(draft.records),
                    "decisions": {d: sum(1 for r in draft.records if r.decision.value == d)
                                  for d in ("accepted", "unclear", "rejected", "bad_document")},
                    "decided": sum(1 for r in draft.records if r.source_id in progress),
                    "model": report.get("model"), "prompt_version": report.get("prompt_version"),
                    "agent_version": report.get("agent_version"), "coverage": report.get("coverage_accepted"),
                    "signed": [{"version": s.version, "list_hash": s.list_hash} for s in signed
                               if s.run_id == draft.run_id]})
    return out


def source_run(store: Store, run_id: str) -> dict:
    draft = _draft(store, run_id)
    progress = _source_progress(store, run_id)
    records = []
    for r in draft.records:
        d = r.model_dump(mode="json")
        d["original_stored"] = bool(r.sha256) and (store.corpus / "originals" /
                                                   (r.sha256.split(":")[1] + ".pdf")).exists()
        d["progress"] = progress.get(r.source_id)
        records.append(d)
    signed = [{"version": s.version, "list_hash": s.list_hash, "signed_off_by": s.signed_off_by,
               "signed_off_at": s.signed_off_at} for s in _signed_lists(store) if s.run_id == run_id]
    return {"run_id": run_id, "list_hash": draft.list_hash, "created_at": draft.created_at, "records": records,
            "decided": len(progress), "signed": signed}


def save_source_progress(store: Store, run_id: str, reviewer: str, rows: list[dict]) -> dict:
    """rows: [{source_id, decision: accept|reject|null (clears), note}]. The sign-off rules run at signing; here
    only the shape is checked, so a reviewer can save a decision and add its note later."""
    check_reviewer(reviewer)
    draft = _draft(store, run_id)
    ids = {r.source_id for r in draft.records}
    if any(s.run_id == run_id for s in _signed_lists(store)):
        raise ApiError(409, f"run {run_id} is already signed; its decisions are final")
    progress = _source_progress(store, run_id)
    at = store.now().isoformat()
    for row in rows:
        sid = row.get("source_id")
        if sid not in ids:
            raise ApiError(400, f"{sid!r} is not in the draft list of {run_id}")
        decision = row.get("decision")
        if decision is None:
            progress.pop(sid, None)
        elif decision in ("accept", "reject"):
            progress[sid] = {"decision": decision, "note": _note(row.get("note")), "reviewer": reviewer, "at": at}
        else:
            raise ApiError(400, f"{sid}: decision must be accept, reject or null")
    _write_json(store.run_dir(run_id) / "review_progress.json",
                {"run_id": run_id, "draft_list_hash": draft.list_hash, "rows": progress})
    store.log("source_progress", reviewer, run_id=run_id, draft_list_hash=draft.list_hash,
              rows=[{k: r.get(k) for k in ("source_id", "decision")} for r in rows])
    return {"decided": len(progress), "records": len(draft.records)}


def sign_source_list(store: Store, run_id: str, reviewer: str, version: str) -> dict:
    check_reviewer(reviewer)
    if not re.match(VERSION_PATTERN, version or ""):
        raise ApiError(400, f"version must look like source_list/v0.1, got {version!r}")
    out = signed_path(store.lists, version)
    if out.exists():
        raise ApiError(409, f"{version} exists; a signed-off version is never overwritten")
    draft = _draft(store, run_id)
    progress = _source_progress(store, run_id)
    rows = [{"source_id": r.source_id, "reviewer_decision": (progress.get(r.source_id) or {}).get("decision", ""),
             "reviewer_note": (progress.get(r.source_id) or {}).get("note") or ""} for r in draft.records]
    try:
        signed = apply_review(draft, rows, reviewer, version, store.now())
    except ReviewError as exc:
        problems = str(exc).splitlines()
        raise ApiError(422, f"not signed: {len(problems)} problem(s)", problems) from None
    _write_new(out, signed.model_dump_json(indent=1))
    result = summary(signed, out)
    store.log("source_list_signed", reviewer, run_id=run_id, draft_list_hash=draft.list_hash,
              version=version, list_hash=signed.list_hash)
    return result


def source_review_csv(store: Store, run_id: str) -> str:
    """The CLI's review table with the saved decisions filled in (`python -m ai.sources.review --csv`)."""
    progress = _source_progress(store, run_id)
    with open(store.run_dir(run_id) / "review.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows, fields = list(reader), reader.fieldnames
    buf = io.StringIO()
    w = csv.DictWriter(buf, fields)
    w.writeheader()
    for r in rows:
        p = progress.get(r["source_id"]) or {}
        w.writerow({**r, "reviewer_decision": p.get("decision", ""), "reviewer_note": p.get("note") or ""})
    return buf.getvalue()


def source_lists(store: Store) -> list[dict]:
    return [{"version": s.version, "run_id": s.run_id, "list_hash": s.list_hash, "signed_off_by": s.signed_off_by,
             "signed_off_at": s.signed_off_at, "records": len(s.records), "accepted": len(s.accepted())}
            for s in _signed_lists(store)]


def source_list(store: Store, version: str) -> dict:
    if not re.match(VERSION_PATTERN, version or ""):
        raise ApiError(400, f"invalid version {version!r}")
    path = signed_path(store.lists, version)
    if not path.exists():
        raise ApiError(404, f"no signed list {version}")
    s = SourceList.model_validate_json(path.read_text(encoding="utf-8"))
    return {**s.model_dump(mode="json"), "list_hash": s.list_hash}


def original_path(store: Store, sha256: str) -> Path:
    """Only originals named in a draft or signed list are served (A7)."""
    if not re.fullmatch(r"[0-9a-f]{64}", sha256 or ""):
        raise ApiError(400, "invalid hash")
    known = {r.sha256 for p in store.lists.glob("*.json")
             for r in SourceList.model_validate_json(p.read_text(encoding="utf-8")).records if r.sha256}
    path = store.corpus / "originals" / f"{sha256}.pdf"
    if f"sha256:{sha256}" not in known or not path.exists():
        raise ApiError(404, "no such original")
    return path


# ---------------------------------------------------------------- corpus and gold labels


def corpus_versions(store: Store) -> list[dict]:
    out = []
    for d in sorted((store.corpus / "versions").glob("*/manifest.json")):
        m = _read_json(d)
        out.append({"corpus_version": m["corpus_version"], "frozen": m["frozen"], "built_at": m["built_at"],
                    "documents": len(m["documents"]), "source_list_version": m["source_list_version"]})
    return out


def corpus_version(store: Store, version: str) -> dict:
    d = store.version_dir(version)
    manifest = _read_json(d / "manifest.json")
    index = _read_json(d / "index" / "manifest.json") if (d / "index" / "manifest.json").exists() else None
    gold = d / "gold"
    metrics = _read_json(gold / "metrics.json") if (gold / "metrics.json").exists() else None
    return {"manifest": manifest, "manifest_hash": content_hash(manifest), "index": index,
            "index_manifest_hash": content_hash(index) if index else None, "metrics": metrics,
            "gold_status": _gold_status(gold)}


def _gold_status(gold: Path) -> str | None:
    if (gold / "gold.json").exists():
        return "reviewed"
    return "draft" if (gold / "draft.json").exists() else None


def _gold_key(question_id: str, chunk_id: str) -> str:
    return f"{question_id}|{chunk_id}"


def _gold_progress(gold: Path) -> dict:
    path = gold / "review_progress.json"
    if path.exists():
        return _read_json(path)["rows"]
    rows = {}
    if (gold / "review.csv").exists():
        with open(gold / "review.csv", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if (r.get("reviewer_label") or "").strip() in GOLD_LABELS:
                    rows[_gold_key(r["question_id"], r["chunk_id"])] = {
                        "label": r["reviewer_label"].strip(), "note": (r.get("reviewer_note") or "").strip() or None,
                        "reviewer": None, "at": None, "from": "review.csv"}
    return rows


def _draft_labels(gold: Path) -> dict:
    if not (gold / "draft.json").exists():
        raise ApiError(404, "no gold-label drafts for this version")
    return _read_json(gold / "draft.json")


def gold_review(store: Store, version: str) -> dict:
    d = store.version_dir(version)
    gold = d / "gold"
    data = _draft_labels(gold)
    sample = review_sample(data["labels"])  # seeded: the same rows as review.csv
    wanted = {r["chunk_id"] for r in sample}
    chunks = {}
    with open(d / "chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c["chunk_id"] in wanted:
                chunks[c["chunk_id"]] = c
    docs = {doc["source_id"]: doc for doc in _read_json(d / "manifest.json")["documents"]}
    progress = _gold_progress(gold)
    rows = []
    for r in sample:
        c = chunks[r["chunk_id"]]
        doc = docs.get(c["source_id"], {})
        rows.append({**{k: r[k] for k in ("question_id", "question", "chunk_id", "label", "quote", "quote_verified")},
                     "passage": c["text"], "page": c["page"], "source_id": c["source_id"],
                     "title": doc.get("title"), "original_sha256": c.get("document_hash"),
                     "progress": progress.get(_gold_key(r["question_id"], r["chunk_id"]))})
    reviewed = _read_json(gold / "gold.json")["meta"].get("review") if (gold / "gold.json").exists() else None
    return {"corpus_version": version, "meta": data["meta"], "status": _gold_status(gold), "review": reviewed,
            "rows": rows, "decided": sum(1 for r in rows if r["progress"])}


def save_gold_progress(store: Store, version: str, reviewer: str, rows: list[dict]) -> dict:
    check_reviewer(reviewer)
    gold = store.version_dir(version) / "gold"
    if (gold / "gold.json").exists():
        raise ApiError(409, "the gold set is already reviewed")
    keys = {_gold_key(r["question_id"], r["chunk_id"]) for r in review_sample(_draft_labels(gold)["labels"])}
    progress = _gold_progress(gold)
    at = store.now().isoformat()
    for row in rows:
        key = _gold_key(row.get("question_id", ""), row.get("chunk_id", ""))
        if key not in keys:
            raise ApiError(400, f"{key} is not in the review sample")
        label = row.get("label")
        if label is None:
            progress.pop(key, None)
        elif label in GOLD_LABELS:
            progress[key] = {"label": label, "note": _note(row.get("note")), "reviewer": reviewer, "at": at}
        else:
            raise ApiError(400, f"{key}: label must be one of {GOLD_LABELS} or null")
    _write_json(gold / "review_progress.json", {"corpus_version": version, "rows": progress})
    store.log("gold_progress", reviewer, corpus_version=version,
              rows=[{k: r.get(k) for k in ("question_id", "chunk_id", "label")} for r in rows])
    return {"decided": len(progress), "rows": len(keys)}


def submit_gold_review(store: Store, version: str, reviewer: str, index_factory) -> dict:
    """Freeze the reviewed gold set (gold.json, never overwritten) and re-measure retrieval on it."""
    check_reviewer(reviewer)
    gold = store.version_dir(version) / "gold"
    if (gold / "gold.json").exists():
        raise ApiError(409, "gold.json exists; a reviewed gold set is never overwritten")
    data = _draft_labels(gold)
    progress = _gold_progress(gold)
    reviewed, missing = [], []
    for r in review_sample(data["labels"]):
        p = progress.get(_gold_key(r["question_id"], r["chunk_id"]))
        if p is None:
            missing.append(f"{r['question_id']} / {r['chunk_id']}: no label")
            continue
        reviewed.append({"question_id": r["question_id"], "chunk_id": r["chunk_id"], "reviewer_label": p["label"],
                         "reviewer_note": p.get("note") or ""})
    if missing:
        raise ApiError(422, f"not submitted: {len(missing)} row(s) without a label", missing)
    result, summary_ = apply_gold_review(data, reviewed, reviewer)
    _write_new(gold / "gold.json", json.dumps(result, ensure_ascii=False, indent=1))
    metrics = measure(index_factory(version), gold)
    store.log("gold_reviewed", reviewer, corpus_version=version,
              index_manifest_hash=data["meta"].get("index_manifest_hash"), agreement=summary_["agreement"])
    return {"review": summary_, "metrics": metrics}


def gold_review_csv(store: Store, version: str) -> str:
    """review.csv with the saved labels filled in, for `python -m ai.corpus.evaluate review`."""
    gold = store.version_dir(version) / "gold"
    progress = _gold_progress(gold)
    with open(gold / "review.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows, fields = list(reader), reader.fieldnames
    buf = io.StringIO()
    w = csv.DictWriter(buf, fields)
    w.writeheader()
    for r in rows:
        p = progress.get(_gold_key(r["question_id"], r["chunk_id"])) or {}
        w.writerow({**r, "reviewer_label": p.get("label", ""), "reviewer_note": p.get("note") or ""})
    return buf.getvalue()


# ---------------------------------------------------------------- search (A6)

PERSONAL = [
    (re.compile(r"(?:\d[\s.\-]?){9,}"), "a long number (ID, account or phone numbers are not allowed)"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "an email address"),
    (re.compile(r"(?:₫\s*\d|\d[\d.,]*\s*(?:₫|đ\b|đồng|vnd\b|vnđ|tỷ(?!\s*lệ)|triệu))", re.IGNORECASE),
     "an amount of money in đồng"),
]


def check_query(q: str) -> str:
    """Typed text goes to OpenAI for its embedding, so anything that looks like personal data is refused before
    it leaves the server (user's decision 2026-10-23, A6). A refused query is neither sent nor logged."""
    q = " ".join((q or "").split())
    if not q:
        raise ApiError(400, "empty query")
    if len(q) > QUERY_CHARS:
        raise ApiError(400, f"a query is at most {QUERY_CHARS} characters")
    for pattern, what in PERSONAL:
        if pattern.search(q):
            raise ApiError(400, f"refused: the query contains {what}. Public research questions only")
    return q


def search(store: Store, index, q: str, mode: str = "hybrid") -> dict:
    if mode not in SEARCH_MODES:
        raise ApiError(400, f"mode must be one of {SEARCH_MODES}")
    q = check_query(q)
    hits = index.search(q, k=8, mode=mode)
    docs = {d["source_id"]: d for d in _read_json(index.dir.parent / "manifest.json")["documents"]}
    for h in hits:
        h["original_sha256"] = docs.get(h["source_id"], {}).get("original_sha256")
    if mode != "keyword":  # the query text went to OpenAI
        store.logs.mkdir(parents=True, exist_ok=True)
        with open(store.logs / "search_queries.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"at": store.now().isoformat(), "query": q, "mode": mode,
                                "index_version": index.manifest.index_version,
                                "embedding_model": index.manifest.embedding_model}, ensure_ascii=False) + "\n")
    return {"query": q, "mode": mode, "index_version": index.manifest.index_version,
            "index_manifest_hash": content_hash(index.manifest), "hits": hits}
