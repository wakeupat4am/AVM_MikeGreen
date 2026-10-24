"""Source agent run (SOURCE_AGENT_PLAN.md): discover -> mechanical checks -> rank and cap -> download -> Qwen
screening -> decision -> draft source list and review table.

    python -m ai.sources.run --run-id <id> [--no-model] [--max-screen 150] [--per-group 15]

Network results, downloads and screenings are cached in the run folder, so a run resumes where it stopped, and
re-running only recomputes the checks and decisions (for example after a publisher's terms are checked).
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from ..catalog import load_catalogue
from ..evaluation.runner import DEFAULT_URL, call_model, git_commit
from ..schemas.source import (Check, Criterion, Decision, FoundBy, Outcome, ProposedBy, Screening, SourceList,
                              SourceRecord)
from . import (AGENT_VERSION, CORPUS, QUERIES_PATH, Candidate, PublisherFile, QueryFile, load_publishers,
               load_queries)
from .checks import check_c1, check_c2, check_c3, check_c7, decide, screening_checks
from .discover import (BIS_FEEDS, NSO_PAGES, SBV_PAGES, OpenAlexBudget, bis_candidates, nso_candidates,
                       nso_pdf_link, openalex_group, sbv_candidates, worldbank_group)
from .http import Blocked, Client, FetchError, sha256_hex
from .screen import MODEL, PROMPT_VERSION, extract_text, screen

VN = re.compile(r"vi[eệ]t\s?nam", re.I)
CITATION_PDF = re.compile(r'<meta[^>]+name="citation_pdf_url"[^>]+content="([^"]+)"', re.I)


def now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------- 1. discover


def discover(client: Client, queries: QueryFile, publishers: PublisherFile, budget_usd: float,
             log) -> tuple[list[Candidate], dict]:
    found: list[Candidate] = []
    notes: dict = {"blocked": [], "errors": []}
    budget = OpenAlexBudget(budget_usd)

    def attempt(label, fn):
        try:
            return fn()
        except Blocked as exc:
            notes["blocked"].append(f"{label}: {exc}")
        except (FetchError, ValueError) as exc:
            notes["errors"].append(f"{label}: {exc}")
        return []

    for g in queries.groups:
        found += attempt(f"openalex/{g.id}", lambda: openalex_group(client, budget, g, publishers))
        found += attempt(f"worldbank/{g.id}", lambda: worldbank_group(client, g))
        log(f"  {g.id}: {len(found)} so far; OpenAlex ${budget.spent:.3f} in {budget.requests} requests")
    for name, url in BIS_FEEDS.items():
        found += attempt(f"bis/{name}", lambda: (lambda r: bis_candidates(r.content, name, r.sha256))(client.get(url)))
    for name, url in SBV_PAGES.items():
        found += attempt(f"sbv/{name}",
                         lambda: (lambda r: sbv_candidates(r.text, url, name, r.sha256))(client.get(url)))
    for name, url in NSO_PAGES.items():
        found += attempt(f"nso/{name}", lambda: (lambda r: nso_candidates(r.text, name, r.sha256))(client.get(url)))
    notes["openalex"] = {"requests": budget.requests, "cost_usd": round(budget.spent, 4), "budget_usd": budget_usd}
    notes["found"] = len(found)
    return dedupe(found), notes


def _title_key(c: Candidate) -> str:
    year = c.publication_date.year if c.publication_date else ""
    return "title:" + re.sub(r"[^\w]+", " ", c.title.lower()).strip() + f"|{year}"


def candidate_keys(c: Candidate) -> set[str]:
    return {c.key, _title_key(c)}


def record_keys(r: SourceRecord) -> set[str]:
    """The keys a candidate for the same document would have: its DOI, its landing URL, its title and year."""
    year = r.publication_date.year if r.publication_date else ""
    keys = {"title:" + re.sub(r"[^\w]+", " ", r.title.lower()).strip() + f"|{year}", f"url:{r.url}"}
    if r.doi:
        keys.add(f"doi:{r.doi}")
    if r.pdf_url:
        keys.add(f"url:{r.pdf_url}")
    return keys


def signed_keys(lists: Path) -> set[str]:
    """Every document already decided in a signed source list (SOURCE_RUN_2_PLAN.md, V2)."""
    keys: set[str] = set()
    for p in lists.glob("source_list-v*.json"):
        for r in SourceList.model_validate_json(p.read_text(encoding="utf-8")).records:
            keys |= record_keys(r)
    return keys


def over_cap_from(run_ids: list[str], groups: set[str]) -> list[Candidate]:
    """Candidates an earlier run passed through the checks but left out over the cap, for these query groups only."""
    out = []
    for rid in run_ids:
        data = _load_json(CORPUS / "runs" / rid / "candidates.json", {})
        out += [c for c in (Candidate.model_validate(x) for x in data.get("candidates", [])) if set(c.groups) & groups]
    return out


def dedupe(found: list[Candidate]) -> list[Candidate]:
    """One candidate per document: the same DOI, or the same title and year (a DOI-less copy of a DOI record)."""
    merged: dict[str, Candidate] = {}
    alias: dict[str, str] = {}
    for c in found:
        keys = [k for k in (c.key if c.key.startswith("doi:") else None, _title_key(c)) if k]
        home = next((alias[k] for k in keys if k in alias), None)
        if home is None:
            home = keys[0]
            merged[home] = c
        else:
            merged[home] = merged[home].merged(c)
        for k in keys:
            alias.setdefault(k, home)
    return list(merged.values())


# ---------------------------------------------------------------- 2-3. checks, ranking


def metadata_checks(c: Candidate, publishers: PublisherFile):
    c1, pub, ptype = check_c1(c, publishers)
    c2, licence, licence_url = check_c2(c, pub)
    c3, lang = check_c3(c, None)
    return {"c1": c1, "c2": c2, "c3": c3, "c7": check_c7(c), "publisher": pub, "publisher_type": ptype,
            "licence": licence, "licence_url": licence_url, "language": lang}


def prescreen_ok(m: dict) -> bool:
    return all(m[k].outcome is not Outcome.FAIL for k in ("c1", "c2", "c3", "c7"))


def group_of(c: Candidate) -> str:
    return c.groups[0] if c.groups else f"listing:{c.adapter}"


def rank(cands: list[Candidate], max_screen: int, per_group: int) -> list[Candidate]:
    """Viet Nam-specific first, then newer first; at most per_group per group and max_screen in all (S6)."""
    def key(c: Candidate):
        vn = bool(VN.search(c.title + " " + (c.abstract or "")))
        return (not vn, -(c.publication_date.toordinal() if c.publication_date else 0))

    taken, per = [], Counter()
    for c in sorted(cands, key=key):
        g = group_of(c)
        if per[g] < per_group and len(taken) < max_screen:
            taken.append(c)
            per[g] += 1
    return taken


# ---------------------------------------------------------------- 4. download


PAGE_PDF = re.compile(r'href="([^"]+\.pdf)"', re.I)


def pdf_link_on_page(page: str, page_url: str) -> str | None:
    """The page's declared PDF (citation_pdf_url), else its first PDF link on the same host."""
    m = CITATION_PDF.search(page)
    if m:
        return m.group(1)
    host = urlsplit(page_url).netloc
    for href in PAGE_PDF.findall(page):
        link = urljoin(page_url, html.unescape(href))
        if urlsplit(link).netloc == host:
            return link
    return None


def manual_name(c: Candidate) -> str:
    return re.sub(r"[^a-z0-9]+", "-", c.key.lower()).strip("-")[:90] + ".pdf"


def store_original(data: bytes) -> tuple[str, Path]:
    digest = sha256_hex(data)
    path = CORPUS / "originals" / (digest.split(":")[1] + ".pdf")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return digest, path


def download(client: Client, c: Candidate, manual: bool) -> dict:
    """C6. Returns {outcome, detail, sha256, path, pdf_url, retrieved_at, retrieved_by}."""
    inbox = CORPUS / "inbox" / manual_name(c)
    if inbox.exists():
        data = inbox.read_bytes()
        if data[:5] != b"%PDF-":
            return {"outcome": "fail", "detail": f"inbox file {inbox.name} is not a PDF"}
        digest, path = store_original(data)
        note = inbox.with_suffix(".source.json")  # written when code fetched another open copy of the work
        if note.exists():
            n = json.loads(note.read_text(encoding="utf-8"))
            return {"outcome": "pass", "detail": f"other open copy fetched by code: {n['url']} ({n['why']})",
                    "sha256": digest, "path": str(path), "pdf_url": n["url"], "retrieved_by": "code",
                    "retrieved_at": n["retrieved_at"]}
        return {"outcome": "pass", "detail": f"downloaded by a person ({inbox.name})", "sha256": digest,
                "path": str(path), "pdf_url": c.pdf_url, "retrieved_by": "person",
                "retrieved_at": datetime.fromtimestamp(inbox.stat().st_mtime, timezone.utc).isoformat()}
    if manual:
        return {"outcome": "unclear", "detail": f"manual download needed: save the PDF as corpus/inbox/{inbox.name}",
                "manual": True}
    tried = []
    urls = [u for u in (c.pdf_url,) if u]
    try:
        if c.item_page:
            page = client.get(c.item_page)
            link = nso_pdf_link(page.text)
            if link:
                urls.append(link)
        if not urls:
            urls.append(c.landing_url)
        for url in urls:
            tried.append(url)
            r = client.get(url, store=False, retries=1)
            if not r.is_pdf and "html" in r.content_type:
                link = pdf_link_on_page(r.text, r.url)
                if link:
                    tried.append(link)
                    r = client.get(link, store=False, retries=1)
            if r.is_pdf:
                digest, path = store_original(r.content)
                return {"outcome": "pass", "detail": f"downloaded and hashed ({len(r.content)} bytes)",
                        "sha256": digest, "path": str(path), "pdf_url": r.url, "retrieved_by": "code",
                        "retrieved_at": r.fetched_at.isoformat()}
        return {"outcome": "fail", "detail": f"no PDF at {', '.join(tried)}"}
    except Blocked as exc:
        return {"outcome": "unclear", "detail": f"{exc}; manual download possible: save as corpus/inbox/"
                                                f"{inbox.name}", "manual": True}
    except FetchError as exc:
        return {"outcome": "fail", "detail": str(exc)}


# ---------------------------------------------------------------- 5. records


def build_record(c: Candidate, m: dict, dl: dict | None, text: str | None, s: Screening | None, err: str | None,
                 proposed: ProposedBy) -> SourceRecord:
    c3, lang = check_c3(c, text) if text else (m["c3"], m["language"])
    checks = [m["c1"], m["c2"], c3, m["c7"]]
    if dl is not None:
        checks.append(Check(criterion=Criterion.C6_RETRIEVABLE, outcome=Outcome(dl["outcome"]),
                            detail=dl["detail"], by="code"))
    if s is not None:
        checks += screening_checks(s)
    elif err:
        checks.append(Check(criterion=Criterion.C4_RELEVANCE, outcome=Outcome.UNCLEAR,
                            detail=f"screening failed: {err}", by="source_agent"))
    decision, failed, reason = decide(checks)
    pub = m["publisher"]
    found = [FoundBy(adapter=c.adapter, query_id=q, raw_sha256=c.raw_sha256) for q in c.query_ids]
    stored = dl is not None and dl["outcome"] == "pass"
    return SourceRecord(
        source_id="src-0000", title=c.title, publisher=pub.name if pub else c.source_name,
        publisher_type=m["publisher_type"], country=pub.country if pub else None, language=lang,
        publication_date=c.publication_date, url=c.landing_url, doi=c.doi,
        pdf_url=(dl or {}).get("pdf_url") or c.pdf_url, licence=m["licence"], licence_url=m["licence_url"],
        retrieved_at=dl["retrieved_at"] if stored else None, retrieved_by=dl["retrieved_by"] if stored else None,
        sha256=dl["sha256"] if stored else None, found_by=found, checks=checks, screening=s,
        qualifies_because=s.qualifies_because if s else None, proposed_by=proposed, decision=decision,
        failed_criterion=failed, decision_reason=reason)


def number(records: list[SourceRecord], keys: list[str], ids_path: Path) -> list[SourceRecord]:
    """A document keeps its source_id for good (one numbering shared by every run, SOURCE_RUN_2_PLAN.md V3), so a
    review table filled in before a re-run still points at the same documents and two runs never reuse an ID. New
    documents get the next free number. Listed accepted first."""
    ids: dict[str, str] = json.loads(ids_path.read_text(encoding="utf-8")) if ids_path.exists() else {}
    order = {Decision.ACCEPTED: 0, Decision.UNCLEAR: 1, Decision.REJECTED: 2, Decision.BAD_DOCUMENT: 3}
    ranked = sorted(zip(records, keys), key=lambda rk: (order[rk[0].decision], rk[0].title.lower()))
    nxt = max((int(v.split("-")[1]) for v in ids.values()), default=0)
    for _, k in ranked:
        if k not in ids:
            nxt += 1
            ids[k] = f"src-{nxt:04d}"
    ids_path.write_text(json.dumps(ids, indent=1), encoding="utf-8")
    return [r.model_copy(update={"source_id": ids[k]}) for r, k in ranked]


REVIEW_COLUMNS = ["source_id", "agent_decision", "reviewer_decision", "reviewer_note", "title", "publisher",
                  "publisher_type", "publication_date", "language", "url", "pdf_url", "licence", "reason",
                  "qualifies_because", "quotes", "unverified_quotes", "metrics", "targets", "geography", "checks"]


def write_review(records: list[SourceRecord], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, REVIEW_COLUMNS)
        w.writeheader()
        for r in records:
            s = r.screening
            w.writerow({
                "source_id": r.source_id, "agent_decision": r.decision.value,
                # rejected by a failed check: pre-filled, change it to overrule; everything else is left blank
                "reviewer_decision": "reject" if r.decision is Decision.REJECTED else "", "reviewer_note": "",
                "title": r.title, "publisher": r.publisher or "",
                "publisher_type": r.publisher_type.value if r.publisher_type else "",
                "publication_date": r.publication_date or "", "language": r.language or "", "url": r.url,
                "pdf_url": r.pdf_url or "", "licence": r.licence or "", "reason": r.decision_reason,
                "qualifies_because": r.qualifies_because or "",
                "quotes": " | ".join(q.quote for q in s.quotes) if s else "",
                "unverified_quotes": " | ".join(s.unverified_quotes) if s else "",
                "metrics": ",".join(s.metrics) if s else "", "targets": ",".join(t.value for t in s.targets) if s else "",
                "geography": s.geography.value if s else "",
                "checks": "; ".join(f"{k.criterion.value} {k.outcome.value}" for k in r.checks)})


def coverage(records: list[SourceRecord], queries: QueryFile) -> dict[str, int]:
    pairs = {f"{m}->{t.value}": 0 for g in queries.groups for m in g.metrics for t in g.targets}
    for r in records:
        if r.screening:
            for m in r.screening.metrics:
                for t in r.screening.targets:
                    k = f"{m}->{t.value}"
                    if k in pairs:
                        pairs[k] += 1
    return pairs


# ---------------------------------------------------------------- run


def _load_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


SHARED_IDS = CORPUS / "source_ids.json"


def shared_ids_path() -> Path:
    """One numbering for all runs; seeded once from run 1's per-run ids.json, without changing any ID."""
    if not SHARED_IDS.exists():
        merged: dict[str, str] = {}
        for p in sorted((CORPUS / "runs").glob("*/ids.json")):
            for k, v in json.loads(p.read_text(encoding="utf-8")).items():
                if k in merged and merged[k] != v:
                    raise SystemExit(f"{k} has two source IDs ({merged[k]}, {v}); fix before numbering across runs")
                merged.setdefault(k, v)
        if len(set(merged.values())) != len(merged):
            raise SystemExit("two documents share a source ID across runs; fix before numbering across runs")
        SHARED_IDS.write_text(json.dumps(merged, indent=1), encoding="utf-8")
    return SHARED_IDS


def run(run_id: str, use_model: bool, max_screen: int, per_group: int, budget_usd: float, url: str,
        queries_path: Path = QUERIES_PATH, include_over_cap: list[str] | None = None) -> Path:
    run_dir = CORPUS / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logf = open(run_dir / "run.log", "a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"{now():%H:%M:%S} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    queries, publishers, catalogue = load_queries(queries_path), load_publishers(), load_catalogue()
    client = Client(run_dir / "raw")

    cand_path = run_dir / "candidates.json"
    if cand_path.exists():
        data = _load_json(cand_path, {})
        cands = dedupe([Candidate.model_validate(c) for c in data["candidates"]])
        notes = data["notes"]
        log(f"discovery: {len(cands)} candidates (cached)")
    else:
        log("discovery")
        cands, notes = discover(client, queries, publishers, budget_usd, log)
        cand_path.write_text(json.dumps({"notes": notes, "candidates": [c.model_dump(mode="json") for c in cands]},
                                        ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"discovery: {notes['found']} found, {len(cands)} after de-duplication; blocked {len(notes['blocked'])}, "
            f"errors {len(notes['errors'])}")

    if include_over_cap:
        cands = dedupe(cands + over_cap_from(include_over_cap, {g.id for g in queries.groups}))
    known = signed_keys(CORPUS / "source_lists")
    skipped = [c for c in cands if candidate_keys(c) & known]
    cands = [c for c in cands if not candidate_keys(c) & known]
    notes["skipped_signed"] = len(skipped)
    if skipped:
        log(f"skipped {len(skipped)} candidates already decided in a signed source list")
    meta = {c.key: metadata_checks(c, publishers) for c in cands}
    passed = [c for c in cands if prescreen_ok(meta[c.key])]
    selected = rank(passed, max_screen, per_group)
    log(f"checks: {len(passed)} of {len(cands)} pass C1, C2, C3, C7 (or unclear); {len(selected)} selected")

    dl_path = run_dir / "downloads.json"
    downloads = _load_json(dl_path, {})
    for i, c in enumerate(selected, 1):
        if c.key in downloads:
            continue
        pub = meta[c.key]["publisher"]
        downloads[c.key] = download(client, c, manual=bool(pub and pub.manual_download))
        dl_path.write_text(json.dumps(downloads, indent=1), encoding="utf-8")
        log(f"  download {i}/{len(selected)} {downloads[c.key]['outcome']}: {c.title[:70]}")

    texts: dict[str, str | None] = {}
    for c in selected:
        d = downloads.get(c.key, {})
        texts[c.key] = None
        if d.get("outcome") == "pass":
            try:
                texts[c.key], pages = extract_text(Path(d["path"]))
                if len(texts[c.key].strip()) < 200:
                    texts[c.key] = None  # scanned: no text layer; screened on title and abstract
            except Exception as exc:  # an unreadable PDF fails C6
                downloads[c.key] = {"outcome": "fail", "detail": f"PDF not readable: {type(exc).__name__}"}

    sc_path = run_dir / "screenings.jsonl"
    screenings: dict[str, dict] = {}
    if sc_path.exists():
        for line in sc_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            screenings[row["key"]] = row
    # a failed call is retried on the next run; a completed screening is kept
    to_screen = [c for c in selected if downloads[c.key]["outcome"] != "fail"
                 and (c.key not in screenings or screenings[c.key].get("error"))]
    if use_model and to_screen:
        log(f"screening {len(to_screen)} candidates with {MODEL}")
        with open(sc_path, "a", encoding="utf-8") as f:
            for i, c in enumerate(to_screen, 1):
                s, err = screen(c, texts[c.key], catalogue, url, call_model)
                row = {"key": c.key, "screening": s.model_dump(mode="json") if s else None, "error": err}
                screenings[c.key] = row
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                what = f"{s.relevant}/{s.geography.value}, {len(s.quotes)} quote(s)" if s else f"error {err}"
                log(f"  screen {i}/{len(to_screen)} {what} {s.seconds if s else 0:.0f}s: {c.title[:60]}")
    elif to_screen:
        log(f"--no-model: {len(to_screen)} candidates not screened")

    proposed = ProposedBy(agent_version=AGENT_VERSION, model=MODEL, prompt_version=PROMPT_VERSION, run_id=run_id,
                          query_file_hash=queries.file_hash, publisher_file_hash=publishers.file_hash)
    sel_keys = {c.key for c in selected}
    records, keys = [], []
    for c in cands:
        m = meta[c.key]
        if c.key in sel_keys:
            row = screenings.get(c.key) or {}
            s = Screening.model_validate(row["screening"]) if row.get("screening") else None
            records.append(build_record(c, m, downloads.get(c.key), texts.get(c.key), s, row.get("error"),
                                        proposed))
            keys.append(c.key)
        elif not prescreen_ok(m):
            records.append(build_record(c, m, None, None, None, None, proposed))
            keys.append(c.key)
        # passed the checks but over the cap: kept in candidates.json for a later run, not on this list
    records = number(records, keys, shared_ids_path())
    source_list = SourceList(version=f"{run_id}-draft", status="draft", run_id=run_id, created_at=now(),
                             records=records)
    lists = CORPUS / "source_lists"
    lists.mkdir(parents=True, exist_ok=True)
    (lists / f"{run_id}-draft.json").write_text(source_list.model_dump_json(indent=1), encoding="utf-8")
    write_review(records, run_dir / "review.csv")
    manual = [(r.source_id, r.title, r.url, k.detail) for r in records for k in r.checks
              if k.criterion is Criterion.C6_RETRIEVABLE and "manual download" in k.detail]
    with open(run_dir / "manual_downloads.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_id", "title", "url", "save_as"])
        for sid, title, link, detail in manual:
            w.writerow([sid, title, link, detail.rsplit("save as ", 1)[-1].rsplit("save the PDF as ", 1)[-1]])

    calls = [Screening.model_validate(r["screening"]) for r in screenings.values() if r.get("screening")]
    report = {
        "run_id": run_id, "agent_version": AGENT_VERSION, "prompt_version": PROMPT_VERSION, "model": MODEL,
        "code_commit": git_commit(), "query_file_hash": queries.file_hash,
        "publisher_file_hash": publishers.file_hash, "catalogue_hash": catalogue.catalogue_hash,
        "list_hash": source_list.list_hash, "created_at": now().isoformat(), "discovery": notes,
        "candidates": len(cands), "passed_metadata_checks": len(passed), "selected": len(selected),
        "over_cap": len(passed) - len(selected),
        "decisions": dict(Counter(r.decision.value for r in records)),
        "failed_criterion": dict(Counter(r.failed_criterion.value for r in records if r.failed_criterion)),
        "by_adapter": dict(Counter(r.found_by[0].adapter for r in records)),
        "downloads": dict(Counter(d["outcome"] for d in downloads.values())),
        "manual_downloads": len(manual),
        "screening": {"calls": len(calls), "errors": sum(1 for r in screenings.values() if r.get("error")),
                      "prompt_tokens": sum(s.prompt_tokens or 0 for s in calls),
                      "completion_tokens": sum(s.completion_tokens or 0 for s in calls),
                      "seconds": round(sum(s.seconds or 0 for s in calls), 1),
                      "unverified_quotes": sum(len(s.unverified_quotes) for s in calls),
                      "verified_quotes": sum(len(s.quotes) for s in calls)},
        "coverage_accepted": coverage([r for r in records if r.decision is Decision.ACCEPTED], queries),
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"decisions {report['decisions']}; review table {run_dir / 'review.csv'}")
    return run_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", default=f"{now():%Y%m%dT%H%M%SZ}")
    p.add_argument("--no-model", action="store_true", help="stop before the Qwen screening")
    p.add_argument("--max-screen", type=int, default=150)
    p.add_argument("--per-group", type=int, default=15)
    p.add_argument("--openalex-budget", type=float, default=0.08, help="USD per run (shared free tier: $0.10/day)")
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--queries", type=Path, default=QUERIES_PATH, help="query file (default queries_v0.yaml)")
    p.add_argument("--include-over-cap", nargs="*", default=[], metavar="RUN_ID",
                   help="earlier runs whose over-cap candidates in these query groups are eligible again")
    a = p.parse_args()
    if shutil.disk_usage(CORPUS.resolve()).free < 2_000_000_000:
        sys.exit(f"less than 2 GB free under {CORPUS.resolve()}")
    run(a.run_id, not a.no_model, a.max_screen, a.per_group, a.openalex_budget, a.url, a.queries, a.include_over_cap)


if __name__ == "__main__":
    main()
