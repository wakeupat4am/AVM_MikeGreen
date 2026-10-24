"""End-to-end runner (RUNNER_PLAN.md): one pass of the online pipeline from a synthetic bulletin and the customer's
property to a verified recommendation, recorded in one run record (`pipeline_run/0.1`).

    bulletin -> OCR (Tesseract, env-ocr) -> extraction + validation -> AVM evidence (Phase 3) -> policy engine
    (provisional mapping) -> signed bundle -> Qwen decision -> verifier

Every stage calls the existing module; nothing is re-implemented here. No stage calls OpenAI. A stage that fails is
recorded with its error and the run stops there; earlier stages stay in the record. A finished record is never
overwritten (a reviewer's action is stored beside it, in review.json).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import yaml

from ..avm import PHASE3_INTERVALS, SYSTEM as AVM_SYSTEM, avm_items, file_sha256, load_row
from ..catalog import Catalogue, load_catalogue
from ..evaluation.runner import DECISION_INSTRUCTIONS, Generation, call_model, git_commit
from ..extraction.macro_document import EXTRACTOR_VERSION, PUBLISHER, extract
from ..ocr import OcrPage
from ..policy import compute_policy, load_policy_config
from ..schemas.bundle import EvidenceBundle, sign_bundle
from ..schemas.recommendation import citable_evidence_ids, generation_schema
from ..sources.checks import detect_language
from ..verifier import VERIFIER_VERSION
from ..verifier.recommendation import verify_recommendation

RUNNER_VERSION = "pipeline_runner/0.1.0"
SCHEMA_VERSION = "pipeline_run/0.1"
# 0.2 (2026-10-23): values the decision may not cite are withheld, items carry their catalogue name, AVM items are marked
# as context. In 0.1 Qwen quoted the figures of an untrusted row it could not cite, and 55 of 62 runs were blocked.
# 0.3 (2026-10-24, "cite the AVM item when stating its figure") made it worse (53 of 62 blocked) and was reverted to 0.2;
# see the findings log. 0.4 (do not state AVM figures; the site shows the band) is to try after the mapping agent.
PROMPT_VERSION = "runner_prompts/0.2"  # DECISION_INSTRUCTIONS (eval_prompts/0.2) + RUNNER_NOTES + CUSTOMER_REQUEST
RUNNER_NOTES = ("Items whose value is 'withheld' cannot be cited: do not state, estimate or describe their values. "
                "AVM items describe the customer's own property (a model band from asking prices, not an appraisal); "
                "they are context, not market signals.")
CUSTOMER_REQUEST = ("Given the latest market bulletin, should the valuation range of my property or my borrowing "
                    "capacity be reviewed?")
from ..paths import PIPELINE_RUNS, ROOT  # noqa: E402
FIXTURES = ROOT / "fixtures"
DEMO_SET = os.environ.get("VINFINX_DEMO_SET", "docs-v0.1-demo-20261023")
HELD_OUT = {"docs-v0.1-holdout-s7"}  # never run, never shown (AGENTS.md)
RUNS = PIPELINE_RUNS
OCR_PYTHON = ROOT / "env-ocr" / "bin" / "python"
QWEN = os.environ.get("VINFINX_QWEN_BASE", "http://127.0.0.1:8061/vinfinx-api")
QWEN_PRECISION = "bf16"
BUNDLE_TTL = timedelta(hours=1)
MESSAGE_CHARS = 3500  # under the adapter's 4,000-character limit per message
MAX_PARTS = 10  # the adapter takes 12 messages: instructions + parts + customer message
KEY_FILE = Path(os.path.expanduser("~/.config/vinfinx/bundle.env"))
CUSTOMERS_PATH = Path(__file__).with_name("customers_v0.yaml")
STAGES = ("document", "ocr", "extraction", "avm", "policy", "bundle", "decision", "verifier")


class RunError(Exception):
    pass


# ---------------------------------------------------------------- inputs


def load_customers(path: Path = CUSTOMERS_PATH) -> dict[str, dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {c["customer_id"]: {**c, "synthetic": data["synthetic"], "file_version": data["version"]}
            for c in data["customers"]}


def set_dir(set_name: str, fixtures: Path = FIXTURES) -> Path:
    if set_name in HELD_OUT:
        raise RunError(f"{set_name} is held out and is never run")
    d = fixtures / set_name
    if not (d / "MANIFEST.json").exists():
        raise RunError(f"no document set {set_name}")
    return d


def documents(set_name: str, fixtures: Path = FIXTURES) -> list[dict]:
    manifest = json.loads((set_dir(set_name, fixtures) / "MANIFEST.json").read_text(encoding="utf-8"))
    return [{"document_id": d["document_id"], "variant": d["variant"], "language": d["language"],
             "params": d.get("params", {}), "pages": sorted(f for f in d["files"] if f.startswith("page-"))}
            for d in manifest["documents"]]


def load_keys() -> dict[str, bytes]:
    """The bundle signing key: environment first, else ~/.config/vinfinx/bundle.env (mode 600). Never logged."""
    key, key_id = os.environ.get("VINFINX_BUNDLE_KEY"), os.environ.get("VINFINX_BUNDLE_KEY_ID")
    if not key and KEY_FILE.exists():
        values = dict(line.split("=", 1) for line in KEY_FILE.read_text().split() if "=" in line)
        key, key_id = values.get("VINFINX_BUNDLE_KEY"), values.get("VINFINX_BUNDLE_KEY_ID")
    if not key or not key_id or len(bytes.fromhex(key)) < 32:
        raise RunError(f"no bundle signing key: set VINFINX_BUNDLE_KEY and VINFINX_BUNDLE_KEY_ID or write {KEY_FILE}")
    return {key_id: bytes.fromhex(key)}


# ---------------------------------------------------------------- stage helpers


def ocr_tesseract(images: list[Path], lang: str = "vie+eng") -> list[OcrPage]:
    out = subprocess.run([str(OCR_PYTHON), "-m", "ai.runner.ocr_pages", "--lang", lang, *map(str, images)],
                         capture_output=True, text=True, timeout=600, cwd=Path(__file__).resolve().parents[2])
    if out.returncode != 0:
        raise RunError(f"OCR failed: {out.stderr.strip()[-400:]}")
    return [OcrPage.model_validate(p) for p in json.loads(out.stdout)]


def qwen_model(messages: list[dict], schema: dict, max_tokens: int = 500) -> Generation:
    try:
        with urllib.request.urlopen(f"{QWEN}/health", timeout=5) as r:
            health = json.load(r)
    except Exception as exc:
        return Generation(text="", error=f"Qwen is not reachable ({type(exc).__name__}); start it (vinfinx-qwen)")
    if health.get("status") != "ready":
        return Generation(text="", error=f"Qwen is {health.get('status')}")
    return call_model(f"{QWEN}/v1/chat/completions", messages, schema, max_tokens=max_tokens)


def render_bundle(bundle: EvidenceBundle, catalogue: Catalogue) -> str:
    """The bundle as Qwen sees it. Like the evaluation's rendering, except that an item the decision may not cite
    (untrusted, stale or excluded by the policy) shows why instead of its value and baseline, and every item carries
    its catalogue name."""
    citable = set(citable_evidence_ids(bundle))
    excluded = {s.evidence_id: s.excluded_reason for t in bundle.policy.targets for s in t.signals if s.excluded_reason}
    items = []
    for i in bundle.items:
        d = catalogue.get(i.metric_id).definition
        row = {"evidence_id": i.evidence_id, "name": d.name_en, "period": i.observation_period.label,
               "value": None if i.value is None else str(i.value), "unit": i.unit.value,
               "baseline": None if i.baseline is None else f"{i.baseline.value} ({i.baseline.period_label})",
               "fresh_until": f"{i.freshness.deadline:%Y-%m-%d}", "source": i.source}
        if d.scope == "avm":
            row["role"] = "context: the customer's property"
        if i.evidence_id not in citable:
            why = ("untrusted" if i.trust != "trusted" else "stale" if i.freshness.deadline < bundle.created_at
                   else f"excluded: {excluded.get(i.evidence_id, '').lower()}")
            row.update(value=f"withheld ({why})", baseline=None)
        items.append(row)
    policy = [{"target": t.target.value, "outcome": t.outcome,
               "signals": [{"evidence_id": s.evidence_id, "effect": s.effect, "material": s.material,
                            **({"excluded": s.excluded_reason} if s.excluded_reason else {})} for s in t.signals],
               "allowed_actions": [a.value for a in t.allowed_actions], "magnitude_cap": t.magnitude_cap.value,
               "must_abstain": t.must_abstain} for t in bundle.policy.targets]
    return json.dumps({"evidence": items, "policy_result": {"mapping_id": bundle.mapping_id, "targets": policy}},
                      ensure_ascii=False)


def decision_messages(bundle: EvidenceBundle, catalogue: Catalogue) -> list[dict[str, str]]:
    """The Qwen adapter takes at most 4,000 characters per message and a real bundle is longer, so the input is split
    into parts, as the source agent splits a document (`MESSAGE_CHARS`)."""
    text = render_bundle(bundle, catalogue)
    parts = [text[i:i + MESSAGE_CHARS] for i in range(0, len(text), MESSAGE_CHARS)]
    if len(parts) > MAX_PARTS:
        raise RunError(f"the bundle needs {len(parts)} messages; the adapter takes at most 12")
    return ([{"role": "user", "content": f"{DECISION_INSTRUCTIONS} {RUNNER_NOTES}\n\nThe input follows in {len(parts)} part(s); "
                                         "answer after the customer message."}]
            + [{"role": "user", "content": f"Input part {i}/{len(parts)}:\n{p}"} for i, p in enumerate(parts, 1)]
            + [{"role": "user", "content": f"Customer message: {CUSTOMER_REQUEST}"}])


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _dump(items) -> list[dict]:
    return [i.model_dump(mode="json") for i in items]


# ---------------------------------------------------------------- the run


def run_pipeline(document_id: str, customer_id: str, *, set_name: str = DEMO_SET, fixtures: Path = FIXTURES,
                 runs_dir: Path = RUNS, ocr_fn: Callable = ocr_tesseract, model_fn: Callable = qwen_model,
                 keys: dict[str, bytes] | None = None, catalogue: Catalogue | None = None,
                 avm_path: Path = PHASE3_INTERVALS, now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                 mapping_fn: Callable | None = None,
                 on_update: Callable[[dict], None] | None = None) -> dict:
    catalogue = catalogue or load_catalogue()
    keys = keys or load_keys()
    if mapping_fn is None:
        from ..mapping import select_mapping
        mapping_fn = select_mapping
    customers = load_customers()
    if customer_id not in customers:
        raise RunError(f"unknown customer {customer_id}")
    customer = customers[customer_id]
    doc = next((d for d in documents(set_name, fixtures) if d["document_id"] == document_id), None)
    if doc is None:
        raise RunError(f"{document_id} is not in {set_name}")
    started = now()
    run_id = f"{started:%Y%m%dT%H%M%SZ}-{document_id}"
    out_dir = runs_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    record = {"schema_version": SCHEMA_VERSION, "run_id": run_id, "status": "running", "started_at": started.isoformat(),
              "finished_at": None, "customer": {k: customer[k] for k in ("customer_id", "display_name", "property_id",
                                                                         "district", "synthetic", "file_version")},
              "stages": [], "versions": {
                  "runner": RUNNER_VERSION, "code_commit": git_commit(), "catalogue": catalogue.version,
                  "catalogue_hash": catalogue.catalogue_hash, "extractor": EXTRACTOR_VERSION,
                  "verifier": VERIFIER_VERSION, "prompt": PROMPT_VERSION}}

    def save():
        tmp = out_dir / "run.json.tmp"
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        os.replace(tmp, out_dir / "run.json")
        if on_update:
            on_update(record)

    def stage(name: str, fn: Callable[[], dict]):
        t0 = time.time()
        entry = {"name": name, "status": "running"}
        record["stages"].append(entry)
        save()
        try:
            entry.update(fn() or {})
            entry["status"] = "ok"
        except Exception as exc:
            entry.update(status="error", error=f"{type(exc).__name__}: {exc}"[:600])
            raise
        finally:
            entry["seconds"] = round(time.time() - t0, 2)
            save()

    s: dict = {}  # values passed between stages
    try:
        def document():
            d = set_dir(set_name, fixtures) / document_id
            s["images"] = [d / p for p in doc["pages"]]
            s["pdf_hash"] = _sha((d / "document.pdf").read_bytes())
            return {"set": set_name, "document_id": document_id, "variant": doc["variant"], "synthetic": True,
                    "publisher": PUBLISHER, "pdf_sha256": s["pdf_hash"],
                    "pages": {p: _sha((d / p).read_bytes()) for p in doc["pages"]}}
        stage("document", document)

        def ocr():
            pages = ocr_fn(s["images"])
            s["pages"] = {p.page: p for p in pages}
            (out_dir / "ocr_pages.json").write_text(json.dumps(_dump(pages), ensure_ascii=False), encoding="utf-8")
            text = " ".join(w.text for w in pages[0].words) if pages else ""
            s["lang"] = detect_language(text) or "en"
            conf = [w.confidence for p in pages for w in p.words]
            return {"engine": pages[0].engine, "engine_version": pages[0].engine_version, "model": pages[0].model,
                    "pages": len(pages), "words": len(conf), "language": s["lang"],
                    "mean_confidence": round(sum(conf) / len(conf), 3) if conf else None,
                    "rotation_applied": {p.page: p.rotation_applied for p in pages}}
        stage("ocr", ocr)

        def extraction():
            ex = extract(s["pages"], lang=s["lang"], catalogue=catalogue, document_id=document_id,
                         document_hash=s["pdf_hash"], retrieved_at=now(), synthetic=True)
            s["items"] = ex.items
            return {"items": _dump(ex.items), "failures": ex.failures, "notes": ex.notes,
                    "publication_date": ex.publication_date, "trusted": sum(i.trust == "trusted" for i in ex.items),
                    "untrusted": sum(i.trust != "trusted" for i in ex.items)}
        stage("extraction", extraction)

        def avm():
            items = avm_items(customer["property_id"], catalogue, now(), avm_path)
            s["avm"] = items
            row = load_row(customer["property_id"], avm_path)
            return {"system": AVM_SYSTEM, "file": avm_path.name, "file_sha256": file_sha256(avm_path),
                    "property": {k: row[k] for k in ("property_id", "district_name", "ward_name", "total_area",
                                                      "number_of_floors", "last_modified_date", "flag",
                                                      "over_P90_pct", "time_adjusted")},
                    "items": _dump(items)}
        stage("avm", avm)

        def policy():
            config = load_policy_config(catalogue=catalogue)
            mapping, source, why = mapping_fn(config)  # MAPPING_AGENT_PLAN.md, M7
            s["policy"] = compute_policy(s["items"] + s["avm"], mapping, config, catalogue=catalogue,
                                         customer_district=customer["district"], case_id=run_id, now=now())
            record["versions"].update(policy=config.policy_version, policy_hash=config.policy_hash,
                                      policy_status=config.status, mapping_id=mapping.mapping_id,
                                      mapping_status=mapping.status, mapping_corpus=mapping.corpus_version,
                                      mapping_source=source)
            return {"result": s["policy"].model_dump(mode="json"), "mapping_provisional": source == "provisional",
                    "mapping_source": source, "mapping_reason": why}
        stage("policy", policy)

        def bundle():
            created = now()
            unsigned = EvidenceBundle(bundle_id=f"bundle.{run_id}", case_id=run_id, created_at=created,
                                      expires_at=created + BUNDLE_TTL, items=s["items"] + s["avm"], policy=s["policy"])
            key_id, key = next(iter(keys.items()))
            s["bundle"] = sign_bundle(unsigned, key, key_id)
            record["versions"]["bundle_hash"] = s["bundle"].bundle_hash
            return {"bundle": s["bundle"].model_dump(mode="json"), "bundle_hash": s["bundle"].bundle_hash,
                    "key_id": key_id}
        stage("bundle", bundle)

        def decision():
            messages = decision_messages(s["bundle"], catalogue)
            gen = model_fn(messages, generation_schema(s["bundle"], catalogue))
            s["raw"] = gen.text
            record["versions"].update(model="Qwen3.5-9B", precision=QWEN_PRECISION)
            out = {"model": "Qwen3.5-9B", "precision": QWEN_PRECISION, "prompt_version": PROMPT_VERSION,
                   "prompt_hash": _sha(json.dumps(messages, ensure_ascii=False).encode()), "raw": gen.text,
                   "prompt_tokens": gen.prompt_tokens, "completion_tokens": gen.completion_tokens,
                   "model_seconds": gen.seconds}
            if gen.error:
                raise RunError(gen.error)
            try:
                out["recommendation"] = json.loads(gen.text)
            except ValueError:
                out["recommendation"] = None  # the verifier blocks it
            return out
        stage("decision", decision)

        def verifier():
            verdict = verify_recommendation(s["raw"], s["bundle"], now=now(), keys=keys, catalogue=catalogue)
            record["verdict"] = verdict.decision
            return {"result": verdict.model_dump(mode="json")}
        stage("verifier", verifier)
        record["status"] = "done"
    except Exception:
        record["status"] = "failed"
    record["finished_at"] = now().isoformat()
    save()
    return record
