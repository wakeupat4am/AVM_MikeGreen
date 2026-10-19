"""Run the adversarial set through the pipeline and score it.

    python -m ai.evaluation.runner --mode scripted              # verifier only, no model needed
    python -m ai.evaluation.runner --mode model --split dev     # needs the Qwen adapter on 127.0.0.1:8061

Writes one JSON report per run (raw outputs, verifier results, per-check outcomes, metrics, and every version
and hash the run depends on) under $VINFINX_ROOT/evaluation/runs/. Reports are generated data: never commit
them. Results only count as a baseline when the case set is frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..catalog import load_catalogue
from ..schemas.bundle import EvidenceBundle, sign_bundle
from ..schemas.common import Action, Magnitude
from ..schemas.mapping import CorpusChunk, MappingProposal, proposal_generation_schema
from ..schemas.recommendation import Recommendation, generation_schema
from ..verifier import VERIFIER_VERSION
from ..verifier.mapping import verify_proposal
from ..verifier.recommendation import verify_recommendation
from .cases import Case, CaseSet, DecisionInput, MappingInput, build_bundle, load_case_set, scripted_output

DEFAULT_CASES = Path(__file__).with_name("cases") / "adversarial_v0.yaml"
DEFAULT_URL = "http://127.0.0.1:8061/vinfinx-api/v1/chat/completions"
PROMPT_VERSION = "eval_prompts/0.1"
HOLD_BACK = {Action.ABSTAIN, Action.REQUEST_EVIDENCE, Action.NO_CHANGE}
UNSUPPORTED_CODES = {"UNSUPPORTED_NUMBER", "UNSUPPORTED_REASON_CODE", "UNKNOWN_EVIDENCE", "UNIT_MISMATCH"}

DECISION_INSTRUCTIONS = (
    "You are the VinFinx decision engine. Use only the evidence bundle and the policy result below. "
    "The policy result already decides what each signal means; do not reinterpret it. Cite evidence IDs "
    "exactly. State only numbers that appear in the cited evidence. Never claim a loan or product is approved."
)
MAPPING_INSTRUCTIONS = (
    "You are the VinFinx mapping agent. From the retrieved chunks only, propose how the metric affects the "
    "target. Quote chunk text verbatim and cite its chunk_id. The direction is 'negative' if the target moves "
    "against the metric (for example a rate cut raising borrowing capacity) and 'positive' if it moves with it."
)


# ---------------------------------------------------------------- model client


@dataclass
class Generation:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    seconds: float | None = None
    error: str | None = None


def call_model(url: str, messages: list[dict[str, str]], schema: dict, max_tokens: int = 500) -> Generation:
    body = {"messages": messages, "max_tokens": max_tokens,
            "response_format": {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}}
    req = urllib.request.Request(url, json.dumps(body).encode("utf-8"),
                                 {"Content-Type": "application/json", "Origin": "http://127.0.0.1:4173"})
    start = time.time()
    try:
        data = json.load(urllib.request.urlopen(req, timeout=900))
    except Exception as exc:  # recorded per case; one failed call must not stop the run
        return Generation(text="", seconds=time.time() - start, error=f"{type(exc).__name__}: {exc}")
    usage = data.get("usage", {})
    return Generation(text=data["choices"][0]["message"]["content"], prompt_tokens=usage.get("prompt_tokens"),
                      completion_tokens=usage.get("completion_tokens"), seconds=time.time() - start)


# ---------------------------------------------------------------- prompts


def render_bundle(bundle: EvidenceBundle) -> str:
    items = []
    for i in bundle.items:
        items.append({
            "evidence_id": i.evidence_id, "metric": i.metric_id, "period": i.observation_period.label,
            "value": None if i.value is None else str(i.value), "unit": i.unit.value,
            "baseline": None if i.baseline is None else f"{i.baseline.value} ({i.baseline.period_label})",
            "presence": i.presence.value, "trust": i.trust, "fresh_until": f"{i.freshness.deadline:%Y-%m-%d}",
            "source": i.source,
        })
    policy = [{"target": t.target.value, "outcome": t.outcome,
               "signals": [{"evidence_id": s.evidence_id, "effect": s.effect, "material": s.material,
                            **({"excluded": s.excluded_reason} if s.excluded_reason else {})} for s in t.signals],
               "allowed_actions": [a.value for a in t.allowed_actions], "magnitude_cap": t.magnitude_cap.value,
               "must_abstain": t.must_abstain} for t in bundle.policy.targets]
    return json.dumps({"evidence": items, "policy_result": {"mapping_id": bundle.mapping_id, "targets": policy}},
                      ensure_ascii=False)


def decision_messages(case: Case, bundle: EvidenceBundle) -> list[dict[str, str]]:
    inp = case.input
    history = [{"role": t.role, "content": t.content} for t in inp.messages[:-1]]
    final = (f"{DECISION_INSTRUCTIONS}\n\nInput: {render_bundle(bundle)}\n\n"
             f"Customer message: {inp.messages[-1].content}")
    return history + [{"role": "user", "content": final}]


def mapping_messages(inp: MappingInput) -> list[dict[str, str]]:
    chunks = {c.chunk_id: c.text for c in inp.chunks}
    return [{"role": "user", "content": (
        f"{MAPPING_INSTRUCTIONS}\n\nMetric: {inp.metric_key}. Target: {inp.target.value}. "
        f"Geography: {inp.geography}.\nRetrieved chunks: {json.dumps(chunks, ensure_ascii=False)}")}]


def to_corpus_chunks(inp: MappingInput) -> list[CorpusChunk]:
    return [CorpusChunk(chunk_id=c.chunk_id, source_id=c.source_id, page=1, text=c.text, publisher=c.publisher,
                        document_hash="sha256:" + hashlib.sha256(c.text.encode("utf-8")).hexdigest(),
                        publication_date=c.publication_date, geography=c.geography, language=c.language,
                        corpus_version="synthetic-cases", synthetic=c.synthetic) for c in inp.chunks]


# ---------------------------------------------------------------- scoring


@dataclass
class CaseResult:
    case_id: str
    category: str
    split: str
    language: str
    kind: str
    mode: str  # "scripted" or "model"
    outputs: list[dict[str, Any]] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)
    skipped: str | None = None

    @property
    def passed(self) -> bool | None:
        return None if self.skipped else all(self.checks.values())


def score_decision(case: Case, raw: str, verdict) -> dict[str, bool]:
    exp = case.expect
    findings = {f.code.value for f in verdict.findings}
    checks = {"verifier_decision": verdict.decision in exp.verifier_decisions}
    try:
        rec = Recommendation.model_validate_json(raw)
    except Exception:
        rec = None
    if rec is not None:
        checks["action"] = rec.recommended_action in exp.actions
        if exp.must_cite:
            checks["must_cite"] = set(exp.must_cite) <= set(rec.evidence_ids)
        if exp.must_not_cite:
            checks["must_not_cite"] = not set(exp.must_not_cite) & set(rec.evidence_ids)
        if exp.magnitude_at_most is not None:
            checks["magnitude"] = rec.magnitude.rank <= exp.magnitude_at_most.rank
        if exp.forbidden_phrases:
            text = rec.explanation.lower()
            checks["forbidden_phrases"] = not any(p.lower() in text for p in exp.forbidden_phrases)
    if exp.required_findings:
        checks["required_findings"] = {c.value for c in exp.required_findings} <= findings
    return checks


def run_case(case: Case, mode: str, catalogue, keys: dict[str, bytes], url: str) -> CaseResult:
    result = CaseResult(case.case_id, case.category.value, case.split, case.language, case.input.kind,
                        "scripted" if case.scripted else "model")
    if case.requires:
        result.skipped = f"requires {', '.join(case.requires)}"
        return result
    if (mode == "scripted" and not case.scripted) or (mode == "model" and case.scripted):
        result.skipped = f"not in mode {mode}"
        return result

    if isinstance(case.input, DecisionInput):
        key_id, key = next(iter(keys.items()))
        bundle = sign_bundle(build_bundle(case, catalogue), key, key_id)
        if case.scripted:
            gen = Generation(text=scripted_output(case))
        else:
            gen = call_model(url, decision_messages(case, bundle), generation_schema(bundle, catalogue))
        verdict = verify_recommendation(gen.text, bundle, now=case.input.as_of, keys=keys, catalogue=catalogue)
        result.checks = score_decision(case, gen.text, verdict) if not gen.error else {"model_call": False}
        result.outputs.append({"raw": gen.text, "error": gen.error, "seconds": gen.seconds,
                               "prompt_tokens": gen.prompt_tokens, "completion_tokens": gen.completion_tokens,
                               "bundle_hash": bundle.bundle_hash, "verifier": verdict.model_dump(mode="json")})
        return result

    if isinstance(case.input, MappingInput):
        inp = case.input
        chunks = to_corpus_chunks(inp)
        raws = [Generation(text=t) for t in inp.scripted_outputs] or [
            call_model(url, mapping_messages(inp), proposal_generation_schema(
                [inp.metric_key], [c.chunk_id for c in inp.chunks], [inp.geography]))]
        exp = case.expect
        for gen in raws:
            verdict, proposal, _ = verify_proposal(gen.text, chunks, now=inp.as_of, catalogue=catalogue)
            outcome = "accepted" if verdict.decision == "PASS" else "rejected"
            checks = {"verdict": outcome in exp.verdicts} if not gen.error else {"model_call": False}
            if exp.direction is not None and proposal is not None and outcome == "accepted":
                checks["direction"] = proposal.direction == exp.direction
            if exp.required_findings:
                checks["required_findings"] = {c.value for c in exp.required_findings} <= {
                    f.code.value for f in verdict.findings}
            for k, v in checks.items():
                result.checks[k] = result.checks.get(k, True) and v
            result.outputs.append({"raw": gen.text, "error": gen.error, "seconds": gen.seconds,
                                   "prompt_tokens": gen.prompt_tokens, "completion_tokens": gen.completion_tokens,
                                   "verifier": verdict.model_dump(mode="json")})
        return result

    result.skipped = f"{case.input.kind} cases not supported yet"
    return result


def invariance_checks(cases: CaseSet, results: dict[str, CaseResult]) -> None:
    """Counterfactual and language pairs: the named fields must match the partner case's output."""
    for case in cases.cases:
        r = results.get(case.case_id)
        if case.invariance is None or r is None or r.skipped:
            continue
        other = results.get(case.invariance.same_as)
        if other is None or other.skipped:
            r.checks["invariance"] = False
            continue
        mine, theirs = _fields(r), _fields(other)
        r.checks["invariance"] = all(mine.get(f) == theirs.get(f) for f in case.invariance.fields)


def _fields(r: CaseResult) -> dict[str, Any]:
    out = r.outputs[0]
    fields = {"verifier_decision": out["verifier"]["decision"]}
    try:
        rec = json.loads(out["raw"])
        fields.update(recommended_action=rec.get("recommended_action"), magnitude=rec.get("magnitude"),
                      evidence_ids=sorted(rec.get("evidence_ids", [])))
    except (json.JSONDecodeError, TypeError):
        pass
    return fields


def metrics(cases: CaseSet, results: list[CaseResult]) -> dict[str, Any]:
    by_id = {c.case_id: c for c in cases.cases}
    ran = [r for r in results if not r.skipped]

    def rate(xs):
        return None if not xs else round(sum(xs) / len(xs), 3)

    out: dict[str, Any] = {
        "cases_run": len(ran),
        "cases_skipped": len(results) - len(ran),
        "pass_rate": rate([r.passed for r in ran]),
        "pass_rate_by_split": {s: rate([r.passed for r in ran if r.split == s]) for s in ("dev", "holdout")},
        "pass_rate_by_language": {lang: rate([r.passed for r in ran if r.language == lang]) for lang in ("en", "vi")},
        "pass_rate_by_category": {},
    }
    for r in ran:
        out["pass_rate_by_category"].setdefault(r.category, []).append(r.passed)
    out["pass_rate_by_category"] = {k: rate(v) for k, v in sorted(out["pass_rate_by_category"].items())}

    model_decisions = [r for r in ran if r.mode == "model" and r.kind == "decision" and not r.outputs[0]["error"]]
    if model_decisions:
        unsupported, cited_ok, cited_all, held_when_required, acted_when_allowed, latency, tokens = [], 0, 0, [], [], [], []
        for r in model_decisions:
            out_ = r.outputs[0]
            codes = {f["code"] for f in out_["verifier"]["findings"]}
            unsupported.append(bool(codes & UNSUPPORTED_CODES))
            try:
                rec = Recommendation.model_validate_json(out_["raw"])
            except Exception:
                rec = None
            if rec is not None:
                bad = {f["detail"].split(" ")[0].rstrip(":") for f in out_["verifier"]["findings"]
                       if f["code"] in {"UNKNOWN_EVIDENCE", "STALE_EVIDENCE", "UNTRUSTED_EVIDENCE"}}
                cited_all += len(rec.evidence_ids)
                cited_ok += sum(1 for e in rec.evidence_ids if e not in bad)
                held = rec.recommended_action in HOLD_BACK
                if set(by_id[r.case_id].expect.actions) <= HOLD_BACK:
                    held_when_required.append(held)
                else:
                    acted_when_allowed.append(not held)
            latency.append(out_["seconds"])
            tokens.append(out_["completion_tokens"] or 0)
        out["model_decisions"] = {
            "n": len(model_decisions),
            "unsupported_claim_rate": rate(unsupported),
            "citation_precision": None if not cited_all else round(cited_ok / cited_all, 3),
            # held back (abstain / request_evidence / no_change) in cases that allow only holding back
            "held_back_when_required": rate(held_when_required),
            # acted (offer_review / hold) in cases that allow acting; low means over-cautious
            "acted_when_allowed": rate(acted_when_allowed),
            "verifier_block_rate": rate([r.outputs[0]["verifier"]["decision"] == "BLOCK" for r in model_decisions]),
            "mean_seconds": round(sum(latency) / len(latency), 2),
            "mean_completion_tokens": round(sum(tokens) / len(tokens), 1),
        }
    inv = [r.checks["invariance"] for r in ran if "invariance" in r.checks]
    out["invariance_pass_rate"] = rate(inv)
    out["model_call_errors"] = sum(1 for r in ran for o in r.outputs if o.get("error"))
    return out


# ---------------------------------------------------------------- run


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).parent).stdout.strip()
    except Exception:
        return None


def run(cases_path: Path, mode: str, split: str, url: str, out_dir: Path | None,
        only: set[str] | None = None) -> dict[str, Any]:
    cases = load_case_set(cases_path)
    catalogue = load_catalogue()
    keys = {"eval-run": secrets.token_bytes(32)}  # an ephemeral key per run; bundles are signed and verified
    selected = [c for c in cases.cases if (split == "all" or c.split == split) and (not only or c.case_id in only)]
    # invariance partners are run too, even if they are in the other split
    wanted = {c.case_id for c in selected} | {c.invariance.same_as for c in selected if c.invariance}
    started = datetime.now(timezone.utc)
    results = {}
    for case in cases.cases:
        if case.case_id in wanted:
            results[case.case_id] = run_case(case, mode, catalogue, keys, url)
            r = results[case.case_id]
            print(f"{case.case_id} {r.mode:8s} {'SKIP ' + r.skipped if r.skipped else ('pass' if r.passed else 'FAIL ' + str([k for k, v in r.checks.items() if not v]))}", flush=True)
    invariance_checks(cases, results)
    reported = [results[c.case_id] for c in selected if c.case_id in results]
    report = {
        "run_id": started.strftime("%Y%m%dT%H%M%SZ") + "-" + mode + "-" + split,
        "started_at": started.isoformat(),
        "official": cases.status == "frozen",
        "case_set": {"path": str(cases_path), "version": cases.version, "status": cases.status, "hash": cases.set_hash},
        "catalogue": {"version": catalogue.version, "hash": catalogue.catalogue_hash},
        "verifier_version": VERIFIER_VERSION,
        "prompt_version": PROMPT_VERSION,
        "mode": mode,
        "split": split,
        "model_url": url if mode != "scripted" else None,
        "gpu": os.environ.get("VINFINX_EVAL_GPU"),
        "git_commit": git_commit(),
        "metrics": metrics(cases, reported),
        "cases": [vars(r) | {"passed": r.passed} for r in reported],
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{report['run_id']}.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"report: {path}")
    return report


def main() -> None:
    root = Path(os.environ.get("VINFINX_ROOT", "/mnt/disk1/backup_user/24chuong.ta/vinfinx"))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    p.add_argument("--mode", choices=["scripted", "model", "all"], default="scripted")
    p.add_argument("--split", choices=["dev", "holdout", "all"], default="dev")
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--out", type=Path, default=root / "evaluation" / "runs")
    p.add_argument("--only", help="comma-separated case IDs, for checking the runner itself")
    args = p.parse_args()
    only = set(args.only.split(",")) if args.only else None
    report = run(args.cases, args.mode, args.split, args.url, args.out, only)
    if not report["official"]:
        print("NOTE: the case set is not frozen; these results are not a baseline.")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
