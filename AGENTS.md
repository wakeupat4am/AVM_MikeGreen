# Project context for agents

## What this repo is
An Automated Valuation Model (AVM) for Vietnamese landed houses (Phases 1–3), plus
VinFinx: a prototype financial-health app with an LLM layer. Read `README.md` for the
AVM, `prototype/vinfinx/README.md` for the prototype.

## My role and current work
I am the AI engineer. Scope: the macro-behavior decision engine, the conversational
assistant and OCR/document intelligence, all governed by
`prototype/vinfinx/AI_ENGINEERING_HANDOFF.md`. The core problem is hallucination control:
an LLM output must never become a trusted fact or decision on its own.

Read before doing anything:
1. `prototype/vinfinx/AI_ENGINEERING_HANDOFF.md`: requirements (source of truth)
2. `prototype/vinfinx/docs/PIPELINE.md`: the intended design
3. `prototype/vinfinx/docs/SERVER_SETUP_PLAN.md`: serving Qwen on this server
4. `prototype/AI_ENGINEER_FINDINGS.md`: background and dated decision log

Status: being built in the build order of `docs/PIPELINE.md` (steps 1–5 done; step 6 in progress). Details in
"Current state" below; progress tracker in `prototype/vinfinx/docs/README.md`.

## Hard rules
- Customer data, identity documents and bank records never go to OpenAI or any
  external API. Only public research documents and the synthetic macro document may.
- The Qwen service binds to 127.0.0.1 only. Access is through an SSH tunnel. Never
  open the port publicly.
- Never commit model weights, API keys, credentials, bank data or generated datasets.
  Keys come from environment variables.
- Synthetic data must be labelled synthetic (the macro-status document has a visible
  "SYNTHETIC, TEST DATA" watermark and a fictional publisher).
- Generation (source agent, mapping agent, decision engine) runs on Qwen3.5-9B on this server.
  OpenAI is used for embeddings, the verifier's support check and gold-label drafts.
  OCR is a pluggable service: Tesseract 5.5.3 (chosen 2026-10-21; PaddleOCR removed, it cannot read Vietnamese),
  Qwen as second reading (not wired in yet); see "OCR service" in `docs/PIPELINE.md`. Customer documents use local
  engines only.
- The LLM never decides what a macro signal means at decision time. That comes from a
  versioned mapping applied by deterministic code.
- Version and hash everything a decision depends on (corpus, index, mapping, prompt,
  model and precision, evidence), so decisions can be replayed.

## AVM facts to quote correctly
- Phase 3 real-time MedAPE is about 21%; band coverage 81.6–95.5%. Do not quote
  random-fold numbers (for example "18%").
- Phase 3 is Hà Nội only and models asking prices. It is a triage engine, not an
  appraisal.
- Phase outputs are raw model output, not macro data.

## Current state (update at the end of every work step)
*Last updated 2026-10-21.* Each pipeline piece follows one pattern: **the model proposes, deterministic code
verifies, a human approves.** Code is under `prototype/vinfinx/ai/`.

| Piece | Where | State and key facts |
|---|---|---|
| Qwen serving | `qwen_api.py`, `serve-qwen.sh` | Qwen3.5-9B bf16 @ `c202236…`, `transformers` 5.19.0, xgrammar 0.2.8, greedy, seed 7, port 8061 loopback. Started on demand in tmux `vinfinx-qwen` (runbook in `SERVER_SETUP_PLAN.md`) |
| Contracts | `catalog/`, `schemas/` | Catalogue v0.1.0 (provisional): 14 metric IDs, 19 reason codes (`<METRIC>_RISE/DROP` + `CONFLICTING_SIGNALS`, `STALE_EVIDENCE`, `INSUFFICIENT_EVIDENCE`); districts `hn-dong-da`, `hn-hai-ba-trung`, `hn-ba-dinh`. Bundles HMAC-signed. Decision output schema built per request: only citable evidence (trusted, fresh, not excluded), `anyOf` requiring ≥1 citation unless abstaining, ≤3 reason codes |
| Verifier v0.1.1 | `verifier/` | Recommendation and mapping verifiers, fail closed. General reason codes need bundle support; all bundle evidence IDs are stripped before numbers are read. Support check is a placeholder |
| Adversarial set | `evaluation/` | v0.1.0 **frozen** (42 cases, 11 `holdout`). Baseline 15/35; dev **21/24** after two fix rounds (prompts `eval_prompts/0.2`). Tuning paused; holdout run waits until step 6 changes the system |
| Policy engine | `policy/` | Built 2026-10-20. P1–P5 approved as **provisional** (`policy_v0.yaml`, no domain sign-off). Uses a hand-built synthetic mapping (`provisional_mapping.py`) until the mapping agent exists. Customer's district gates district metrics. Reproduces all 30 frozen policy results |
| Document generator | `documents/` | v0.1.0, every spec variant EN+VI, exact ground truth. Sets on the server: `docs-v0.1-draft` (seed 42), `docs-v0.1-holdout-s7` (seed 7, held out); 62 docs each; not frozen |
| OCR | `ocr/` | Tesseract 5.5.3, `tessdata_best@e12c65a`, `vie+eng`, psm 3, OSD orientation. Runs in `env-ocr` |
| Extraction + validation | `extraction/` | `macro_doc_extractor/0.1`. Trusted only if every check passes (strict per-language numbers, Change = Value − Baseline, ranges, units/labels vs catalogue). **0 wrong-and-trusted** on both sets (868 items each; 590/579 trusted and correct). Results: `docs/OCR_RESULTS.md` |
| OCR confidence threshold | `docs/OCR_CONFIDENCE_THRESHOLD.md` | Proposal (no minimum for bulletin values) **awaiting the risk owner** |
| Qwen second reading, research corpus, chunker, indexer, mapping agent, support check, mapping registry, online pipeline runner, audit log, cost logging | | **Not built** |

**Next (agreed direction, 2026-10-21):** the offline research half, in this order:
1. **Source agent** (PIPELINE.md offline step 1): Qwen proposes sources against `docs/CORPUS_SOURCE_CRITERIA.md`
   (C1–C7). Qwen has no web access, so it must be given search tools (scholarly/institutional APIs such as OpenAlex,
   World Bank, IMF, ADB, BIS, SBV, GSO; optionally a web-search API). Qwen judges relevance and geography with quotes
   checked verbatim; code checks URL, download, hash, licence, language and date; a human signs off.
   **Do not hand-draft the source list yourself**: the agent finds it.
2. Ingestion (fetch, hash, store original, extract text; OCR for scanned PDFs) → chunker (page + character offsets)
   → indexer (OpenAI embeddings + keyword index) → freeze with version ID and hash → retrieval evaluation (recall@k on gold passages).
3. Mapping agent → mapping verifier with the OpenAI support check → mapping registry replacing the provisional mapping
   → online pipeline runner (document → evidence → policy → decision → verifier).

**Waiting on the user:** search backend for the source agent (free APIs only, or add a paid web search); an
`OPENAI_API_KEY` on the server (not set); who signs off the source list; embedding model and index storage;
risk-owner answer on the OCR threshold. **Later:** Qwen second reading, freeze a document set, holdout run of the
adversarial set, re-measure generation speed, cost accounting (handoff §9).

## Server and environment
- Host `vishc-server-1`: 8 × RTX 3090 24 GiB, shared with other users, 64 CPU threads. Use any GPU with ~19 GiB free
  and record its index with each run. CPU contention from other users slows both Qwen and OCR.
- `VINFINX_ROOT=/mnt/disk1/backup_user/24chuong.ta/vinfinx`: `env/` (main), `env-ocr/` (Tesseract), `models/`,
  `fixtures/`, `evaluation/{runs,ocr,extraction}/`, `corpus/` (empty), `logs/`, `cache/`, lock files. Nothing there is committed.
- **Disk:** our allowance on `/mnt/disk1` is 400 GB, but the shared disk is often almost full (230 MB free on
  2026-10-21). Check `df -h /mnt/disk1` before writing anything large.
- Record every install command and pinned version in `docs/SERVER_SETUP_PLAN.md`.
- Run long jobs in tmux (`vinfinx-qwen`, `vinfinx-eval`, `vinfinx-ocr`) with logs under `$VINFINX_ROOT/logs/`: the
  user may close the window.
- The login shell is zsh: unquoted variables are not word-split, and `=word` at the start of an argument expands.
  Use explicit commands and literal paths (or `"${VAR:?}"`) for `rm`.

## Known issues
- `qwen_api.py` no longer carries the demo context (removed 2026-10-19), so the web demo's chat no longer
  knows the demo customer's figures; the app must send them in the conversation if needed.
- Generation has run at ~3 tokens/s since 2026-10-19 (15–20 on 2026-10-18); cause not confirmed
  (likely CPU contention). Do not quote latency until re-measured.
- OCR runs in its own environment (`$VINFINX_ROOT/env-ocr`: Tesseract); run `ai.ocr.*` and `ai.extraction.evaluate`
  with `env-ocr/bin/python`. Tests that need Tesseract are skipped in the main env; run them in `env-ocr`.
- Tesseract always reads "±" as "+", so uncertainty intervals are dropped (flagged), never extracted.
- Load Qwen with `device_map="auto"`: an explicit single device (`{"": 0}`) runs ~7× slower.
- xgrammar enforces `maxItems` and `anyOf` but ignores `uniqueItems`; `maxLength` cuts text mid-word.
- Run Python from the VinFinx environment (`/mnt/disk1/backup_user/24chuong.ta/vinfinx/env`); tests run
  from `prototype/vinfinx` with `python -m unittest discover -s ai/tests -t .` (114 pass, 2 skipped, 2026-10-21).
- Phase 1 processed data and the `final_scorer.joblib` model are gitignored, so Phase 1
  scripts will not run from a fresh clone.

## Lessons learned
- Every value a generation schema offers needs a verifier rule saying when it is supported: when one route is
  closed, the model moves to the next unchecked field (Qwen used `CONFLICTING_SIGNALS` as filler).
- Fix bugs in the measuring instrument (verifier, scorer, generator) before measuring; measure model, prompt
  and schema changes afterwards, so their effect is visible.
- OCR confidence does not protect against wrong values (Tesseract is overconfident on degraded pages); format and
  arithmetic cross-checks do.
- After generating a set, check that the document count matches the manifest (a document-ID collision once
  silently overwrote documents).

## Working style
- The user commits the work themselves; do not commit.
- Give a recommendation, not a survey of options. Propose a plan and get approval before building a new component.
- Don't modify Phase 1–3 code or data unless asked.
- New AI-layer code goes under `prototype/vinfinx/`; new plans go in `prototype/vinfinx/docs/`.
- Record decisions as a dated entry in `prototype/AI_ENGINEER_FINDINGS.md`, update the tracker in
  `prototype/vinfinx/docs/README.md`, and update "Current state" above at the end of each step.
- Ask before choosing anything listed under "Open" in `docs/PIPELINE.md`.
- Never tune the verifier or prompts on the adversarial set's `holdout` cases, and never edit a frozen
  case set; changes go into a new version. The same applies to the held-out document set (`docs-v0.1-holdout-s7`).
