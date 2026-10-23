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
*Last updated 2026-10-23.* Each pipeline piece follows one pattern: **the model proposes, deterministic code
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
| Source agent | `sources/`, `schemas/source.py` | Built 2026-10-22 (S1–S7 approved). First run `20261022-sources-1` (prompt `source_agent_prompts/0.3`): **14 accepted, 9 unclear, 294 rejected** after the user decided VinFinx is **non-commercial use** (2026-10-22: World Bank and BIS terms pass C2; IMF and SBV terms unread). Draft list `corpus/source_lists/20261022-sources-1-draft.json`; source IDs stable per run (`ids.json`). **Signed 2026-10-23 in the site:** `source_list/v0.1` (`corpus/source_lists/source_list-v0.1.json`, hash `f8d97c9f…`): 14 accepted (the agent's 14), 303 rejected (incl. all 9 unclear), 0 overruled |
| Corpus + index | `corpus/`, `schemas/corpus.py` | Built 2026-10-22 (I1–I7). Dev build `corpus/versions/corpus-dev` (17 docs, 1,633 chunks, `text-embedding-3-large`); search `python -m ai.corpus.index search --version corpus-dev "<query>"`. Broken Vietnamese text layers are OCR'd. OpenAI calls logged with cost in `/mnt/sdb/shared/chuong/vinfinx/logs/openai_calls.jsonl` (price table `openai-pricing-2026-10-22` in `openai_client.py`). Gold labels drafted with `gpt-5-mini-2025-08-07` (user's choice after gpt-5-nano labelled nothing; $0.12/run): 91 gold passages, 22/24 questions; review sample `corpus-dev/gold/review.csv` (189 rows) **awaiting the user**. Draft retrieval: vector recall@8 0.40 > hybrid 0.34 > keyword 0.25; keyword fix (stopwords, weighted fusion) waits for the reviewed labels. **Frozen `corpus-v0.1` + `index-v0.1` built 2026-10-23** from `source_list/v0.1`: 14 documents, 452 pages (30 OCR'd), 1,452 chunks; all embeddings from the cache ($0). No gold labels for v0.1 yet (the 189-row review is on `corpus-dev`; chunk IDs of the 14 shared documents match) |
| Pipeline API + Evidence sources view | `pipeline_api.py`, `serve-pipeline.sh`, `ai/api/service.py`, `evidence.js` | Built 2026-10-23 (A1–A7 approved; `docs/PIPELINE_API_PLAN.md`). CPU-only FastAPI on 127.0.0.1:**8062** (`/vinfinx-pipeline`), tmux `vinfinx-api`, log `/mnt/sdb/shared/chuong/vinfinx/logs/pipeline_api.log`; the bridge forwards 8061 and 8062. Authenticator → Evidence sources (tabs: Source list, Gold labels, Corpus). Sign-off and gold submit call the CLI's functions; progress saved per row (`review_progress.json`); writes logged in `logs/review_actions.jsonl`. Search by meaning allowed (user, 2026-10-23) behind a personal-data guard (`check_query`); sent queries logged in `logs/search_queries.jsonl`. The agent's pre-filled `reject` rows in `review.csv` are **not** treated as decisions. Writes are serialised (`_WRITE` lock): two parallel saves once dropped a decision (2026-10-23); the page reloads from the server after any failed write. Corpus tab has a version picker (defaults to the newest frozen version). Browser-tested in headless Chromium (`env-webtest`) on a scratch corpus copy; **not yet tried from the Mac through the tunnel** |
| Qwen second reading, mapping agent, support check, mapping registry, online pipeline runner, audit log | | **Not built** |

**Next (agreed direction, 2026-10-21):** the offline research half, in this order:
1. **Source agent** (PIPELINE.md offline step 1): Qwen proposes sources against `docs/CORPUS_SOURCE_CRITERIA.md`
   (C1–C7). Qwen has no web access, so code gives it search tools. **Decided 2026-10-21: free scholarly and
   institutional sources only** (no paid web search): OpenAlex, World Bank Documents & Reports API, and IMF, ADB, BIS,
   SBV and statistics-office (`nso.gov.vn`) publications. IMF and ADB web pages block scripts (403), so they are found
   through OpenAlex (DOI prefixes `10.5089`, `10.22617`); IMF PDFs need a person to download them; ADB PDFs download.
   Access facts: `SERVER_SETUP_PLAN.md`, "Source access check". Qwen judges relevance and geography with quotes
   checked verbatim; code checks URL, download, hash, licence, language and date; **jimmytrn154 signs off the source
   list** (decided 2026-10-21). **Do not hand-draft the source list yourself**: the agent finds it. Respect
   `robots.txt`; never get around bot protection. Contact address for OpenAlex `mailto=` and the User-Agent:
   `jimmytrn154@gmail.com` (given by the user); send no other personal data. PDF library: `pypdfium2` 5.14.0
   (installed; not PyMuPDF, AGPL). **Plan: `docs/SOURCE_AGENT_PLAN.md` (S1–S7), awaiting approval; build after it.**
2. Ingestion (fetch, hash, store original, extract text; OCR for scanned PDFs) → chunker (page + character offsets)
   → indexer (OpenAI embeddings + keyword index) → freeze with version ID and hash → retrieval evaluation (recall@k on gold passages).
   **Plan: `docs/CORPUS_INDEX_PLAN.md` (I1–I7), approved 2026-10-22; being built.** OpenAI is called via REST
   with `requests` (no `openai` package: `/mnt/disk1` is full); key in `~/.config/vinfinx/openai.env`. The user has
   OpenAI credits (2026-10-22). Also agreed: an end-to-end runner (document → evidence → policy → decision →
   verifier) on the provisional mapping can be built before the research half is finished, labelled provisional.
   The runner **also takes the Phase 3 AVM evidence** for the customer's property (user's decision 2026-10-22).
   Agreed order: (1) corpus + index on the dev corpus → (2) pipeline API + Evidence sources view (sign v0.1 in the
   site) → (3) end-to-end runner with AVM evidence, wired into the site → (4) mapping agent + support check →
   (5) second source run, signed live in the demo as v0.2.
3. Mapping agent → mapping verifier with the OpenAI support check → mapping registry replacing the provisional mapping
   → online pipeline runner (document → evidence → policy → decision → verifier).

**Waiting on the user:** the 189-row gold review (Authenticator → Evidence sources → Gold labels; then the keyword-search
fix and gold for `corpus-v0.1`). Source list v0.1 is signed and `corpus-v0.1` frozen (2026-10-23). **Next build step: step 3**
(end-to-end runner with AVM evidence, wired into the site); write its plan first. The site works from the Mac through
the tunnel (SSH alias `vishc-server-1`, over NetBird). The OpenAI key is in `~/.config/vinfinx/openai.env` (mode 600), validated 2026-10-22:
`text-embedding-3-large` works (3,072 dims); request `encoding_format: base64` (a plain-JSON response once stalled
past a 60 s read timeout). Optionally an `OPENALEX_API_KEY` (without one, OpenAlex allows $0.10/day
shared by the whole server IP; a run costs ~$0.05); embedding model and index storage;
risk-owner answer on the OCR threshold. **Later:** Qwen second reading, freeze a document set, holdout run of the
adversarial set, re-measure generation speed, cost accounting (handoff §9).

## Demo website (the user's requirement, 2026-10-22)
The demo is the website `prototype/vinfinx/index.html`: one static page (served on the Mac) with three roles,
Customer, Advisor and Authenticator, that reaches the server only through the SSH tunnel to loopback
(`AI_BASE = http://127.0.0.1:8061/vinfinx-api`; `qwen_api.py` checks the `Origin`; `PIPELINE_BASE =
http://127.0.0.1:8062/vinfinx-pipeline` in `evidence.js` for pipeline results). The site is served on the Mac at
`http://127.0.0.1:4173` (an allowed origin). To test it on the server, use `env-webtest` (Playwright) against a scratch
copy of the corpus, never the real one. **Build every pipeline step so
its results can be shown there**:
- Every component writes machine-readable results (JSON with versions and hashes) that a loopback API can serve;
  no result exists only in a CSV or a terminal.
- Where each result goes in the site:
  - Authenticator → **Evidence sources**: source-list review and sign-off (live in the demo, as a new version such as
    v0.2; v0.1 is signed before), corpus and mappings with their verified quotes.
  - Authenticator → **Market signals**: evidence items from the OCR'd bulletin, with page boxes.
  - Authenticator → **Review queue**: decisions the verifier routes to a human.
  - Authenticator → **Audit log**: versions and hashes behind each decision.
  - Customer view: the recommendation with source labels and timestamps (today its figures are hardcoded).
- Writes from the site (sign-off, review actions) are explicit, carry the reviewer's name and run the same checks as
  the command line (`ai.sources.review`). The server stays loopback-only; no new public port.
- New pages may be added beside `index.html` when a view does not fit it (the user's permission, 2026-10-22); link
  them from the existing navigation.
- Match the page's existing style (CSS variables, cards, `Manrope`/`DM Sans`). The Authenticator's navigation switches
  views (`showAuthView` in `evidence.js`); Evidence sources is live, the other three are still the static mock-up.

## Server and environment
- Host `vishc-server-1`: 8 × RTX 3090 24 GiB, shared with other users, 64 CPU threads. Use any GPU with ~19 GiB free
  and record its index with each run. CPU contention from other users slows both Qwen and OCR.
- `VINFINX_ROOT=/mnt/disk1/backup_user/24chuong.ta/vinfinx`: `env/` (main), `env-ocr/` (Tesseract), `models/`,
  `fixtures/`, `evaluation/{runs,ocr,extraction}/`, `logs/`, `cache/`, lock files. Nothing there is committed.
- **Research corpus:** `/mnt/sdb/shared/chuong/vinfinx/corpus/` (second disk, 1.2 TB free; `vinfinx/` is mode 700),
  reached as `$VINFINX_ROOT/corpus` through a symlink. The rest of `/mnt/sdb/shared/chuong/` is another project's data: don't touch it.
- **Disk:** both disks are shared and swing widely. `/mnt/disk1` (NVMe SSD; our allowance 400 GB) went from 0 bytes
  free to 1.1 TB free on 2026-10-22; `/mnt/sdb` (spinning HDD) from 1.2 TB to 703 GB the same day. Check `df -h`
  before writing anything large. Keep environments and model weights on `/mnt/disk1` (fast loading); the corpus
  lives on `/mnt/sdb`; if `/mnt/disk1` fills again, logs and new data go to `/mnt/sdb/shared/chuong/vinfinx/`.
- Record every install command and pinned version in `docs/SERVER_SETUP_PLAN.md`.
- Run long jobs in tmux (`vinfinx-qwen`, `vinfinx-eval`, `vinfinx-ocr`) with logs under `$VINFINX_ROOT/logs/`: the
  user may close the window.
- The login shell is zsh: unquoted variables are not word-split, and `=word` at the start of an argument expands.
  Use explicit commands and literal paths (or `"${VAR:?}"`) for `rm`.

## Known issues
- `qwen_api.py` no longer carries the demo context (removed 2026-10-19), so the web demo's chat no longer
  knows the demo customer's figures; the app must send them in the conversation if needed.
- Generation speed depends on server load: ~3 tokens/s 2026-10-19 to 10-21 under heavy CPU contention, ~22 on
  2026-10-22 at load ~11 (GPU 0). Quote latency only with the load and date it was measured at.
- OCR runs in its own environment (`$VINFINX_ROOT/env-ocr`: Tesseract); run `ai.ocr.*` and `ai.extraction.evaluate`
  with `env-ocr/bin/python`. Tests that need Tesseract are skipped in the main env; run them in `env-ocr`.
- Tesseract always reads "±" as "+", so uncertainty intervals are dropped (flagged), never extracted.
- Load Qwen with `device_map="auto"`: an explicit single device (`{"": 0}`) runs ~7× slower.
- xgrammar enforces `maxItems` and `anyOf` but ignores `uniqueItems`; `maxLength` cuts text mid-word.
- Run Python from the VinFinx environment (`/mnt/disk1/backup_user/24chuong.ta/vinfinx/env`); tests run
  from `prototype/vinfinx` with `python -m unittest discover -s ai/tests -t .` (170 pass, 2 skipped, 2026-10-23).
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
