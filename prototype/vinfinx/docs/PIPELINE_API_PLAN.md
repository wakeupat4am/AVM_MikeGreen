# Pipeline API and Evidence sources view plan

*Status: **A1–A7 approved 2026-10-23** (AI engineer, jimmytrn154; A6 with search by meaning); **built 2026-10-23**. Step 2 of the agreed order in `AGENTS.md`: a loopback API that
serves the pipeline's results to the demo website, and the Authenticator's **Evidence sources** view, where the source
list is signed and the gold labels are reviewed. Later steps (end-to-end runner, mapping agent) add their own routes and
views to the same API.*

## What it does and why

Today the source list and the gold labels can only be reviewed in CSV files and signed from the command line, and the
site's Authenticator views are static mock-ups. This step makes the site the place where a person approves what the
models proposed:

1. **Show** the draft source list (every check, the verified quotes, the agent's decision) and the corpus built from it.
2. **Sign** a source list (v0.1 now, v0.2 live in the demo) and **review** the gold-label sample, with the same rules as
   the command line, so the site cannot sign anything the CLI would refuse.
3. **Record** every write with the reviewer's name, time and the hashes it applied to.

No model runs in this step. The only external call is the optional corpus search (A6).

## Flow

```
Mac: index.html (served on 127.0.0.1:4173)
   │  fetch, through the SSH tunnel (start-qwen-bridge.sh forwards 8061 and 8062)
   ▼
vishc-server-1: pipeline_api.py on 127.0.0.1:8062   (CPU only; no GPU, no Qwen needed)
   │  reads                                    │  writes (explicit, named reviewer, Origin-checked)
   ▼                                           ▼
corpus/source_lists/*-draft.json         corpus/runs/<run>/review_progress.json   (row decisions so far)
corpus/runs/<run>/report.json            corpus/source_lists/source_list-v0.1.json (signed; never overwritten)
corpus/versions/<v>/manifest.json, gold/ corpus/versions/<v>/gold/review_progress.json, gold.json, metrics.json
corpus/originals/<sha256>.pdf            logs/review_actions.jsonl                  (one line per write)
```

## Design choices (A1–A7), for approval

| # | Choice | Proposal |
|---|---|---|
| A1 | Process | A **separate FastAPI service**, `pipeline_api.py`, on **127.0.0.1:8062**, started by `serve-pipeline.sh` in tmux `vinfinx-api`. It needs no GPU, so sign-off and review work while Qwen is stopped, and restarting one never interrupts the other. Same packages as `qwen_api.py` (FastAPI, uvicorn), nothing new installed. `start-qwen-bridge.sh` forwards both ports; the site gets `PIPELINE_BASE = http://127.0.0.1:8062/vinfinx-pipeline` beside `AI_BASE` |
| A2 | Access | Loopback only, as for Qwen: the SSH tunnel is the access control. Every request's `Origin` must be in `VINFINX_ALLOWED_ORIGINS` (the same list as Qwen); writes are `POST` with a JSON body, so a browser preflight is required and other sites cannot post. A write names a reviewer from a short allow-list in config (today `jimmytrn154`), which guards against typos, not against an attacker |
| A3 | Same checks as the CLI | The routes call the existing functions: `ai.sources.review.apply_review` for sign-off, `ai.corpus.evaluate.apply_review` and `retrieval_metrics` for the gold review. Their refusal messages are shown in the site unchanged. The CLI and the site stay interchangeable: progress saved in the site can be exported to the CSV the CLI reads, and the reverse |
| A4 | Saving progress | Row decisions (accept or reject with a note; gold label with a note) are saved on the server as they are made, in `review_progress.json` beside the run or the gold set, so a review can be finished later or on another machine. Nothing is signed or frozen until the reviewer presses **Sign** (source list) or **Submit review** (gold), confirms, and every check passes |
| A5 | Where it lives in the site | The Authenticator's four navigation buttons start switching views (today they only change labels). **Evidence sources** becomes a real view in `index.html` with three tabs: *Source list*, *Gold labels*, *Corpus*. Its code goes in a new `evidence.js` loaded by `index.html`, so the page stays readable; styles reuse the page's variables, cards and fonts. The other three Authenticator views stay as they are until their steps |
| A6 | Corpus tab | Read-only: corpus and index versions with hashes, documents with page and chunk counts, the retrieval metrics and their label status (draft or reviewed). A **search box** runs hybrid search; the query text is sent to OpenAI for its embedding, so the box is labelled "public research questions only". **Decided 2026-10-23: search by meaning allowed**, with a server-side guard: a query over 200 characters, or containing a long number (9+ digits: ID, account, phone), an email address or an amount in đồng, is refused before anything is sent. Every query that is sent is logged; a refused one is neither sent nor logged. Keyword mode never leaves the server |
| A7 | Original documents | The view links each source to its stored original (`GET .../originals/<sha256>`, only hashes that are in a draft or signed list) and each quote to its page, so the reviewer can check a quote against the PDF. PDFs open in the browser's own viewer |

## Routes

All under `/vinfinx-pipeline`; every response carries the versions and hashes it was built from.

| Route | Purpose |
|---|---|
| `GET /health` | Service status and the corpus root it reads |
| `GET /source-runs` | Runs with a draft list: counts by decision, report summary |
| `GET /source-runs/{run}` | Draft records (checks, quotes, decision, reasons) merged with saved progress |
| `POST /source-runs/{run}/progress` | Save one row's decision and note (`reviewer`, `source_id`, `decision`, `note`) |
| `POST /source-runs/{run}/sign` | `reviewer`, `version` (e.g. `source_list/v0.1`): run `apply_review`; on success write the signed list and return its hash; on failure return every problem and write nothing |
| `GET /source-lists` / `GET /source-lists/{version}` | Signed lists with hash, signer and time |
| `GET /originals/{sha256}` | The stored PDF (only hashes listed in a draft or signed list) |
| `GET /corpus/{version}` | Corpus and index manifests, documents, metrics |
| `GET /corpus/{version}/gold` | Review sample with passages, quotes, verification and saved progress |
| `POST /corpus/{version}/gold/progress` | Save one gold row's label and note |
| `POST /corpus/{version}/gold/submit` | Run the gold `apply_review`, write `gold.json` (never overwritten), recompute and return the metrics |
| `GET /corpus/{version}/search?q=&mode=` | Hybrid (or keyword) search, top 8 with source, page and offsets (A6) |

## Review in the site

- **Source list:** one row per source, filterable by the agent's decision (accepted, unclear, rejected). A row opens to
  show the C1–C7 checks with their outcomes, the verified quotes and the link to the original. The reviewer picks
  accept or reject; overruling the agent asks for a note before it saves; a source without a stored original cannot be
  accepted (the same rules as the CLI). A progress bar shows decided / total; **Sign as v0.1** is enabled when every row
  is decided and asks for confirmation. After signing, the list shows its hash, signer and time and becomes read-only.
- **Bulk decisions:** "accept all the agent accepted" and "reject all the agent rejected" save time on the 294
  rejections; each still records the reviewer, and the unclear rows must be decided one by one.
- **Gold labels:** one card per sampled row: question, passage with the drafted quote highlighted, the model's label;
  the reviewer clicks *supports*, *related* or *irrelevant* (keys 1–3). **Submit review** shows agreement and the
  re-measured recall@k.

## Build

1. `ai/api/` (new): read and write functions (no web code) that wrap the source review, the gold review and the
   corpus reads; `review_progress.json` handling; the action log.
2. `pipeline_api.py` (thin FastAPI routes, Origin check, reviewer allow-list) and `serve-pipeline.sh`; the bridge
   script forwards 8062 too.
3. `index.html`: Authenticator navigation switches views; the Evidence sources view; `evidence.js`. The page's
   remaining "cecs-server1" labels become `vishc-server-1`.
4. Tests: the functions in `ai/api/` directly (signing refuses what the CLI refuses, never overwrites, progress
   round-trips to the CLI's CSV, unknown hashes are not served, the Origin and reviewer checks). No browser test
   framework; the view is checked by hand in the browser through the tunnel.
5. Update `SERVER_SETUP_PLAN.md` (runbook: start the API, tunnel both ports), `AGENTS.md`, the tracker and the
   findings log.

## Needed from you

- Approval of A1–A7. The one real choice is **A6**: the corpus search box sends the typed query to OpenAI for its
  embedding (recommended, labelled "public research questions only"), or keyword search only.
- After the build: open the site through the tunnel, review the source list and sign v0.1, and review the gold
  sample, both in the Evidence sources view.

## Not in this step

- Market signals, Review queue, Audit log and the Customer view (step 3, end-to-end runner).
- Mappings and their quotes in the Evidence sources view (step 4, mapping agent).
- Starting a source-agent run from the site (step 5 decides whether the live demo starts a run or only signs one).
- Logins or roles: the site's role switch remains a demo control; the tunnel is the access control.

## As built (2026-10-23)

| Piece | Where |
|---|---|
| Service functions (reads, saved progress, sign-off, gold submit, query guard, action log) | `ai/api/service.py` |
| Routes (Origin check, error mapping) | `pipeline_api.py`; started by `serve-pipeline.sh` in tmux `vinfinx-api`, port 8062 |
| Bridge | `start-qwen-bridge.sh` forwards 8061 and 8062 (only the ports not already forwarded) |
| View | `index.html` (Evidence sources view, styles, Authenticator navigation) + `evidence.js` |
| Shared with the CLI | `ai.sources.review`: `signed_path`, `summary`, `VERSION_PATTERN` (the CLI now also checks the version name); `ai.corpus.evaluate.measure` |
| Tests | `ai/tests/test_api.py`; full suite 170 pass, 2 skipped |

- **Origin rule as built:** a request carrying an `Origin` header must come from an allowed origin; a `POST` must carry one. A `GET`
  without one is allowed, because a reviewer opening a stored PDF in a new tab sends none, and the tunnel is the access control.
- **The source agent's pre-filled rejections are not decisions.** The agent writes `reject` into `review.csv` on every row it
  rejected; the site ignores that default (it would otherwise count 294 rows as decided by nobody) and the reviewer confirms them
  with "Reject all the agent rejected". Decisions a person typed into the CSV are picked up.
- **Quotes and pages:** gold passages and search hits link to their PDF page (`#page=N`). A source's screening quotes have offsets
  into the screened text, not page numbers, so they link to the whole PDF.
- **Files written:** `runs/<run>/review_progress.json`, `source_lists/source_list-v<x>.json` (never overwritten),
  `versions/<v>/gold/review_progress.json`, `gold.json` (never overwritten), `metrics.json`; logs `review_actions.jsonl` and
  `search_queries.jsonl` under `/mnt/sdb/shared/chuong/vinfinx/logs/`.
- **Checked in a browser:** headless Chromium (Playwright, test environment `env-webtest`) against a second API instance on a
  scratch copy of the corpus: bulk decisions, an unclear row decided with a note, sign-off (button disabled until every row is
  decided), gold labelling with keys, corpus metrics, search (8 hits) and a refused query, navigation back to Review queue, and
  no horizontal overflow at 390 px. The real corpus was not written to. Not yet checked: the site from the Mac through the tunnel.
