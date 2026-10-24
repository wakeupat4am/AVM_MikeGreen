# End-to-end runner plan

*Status: **R1–R8 approved 2026-10-23** (AI engineer, jimmytrn154; R3 AVM band metrics, R4 Minh moved to Đống Đa, Hà Nội); **built 2026-10-23**. Step 3 of the agreed order in `AGENTS.md`: one run of the online
pipeline ([`PIPELINE.md`](PIPELINE.md), "Online decision") from the synthetic bulletin to a verified recommendation, with the
Phase 3 AVM evidence for the customer's property, shown in the site's Market signals, Review queue, Audit log and Customer
views. It uses the **provisional mapping** (hand-built, synthetic, status `unvalidated`) until the mapping agent exists (step 4),
and says so everywhere it appears.*

## What it does and why

Every stage exists and is tested on its own: the document generator, OCR, extraction and validation, the bundle, the policy
engine, the Qwen decision step and the verifier. Nothing yet runs them in one pass on one input, records what each stage
saw and produced, or shows the result to a person. This step does that:

1. **One run** turns a bulletin and the customer's property into a recommendation and a verifier verdict.
2. **One record** per run holds every input hash, every version and every intermediate result, so the decision can be replayed
   and audited.
3. **The site shows it:** what was read from the bulletin (with page boxes), what the verifier did, what a person must review,
   and what the customer is told.

## Flow

```
synthetic bulletin (docs-v0.1-draft; PDF + page images, SYNTHETIC watermark)
   │  hash
   ▼
1. OCR        Tesseract 5.5.3 (env-ocr), per page: words, boxes, confidence
   ▼
2. Extract    macro_doc_extractor/0.1 → EvidenceItems, trusted only if every check passes
   ▼
3. AVM        Phase 3 row for the customer's property → EvidenceItems (system provenance, model output, asking prices)
   ▼
4. Bundle     items + policy result, HMAC-signed, time-bounded
   │  4a. Policy engine: provisional mapping (unvalidated) + policy_v0 (provisional) → per-target outcome
   ▼
5. Decide     Qwen3.5-9B (loopback API), schema built from the bundle → recommendation JSON
   ▼
6. Verify     verifier v0.1.1 → PASS / ROUTE_TO_HUMAN / BLOCK
   ▼
run record (pipeline_run/0.1): inputs, versions, hashes, every stage's output, timings  →  site
```

No OpenAI call anywhere in a run: OCR, extraction, policy and verification are local code, and generation is Qwen on this
server. The support check (OpenAI) arrives with the mapping agent in step 4.

## Design choices (R1–R8), for approval

| # | Choice | Proposal |
|---|---|---|
| R1 | Shape | A library function `run_pipeline(document, customer, ...)` in `ai/runner/`, a CLI (`python -m ai.runner run --document <id> --customer <id>`) and a pipeline-API route that starts a run as a background job. Stages call the existing modules; nothing is re-implemented. OCR runs in `env-ocr` as a subprocess (as the corpus build already does); Qwen through its loopback API |
| R2 | Bulletin input | **Only documents from the generated draft set `docs-v0.1-draft`**, chosen from a list in the site (clean, degraded, stale, missing-value and causal-commentary variants, English and Vietnamese). No upload: every bulletin is synthetic, watermarked and has known ground truth, and the held-out set `docs-v0.1-holdout-s7` is never touched |
| R3 | AVM evidence | The Phase 3 row for the customer's property (`Phase 3/data/phase3_available_intervals_v2.csv`, read-only) becomes EvidenceItems with system provenance (`avm.phase3`, property ID, file hash): **P10, P50, P90 and asking price** (billion VND), with the `flag` and `over_P90_pct` as quality flags. This needs four AVM metrics in the catalogue (**catalogue v0.2.0**, provisional; freshness provisionally 180 days, since the latest Phase 3 rows are from July 2026). Every item carries the caveats "model output, asking prices, Hà Nội only, triage not appraisal". AVM items **have no mapping, so they never move the policy result**; they are citable context for the valuation range, and the verifier checks any number quoted from them as it does for macro items |
| R4 | Demo customer | The site's customer (Minh) owns a property in Thảo Điền (Hồ Chí Minh City), but Phase 3 and the catalogue's districts are **Hà Nội only**. Proposal: give Minh a real Phase 3 property in Đống Đa (a catalogue district) and keep a small **synthetic customer file** (`customer_id`, property ID, district; labelled synthetic). The street map stays as it is, labelled "illustrative demo layer". The customer file stays on the server and never goes to OpenAI |
| R5 | Live run and replay | In the demo, Authenticator → Market signals → **Run pipeline** starts a run and shows each stage as it completes (OCR is ~15 s for three pages; the Qwen decision ~15–60 s depending on server load). Qwen must be running; if it is not, the run stops at stage 5 with a clear message, and earlier stages are still shown. Every run is stored, so the demo can also open earlier runs |
| R6 | Run record | `pipeline_run/0.1` JSON per run under `/mnt/sdb/shared/chuong/vinfinx/pipeline_runs/<run_id>/`: document ID and hash, page hashes, OCR engine and model, extractor version, every EvidenceItem (trusted or not, with reasons and boxes), the AVM export hash and row, the signed bundle and its hash, policy version and hash, mapping ID and status, catalogue hash, Qwen model, precision and prompt version, raw output and tokens, verifier verdict, code commit, stage timings and errors. Never overwritten; generated data, never committed |
| R7 | Review queue | Runs the verifier routes to a human (`ROUTE_TO_HUMAN`) appear in the Review queue with the bundle, the policy result and the verifier findings. The reviewer **approves** (releases to the customer), **rejects** or **requests evidence**, with a note; the action is saved in the run's record and in `logs/review_actions.jsonl`. `BLOCK`ed runs are listed for inspection but cannot be released. The current mock queue (Hồ Chí Minh City price-update cases) is replaced |
| R8 | Customer view | The Customer overview's property card and "safe borrowing capacity" area show the **latest released recommendation** for Minh: the AVM band (P10–P90, labelled "model estimate from asking prices"), the action and explanation, and source labels with publication dates ("Lotus Bay Economic Observatory, SYNTHETIC TEST DATA, published …"; "VinFinx AVM Phase 3, model output, …"). A recommendation reaches the customer only if the verifier passed it, or a reviewer approved it. Figures that have no pipeline source (health score, net worth, cash flow) stay as they are, and stay labelled demo |

## Site views

- **Market signals:** a document picker and **Run pipeline**; a list of runs (time, document, verdict). A run shows the page images
  with a box over each extracted value (green trusted, amber untrusted, with the reason on hover) and a table of evidence items:
  metric, value, baseline, change, unit, OCR confidence, trust and reasons. Below it, the AVM items for the property.
- **Review queue:** routed runs; for one run: the recommendation, the policy outcome per target, the verifier findings, the cited
  evidence, and the three reviewer actions.
- **Audit log:** one row per run with its versions and hashes (document, bundle, policy, mapping, catalogue, model, prompt, code
  commit), and the reviewer actions, with every hash copyable. Also the source-list and gold-review actions from step 2.
- **Customer:** as in R8.

## Build

1. Catalogue v0.2.0 with the four AVM metrics (provisional), and `ai/avm/` turning a Phase 3 row into EvidenceItems.
2. `ai/runner/`: stages, run record, CLI; tests with a fake OCR result and a scripted model output (no GPU, no Tesseract), plus
   one test run end to end in `env-ocr` with a scripted decision.
3. Pipeline API: list documents and customers, start a run (background job, one at a time), run status, run records, page
   images (only for documents in the set), reviewer actions on routed runs.
4. Site: Market signals, Review queue, Audit log and the Customer-view changes; checked in headless Chromium against a scratch
   copy of the run directory, then from the Mac through the tunnel.
5. Measure: run every draft-set document through the runner with Qwen; record verdicts, trusted-item counts and stage timings
   (with server load and date) in the findings log. The 42-case adversarial set is not affected.
6. Update `AGENTS.md`, the tracker, the findings log and `SERVER_SETUP_PLAN.md` (runbook).

## Needed from you

- Approval of R1–R8. Two choices are yours in particular:
  - **R3:** adding the four AVM metrics to the catalogue (v0.2.0, provisional), with a provisional freshness of 180 days.
  - **R4:** moving the demo customer's property to Hà Nội (Đống Đa) so that the Phase 3 evidence applies, and keeping the
    street map as an illustrative layer.

## Not in this step

- The mapping agent, the support check and the mapping registry (step 4): until then the mapping is provisional and every
  run says so; an `unvalidated` mapping allows only low-impact actions (policy rule).
- The Qwen second reading of the bulletin (not wired in yet; the extractor already marks a field untrusted on any failed check).
- Customer documents (salary statements and similar) and the conversational assistant.
- Policy weights and thresholds stay provisional until the domain owner signs them off; the OCR threshold stays as proposed until
  the risk owner answers.

## As built (2026-10-23)

| Piece | Where |
|---|---|
| Catalogue v0.2.0 | `ai/catalog/metrics.yaml`: four AVM metrics (`value_p10`, `value_p50`, `value_p90`, `asking_price`, billion VND, snapshot, freshness 180 days provisional) in a separate **`context_metrics`** section. Context metrics have no reason codes and no policy weight; `Catalogue.instances` and `metrics` still list only the 8 signal metrics (14 IDs, 19 reason codes), so the bulletin, its extraction and the agents' prompts are unchanged (checked: regenerated draft pages and PDFs are byte-identical). `Catalogue.get` finds both |
| AVM evidence | `ai/avm/`: a Phase 3 row → 4 EvidenceItems (system provenance `avm.phase3`, file hash, property ID; the P10–P90 band also attached to P50 as an 80% interval; flag and over-P90 as quality flags; caveats) |
| Demo customer | `ai/runner/customers_v0.yaml` (synthetic): `demo-minh`, property `7b99d5ef…` (P. Quang Trung, Q. Đống Đa, 52.3 m², listed 2026-07-06, band 8.36 / 18.41 / 38.54, asking 33.0, flag fair; fresh until 2027-01-02) |
| Runner | `ai/runner/` (`run_pipeline`, CLI `python -m ai.runner`), `ai/runner/ocr_pages.py` (env-ocr). Stages document → ocr → extraction → avm → policy → bundle → decision → verifier; record `pipeline_run/0.1` in `/mnt/sdb/shared/chuong/vinfinx/pipeline_runs/<run_id>/run.json` (+ `ocr_pages.json`, `review.json`) |
| Demo bulletins | `fixtures/docs-v0.1-demo-20261023`: the generator's full set (62 documents, seed 42) dated **2026-10-23**, so their figures are fresh for the demo (the draft set is dated 2026-10-05 and its 14-day inventory figures are already stale). Synthetic, watermarked; never the held-out set |
| Signing key | `~/.config/vinfinx/bundle.env` (mode 600; key ID `vinfinx-demo-2026-10`), or `VINFINX_BUNDLE_KEY` / `VINFINX_BUNDLE_KEY_ID` |
| API | `ai/api/runs.py` + routes `/pipeline/inputs`, `/pipeline/runs` (POST starts one run at a time in a thread), `/pipeline/runs/{id}`, `/pages/{n}`, `/review`, `/pipeline/customers/{id}/latest`, `/pipeline/audit` |
| Site | `pipeline.js`; `index.html`: Review queue (replaces the mock), Market signals, Audit log, the Customer property card (Đống Đa band and the latest released recommendation with sources), map labelled illustrative |
| Tests | `ai/tests/test_pipeline_runner.py` (6); full suite 177 pass, 2 skipped |

- **Decision prompt split:** the Qwen adapter takes at most 4,000 characters per message and a real bundle is ~8,000, so the
  bundle is sent in parts of 3,500 characters (as the source agent does), `runner_prompts/0.1`.
- **First real runs (clean English bulletin):** OCR ~10 s, Qwen 11–16 s (GPU 1, load ~10–11). One run was routed (the output
  asked for review), one blocked (Qwen quoted the untrusted inventory row's figures and used `INVENTORY_DROP` without
  evidence). Both explanations also contained a **false sentence the verifier does not check**: "the AVM data from July 2026 …
  is considered stale" (it is fresh until 2027-01-02) and "transaction volume (255) is below baseline (217)". The verifier checks
  numbers, citations, codes and actions, not the meaning of sentences: see the findings log, 2026-10-23.
- **Checked in a browser:** headless Chromium against a second API instance with scratch run and log directories: run, page boxes
  (11 trusted, 3 untrusted on page 1), queue, audit, customer card; no horizontal overflow at 390 px after a fix.
- **Fixes after the measurement (approved 2026-10-23):** `runner_prompts/0.2` withholds values the decision may not cite and names
  every item; verifier v0.2.0 ignores percentile labels and blocks sentences that contradict a cited item's movement or call
  fresh evidence stale. Re-measurement on the demo set pending (needs Qwen).

