# VinFinx AI engineering docs

Working plans and progress for the AI engineering scope. Background, decisions and daily notes live in [`../../AI_ENGINEER_FINDINGS.md`](../../AI_ENGINEER_FINDINGS.md); the requirements are in [`../AI_ENGINEERING_HANDOFF.md`](../AI_ENGINEERING_HANDOFF.md).

*Progress tracker last updated 2026-10-21.*

## Where we stand

### The handoff's philosophy

The LLM must never be the sole authority. Every fact it uses comes from validated evidence, every output is checked by deterministic code, and anything uncertain goes to a human. The handoff asks how to control hallucination (§3, §4, §5) and answers with a stack of independent controls (§6), not a better prompt, then requires measuring whether those controls work (§7).

[`PIPELINE.md`](PIPELINE.md) turns this into a concrete system for macro decisions:
- **Offline:** an agent reads real research documents and proposes how each macro change affects a target. Every proposal cites an exact quote, is verified, and is stored as a versioned mapping.
- **Online, per case:** a synthetic macro document is read by OCR into trusted evidence. Deterministic code applies the mapping. Qwen only writes the recommendation, a verifier checks it, and a human reviews what is routed.

### Against the handoff's controls (§6)

| §6 control | What it means | What we have | Status |
|---|---|---|---|
| Constrained inputs | Validated evidence objects, not free text | `EvidenceItem`, signed `EvidenceBundle`, metric catalogue | **Built** |
| Structured outputs | Strict schema, enumerated actions | `Recommendation` and `MappingProposal`; per-request output schemas built from the bundle, so Qwen cannot cite evidence that does not exist | **Built, tested on Qwen** |
| Citation enforcement | Every claim maps to evidence | Verifier: citations in the bundle; quotes word for word; numbers traced to evidence | **Built** (verifier v0) |
| Deterministic verification | Units, policy and arithmetic checked outside the LLM | Verifier v0.1.1 checks units, freshness, trust, allowed actions, magnitude cap, reason codes; the policy engine applies weights and thresholds in code | **Built** (policy values provisional) |
| Abstention | Explicit outcomes for missing, conflicting or stale data | Conflict forces abstention; abstaining routes to a human | **Built** in schemas and verifier |
| Human review | Mandatory for high impact and low confidence | `ROUTE_TO_HUMAN` decisions exist; the review screen is only a UI mock-up | **Partly** |
| Model separation | Generator, verifier and policy engine are distinct | Qwen generates, rules verify, OpenAI support check planned | **Partly**: the support check is a placeholder |
| Adversarial evaluation | Test injection, forgery, stale data, contradictions | 42 frozen cases (11 held out) and an evaluation runner | **Built** |
| Auditability | Log versions, hashes, outputs, verifier results | Everything is hashed and versioned; the audit log is not built | **Partly** |
| Monitoring | Track unsupported claims, overrides, cost | Requirements written (handoff §9) | **Not started** |

### Against the pipeline

| Component | Status |
|---|---|
| Qwen on our server | **Running**, pinned versions, ~20 tokens/s |
| Contracts (schemas, catalogue, document spec, source criteria) | **Done** |
| Verifiers (rules) | **Done** (v0, approved) |
| Adversarial test set and runner | **Frozen** (v0.1.0); runner built |
| Synthetic document generator | **Built** 2026-10-20 (v0.1.0); draft and held-out sets on the server |
| OCR service, extraction and validation | **Built** 2026-10-21 (Tesseract); 0 wrong-and-trusted items; Qwen second reading not wired in |
| Source agent, research corpus, RAG, mapping agent, support check | Designed, not built (next) |
| Policy engine | **Built** 2026-10-20; weights and thresholds provisional |
| Decision engine | Proven in live checks; not wired into a service |
| Audit log, Authenticator review flow, cost logging | Not started |

**In short:** we have built the safety framework the handoff asks for: the contracts, the constraints on the model, and the checker. Since then: the baseline and two rounds of fixes are measured (dev 21/24), and the policy engine is built (2026-10-20). Still not built: the parts that *produce* evidence (document, OCR, research corpus, mapping agent).

### Why the adversarial test set exists

The handoff's question is "how can we control hallucination?". Everything built so far is a proposed answer; the adversarial set is how we prove it, or find where it fails.

1. **The handoff requires it.** §6 lists adversarial evaluation as a control. §7 says no model update ships without a frozen evaluation set, and its gates are all rates (unsupported claims, citation precision, abstention accuracy, false acceptance of tampered documents). A rate needs a fixed set of cases to be computed over.
2. **Ordinary inputs do not test the controls.** On a normal question Qwen usually behaves, so the verifier has nothing to catch. Hallucinations appear under pressure: conflicting signals, missing data, injected instructions, forged history. The cases apply that pressure on purpose. The verifier is the security guard; the adversarial set is the break-in test.
3. **We have already seen why.** In the 2026-10-18 live check, Qwen reversed a mapping direction, copied "VND 3.20B" from the demo context, gave reason codes for metrics not in the evidence, and said the rate rose when it fell, all in valid JSON that no schema could reject. The first three are now cases ADV-034, ADV-010 and ADV-021, so they are measured every time something changes instead of remaining anecdotes; the fourth is covered by a verifier unit test and is a candidate case to add.
4. **It makes progress comparable.** The Qwen baseline, every verifier change, another model, vLLM, and the cost comparison (handoff §9: cheaper *at equal quality*) are all measured on the same frozen set. Without it, "better" is an impression.
5. **It must be independent of the builder.** The verifier was written by the same hand that drafted the cases, so the AI engineer reviews and freezes the set before the verifier is tuned; otherwise the tests quietly fit the code.

**What the freeze unlocks:** the Qwen baseline on the frozen set gives the first *measured* answer to the handoff's macro question (how often the current system makes unsupported claims, abstains correctly, resists injection). After that, the evidence-producing components are built and each is measured on the same set as it is added.

## Documents

| Document | What it covers | Status |
|---|---|---|
| [`PIPELINE.md`](PIPELINE.md) | The macro-behavior pipeline: offline research, online decision, OCR service, models, versioning, evaluation, build order | Intended design; being built in build order |
| [`SERVER_SETUP_PLAN.md`](SERVER_SETUP_PLAN.md) | Serving Qwen3.5-9B on `vishc-server-1` with plain `transformers`: environment, pinned versions, test results, runbook; vLLM deferred | Installed and tested on the server, 2026-10-16 |
| [`SPECS_AND_SCHEMAS_PLAN.md`](SPECS_AND_SCHEMAS_PLAN.md) | Build steps 1–2: metric catalogue, document spec, source criteria, schemas; decisions D1–D10 | Approved 2026-10-17; built 2026-10-18 |
| [`MACRO_DOCUMENT_SPEC.md`](MACRO_DOCUMENT_SPEC.md) | The synthetic macro-status document: layout, variants, ground truth, traceability to handoff §3 and §5 | Approved 2026-10-17; generator not built |
| [`CORPUS_SOURCE_CRITERIA.md`](CORPUS_SOURCE_CRITERIA.md) | What may enter the research corpus, source records, deliberately bad documents | Approved 2026-10-17 |
| [`SOURCE_AGENT_PLAN.md`](SOURCE_AGENT_PLAN.md) | How the source agent finds, checks and screens candidate sources for sign-off (S1–S7) | Approved 2026-10-22; built, first run done |
| [`CORPUS_INDEX_PLAN.md`](CORPUS_INDEX_PLAN.md) | Ingestion, chunking, OpenAI embeddings + SQLite keyword index, freeze, retrieval evaluation (I1–I7) | Approved 2026-10-22; dev build done |
| [`PIPELINE_API_PLAN.md`](PIPELINE_API_PLAN.md) | Loopback pipeline API + Evidence sources view: source-list sign-off and gold review in the site (A1–A7) | Approved and built 2026-10-23 |
| [`RUNNER_PLAN.md`](RUNNER_PLAN.md) | End-to-end runner: bulletin → OCR → evidence + Phase 3 AVM → policy → Qwen → verifier, shown in Market signals, Review queue, Audit log and Customer views (R1–R8) | Approved and built 2026-10-23 |
| [`MAPPING_AGENT_PLAN.md`](MAPPING_AGENT_PLAN.md) | Mapping agent (Qwen) → rules → OpenAI support check → versioned mapping registry, reviewed in Evidence sources (M1–M8) | Approved and built 2026-10-24 |
| [`OFFLINE_DEMO_PLAN.md`](OFFLINE_DEMO_PLAN.md) | Showing the offline half live: "Run mapping agent" button, pipeline overview page; source agent stays pre-run | Noted 2026-10-24, not started |
| [`SOURCE_RUN_2_PLAN.md`](SOURCE_RUN_2_PLAN.md) | Second source run for the three weighted pairs; source list v0.2 on top of v0.1; corpus-v0.2 and a new mapping run (V1–V7) | Proposal 2026-10-24 |
| [`HANDOFF_2026-10.md`](HANDOFF_2026-10.md) | Handoff to the next AI engineer: rebuild from scratch on a new server (setup, rebuild order, remaining tasks, decision owners, rules) | Written 2026-10-24 |
| [`EVALUATION_RESULTS.md`](EVALUATION_RESULTS.md) | Results of the adversarial set over time: baseline and every later run, with causes of failure | Baseline recorded 2026-10-18 |

## Build order progress

The six steps from [`PIPELINE.md`](PIPELINE.md#build-order):

| # | Step | Status | Notes |
|---|---|---|---|
| 1 | Specs: synthetic macro-status document, research-corpus source criteria | **Done** 2026-10-17 | Plus the metric catalogue (`ai/catalog/metrics.yaml`): 14 metric IDs, 19 reason codes, provisional freshness |
| 2 | Schemas | **Done** 2026-10-18 | `ai/schemas/`; live checks on Qwen passed for all three generation schemas |
| 3 | Verifiers v0 (rules) | **Done**, approved 2026-10-18 | `ai/verifier/`: recommendation and mapping verifiers; 24 tests including the four worked cases. Support check is a placeholder |
| 4 | Frozen adversarial test set | **Done**: frozen 2026-10-18 by jimmytrn154 (v0.1.0) | `ai/evaluation/`: 42 cases, 16 categories, 11 held out; evaluation runner built ([README](../ai/evaluation/README.md)) |
| 5 | Qwen on our server; baseline | **Done**: serving 2026-10-16; baseline 2026-10-18 (15/35 passed; see [`EVALUATION_RESULTS.md`](EVALUATION_RESULTS.md)) | Speed partly explained (see [Other workstreams](#other-workstreams)) |
| 6 | Document generator, OCR service, RAG, mapping agent, policy engine, decision engine | **In progress:** policy engine and document generator 2026-10-20; OCR (Tesseract), extraction and validation 2026-10-21; source agent 2026-10-22, source list v0.1 signed and `corpus-v0.1` frozen 2026-10-23 (RAG); pipeline API, site views and end-to-end runner (decision engine on the provisional mapping, verifier v0.2.0) 2026-10-23. **Not started:** mapping agent, support check, mapping registry, Qwen second reading | `POLICY_ENGINE_PLAN.md`, `SOURCE_AGENT_PLAN.md`, `CORPUS_INDEX_PLAN.md`, `PIPELINE_API_PLAN.md`, `RUNNER_PLAN.md` |

## Done so far

| Date | Work | Where |
|---|---|---|
| 2026-10-11 | Context review, findings, verifier design sketch | Findings log |
| 2026-10-13 | Phase outputs confirmed raw, not macro; synthetic macro document plus OCR chosen as the macro source | Findings log |
| 2026-10-14 | Pipeline designed: agent-owned mapping with cited evidence, real research corpus as RAG, model split (Qwen for generation, OpenAI for embeddings and support check), pluggable OCR | `PIPELINE.md` |
| 2026-10-16 | Server surveyed; dedicated environment installed (Python 3.12, torch 2.14.1, transformers 5.19.0, xgrammar 0.2.8); weights pinned and hashed; adapter reworked (JSON-schema output, stop-token fix); flow tested on the server | `SERVER_SETUP_PLAN.md` |
| 2026-10-16 | Inference engine: plain `transformers` now, vLLM after experiments | Findings log |
| 2026-10-16 | Access: SSH over NetBird only; public relays dropped per VinSoc policy | `SERVER_SETUP_PLAN.md` |
| 2026-10-17 | Token and cost accounting added to the requirements (§9, §7 gate, backlog item 9); parked for now | Handoff §9 |
| 2026-10-17 | Specs and schemas plan approved (D1–D10); metric catalogue and `EvidenceItem` built; document spec and source criteria approved | `SPECS_AND_SCHEMAS_PLAN.md` |
| 2026-10-17 | Server questions settled: any free GPU may be used; NetBird works from the Mac; `backup_user` storage is not purged | `SERVER_SETUP_PLAN.md` |
| 2026-10-18 | Remaining schemas built (`RelationshipEvidence`, `MappingProposal`/`MappingSet`, `PolicyResult`, `EvidenceBundle` with HMAC signing, `Recommendation` with per-request generation schema, `VerifierResult`); live checks on Qwen; two adapter fixes | `SPECS_AND_SCHEMAS_PLAN.md`, findings log |
| 2026-10-18 | Verifier v0 built (recommendation and mapping); adversarial case format and 42 draft cases; live checks passed after adapter fixes; speed test traced the direct-load slowness to `device_map` | `ai/verifier/`, `ai/evaluation/`, `SERVER_SETUP_PLAN.md` |
| 2026-10-18 | Adversarial set and verifier v0 approved; held-out split proposed (11 cases); evaluation runner built and checked on three cases through Qwen; verifier false positive on freshness dates fixed; `AGENTS.md` updated | `ai/evaluation/`, findings log |
| 2026-10-18 | Adversarial set frozen (v0.1.0, by jimmytrn154); baseline run: 15/35 passed; main cause: reason codes for metrics not in the evidence (13 of 20 model-run decisions), sometimes looping until the token limit | `EVALUATION_RESULTS.md`, findings log |
| 2026-10-19 | First fixes applied and run on the dev split: 18/24 passed (baseline 11/24), but 4 of those passes used `CONFLICTING_SIGNALS` without a conflict, which verifier v0 does not check (14/24 counting them as failures). No more invented metric codes or loops; Vietnamese 3/3. Run was slow (75 s per decision) because of server load | `EVALUATION_RESULTS.md`, findings log |
| 2026-10-19 | Second fixes (verifier v0.1, citable evidence only, citation required unless abstaining) run on the dev split: 21/24 passed, all genuine; remaining: one real excluded-figure statement, one customer-figure echo, one verifier bug (evidence-ID digits counted as numbers in uncited abstentions). Generation still slow (~70 s per decision) | `EVALUATION_RESULTS.md`, findings log |
| 2026-10-19 | Verifier v0.1.1 (evidence-ID digits no longer counted as numbers); decision-step tuning paused at dev 21/24; policy-engine plan drafted | `POLICY_ENGINE_PLAN.md`, findings log |
| 2026-10-20 | P1–P5 approved as provisional; **policy engine built** (`ai/policy/`): reproduces all 30 frozen decision cases' policy results, runner `--policy engine`; schema `policy_result/0.2` | `POLICY_ENGINE_PLAN.md`, findings log |
| 2026-10-20 | **Document generator built** (`ai/documents/`, v0.1.0): every spec variant in English and Vietnamese, image-only pages and PDF, exact ground truth; draft set `docs-v0.1-draft` (62 documents) on the server | `MACRO_DOCUMENT_SPEC.md`, findings log |
| 2026-10-21 | OCR: PaddleOCR (CPU) installed but cannot read Vietnamese (recognizer lacks 44 of 67 letters) and takes ~200 s per page; Tesseract 5.5.3 added: reads Vietnamese, 4–6 s per page, accurate on clean and mildly degraded pages, fails on strong blur, 100 dpi and 90° rotation. OCR adapter and evaluation built (`ai/ocr/`); document-ID collision in the generator fixed | `OCR_RESULTS.md`, `SERVER_SETUP_PLAN.md`, findings log |
| 2026-10-21 | **Tesseract chosen, PaddleOCR removed.** Orientation detection, field extraction and validation built (`ai/extraction/`). End to end: **0 wrong-and-trusted items** of 868 on the draft set and 868 on a held-out set (seed 7); 67–68% of items trusted and correct; a confidence threshold only loses coverage. Qwen second reading not wired in yet | `OCR_RESULTS.md`, findings log |
| 2026-10-21 | Confidence-threshold analysis: no misreading survives the other checks in any field, so a threshold only adds review work; Tesseract is overconfident on degraded pages (95% right at 0.90–0.95). Proposal for the risk owner: no minimum for bulletin values. Also fixed: page 2 had lost its Source column (generator), Vietnamese page 2 was not read (extractor) | `OCR_CONFIDENCE_THRESHOLD.md`, findings log |
| 2026-10-21 | Source agent direction: free scholarly and institutional sources only (OpenAlex, World Bank API, IMF, ADB, BIS, SBV, statistics office); access checked from the server (IMF and ADB pages block scripts, found via OpenAlex). Corpus storage moved to `/mnt/sdb` (`/mnt/disk1` full) | `SERVER_SETUP_PLAN.md`, findings log |
| 2026-10-21 | Source list signed off by jimmytrn154; pypdfium2 installed; source agent plan (S1–S7) drafted for approval | `SOURCE_AGENT_PLAN.md`, findings log |
| 2026-10-22 | S1–S7 approved; **source agent built** (`ai/sources/`); first run: 342 candidates, 110 screened by Qwen, 2 accepted, 21 unclear (13 wait only for a publisher-terms check), 294 rejected; Qwen back at ~22 tokens/s | `SOURCE_AGENT_PLAN.md`, findings log |
| 2026-10-23 | A1–A7 approved (A6: search by meaning, with a personal-data guard); **pipeline API and Evidence sources view built** (`pipeline_api.py`, `ai/api/`, `evidence.js`): source-list review and sign-off, gold-label review, corpus metrics and search in the site, same checks as the CLI; checked in headless Chromium on a scratch copy | `PIPELINE_API_PLAN.md`, findings log |
| 2026-10-23 | **Source list v0.1 signed** in the site by jimmytrn154 (14 accepted, 303 rejected, 0 overruled); **`corpus-v0.1` frozen** (14 documents, 1,452 chunks, embeddings from cache). Fixed: parallel saves could drop a review decision (now serialised) | `PIPELINE_API_PLAN.md`, `CORPUS_INDEX_PLAN.md`, findings log |
| 2026-10-23 | R1–R8 approved; **end-to-end runner built** (`ai/runner/`, `ai/avm/`, catalogue v0.2.0 with AVM context metrics, demo customer in Đống Đa, demo bulletins dated 2026-10-23); Market signals, Review queue, Audit log and the Customer card are live. Found: the verifier does not check the meaning of sentences | `RUNNER_PLAN.md`, findings log |
| 2026-10-23 | Measured on 62 demo bulletins: 55 blocked, 5 passed, 2 routed. Fixes approved and built: `runner_prompts/0.2` (uncitable values withheld) and **verifier v0.2.0** (percentile labels; direction and staleness claims checked); no new blocks on past real outputs. Re-measurement pending | `RUNNER_PLAN.md`, findings log |
| 2026-10-24 | Re-measured: (a)–(c) 39 blocked / 6 passed / 17 routed; (d) backfired and was reverted, (e) kept. M1–M8 approved; **mapping agent, support check (`gpt-4.1-mini`) and registry built**; first run on `corpus-v0.1`: 0 accepted (9 no mapping, 3 rejected by rules), runner keeps the provisional mapping | `MAPPING_AGENT_PLAN.md`, findings log |

## Other workstreams

| Workstream | Status |
|---|---|
| Qwen serving | Running on demand (`tmux`, any free GPU). The full prototype check from the Mac through the bridge has not been run yet |
| Generation speed | **Explained enough.** Loading with an explicit device (`device_map={"": 0}`) runs at ~3.2 tokens/s; `device_map="auto"` (the adapter's setting) at ~21 on the same GPU (2026-10-18). The adapter on GPU 2 ran at 15–20 tokens/s on 2026-10-18, so its slow run on 2026-10-17 was transient. All measurements: [`SERVER_SETUP_PLAN.md`](SERVER_SETUP_PLAN.md#generation-speed-log) |
| Cost accounting (handoff §9) | Requirements written; per-call logging not built; parked until the pipeline exists |
| vLLM | Deferred until measurements show plain `transformers` is too slow |
| Conversational AI, customer-document OCR | Not started; they reuse the contracts built here |

## Next steps

1. Decision-step tuning is paused (dev 21/24 with the second fixes; verifier v0.1.1 fixes the evidence-ID bug). The holdout run waits until step 6 changes the system again. Re-measure generation speed separately.
2. **Step 6 continues:** the document chain works end to end without wrong-and-trusted items ([`OCR_RESULTS.md`](OCR_RESULTS.md)). Open: the OCR confidence threshold (proposal in [`OCR_CONFIDENCE_THRESHOLD.md`](OCR_CONFIDENCE_THRESHOLD.md), awaiting the risk owner), wiring in the Qwen second reading, freezing a document set. Then research corpus, mapping agent and support check.

## Open items

| Item | Owner | Status |
|---|---|---|
| Who signs off the source list | jimmytrn154 | Decided 2026-10-21 |
| Who reviews the gold-label sample | Not assigned | Open |
| Policy-engine weights and thresholds | Domain owner, not assigned | Open |
| Freshness defaults in the catalogue | Risk owner | Provisional |
| Cost baseline model(s) and the hourly GPU rate | | Parked |
| Evidence ID scheme for per-property AVM (Phase 3) evidence | AI engineer | Deferred to the Phase 3 evidence adapter |
| Demo property scope (Thảo Điền vs Hà Nội) | Project lead | Hà Nội districts used for market metrics (D3); the prototype UI still shows Thảo Điền |

## Code map

| Path | Contents |
|---|---|
| `../ai/catalog/` | Metric catalogue (`metrics.yaml`) and its loader |
| `../ai/schemas/` | `common`, `evidence`, `mapping`, `policy`, `bundle`, `recommendation`, `verifier` |
| `../ai/verifier/` | Verifier v0: `recommendation`, `mapping`, `numbers` |
| `../ai/evaluation/` | Case format (`cases.py`), the case set (`cases/adversarial_v0.yaml`), the runner (`runner.py`) |
| `../ai/policy/` | Policy engine (`engine.py`), config (`policy_v0.yaml`), provisional synthetic mapping |
| `../ai/documents/` | Synthetic macro-document generator (content, render, degrade, ground truth) |
| `../ai/ocr/` | OCR interface, Tesseract adapter, field scoring, OCR evaluation (run in `env-ocr`) |
| `../ai/extraction/` | Parsers, layout reader, bulletin extractor, end-to-end and confidence evaluation |
| `../ai/tests/` | Tests and shared fixtures (worked-case bundles X and Y); 114 passing, 2 skipped in the main env |
| `../qwen_api.py`, `../serve-qwen.sh`, `../start-qwen-bridge.sh` | Qwen adapter, server launcher, Mac bridge |

Run the tests from `prototype/vinfinx` on the server:

```bash
/mnt/disk1/backup_user/24chuong.ta/vinfinx/env/bin/python -m unittest discover -s ai/tests -t .
```
