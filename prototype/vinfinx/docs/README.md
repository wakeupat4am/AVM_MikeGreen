# VinFinx AI engineering docs

Working plans and progress for the AI engineering scope. Background, decisions and daily notes live in [`../../AI_ENGINEER_FINDINGS.md`](../../AI_ENGINEER_FINDINGS.md); the requirements are in [`../AI_ENGINEERING_HANDOFF.md`](../AI_ENGINEERING_HANDOFF.md).

*Progress tracker last updated 2026-10-18 (evening).*

## Documents

| Document | What it covers | Status |
|---|---|---|
| [`PIPELINE.md`](PIPELINE.md) | The macro-behavior pipeline: offline research, online decision, OCR service, models, versioning, evaluation, build order | Intended design; being built in build order |
| [`SERVER_SETUP_PLAN.md`](SERVER_SETUP_PLAN.md) | Serving Qwen3.5-9B on `vishc-server-1` with plain `transformers`: environment, pinned versions, test results, runbook; vLLM deferred | Installed and tested on the server, 2026-10-16 |
| [`SPECS_AND_SCHEMAS_PLAN.md`](SPECS_AND_SCHEMAS_PLAN.md) | Build steps 1–2: metric catalogue, document spec, source criteria, schemas; decisions D1–D10 | Approved 2026-10-17; built 2026-10-18 |
| [`MACRO_DOCUMENT_SPEC.md`](MACRO_DOCUMENT_SPEC.md) | The synthetic macro-status document: layout, variants, ground truth, traceability to handoff §3 and §5 | Approved 2026-10-17; generator not built |
| [`CORPUS_SOURCE_CRITERIA.md`](CORPUS_SOURCE_CRITERIA.md) | What may enter the research corpus, source records, deliberately bad documents | Approved 2026-10-17 |

## Build order progress

The six steps from [`PIPELINE.md`](PIPELINE.md#build-order):

| # | Step | Status | Notes |
|---|---|---|---|
| 1 | Specs: synthetic macro-status document, research-corpus source criteria | **Done** 2026-10-17 | Plus the metric catalogue (`ai/catalog/metrics.yaml`): 14 metric IDs, 19 reason codes, provisional freshness |
| 2 | Schemas | **Done** 2026-10-18 | `ai/schemas/`; live checks on Qwen passed for all three generation schemas |
| 3 | Verifiers v0 (rules) | **Built** 2026-10-18, awaiting review | `ai/verifier/`: recommendation and mapping verifiers; 24 tests including the four worked cases. Support check is a placeholder |
| 4 | Frozen adversarial test set | **Draft ready** 2026-10-18; **AI engineer to edit and freeze** | `ai/evaluation/`: case format and 42 draft cases across 16 categories ([README](../ai/evaluation/README.md)) |
| 5 | Qwen on our server; baseline | Serving **done** 2026-10-16; baseline **waiting** on step 4 | Generation slowdown under investigation |
| 6 | Document generator, OCR service, RAG, mapping agent, policy engine, decision engine | Not started | |

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

## Other workstreams

| Workstream | Status |
|---|---|
| Qwen serving | Running on demand (`tmux`, any free GPU). The full prototype check from the Mac through the bridge has not been run yet |
| Generation speed | **Partly explained.** Loading with an explicit device (`device_map={"": 0}`) runs at ~3.2 tokens/s; `device_map="auto"` (the adapter's setting) at ~21 on the same GPU (2026-10-18). The adapter's slow run on GPU 2 on 2026-10-17 is still unexplained. All measurements: [`SERVER_SETUP_PLAN.md`](SERVER_SETUP_PLAN.md#generation-speed-log) |
| Cost accounting (handoff §9) | Requirements written; per-call logging not built; parked until the pipeline exists |
| vLLM | Deferred until measurements show plain `transformers` is too slow |
| Conversational AI, customer-document OCR | Not started; they reuse the contracts built here |

## Next steps

1. **AI engineer: edit and freeze the adversarial set** (`ai/evaluation/cases/adversarial_v0.yaml`), including the five points listed in its [README](../ai/evaluation/README.md#points-to-decide-while-editing).
2. **AI engineer: review verifier v0** (`ai/verifier/`), especially the provisional limits (minimum quote length 20 characters, maximum source age 10 years) and the rule that one conflicting target forces abstention.
3. **After the freeze:** an evaluation runner that scores the pipeline on the set, then the Qwen baseline (build step 5).
4. **Speed:** check the adapter on GPU 2 or 3 again (still unexplained slowness on 2026-10-17).

## Open items

| Item | Owner | Status |
|---|---|---|
| Who signs off the source list and reviews the gold-label sample | Not assigned | Open |
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
| `../ai/evaluation/` | Adversarial case format (`cases.py`) and the draft set (`cases/adversarial_v0.yaml`) |
| `../ai/tests/` | Tests and shared fixtures (worked-case bundles X and Y); 69 passing |
| `../qwen_api.py`, `../serve-qwen.sh`, `../start-qwen-bridge.sh` | Qwen adapter, server launcher, Mac bridge |

Run the tests from `prototype/vinfinx` on the server:

```bash
/mnt/disk1/backup_user/24chuong.ta/vinfinx/env/bin/python -m unittest discover -s ai/tests -t .
```
