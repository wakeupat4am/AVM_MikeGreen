# Macro-behavior pipeline

*Status: intended design, 2026-10-14. Nothing is built yet. Decisions and their reasoning are in the daily notes of [`../../AI_ENGINEER_FINDINGS.md`](../../AI_ENGINEER_FINDINGS.md).*

## Purpose

Let the macro-behavior decision engine change a recommendation because of macro conditions, without letting an LLM hallucination become a decision. This answers the handoff §3 primary question: *how can we control AI hallucination when macro behavior changes the final recommendation or decision?*

The pipeline has two halves:
- **Offline research** builds a *mapping*: what a macro change means (for example, a rate cut raises borrowing capacity). It runs occasionally, and every output is versioned.
- **Online decision** runs per case: it reads the current macro status, applies an approved mapping version, and produces a verified recommendation.

## Overview

```
OFFLINE (research, run occasionally, outputs are versioned)
──────────────────────────────────────────────────────────────────────
 Source agent (Qwen) ──proposes──► Source list ──human sign-off──► v1
                                        │
                                        ▼
 Ingestion: fetch → hash → store original → extract text
            (PDF text layer, or the OCR service for scanned PDFs)
                                        │
                                        ▼
 Chunker (page + character offsets) → Indexer (OpenAI embeddings
                                       + keyword index)
                                        │
                                        ▼
                         FROZEN CORPUS + INDEX  (version ID, hash)
                                        │
            ┌───────────────────────────┴──────────────────────┐
            ▼                                                  ▼
 Mapping agent (Qwen)                        Gold labels: OpenAI drafts,
 retrieves chunks → MappingProposal          human reviews ~20% → frozen
 (metric, target, direction, lag,            evaluation set
  size, geography, chunk ID, quote)
            │
            ▼
 Mapping verifier
   rules: quote verbatim in chunk? fresh? geography/units? conflicts?
   support check (OpenAI): does the quote back the direction?
            │
            ▼
 MAPPING REGISTRY  (version ID, hash, status: unvalidated → validated/rejected)
            │                                    ▲
            │                                    └── human validates later
            ▼
ONLINE (per decision)
──────────────────────────────────────────────────────────────────────
 Macro-status document (synthetic)
   → file and quality checks
   → OCR service (pluggable engine: boxes + confidence), Qwen second reading
   → field extraction → deterministic validation
   → observation EvidenceItems (doc hash, page, box, confidence)
                                        │
 AVM evidence (Phase 3 bands, flags) ───┤
 Customer records (authorized only) ────┤
                                        ▼
 Evidence bundle builder (signed, time-bounded)
                                        │
                                        ▼
 Policy engine (deterministic)
   applies mapping version → per-target effects, materiality,
   weights, conflicts, magnitude cap
   (unvalidated mapping → low-impact actions only)
                                        │
                                        ▼
 Decision engine (Qwen) → Recommendation JSON
                                        │
                                        ▼
 Verifier (rules) → PASS / ROUTE_TO_HUMAN / BLOCK
                                        │
                     ┌──────────────────┴─────────────────┐
                     ▼                                    ▼
          Human review (Authenticator view)     Customer explanation
                                        │
                                        ▼
 AUDIT LOG: every version and hash (corpus, index, mapping, prompt,
 model, evidence), verifier results, reviewer actions
```

## Inputs

| Input | Nature | Role |
|---|---|---|
| Macro-status document | **Synthetic**, fictional publisher, visible "SYNTHETIC, TEST DATA" watermark | The current macro observations (rates, inflation, inventory, transactions). Generated with known ground truth and degraded variants |
| Research corpus | **Real public documents** (central bank, statistics office, IMF / World Bank country reports, BIS and central-bank research, open-access papers) | Evidence for the mapping: what a macro change does |
| Adversarial set | **Synthetic**, kept separate and labelled | Injected instructions, fabricated quotes and other cases real documents will not contain |
| AVM evidence | Real Phase 3 output (`phase3_*_intervals_v2.csv`) | Listing bands and flags. Hà Nội only; asking prices, not appraisals |
| Customer records | Authorized records of the authenticated customer | Personalization. Never leaves our server |

## Offline research

### 1. Source list
The source agent (Qwen) proposes sources and records why each qualifies: publisher type, country, date, open licence. A human signs the list off before the corpus is frozen. Selection criteria are written down.

### 2. Ingestion and indexing (RAG)
1. **Ingest:** store the original file with URL, publisher, publication date, retrieval date and hash.
2. **Extract:** PDF text layer where present; the [OCR service](#ocr-service-pluggable) for scanned PDFs.
3. **Chunk:** keep page numbers and character offsets, so a quote traces to an exact span.
4. **Embed** with the OpenAI API; record the embedding model name and version.
5. **Index** in a local vector store with metadata filters (geography, date, publisher), combined with keyword search for exact terms.
6. **Freeze:** corpus and index get a version ID and hash. Any new document or re-embedding creates a new version.

### 3. Mapping proposals
The mapping agent (Qwen) retrieves chunks and drafts proposals in a strict structure:

```json
{
  "metric": "macro.vn.refi_rate",
  "target": "borrowing_capacity",
  "direction": "negative",
  "lag_months": [1, 6],
  "size_range": "qualitative|low|medium|high",
  "geography": "VN",
  "evidence": [{"chunk_id": "...", "quote": "verbatim text from the chunk"}]
}
```

A mapping is keyed by **target** (borrowing capacity, valuation range), not by one global "value". Signals affecting different targets do not conflict.

`direction` is `positive` (the target moves with the metric) or `negative` (it moves against it); a rate cut raising borrowing capacity is `negative`. Field-level definitions of all schemas are in [`SPECS_AND_SCHEMAS_PLAN.md`](SPECS_AND_SCHEMAS_PLAN.md).

### 4. Mapping verifier
- **Rules:** the quote exists verbatim in the cited chunk; the source is fresh; geography and units match; conflicting sources are flagged.
- **Support check (OpenAI):** does the quote actually back the stated direction? A different model from the mapping agent, per handoff §6 model separation.

### 5. Mapping registry
Each verified mapping set gets a version ID and hash and a status: `unvalidated`, `validated` or `rejected`. Mappings take effect automatically; a human validates later. Until validated, a mapping may only drive low-impact actions.

## Online decision

### 1. Macro-document intake (handoff §5 stages)
File-type and quality checks → the [OCR service](#ocr-service-pluggable) with word boxes and confidence → Qwen second reading → schema-constrained field extraction → deterministic validation (dates, units, ranges, cross-field) → observation `EvidenceItem`s. Disagreement between the two readings, or low confidence, marks the field untrusted.

`EvidenceItem` carries the handoff §3 input contract plus OCR provenance: metric ID, source, observation and publication timestamps, geography, value, unit, baseline, revision status, confidence, freshness deadline, document hash, page, bounding box, OCR text, normalized value, extractor version, reviewer action.

### 2. Evidence bundle
Observations, AVM evidence and authorized customer records, signed and time-bounded.

### 3. Policy engine (deterministic)
Applies the current mapping version to the observations, per target:
1. keep material signals only (each metric has a materiality threshold); drop stale or untrusted ones;
2. sum the signed weights;
3. dominant direction if the net clears the threshold; conflict (must abstain) if signals oppose and the net is small;
4. cap magnitude; an unvalidated mapping allows only `offer_review`, `request_evidence` or widening a range.

Thresholds and weights are policy. Documents can support a direction, not a rule like "act when the change is at least 0.25 pp".

### 4. Decision engine (Qwen)
Writes the recommendation and explanation. It never decides what a signal means; it can only cite the policy engine's result. Output contract (handoff §3, plus the mapping ID):

```json
{
  "recommended_action": "hold|offer_review|request_evidence|no_change|abstain",
  "magnitude": "none|low|medium|high",
  "reason_codes": ["RATE_DROP", "LOCAL_INVENTORY_RISE"],
  "evidence_ids": ["macro.vn.refi_rate.2026-09", "market.thao_dien.inventory.2026-10-01"],
  "mapping_id": "mapping.v3.<hash>",
  "customer_impacts": ["borrowing_capacity", "valuation_range"],
  "uncertainty": 0.21,
  "requires_human_review": true,
  "explanation": "..."
}
```

### 5. Verifier (rules)
Fails closed. Blocks on: invalid schema or unknown action, missing or unknown citations, stale evidence, unit mismatches, numbers in the explanation that do not appear in cited evidence, and conflicts the engine did not abstain on. Routes to a human when the magnitude exceeds the automatic cap or review is requested. Returns `PASS`, `ROUTE_TO_HUMAN` or `BLOCK` with reason codes.

### 6. Human review and explanation
Routed cases go to the Authenticator view. Passed recommendations reach the customer with source labels and timestamps.

## OCR service (pluggable)

OCR sits behind one interface, so the engine can be chosen per document type and swapped without touching the rest of the pipeline. OCR is an evidence extractor, not a truth oracle (handoff §5).

### Interface
Every engine is wrapped in an adapter that takes a page image and returns the same structure:

```json
{
  "engine": "paddleocr",
  "engine_version": "...",
  "model": "...",
  "document_hash": "sha256:...",
  "page": 1,
  "words": [
    {"text": "4.0", "bbox": [x0, y0, x1, y1], "confidence": 0.97, "line_id": 12}
  ]
}
```

An engine that cannot return word or line boxes and a confidence score cannot be the primary engine, because every extracted field must link back to a visible span on the page.

### Candidate engines

| Engine | Role | Notes |
|---|---|---|
| PaddleOCR | ~~Default primary~~ Removed 2026-10-21: its recognizers lack most Vietnamese letters | Detection plus recognition with boxes and confidence; Vietnamese-capable models |
| Tesseract (`vie`) | Baseline / fallback | Mature, word boxes and confidence; weaker on degraded scans |
| VietOCR | Vietnamese recognizer | Recognition only; pair with a detector (for example PaddleOCR's) |
| EasyOCR | Alternative primary | Vietnamese support, boxes and confidence |
| Qwen3.5-9B | Second reading only | Runs on our server; no reliable boxes or confidence, can hallucinate values |
| OpenAI | Optional, public documents only | Never for customer documents |

Check current Vietnamese support and versions before adopting any engine.

### Configuration
The engine is chosen in config per document type, for example:

```yaml
ocr:
  macro_status_document: {primary: paddleocr, second_reading: qwen}
  research_pdf_scanned:  {primary: paddleocr}
  customer_document:     {primary: paddleocr, second_reading: qwen}   # local engines only
```

The engine name, version and model are recorded with every extracted field (`extractor_version` in `EvidenceItem`), so a change of engine is a new version for replay.

### Second reading and trust
When the primary engine and the second reading disagree on a field, or the primary confidence is below the field's threshold, the field is marked untrusted. Untrusted fields cannot feed the policy engine without human review.

### Choosing the engine
Choose by measurement: run the candidate engines on the synthetic macro document's clean and degraded variants and compare field precision/recall and false acceptance of altered documents against its ground truth. Record the comparison and the chosen configuration here.

**2026-10-21:** PaddleOCR 3.7.0's recognizers lack most Vietnamese letters (44 of 67 lowercase), so it cannot be the primary engine for Vietnamese documents. Tesseract 5.5.3 (`tessdata_best`, `vie+eng`) reads both languages; it is accurate on clean, JPEG, noisy, cropped and slightly rotated pages, but fails on strong blur, 100 dpi and 90° rotation, and makes some confident value errors on degraded pages. Results: [`OCR_RESULTS.md`](OCR_RESULTS.md).

**Decision (2026-10-21, AI engineer): Tesseract 5.5.3 is the OCR engine** (`tessdata_best` at `e12c65a`, `vie+eng` for Vietnamese documents, `eng` for English; page segmentation mode 3; orientation detection before reading). PaddleOCR was removed. With extraction and validation, the chain produced **no wrong-and-trusted evidence item** on 868 items of the draft set and 868 of a held-out set.

## Model split and data boundary

| Component | Model | Sees customer data? |
|---|---|---|
| Source agent, mapping agent | Qwen3.5-9B | No |
| Embeddings, support check, gold-label drafts | OpenAI | No |
| OCR (macro document, scanned research PDFs) | OCR service: Tesseract (chosen 2026-10-21), engine configurable; Qwen second reading | No |
| Customer-document OCR (later) | OCR service, local engines only | Yes |
| Policy engine, verifiers, bundle builder | Deterministic code | Yes |
| Decision engine | Qwen3.5-9B on our own server ([`SERVER_SETUP_PLAN.md`](SERVER_SETUP_PLAN.md)) | Yes |

**Rule:** only public research documents and the synthetic macro document may go to OpenAI. Customer data, identity documents and bank records never do; their OCR runs on local engines only.

## Versioning and replay

Every decision records the versions and hashes of: corpus, index, embedding model, mapping (and its validation status), prompts, model and precision, evidence items, verifier results and reviewer actions. Any decision can be replayed exactly (handoff Q8), and rejecting a mapping finds every decision that used it.

## Evaluation

| Area | Metric |
|---|---|
| OCR | Field precision/recall against the synthetic document's ground truth; false acceptance of altered documents |
| Retrieval | Recall@k on gold passages |
| Mapping | Citation precision; fabricated-quote rate (verifier must catch all); conflict handling; agreement with the provisional mapping table |
| Decisions | Unsupported-claim rate, abstention accuracy, counterfactual stability (irrelevant evidence must not change the output) |
| Humans | Override rate and reasons |
| Cost | Tokens, latency and GPU-seconds for every model call; cost per passed task against an all-API baseline on the same frozen set (handoff §9) |

Gold passages: OpenAI drafts the labels; a human reviews a random ~20% and records agreement before they are frozen. The labelling model or prompt differs from the support check's. The adversarial test set is written and frozen before the checks are built, so they are not tuned to it.

## Build order

1. Specs: the synthetic macro-status document (fields, layout, variants, ground-truth format) and the research-corpus source criteria.
2. Schemas: `EvidenceItem`, `RelationshipEvidence`, `MappingProposal`, `PolicyResult`, evidence bundle, recommendation, verifier result (plan: [`SPECS_AND_SCHEMAS_PLAN.md`](SPECS_AND_SCHEMAS_PLAN.md), approved 2026-10-17).
3. Verifiers v0: recommendation verifier and mapping verifier (rules), with a placeholder for the support check.
4. Frozen adversarial test set.
5. Qwen on our own server; baseline the current setup on the frozen set.
6. Document generator and OCR service (adapter interface, PaddleOCR first, engine comparison); RAG ingestion and index; mapping agent; policy engine; decision engine.

## Open

- Server details for Qwen (see [`SERVER_SETUP_PLAN.md`](SERVER_SETUP_PLAN.md)).
- ~~Final OCR engine per document type~~: Tesseract chosen 2026-10-21 for the macro document (see [OCR service](#ocr-service-pluggable)); customer documents still need their own measurement.
- Source validity criteria: draft approved 2026-10-17 (see [`SPECS_AND_SCHEMAS_PLAN.md`](SPECS_AND_SCHEMAS_PLAN.md) Part C; to be written up as `CORPUS_SOURCE_CRITERIA.md`). Who signs off each source list is still open.
- Who reviews the gold-label sample and signs off the source list.
- Weights and thresholds in the policy engine need a domain owner's sign-off.
