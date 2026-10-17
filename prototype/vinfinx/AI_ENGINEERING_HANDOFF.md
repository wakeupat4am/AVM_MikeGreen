# VinFinx AI engineering handoff

## 1. Product intent

VinFinx combines property valuation, transaction and cash-flow signals, uploaded evidence, and bank product rules into a personalized financial-health report. The LLM should help interpret evidence, explain uncertainty, and propose a next action. It must not become the sole authority for appraisal, credit approval, eligibility, or suitability.

The current prototype uses synthetic values. It proves the interaction model, not the accuracy or regulatory validity of a financial decision.

## 2. Proposed decision architecture

```text
Bank + property + document sources
                │
                ▼
     validation and provenance layer
                │
       ┌────────┴────────┐
       ▼                 ▼
 AVM / score models   macro data service
       │                 │
       └────────┬────────┘
                ▼
 deterministic policy and eligibility engine
                │
                ▼
 LLM recommendation and explanation layer
                │
       ┌────────┴────────┐
       ▼                 ▼
 human review       customer explanation
       │
       ▼
 final bank decision / recorded action
```

The LLM receives a signed, time-bounded evidence bundle. It returns structured recommendations, citations to evidence IDs, uncertainty and an abstention flag. A deterministic service checks the response against policy before anything appears as an actionable recommendation.

## 3. Macro-behavior decision engine

Macro behavior means changing a recommendation because of interest rates, inflation, liquidity, unemployment, inventory, transaction volume, sentiment, seasonality or a detected market regime.

### Required input contract

Every macro feature should contain:

- a canonical metric ID and source;
- observation and publication timestamps;
- geography and applicable customer/property segment;
- revision status and data quality;
- value, unit, transformation and baseline;
- confidence or uncertainty where available; and
- a freshness deadline.

The LLM must not retrieve or invent missing macro values. A separate service should fetch and validate them. The model should only reason over the supplied evidence bundle.

### Suggested output contract

```json
{
  "recommended_action": "hold|offer_review|request_evidence|no_change|abstain",
  "magnitude": "none|low|medium|high",
  "reason_codes": ["RATE_SHOCK", "LOCAL_INVENTORY_RISE"],
  "evidence_ids": ["macro.vn.refi_rate.2026-09", "market.thao_dien.inventory.2026-10-01"],
  "customer_impacts": ["borrowing_capacity", "valuation_range"],
  "uncertainty": 0.21,
  "requires_human_review": true,
  "explanation": "..."
}
```

Unknown actions, missing citations, expired evidence, conflicting units, or unsupported magnitude changes should fail closed. A macro recommendation may widen a range or trigger review; it should not directly alter a legal appraisal or approve a product.

### Primary open question

> **How can we control AI hallucination when macro behavior changes the final recommendation or decision?**

Sub-questions for the AI engineer:

1. Can every material claim and direction-of-effect be grounded to an evidence ID supplied in the request?
2. How should the system detect a plausible but unsupported causal story—for example, “rates fell, therefore this district must appreciate”?
3. What freshness limit and revision policy should apply to each macro series?
4. How should contradictory macro signals be represented, and when must the model abstain?
5. What maximum change may an LLM recommend before deterministic rules or a human must approve it?
6. Can a smaller verifier model or rules engine independently validate units, direction, citations and allowed actions?
7. How will counterfactual tests prove that irrelevant macro evidence does not change the result?
8. How will decisions be replayed exactly after a data revision or prompt/model update?

## 4. Conversational AI

The conversation layer should explain existing results, collect missing non-sensitive context, and route the user to the next safe step. It should not silently modify profile fields, claim bank approval, infer protected attributes, or present a generated number as a verified fact.

### Grounding and tool boundaries

- Retrieve only the authenticated customer's authorized records.
- Separate verified facts, model estimates and user statements in the context object.
- Require tool calls for current balances, product rules, rates and application status.
- Make write operations explicit, reviewable and idempotent.
- Attach a source label and timestamp to figures shown in an answer.
- Refuse or abstain when the required source is missing or stale.
- Redact identity numbers and document images from logs and traces.

### Primary open question

> **How can we control hallucination in the conversational AI while still allowing natural, personalized financial explanations?**

Evaluation should include unsupported-number rate, citation correctness, stale-data use, policy violations, cross-customer leakage, prompt injection, overconfident eligibility language, Vietnamese/English consistency and calibrated abstention.

## 5. OCR and document intelligence

OCR is an evidence-extraction subsystem, not a truth oracle. Recommended stages:

1. malware and file-type validation;
2. image quality, orientation and tamper checks;
3. OCR with word/line bounding boxes and token confidence;
4. document classification and page grouping;
5. schema-constrained field extraction;
6. deterministic validation of formats, totals, dates and cross-field relationships;
7. comparison against bank/customer records; and
8. human review for low confidence, conflict or high-impact fields.

Store the original document hash, page, bounding box, OCR text, normalized value, confidence, extractor version and reviewer action for each field. Never keep only the normalized output.

### Primary open question

> **How can we prevent OCR or a multimodal LLM hallucination from becoming a trusted salary, identity, ownership or property fact?**

Sub-questions:

1. What per-field thresholds should trigger review rather than auto-acceptance?
2. Can extracted values always link back to a visible bounding box on the source page?
3. Which validations are deterministic—for example checksums, date logic, subtotals and account-name matching?
4. How should handwritten, blurred, cropped, altered or adversarial documents be handled?
5. How will the system distinguish “not present” from “present but unreadable”?
6. What data must be encrypted, masked, excluded from model training, or deleted after processing?

## 6. Hallucination-control stack

Use multiple independent controls rather than relying on a stronger prompt:

- **Constrained inputs:** validated evidence objects instead of free-form source dumps.
- **Structured outputs:** strict JSON schema, enumerated actions and bounded magnitudes.
- **Citation enforcement:** every factual claim maps to an evidence ID or OCR span.
- **Deterministic verification:** recompute arithmetic, units, eligibility and policy outside the LLM.
- **Abstention:** explicit missing/conflicting/stale-data outcomes.
- **Human review:** mandatory for high-impact changes and low-confidence evidence.
- **Model separation:** generator, verifier and policy engine have distinct responsibilities.
- **Adversarial evaluation:** prompt injection, forged documents, stale macro data and contradictory evidence.
- **Auditability:** log model/prompt versions, evidence hashes, outputs, verifier results and final reviewer action.
- **Monitoring:** track unsupported claims, overrides, drift, latency, cost and subgroup performance.

## 7. Minimum evaluation gates

Before a production pilot, define thresholds for:

- AVM interval coverage and calibration by geography/property type;
- macro recommendation stability and counterfactual sensitivity;
- unsupported factual claims and citation precision;
- OCR field precision/recall by document and field type;
- false acceptance of altered or low-quality documents;
- conversation policy compliance and privacy leakage;
- abstention accuracy; and
- human override rate and reason distribution; and
- cost per task against the paid-API baseline, at equal quality (§9).

No model update should ship without a frozen evaluation set, change log, rollback target and approval owner.

## 8. Near-term implementation backlog

1. Replace synthetic values with versioned API contracts and provenance IDs.
2. Split the Qwen adapter into authentication, retrieval, policy, inference and audit services.
3. Define the macro evidence and recommendation JSON schemas.
4. Add a verifier that rejects uncited or out-of-policy LLM output.
5. Build the OCR evidence viewer with bounding-box review.
6. Create conversation and document red-team suites in Vietnamese and English.
7. Add model/prompt versioning, trace IDs, latency limits and safe fallback responses.
8. Run a shadow-mode pilot in which humans make every final decision.
9. Record tokens, latency and compute for every model call in the pipeline (§9).

## 9. Token and cost accounting

*Added 2026-10-17 by the AI engineer.*

The project must show that the system built on an open-source LLM (Qwen3.5-9B on our own GPU server) costs less than the same system built on a paid API such as OpenAI, **at equal quality**. That claim needs measured numbers from day one, so every service in the pipeline records its usage on every call.

### What to record per call

| Field | Notes |
|---|---|
| Trace ID, case ID, service and stage | For example source agent, mapping agent, support check, embeddings, gold-label drafts, OCR primary, OCR second reading, decision engine, conversation |
| Model, version, engine and precision | For example `Qwen3.5-9B` bf16 on `transformers`, or the API model name |
| Prompt, completion and cached tokens | From the tokenizer for local models, from the response `usage` for APIs |
| Images or pages | For OCR and multimodal calls |
| Wall-clock latency | Per call |
| GPU-seconds and GPU indices | Local calls only |
| Price-table version | For API calls |
| Outcome | Verifier result (`PASS`, `ROUTE_TO_HUMAN`, `BLOCK`), retry or failure |

Retries, blocked outputs and failed calls are recorded too: they cost the same as successful ones.

### How costs are computed

- **Paid API:** tokens (and images) × the published price, from a dated, versioned price table.
- **Self-hosted:** GPU time × an hourly GPU rate, stated in two ways:
  - **marginal:** only the GPU-seconds spent on our requests, priced at a stated rate (amortized RTX 3090 hardware plus power, or a cloud rate for a comparable GPU);
  - **reserved:** the whole time the GPU is held for the service, idle time included.
  Report both, because the server is shared and the service may sit idle between requests.
- **The current system is hybrid.** OpenAI is still used for embeddings, the support check and gold-label drafts, so their API cost counts towards our system, not the baseline.
- Engineering and operations effort is not in these numbers; state that wherever results are reported.

### How the comparison is run

- Run the **same frozen evaluation set** through our system and through an all-API baseline (the same pipeline with the Qwen calls replaced by a paid model).
- Compare **cost per task**: per decision, per mapping proposal, per document and per conversation turn. Do not compare cost per token, because tokenizers differ between models.
- Compare at **equal quality**: report cost per *passed* output against the §7 gates. A cheaper system that fails more gates is not more cost-efficient.
- **The data boundary still applies:** the baseline may only run on public documents and synthetic data. Customer data never goes to a paid API, even for this comparison.
- State the inference engine with every result. Plain `transformers` serves one request at a time; batching (for example vLLM) lowers the cost per task, so it is part of the configuration being compared.

### Open

- Which paid model(s) form the baseline.
- The hourly GPU rate and how it is derived (hardware amortization or a cloud-equivalent rate).
- Who signs off on the cost methodology.
