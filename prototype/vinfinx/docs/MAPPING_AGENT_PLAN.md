# Mapping agent, support check and registry plan

*Status: **proposal, awaiting approval** (2026-10-24). Step 4 of the agreed order in `AGENTS.md`; offline research steps 3–5
in [`PIPELINE.md`](PIPELINE.md) ("Mapping proposals", "Mapping verifier", "Mapping registry"). It turns the frozen corpus
`corpus-v0.1` into a versioned, verified mapping set: for each catalogue metric and target, which direction the research
supports, with verbatim quotes. It replaces the hand-built provisional mapping once it covers what the policy uses.*

## What it does and why

Today the policy engine applies a hand-built, synthetic mapping (`provisional_mapping.py`): three directions written by us,
backed by made-up quotes. PIPELINE.md's rule is that what a macro signal means comes from research, through a versioned
mapping applied by code, never from the model at decision time. This step builds that research path with the usual pattern:

1. **Qwen proposes** a mapping for one (metric, target) pair from retrieved passages, quoting them, or says the passages do
   not establish one.
2. **Code verifies** every quote word for word, its geography and age (the existing mapping verifier), and **a different model
   (OpenAI) checks** that each quote actually supports the stated direction.
3. **A person validates** the resulting mapping set in the site before it counts as validated; until then it is
   `unvalidated` and the policy allows only low-impact actions (as now).

## Flow

```
catalogue pairs (12, from queries_v0.yaml: metric -> target)
   │  per pair
   ▼
1. Retrieve   hybrid search on corpus-v0.1 (English and Vietnamese question), geography vn, top 8 passages
   ▼
2. Propose    Qwen3.5-9B, schema-constrained: a MappingProposal citing only the retrieved chunks, or "no_mapping"
   ▼
3. Rules      mapping verifier: quotes verbatim, geography, source age, metric known
   ▼
4. Support    OpenAI, per verified quote: supports / contradicts / insufficient for (metric, target, direction)
   ▼
5. Conflicts  two accepted proposals for one pair in opposite directions -> neither enters the set
   ▼
6. Registry   mapping_set v<N> (content-hashed ID, unvalidated), never overwritten; a person validates or rejects it
   ▼
7. Runner     uses the newest non-rejected registry set that covers the policy's weighted pairs; else the provisional one
```

## Design choices (M1–M8), for approval

| # | Choice | Proposal |
|---|---|---|
| M1 | Pairs and retrieval | The 12 (metric, target) pairs of `queries_v0.yaml`. For each, one English and one Vietnamese question (the gold-label questions), hybrid search on **`corpus-v0.1`** filtered to geography `vn`, merged to the **top 8 distinct passages**. The retrieval log (queries, filters, chunk IDs, index hash) is stored with the proposal, so a reviewer sees exactly what the agent saw |
| M2 | "No mapping" is an answer | The generation schema gets a second branch, `{"no_mapping": true, "reason": …}`, so Qwen can say the passages do not establish a direction instead of being forced to invent one. A pair with no accepted proposal simply has no mapping (the policy then ignores the metric for that target) |
| M3 | Geography | Only `vn` evidence supports a mapping in v0.1. The two emerging-market documents (`em`) stay in the corpus but cannot back a Viet Nam mapping (the verifier's existing rule); whether `em` may count as weaker evidence is left for later |
| M4 | Support check model | **OpenAI `gpt-4.1-mini`**, a different model from the mapping agent (Qwen) and from the gold-label drafter (`gpt-5-mini`), so neither the proposer nor the labeller grades its own kind of output. Strict JSON output per quote: verdict and a one-sentence rationale. Public corpus text only. Cost: about 12 pairs × up to 3 quotes, well under $0.05 per run; every call logged with cost |
| M5 | Acceptance rule | A proposal is accepted when every rule passes, **at least one quote `supports`** and **none `contradicts`**; `insufficient` quotes are dropped from the entry. The schema stores one support verdict per quote (`MappingVerification` gets a per-quote list: schema `mapping_set/0.2`) |
| M6 | Registry and review | `mappings/mapping-v<N>.json` under the corpus directory, never overwritten, ID derived from the content; status `unvalidated` → `validated` or `rejected` by a named reviewer in the site (Evidence sources → **Mappings** tab: each entry with direction, lag, size, verified quotes linked to their PDF page, and the support verdicts). Validation changes the status, never the content or ID; every action is logged |
| M7 | Runner switch | The runner uses the newest non-rejected registry set **only if it covers every pair the policy weights** (today: refinancing rate, mortgage rate and listing inventory → borrowing capacity); otherwise it keeps the provisional mapping and says so in the run record and the site. Reason: the signed corpus has **no accepted source for those three pairs** (source-run coverage report), so a v1 set will almost certainly not cover them; switching anyway would turn every decision into "no signal". Step 5 (second source run, aimed at those gaps) is what fills them |
| M8 | Measures | Per pair: proposed / no_mapping / rejected by rules / rejected by support / accepted. On the frozen adversarial set: re-run the **mapping cases of the dev split** with the support check in place (it was a placeholder), never the holdout. Retrieval quality stays on the draft gold labels until the gold review is done |

## Build

1. Schemas: the `no_mapping` branch; per-quote support verdicts (`mapping_set/0.2`); `MappingSet` stays content-addressed.
2. `ai/mapping/`: retrieval per pair, Qwen proposal (prompt `mapping_agent_prompts/0.1`, input split under the 4,000-character
   message limit as elsewhere), rules (existing verifier), support check (`ai/corpus/openai_client.py`, prompt
   `support_check_prompts/0.1`), conflicts, registry write; CLI `python -m ai.mapping run --corpus corpus-v0.1`.
3. Pipeline API and site: Mappings tab in Evidence sources (read, validate, reject with a note); the runner's mapping choice (M7)
   shown in Market signals and the Audit log.
4. Tests with a fake Qwen and a fake support checker (no GPU, no network); then one real run (Qwen on a free GPU, stopped
   afterwards) and the dev-split mapping cases.
5. Update `AGENTS.md`, the tracker, the findings log and `SERVER_SETUP_PLAN.md`.

## Needed from you

- Approval of M1–M8. The choices that are yours in particular:
  - **M4:** `gpt-4.1-mini` for the support check (OpenAI credits; public text only).
  - **M7:** keep the provisional mapping in the runner until a registry set covers the three weighted pairs.
- Later, not for this step: policy weights for the valuation-range pairs (PIPELINE.md "Open", the domain owner), without which
  any valuation-range mapping is recorded but not used.

## Not in this step

- The second source run (step 5), which targets the missing pairs.
- Weights for new pairs in the policy config (domain owner).
- Retrieval tuning (keyword fix) before the gold review.
