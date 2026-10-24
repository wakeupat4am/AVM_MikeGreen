# Mapping agent, support check and registry plan

*Status: **M1–M8 approved 2026-10-24** (AI engineer, jimmytrn154; support check kept, `gpt-4.1-mini`); **built 2026-10-24**, first run done. Step 4 of the agreed order in `AGENTS.md`; offline research steps 3–5
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

## As built (2026-10-24)

| Piece | Where |
|---|---|
| Schemas | `NoMapping` and `proposal_generation_schema(..., allow_no_mapping=True)` (the frozen evaluation keeps the proposal-only schema); per-quote verdicts `QuoteSupport` inside `SupportCheck.quotes` (**not** on `MappingVerification`, as M5 said: that would have changed the provisional mapping's content-derived ID, `mapping.v1.d87155943b20`, recorded in every earlier run) |
| Agent | `ai/mapping/` (`run`, `retrieve`, `check_support`, `select_mapping`, `review_set`), CLI `python -m ai.mapping run --corpus corpus-v0.1` / `sets`; registry `corpus/mappings/` (`mapping-v<N>.json`, `.review.json`, `runs/<run_id>.json`); quotes capped at 3 (`maxItems`) |
| Runner | `run_pipeline(..., mapping_fn=select_mapping)`; the run record and Market signals show the mapping source and why |
| API and site | `ai/api/mappings.py`; routes `/mappings`, `/mappings/runs/{id}`, `/mappings/sets/{v}/review`; Evidence sources → **Mappings** tab |
| Tests | `ai/tests/test_mapping_agent.py` (fake Qwen, fake support model, fake embeddings) |

### First run `20261024T023241Z-corpus-v0.1` (Qwen on GPU 1, 2 min 23 s; stopped afterwards)

**0 accepted, 9 no mapping, 3 rejected by rules, 0 support calls ($0).** No mapping set was written; the runner keeps the
provisional mapping. Reading the outputs:
- **Correct refusals:** most pairs (unemployment, inventory, transaction volume, price index, CPI → borrowing capacity) have no
  passage that states the effect, as the source run's coverage report predicted.
- **A false negative:** credit growth → valuation range answered "no mapping" although a retrieved passage says "a positive
  impact of bank credit on the house price index in the long term … in both Ho Chi Minh City … and Hanoi" (`src-0002.p6.c2`).
  The agent treated bank credit as a different variable from credit growth.
- **A good proposal lost to a weak extra quote:** refinancing rate → valuation range, direction negative, first quote verbatim
  ("an increase in interest rates tends to make a decrease in real estate prices", `src-0003`), but two paraphrased Vietnamese
  quotes were not verbatim, and one non-verbatim quote rejects the proposal (by design).
- **Right rejections:** a CPI proposal quoting a variable definition and an equation; a mortgage proposal leaning on a 2014
  source (over the 10-year limit) and a credit-risk sentence.

### Second run `20261024T030058Z-corpus-v0.1` (prompts 0.2: one quote, named proxies; 1 min 32 s; support $0.002)

**2 accepted, 3 no mapping, 3 rejected by rules, 4 rejected by the support check.** First set: **`mapping.v1.a8086a001d3f`**
(unvalidated). The runner keeps the provisional mapping: the set does not cover the three weighted pairs.
- **Accepted:** refinancing rate → valuation range, negative ("an increase in interest rates tends to make a decrease in real
  estate prices", `src-0003`, 2024); credit growth → borrowing capacity, positive ("increased access to credit can encourage
  households and companies to borrow more …", `src-0002`, 2025). The second is close to circular (more credit = more lending):
  a case for the reviewer's judgment.
- **The support check earned its place:** with one quote allowed, Qwen proposed more often, and 4 proposals rested on quotes that
  state no effect (CPI as a control variable, twice; a row of unemployment figures; borrowing costs rather than capacity). The
  rules cannot see this; the support check rejected all four as `insufficient`.
- **Rules:** three proposals quoted 2014 sources (over the 10-year limit).
- **Still a false negative:** credit growth → valuation range answered "no mapping" although the accepted quote itself ends
  "… the demand for housing rises, leading to an increase in housing prices".
- **Unverified fields:** `lag_months` and `size_range` come from Qwen and no quote states them (accepted: 12–24 months, medium;
  0–12, medium). The policy engine does not use them today; they must not be used before something checks them.
