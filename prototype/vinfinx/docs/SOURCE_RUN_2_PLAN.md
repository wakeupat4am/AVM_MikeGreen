# Second source run plan (source list v0.2)

*Status: **proposal, awaiting approval** (2026-10-24). Step 5 of the agreed order in `AGENTS.md`. Goal: research evidence for the
three pairs the policy weights, so that the mapping agent can replace the hand-built provisional mapping (MAPPING_AGENT_PLAN.md,
M7). The source agent itself is unchanged (SOURCE_AGENT_PLAN.md, S1–S7); this run changes what it searches for and how its
result joins v0.1.*

## Why

The provisional mapping stays until a registry set covers **refinancing rate, average mortgage rate and listing inventory →
borrowing capacity**. Run 1's coverage report found **no accepted source for any of the three** (it was spread over all 12
pairs, and 147 candidates failed C1 publisher, 60 failed C2 licence). The first mapping run on `corpus-v0.1` confirmed it: no
mapping for those pairs.

## Flow

```
queries_v1 (the three pairs)  →  source agent run 2 (discover, check, Qwen screening)  →  draft list
   + IMF terms read, IMF PDFs downloaded by a person into corpus/inbox
                                                          │
you review run 2 in the site and sign  source_list/v0.2 = v0.1's signed records + run 2's reviewed records
                                                          │
corpus-v0.2 (frozen; v0.1's embeddings come from the cache)  →  mapping agent on corpus-v0.2  →  runner switches if all three pairs are covered
```

## Design choices (V1–V7), for approval

| # | Choice | Proposal |
|---|---|---|
| V1 | Queries | A new `queries_v1.yaml` with **only the three pairs**, more specific than v0 (which mixed all 12): English, e.g. "lending interest rate household credit Vietnam", "interest rate pass-through mortgage lending Vietnam", "policy rate bank lending households emerging markets", "housing inventory unsold stock credit Vietnam"; Vietnamese, e.g. "lãi suất cho vay tín dụng hộ gia đình", "truyền dẫn lãi suất cho vay mua nhà", "tồn kho bất động sản tín dụng ngân hàng". Written by me, not the model (as v0); the list is in the file for you to see before the run |
| V2 | Known documents | Documents already decided in a signed list are skipped (same DOI or title-and-year key), so nothing signed in v0.1 is screened or reviewed again. Run 1's 25 candidates that passed the checks but were over the cap are eligible |
| V3 | Source IDs | One shared numbering across runs: `corpus/source_ids.json`, seeded from run 1's `ids.json`; run 2's documents continue after `src-0317`. Today the numbering is per run, so run 2 would reuse `src-0001` and collide with v0.1 in the corpus |
| V4 | IMF | I read the IMF's copyright and usage terms, as I did for the World Bank and BIS, and record whether they pass C2 under the non-commercial-use decision; you confirm. IMF pages block scripts, so the agent finds IMF documents through OpenAlex and lists the PDFs; **you download the ones that pass screening** into `corpus/inbox` (the run's manual-download list), and the agent picks them up on a re-run. The same for any other blocked PDF |
| V5 | Signing v0.2 | `source_list/v0.2` = every record of the signed v0.1 (with its review, unchanged) + run 2's records with your decisions. The site's Source list tab shows run 2 and signs it "on top of v0.1"; the same checks as the CLI (`python -m ai.sources.review --base source_list/v0.1`). Run before the demo, signed live in the demo, as agreed |
| V6 | Corpus and mappings | A frozen `corpus-v0.2` from the signed v0.2 (v0.1's chunks re-use their cached embeddings; only new documents are embedded), then the mapping agent on `corpus-v0.2`. If all three pairs get an accepted mapping, the runner switches to it automatically (M7); the run record shows it |
| V7 | Measures | Per pair: candidates found, passed checks, screened relevant, accepted, signed; then mapping outcomes on `corpus-v0.2`. Costs: OpenAlex ~$0.05 (shared free tier), Qwen screening ~10 minutes on a free GPU (stopped afterwards), embeddings for new documents < $0.10, support check a few cents |

## Build

1. `queries_v1.yaml`; the source agent's `--queries` option (default stays v0) and the skip of signed documents (V2).
2. Shared IDs (V3), with a migration that writes run 1's IDs into `corpus/source_ids.json` without changing any of them.
3. IMF terms read and recorded in `publishers_v0.yaml` (`terms_checked_by`, after your confirmation).
4. Signing on top of a base list (V5): `apply_review` with a base; the site's Sign button names the base; tests (a v0.1 record
   can neither be dropped nor re-decided).
5. Run 2 (Qwen on a free GPU, stopped afterwards); manual-download list for you; re-run after your downloads.
6. After you sign: `corpus-v0.2`, the mapping agent, and the measures in the findings log.

## Needed from you

- Approval of V1–V7.
- During the step: confirm the IMF terms reading (V4); download the PDFs on the manual list; review run 2 and sign v0.2.
- **For the domain owner:** whether listing inventory should carry weight for borrowing capacity at all. Research on that link is
  thin; if run 2 finds none, the provisional mapping stays because of that one pair, unless the policy drops it (a policy
  decision, PIPELINE.md "Open").

## Honest odds

The two interest-rate pairs are well studied (rate pass-through, household credit), including for Viet Nam, so evidence is
likely, partly in IMF reports. Listing inventory → borrowing capacity is rarely studied anywhere.

## Not in this step

- A "Run source agent" button in the site (a run takes 10+ minutes; OFFLINE_DEMO_PLAN.md).
- Policy weights (domain owner).
