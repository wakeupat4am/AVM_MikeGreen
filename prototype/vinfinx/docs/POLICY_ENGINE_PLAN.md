# Policy engine plan

*Status: **P1–P5 approved as provisional 2026-10-20** (AI engineer, jimmytrn154); **built** 2026-10-20 (`ai/policy/`, config `ai/policy/policy_v0.yaml`, tests `ai/tests/test_policy.py`). First component of build step 6 (`PIPELINE.md`). The code is deterministic; the numbers it uses are policy and are listed under "Open" in `PIPELINE.md` (a domain owner signs them off). This plan proposes **provisional** values so the engine can be built and tested now; they are marked provisional in the config, like `freshness_days` in the catalogue.*

## What it does

Input: the evidence items (from OCR, or today from the test cases), the current `MappingSet`, a versioned policy config, and the **customer's property district** (district-level metrics count only for that district). Output: a `PolicyResult` (schema already built, `ai/schemas/policy.py`). No model is involved.

For each target the mapping covers:
1. **Signals.** For each evidence item whose metric the mapping links to the target (an item with no mapping for the target is ignored, e.g. unemployment for borrowing capacity):
   - **Effect** = the observed change's sign × the mapping's direction. A metric that fell with direction `negative` (rate cut) **raises** the target; inventory that rose with direction `negative` **lowers** it.
   - **Weight** comes from the policy config for (metric, target), not from the mapping or the LLM.
   - **Excluded** (kept in the result, `material: false`) with a reason: `STALE` (past its freshness deadline), `UNTRUSTED` (untrusted, missing or unreadable), `NO_BASELINE` (no direction can be computed) or `BELOW_MATERIALITY` (the change is below the metric's threshold).
2. **Net weight** = signed sum of the material signals' weights.
3. **Outcome:**
   - `none` if no signal is material;
   - `conflict` if material signals point both ways and |net| is below the conflict threshold (must abstain);
   - otherwise `raises` or `lowers` by the net's sign.
4. **Allowed actions and magnitude cap** from the outcome and the mapping's status (table below).

The result records `policy_version` and the config's hash, so a decision can be replayed and a policy change finds every decision it affected.

## Proposed provisional values (P1–P5)

| # | Item | Proposal | Where it comes from |
|---|---|---|---|
| P1 | Weights (metric → target) | Borrowing capacity: refinancing rate 1.0, mortgage rate 0.8, listing inventory **of the customer's district** 0.9 (another district's inventory is ignored). Other pairs: no weight, so not used until a domain owner adds one | The values the frozen cases were written with; ADV-029 (another district's inventory must not change the result) |
| P2 | Materiality threshold per metric (absolute change from the baseline, in the metric's unit) | Rates (refi, mortgage, CPI, credit growth, unemployment): 0.25 pp; listing inventory and transaction volume: 5% of the baseline; price index: 1 point | Placeholder; every frozen case's change clears it (refi 0.5 pp, inventory +13.6%) |
| P3 | Conflict threshold | Conflict if material signals oppose and \|net\| < 0.5 | Frozen cases: net 0.1 must be a conflict |
| P4 | Actions and cap, **unvalidated** mapping | `raises`/`lowers`: offer_review, request_evidence, no_change, cap low. `none`: request_evidence, no_change, abstain, cap none. `conflict`: abstain, request_evidence, cap none | The frozen cases' policy results |
| P5 | Actions and cap, **validated** mapping | Same actions as P4 plus `hold` for `lowers`; cap `medium` if \|net\| ≥ 1.5, else `low` | Placeholder; no frozen case uses a validated mapping |

## Acceptance test

1. **Reproduces the frozen cases.** For every decision case in `adversarial_v0.yaml`, the engine, given the case's evidence and a mapping with the P1 directions, produces the same policy result the case author wrote (signals, effects, weights, exclusions, outcome, actions, cap). The cases' customer is in Đống Đa (ADV-005, 006). It is a consistency check of the hand-written inputs, not a score, so it covers all cases.
2. **Unit tests:** each exclusion reason, effect sign for both directions, conflict vs dominant direction at the threshold, multiple targets, an item with no mapping, validated vs unvalidated.
3. **Runner switch:** `--policy engine` makes decision cases use the engine's result instead of the hand-written one. Once (1) passes, the two give identical bundles.

## Not in this step

- The mapping set comes from the mapping agent later. Until then the engine is tested with a hand-built `MappingSet` (synthetic, labelled), with directions matching P1.
- The domain owner's sign-off of P1–P5 (`PIPELINE.md` "Open").

## As built (2026-10-20)

- `compute_policy(items, mapping, config, catalogue=, customer_district=, case_id=, now=)` in `ai/policy/engine.py`. Exclusions are checked in the order STALE, UNTRUSTED, NO_BASELINE, BELOW_MATERIALITY. A district-level item counts only for `customer_district`. The mapping is looked up at the item's geography, then its parent (Viet Nam research applies to its districts). A target is included when the config weights it and the mapping covers it.
- **Schema `policy_result/0.2`:** `SignalUse.effect` may be empty for an excluded signal whose change cannot be computed (no value, no baseline, no change), and `PolicyResult.policy_hash` records the config's content hash.
- **Provisional mapping:** `ai/policy/provisional_mapping.py`, a hand-built synthetic `MappingSet` with the P1 directions, until the mapping agent exists.
- **Acceptance:**
  1. The engine reproduces all 30 frozen decision cases' policy results. The one known difference: for an unreadable value (ADV-014, 015), the case author wrote `effect: raises` and the engine leaves it empty. The signal is excluded either way.
  2. Unit tests cover each rule.
  3. The runner's `--policy engine` passes all 13 scripted cases.
- **Still open:** the domain owner's sign-off of P1–P5. Step 6 continues with the document generator and OCR (order proposed 2026-10-19, not yet confirmed).
