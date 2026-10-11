# AI engineer findings — context review and next step

*Written 2026-10-11 from a read-through of the project README, the VinFinx prototype README and handoff, the Phase 3 model card, and the prototype code. Nothing has been implemented yet.*

Sources read: [`../README.md`](../README.md), [`vinfinx/README.md`](vinfinx/README.md), [`vinfinx/AI_ENGINEERING_HANDOFF.md`](vinfinx/AI_ENGINEERING_HANDOFF.md), [`../Phase 3/MODEL_CARD.md`](../Phase%203/MODEL_CARD.md), [`vinfinx/qwen_api.py`](vinfinx/qwen_api.py), [`vinfinx/index.html`](vinfinx/index.html), [`vinfinx/serve-qwen.sh`](vinfinx/serve-qwen.sh), [`vinfinx/start-qwen-bridge.sh`](vinfinx/start-qwen-bridge.sh), [`../Phase 1/Rule-based Scene/src/score_listing.py`](../Phase%201/Rule-based%20Scene/src/score_listing.py).

---

## 1. Project context (AVM, three phases)

| Phase | Data | Result |
|---|---|---|
| Phase 1 | 704k national scraped listings | Shrinkage-blend AVM, 12.7% MedAPE / 64.6% within 20% on the serving population; 3-tier serving policy (A auto-serve, B serve with range, C human review) |
| Phase 2 | 10.5k Hà Nội brokerage records | Same accuracy (~13% MedAPE) on 70× less data: data quality was the ceiling, not the model |
| Phase 3 | 27.8k merged Hà Nội | Listing-triage engine: calibrated `[P10, P50, P90]` clearing band and an under / fair / over flag |

Project-level conclusion: the binding constraint is data acquisition (realized transaction prices, time-on-market), not more modelling. The honest, real-time numbers come from Phase 3's rolling-origin backtest, not random-fold validation.

**Phase 3 real-time numbers (the ones to quote):**

- MedAPE about 21% overall; 19–25% in 2024 and 27–37% in 2025 as the market drifts.
- Band coverage (v2, nominal 90%) is 81.6–95.5% at every backtest origin. The median band is wide: P90/P10 about ×4.5.
- 81% of live AVAILABLE inventory is flagged `fair`. The `fair` share holds at 75–81% under ±20% error in the late index.
- Tail flags (`UNDERPRICED`, `STRONGLY_OVER`) on 2025+ listings are sensitive to the thin late index and should be treated as "review", not "confirmed".
- The "will it sell?" classifier has AUC about 0.59, so it is a soft ranking hint at most.

## 2. Your scope (from the VinFinx handoff)

You own the AI layer around the AVM. The LLM interprets evidence, explains uncertainty and proposes a next action. It must never be the sole authority for appraisal, credit approval, eligibility or suitability.

Three subsystems, each with one primary open question:

| Subsystem | Primary open question |
|---|---|
| Macro-behavior decision engine | How do we control hallucination when macro behavior changes the final recommendation or decision? |
| Conversational AI | How do we control hallucination while keeping explanations natural and personalized? |
| OCR / document intelligence | How do we stop an OCR or multimodal-LLM hallucination from becoming a trusted salary, identity, ownership or property fact? |

Architecture in the handoff: validation and provenance layer → AVM / score models and macro data service → deterministic policy and eligibility engine → LLM recommendation and explanation layer → human review → final decision. The LLM receives a signed, time-bounded evidence bundle and returns structured JSON with evidence IDs, uncertainty and an abstention flag; a deterministic service checks it before anything is shown as actionable.

Handoff §8 backlog, in its own order:

1. Versioned API contracts and provenance IDs instead of synthetic values.
2. Split the Qwen adapter into auth, retrieval, policy, inference and audit services.
3. Macro evidence and recommendation JSON schemas.
4. Verifier that rejects uncited or out-of-policy LLM output.
5. OCR evidence viewer with bounding-box review.
6. Vietnamese and English red-team suites for conversation and documents.
7. Model/prompt versioning, trace IDs, latency limits and safe fallbacks.
8. Shadow-mode pilot where humans make every final decision.

Minimum evaluation gates (handoff §7) include AVM interval coverage by geography and property type, macro recommendation stability and counterfactual sensitivity, unsupported-claim rate and citation precision, OCR field precision/recall, false acceptance of altered documents, privacy leakage, abstention accuracy and human override rate. No model update ships without a frozen evaluation set, change log, rollback target and approval owner.

## 3. What exists today

- **LLM service** ([`qwen_api.py`](vinfinx/qwen_api.py)): a FastAPI chat endpoint over Qwen3.5-9B (`image-text-to-text` pipeline, greedy decoding, `enable_thinking=False`, a lock that serializes requests, 12 messages max, 600 tokens max). A hardcoded system prompt carries the demo context (score 782, Thảo Điền townhouse ₫12.48B, range ₫11.95B–13.06B, "confidence 91%", borrowing capacity ₫3.20B). There is no schema, no citations, no verifier, no abstention and no audit log.
- **Frontend** ([`index.html`](vinfinx/index.html)): a single-page mock with Customer, Advisor and Authenticator views. All values are synthetic. The chat panel posts the last 8 messages to `http://127.0.0.1:8061/vinfinx-api/v1/chat/completions`. The street-price map layer uses made-up per-road prices and observation counts.
- **Infra scripts:** [`serve-qwen.sh`](vinfinx/serve-qwen.sh) runs on `cecs-server1` bound to loopback; [`start-qwen-bridge.sh`](vinfinx/start-qwen-bridge.sh) opens an SSH tunnel on port 8061.
- **Not present in the repo:** any macro data or macro service, any OCR or document code, any evaluation or red-team set, any model/prompt versioning or audit logging.

**Real AVM outputs you can build evidence items from:**

- Phase 3: [`phase3_available_intervals_v2.csv`](../Phase%203/data/phase3_available_intervals_v2.csv) and `phase3_unknown_intervals_v2.csv`. Columns: `property_id, dataset_source, district_name, ward_name, total_area, number_of_floors, last_modified_date, asking_B, P10_B, P50_B, P90_B, over_P90_pct, flag, time_adjusted`.
- Phase 3 hedonic market index (+111% from 2023-09 to 2026-07), the only macro-like series in the repo, with thin 2025–26 months.
- Phase 1: [`score_listing.py`](../Phase%201/Rule-based%20Scene/src/score_listing.py) returns prediction, interval, Tier A/B/C and a decision. Its model artifact (`final_scorer.joblib`) is not tracked in git.
- Validation numbers for the evaluation gates come from [`MODEL_CARD.md`](../Phase%203/MODEL_CARD.md).

## 4. Gaps and risks found

1. **The demo property has no model behind it.** It is a Thảo Điền (HCM) townhouse, but Phase 3 covers Hà Nội only, and Phase 1's HCM is its weakest region (23.7% MedAPE, 44% within 20%). The "91% confidence" figure is not a calibrated quantity; Phase 3's band is an 80%-target calibrated interval. The prototype should not imply otherwise.
2. **Phase 3 is a triage engine, not an appraiser.** Every training price is an asking price and `SOLD` is a label, not a sale value. The LLM must never describe Phase 3 output as an appraisal or sale price; the evidence bundle should carry that caveat as a field.
3. **Launch mismatch.** `serve-qwen.sh` runs `uvicorn vinfinx_qwen_api:app`, but the README says to copy `qwen_api.py` into the server root. Unless the file is renamed to `vinfinx_qwen_api.py` on the server, the service will not start.
4. **Forgeable history.** The client supplies the `assistant` turns, and any role other than `assistant` is treated as `user`. Conversation history cannot be trusted, so grounded claims must be re-verified server-side against the evidence bundle rather than against earlier turns.
5. **Confidence vs coverage.** Phase 3 bands are wide (×4.5) because they reflect true drift uncertainty. A UI that shows a tight range with a high-confidence badge would misrepresent the model.
6. **No macro data source.** The handoff requires a separate service to fetch and validate macro series. Whether a real feed exists, and who owns it, is undecided.
7. **Sensitive-data handling is unspecified.** The handoff requires redaction of identity numbers and document images from logs, but there is no logging or redaction code yet.

## 5. Recommended next step

Start with handoff backlog items 3 and 4 (schemas and verifier). Every other control depends on them, and they need no macro feed to begin.

1. **Define the contracts.** Pydantic models for `EvidenceItem` (id, source, observed and published timestamps, unit, value, uncertainty, freshness deadline, revision status, caveats), `EvidenceBundle` (signed, time-bounded) and `MacroRecommendation` (the handoff §3 JSON). Place them in a new `prototype/vinfinx/ai/` package. Where to take the macro?
2. **First real evidence adapter.** Convert Phase 3 band/flag rows and the hedonic index into `EvidenceItem`s with provenance. Mark the thin 2025+ index and tail flags as low-confidence, per the model card.
3. **Verifier v0 (rules only, no LLM).** Reject unknown actions, uncited claims, expired evidence, unit mismatches and numbers in the explanation that do not appear in the cited evidence. Enforce a magnitude cap and fail closed.
4. **Frozen test set before touching the model.** 30–50 cases in Vietnamese and English: clean, stale evidence, contradictory signals, the unsupported causal story ("rates fell, so the district must appreciate"), prompt injection, and counterfactuals where irrelevant evidence is added and the output must not change.
5. **Baseline the current Qwen setup** on that set (unsupported-claim rate, citation precision, abstention accuracy). All later work is measured against it.
6. **Then** refactor `qwen_api.py` to accept an evidence bundle, remove the hardcoded context, use constrained JSON decoding with the verifier in the loop, and add trace IDs and audit logging. OCR and conversation red-team suites follow.

## 6. Decisions needed

- **Which open question first?** Recommended: macro, because its contracts (evidence, citations, verifier, abstention) are reused by the other two. Conversation is the most customer-visible.
- **Is Qwen3.5-9B fixed?** This affects how constrained decoding is designed.
- **Macro data source:** a real feed, or a versioned mock fixture for now?
- **Scope of the demo property:** keep the synthetic Thảo Điền example, or re-point the prototype at Hà Nội listings where Phase 3 actually has validated output?

---

# Daily notes

## 2026-10-11 — verifier design and macro-engine concepts

Follow-up discussion after the findings above. Still no implementation; this records the concepts and the proposed starting design.

### Do the open questions need answering before building?

Not as a separate first step. Most are answered by building and measuring, but four macro sub-questions need a **provisional answer first**, because the schema and verifier take them as config. Mark each provisional and get a risk or business owner to confirm later.

| Sub-question | Type | Starting position |
|---|---|---|
| Q3 Freshness and revision policy per series | Settle up front | A per-series default table: how long each metric stays valid and what happens on expiry |
| Q4 Contradictory signals, when to abstain | Settle up front | Abstain when two cited signals point in opposite directions and neither is dominant |
| Q5 Maximum change before a human approves | Settle up front (risk owner decides) | LLM may widen a range, trigger review or request evidence; it may not change an appraisal or approve a product |
| Q6 Verifier model or rules engine | Settle up front | Start with rules; add a model-based verifier only if rules miss cases in the test set |
| Q1 Grounding every claim | Answer by building | Citation schema + verifier; measure citation precision |
| Q2 Plausible but unsupported causal stories | Answer by building | Write adversarial test cases and see whether the verifier catches them; likely needs iteration |
| Q7 Counterfactual tests | Answer by building | Add irrelevant evidence to a case; the output must not change |
| Q8 Exact replay | Answer by building | Log evidence hashes, prompt and model versions, pinned seed (current greedy decoding with `torch.manual_seed(7)` is a start) |

### Q3–Q6 explained intuitively

Analogy: the LLM is a junior analyst writing a short memo; the verifier is a compliance checker who reads each memo before it goes anywhere.

- **Q3, freshness:** evidence has an expiry date that depends on the item. Local listing inventory changes weekly (say 14 days), a policy rate changes a few times a year (say 60 days), and a thin-data AVM estimate should get a shorter life. On expiry the verifier rejects anything citing it, and the system says it has no fresh data instead of guessing. *Revisions*: some numbers are corrected after publication (for example inflation), so the policy must say whether to use the latest version and what to do about recommendations made on a number later corrected. This is also what makes decisions replayable.
- **Q4, contradictions:** if mortgage rates fell (pushes value up) but local inventory is rising (pushes value down), the LLM must not pick the more convincing story. Opposing signals with no clear winner means abstain and send to a human, like two doctors disagreeing. The hard part is defining "dominant" (for example a much higher confidence or a much larger measured effect), which is tuned on test cases.
- **Q5, authority limit:** like a junior employee's spending limit. The limit is a risk-appetite call, so a risk owner confirms it; the engineer's job is to enforce it in code so the LLM cannot exceed it.
- **Q6, who checks the checker:** plain rules are fast, predictable and cannot hallucinate (does the cited ID exist, do numbers and units match, is the action allowed). A second model can judge whether reasoning follows from the evidence, but is slower and can itself be wrong. Start with rules.

Worked example: for the memo "Rates fell, so I recommend raising this customer's borrowing capacity", the verifier checks, in order, that the rate data is fresh (Q3), that nothing else in the bundle contradicts it (Q4), that the change is within the LLM's authority (Q5), and that the cited ID, number and unit match the evidence (Q6).

### What "macro series" means

A measurement of the wider economy tracked repeatedly over time ("series" is a list of dated values; "macro" means the whole economy or market, not one customer or property). Examples from the handoff §3: interest and refinancing rates (`macro.vn.refi_rate.2026-09`), inflation, liquidity, unemployment, and property-market data such as inventory and transaction volume (`market.thao_dien.inventory.2026-10-01`), plus seasonality. Each series needs its own freshness deadline and revision policy because they change at very different speeds.

The repo has no macro series yet. The closest is the Phase 3 hedonic market index (+111% from 2023-09 to 2026-07). It is a Hà Nội market series, not strictly macro, but is a usable real example while building the evidence schema. Someone has to collect, validate and supply the series; the LLM must never fetch or invent them.

### What the verifier is, and the other roles

The verifier is deterministic code (not an AI) that checks the LLM's output before anyone sees it as an actionable recommendation. It is handoff backlog item 4 and **does not exist in the repo yet**.

| Role (handoff §2, §6) | Job | Exists today? |
|---|---|---|
| Generator | The LLM (Qwen) writing recommendations and explanations | Yes, crudely, in `qwen_api.py` |
| Verifier | Checks LLM output against the evidence, after the LLM runs | No |
| Policy / eligibility engine | Deterministic eligibility and allowed-action rules, before the LLM runs | No |
| Validation and provenance layer | Checks incoming data and attaches source and timestamps | No |
| AVM / score models | Produce valuation and bands | Yes (Phase 1, Phase 3) |
| Macro data service | Fetches and validates macro series | No |
| Human reviewer | Final say on high-impact or low-confidence cases | Only mocked in the UI |

The prototype UI also has three user roles, all mocked: Customer, Advisor, and Authenticator (the closest thing to the human reviewer). One existing guard is verifier-like: Phase 1's `score_listing.py` and serving policy route listings below 20M VND/m² or with flagged prices to Tier C (human review) instead of guessing, which is the style of rule the new verifier should follow.

### Verifier sketch (illustrative, not in the repo)

Input: the LLM's raw JSON and the evidence bundle. Output: `PASS`, `ROUTE_TO_HUMAN` or `BLOCK`, plus reason codes for the audit log.

```python
import re
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field, ValidationError

# ---- contracts (the schemas from step 1) ----
class Evidence(BaseModel):
    id: str
    value: float
    unit: str                                   # "pct", "vnd_bn", ...
    effect_on_value: Literal["raises", "lowers", "neutral"]  # set by the policy engine, NOT the LLM
    expires_at: datetime

class Recommendation(BaseModel):                # strict: unknown actions fail to parse
    recommended_action: Literal["hold", "offer_review", "request_evidence", "no_change", "abstain"]
    magnitude: Literal["none", "low", "medium", "high"]
    reason_codes: list[str]
    evidence_ids: list[str]
    uncertainty: float = Field(ge=0, le=1)
    requires_human_review: bool
    explanation: str

class Result(BaseModel):
    decision: Literal["PASS", "ROUTE_TO_HUMAN", "BLOCK"]
    violations: list[str] = []

ORDER = ["none", "low", "medium", "high"]
MAX_AUTO_MAGNITUDE = "low"                      # Q5: provisional, owned by risk

# ---- the verifier ----
def verify(raw_json: str, bundle: dict[str, Evidence], now: datetime) -> Result:
    try:
        rec = Recommendation.model_validate_json(raw_json)
    except ValidationError:
        return Result(decision="BLOCK", violations=["SCHEMA_INVALID"])

    if rec.recommended_action == "abstain":     # abstaining is always safe
        return Result(decision="ROUTE_TO_HUMAN")

    v = []

    # 1. citations: present, real, fresh (Q3)
    if not rec.evidence_ids:
        v.append("NO_CITATIONS")
    cited = []
    for eid in rec.evidence_ids:
        ev = bundle.get(eid)
        if ev is None:
            v.append(f"UNKNOWN_EVIDENCE:{eid}")
        elif ev.expires_at < now:
            v.append(f"STALE_EVIDENCE:{eid}")
        else:
            cited.append(ev)

    # 2. numbers in the explanation must come from cited evidence
    text = rec.explanation
    for eid in rec.evidence_ids:                # ignore IDs like "...2026-09"
        text = text.replace(eid, "")
    allowed = {f"{ev.value:g}" for ev in cited}
    for num in re.findall(r"\d+(?:\.\d+)?", text):
        if num not in allowed:
            v.append(f"UNSUPPORTED_NUMBER:{num}")

    # 3. contradictions are checked against the WHOLE bundle, not just what was cited (Q4)
    effects = {e.effect_on_value for e in bundle.values() if e.expires_at >= now} - {"neutral"}
    if effects == {"raises", "lowers"}:
        v.append("CONFLICTING_SIGNALS_MUST_ABSTAIN")

    # 4. size of the change (Q5)
    big = ORDER.index(rec.magnitude) > ORDER.index(MAX_AUTO_MAGNITUDE)

    if v:
        return Result(decision="BLOCK", violations=v)
    if big or rec.requires_human_review:
        return Result(decision="ROUTE_TO_HUMAN")
    return Result(decision="PASS")
```

Four worked cases. Bundle X holds one item: refinancing rate 4.0, which raises value. Bundle Y holds that item plus local inventory 18, which lowers value.

| LLM output | Bundle | Result | Why |
|---|---|---|---|
| Cites the rate: "refi rate is 4.0%, offer review", magnitude `low` | X | `PASS` | Everything checks out |
| Same output | Y | `BLOCK: CONFLICTING_SIGNALS_MUST_ABSTAIN` | Cherry-picking: it cited only the supportive signal. Checking the whole bundle catches this. |
| Cites the rate: "rate is 4.0%, so the district will appreciate 15%" | X | `BLOCK: UNSUPPORTED_NUMBER:15` | The 15 is in no evidence; the LLM invented it |
| Cites `macro.vn.cpi.2026-09`, not in the bundle | X | `BLOCK: UNKNOWN_EVIDENCE` | Citation to evidence that does not exist |

Invalid JSON or an action such as `approve_loan` gives `BLOCK: SCHEMA_INVALID` (fail closed).

Design choices: the LLM never decides what a signal means (`effect_on_value` comes from the policy engine, since "rates fell, so the district must appreciate" is exactly the causal claim the LLM cannot be trusted with); abstaining always passes, so the verifier does not push the LLM to bluff; reason codes are specific so they feed the hallucination metrics (unsupported-number rate, unknown-citation rate).

Known gaps in the sketch:
- Number matching is crude and would reject paraphrases such as "about 4%"; a real version needs unit and format normalization (₫12.48B versus 12,480M).
- "Dominant" is not implemented: any two opposing signals force abstention. Q4 allows one signal to outweigh the other, which needs a weight field.
- It checks form only (citations, numbers, freshness, size), not whether the reasoning is valid. A causal story that uses only cited numbers would pass; the adversarial test cases are what catch it, and this is where a model-based verifier might come in.

### Open items after today

- Macro data source (real feed or versioned mock fixture) is still undecided; the question "Where to take the macro?" is noted inline in §5 step 1.
- Draft provisional defaults for Q3–Q6 as a one-page table with a proposed owner for each (offered, not yet done).
- Turn the verifier sketch into a real module with a test file, using the four cases above as the first tests (offered, not yet done).
- The §6 decisions (first open question, whether Qwen3.5-9B is fixed, demo property scope) are still open.
