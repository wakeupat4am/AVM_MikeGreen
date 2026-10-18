# Adversarial test set

*Status, 2026-10-18: **draft** (`cases/adversarial_v0.yaml`, version `0.1.0-draft`, 42 cases). Drafted by Claude from the handoff's requirements, for the AI engineer to edit, extend and freeze. The verifier has not been run against these cases and must not be tuned on them before they are frozen.*

## What it is

A fixed set of cases built to make the system fail: missing data, contradictory signals, misleading text, tampered documents, injected instructions. Each case pins down an input and the **outcome** it requires (action, citations, verifier decision), not the wording. Once frozen, every later number (baseline, verifier changes, model bake-off, cost per passed task) is measured on the same set, so results stay comparable and replayable (handoff §7: no model update ships without a frozen evaluation set).

## Case format

Defined in [`cases.py`](cases.py). Three kinds of case:

| Kind | Input | Tests | Expectation fields |
|---|---|---|---|
| `decision` | Evidence items, the policy engine's result, the conversation | Decision engine + recommendation verifier | `actions`, `verifier_decisions`, `must_cite`, `must_not_cite`, `magnitude_at_most`, `required_findings`, `forbidden_phrases` |
| `mapping` | Retrieved corpus chunks and the metric/target asked about | Mapping agent + mapping verifier | `verdicts`, `direction`, `required_findings` |
| `document` | A variant of the synthetic macro document | Generator + OCR + extraction | per-field `presence` / `trust` / `reason`, or `measured_only` |

- **Scripted vs model run.** A case with `scripted_output` (decision) or `scripted_outputs` (mapping) tests the verifier on that exact output, with no model call. Without one, the model generates the output and the case tests model and verifier together. In scripted outputs, write `${MAPPING_ID}`; it is replaced with the case's mapping ID.
- **Evidence items** are written compactly (`metric_id`, `period`, `value`, optional `baseline`, `presence`, `trust`); the builder fills the rest from the catalogue, including freshness. All items are labelled synthetic.
- **The policy result is written by the case author**, because the policy engine is not built yet. A decision case therefore tests the decision engine and verifier *given* a policy result.
- **`invariance`** links counterfactual and language pairs: the named output fields must equal the other case's.
- **`requires`** marks cases that need components not built yet (`generator`, `ocr`, `support_check`); they are skipped until then.
- Top-level keys starting with `x-` hold shared YAML anchors (common items, policies, chunks) and are ignored by the loader.

## What the draft covers

| Category | Cases | Notes |
|---|---|---|
| `ordinary` (controls) | ADV-001, 002 | The system must not abstain on everything |
| `language` | ADV-003, 004, 037 | Vietnamese versions; 003 and 004 are invariance pairs |
| `contradictory_signals` | ADV-005, 006 | Bundle Y from the worked cases |
| `cherry_picking` | ADV-007 | Worked case 2 (scripted) |
| `unsupported_causal` | ADV-008–011 | ADV-009 is worked case 3; ADV-010 is the demo-context number seen in the 2026-10-18 live check |
| `stale_evidence` | ADV-012, 013 | |
| `missing_unreadable` | ADV-014–016 | |
| `unit_conflict` | ADV-017 | |
| `out_of_policy_action` | ADV-018–021 | ADV-021 is the irrelevant reason codes seen in the live check |
| `prompt_injection` | ADV-022–024, 041 | User message, evidence field, Vietnamese, hidden text on the document |
| `forged_history` | ADV-025–027 | ADV-027 is worked case 4 |
| `counterfactual` | ADV-028–030 | Irrelevant evidence must not change the output |
| `fabricated_quote` | ADV-031–033 | |
| `wrong_source` | ADV-034–036 | ADV-034 is the reversed direction seen in the live check |
| `tampered_document`, `document_quality` | ADV-038–042 | Need the generator and OCR |

30 decision cases (20 run the model, 10 scripted), 7 mapping cases (2 run the model, 5 scripted; ADV-034 and 036 need the support check), 5 document cases (need the generator and OCR). 35 cases are runnable today.

## Points to decide while editing

1. **ADV-021:** should an unsupported reason code block an *abstention*, or only be recorded? Drafted as BLOCK.
2. **Internal numbers in explanations.** The verifier treats policy weights (e.g. "net weight 0.1", seen in the live check) as unsupported numbers, so they block. Agree, or allow them?
3. **A conflict on one target forces abstention for the whole recommendation** (verifier v0). Should a clear signal on another target still be actionable?
4. **Coverage gaps worth adding:** more Vietnamese cases, more ordinary cases (abstention accuracy needs both sides), customer-data cases once the conversation layer exists.
5. **Size.** 42 cases is a start; the handoff's gates are better served by 30–50 *runnable* cases than by many placeholders.

## How to freeze

1. Edit `cases/adversarial_v0.yaml`. Check it still loads: `python -m unittest ai.tests.test_cases` (from `prototype/vinfinx`).
2. Set `status: frozen`, `version: "0.1.0"`, `frozen_by: <you>`, `frozen_at: <timestamp with timezone>`.
3. Record the hash in `cases/MANIFEST.md`:
   ```bash
   python -c "from pathlib import Path; from ai.evaluation.cases import load_case_set; s = load_case_set(Path('ai/evaluation/cases/adversarial_v0.yaml')); print(s.version, s.set_hash)"
   ```
4. Commit. From then on the file is never edited; a change is a new file and version (`adversarial_v1.yaml`).
