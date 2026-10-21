# Project context for agents

## What this repo is
An Automated Valuation Model (AVM) for Vietnamese landed houses (Phases 1–3), plus
VinFinx: a prototype financial-health app with an LLM layer. Read `README.md` for the
AVM, `prototype/vinfinx/README.md` for the prototype.

## My role and current work
I am the AI engineer. Scope: the macro-behavior decision engine, the conversational
assistant and OCR/document intelligence, all governed by
`prototype/vinfinx/AI_ENGINEERING_HANDOFF.md`. The core problem is hallucination control:
an LLM output must never become a trusted fact or decision on its own.

Read before doing anything:
1. `prototype/vinfinx/AI_ENGINEERING_HANDOFF.md`: requirements (source of truth)
2. `prototype/vinfinx/docs/PIPELINE.md`: the intended design
3. `prototype/vinfinx/docs/SERVER_SETUP_PLAN.md`: serving Qwen on this server
4. `prototype/AI_ENGINEER_FINDINGS.md`: background and dated decision log

Status: being built in the build order of `docs/PIPELINE.md`. Built so far: Qwen serving on our
server, the contracts (`ai/catalog/`, `ai/schemas/`), verifier v0.1.1 (`ai/verifier/`), the adversarial
test set and its runner (`ai/evaluation/`), the policy engine (`ai/policy/`, provisional weights and
thresholds), the synthetic document generator (`ai/documents/`), the OCR adapter (`ai/ocr/`, Tesseract with orientation
detection), field extraction and validation (`ai/extraction/`; results in `docs/OCR_RESULTS.md`). Not built: Qwen
second reading, research corpus, mapping agent. Current progress: `prototype/vinfinx/docs/README.md`.

## Hard rules
- Customer data, identity documents and bank records never go to OpenAI or any
  external API. Only public research documents and the synthetic macro document may.
- The Qwen service binds to 127.0.0.1 only. Access is through an SSH tunnel. Never
  open the port publicly.
- Never commit model weights, API keys, credentials, bank data or generated datasets.
  Keys come from environment variables.
- Synthetic data must be labelled synthetic (the macro-status document has a visible
  "SYNTHETIC, TEST DATA" watermark and a fictional publisher).
- Generation (mapping agent, decision engine) runs on Qwen3.5-9B on this server.
  OpenAI is used for embeddings, the verifier's support check and gold-label drafts.
  OCR is a pluggable service: Tesseract 5.5.3 (chosen 2026-10-21; PaddleOCR removed, it cannot read Vietnamese),
  Qwen as second reading (not wired in yet); see "OCR service" in `docs/PIPELINE.md`. Customer documents use local
  engines only.
- The LLM never decides what a macro signal means at decision time. That comes from a
  versioned mapping applied by deterministic code.
- Version and hash everything a decision depends on (corpus, index, mapping, prompt,
  model and precision, evidence), so decisions can be replayed.

## AVM facts to quote correctly
- Phase 3 real-time MedAPE is about 21%; band coverage 81.6–95.5%. Do not quote
  random-fold numbers (for example "18%").
- Phase 3 is Hà Nội only and models asking prices. It is a triage engine, not an
  appraisal.
- Phase outputs are raw model output, not macro data.

## Known issues
- `qwen_api.py` no longer carries the demo context (removed 2026-10-19), so the web demo's chat no longer
  knows the demo customer's figures; the app must send them in the conversation if needed.
- Generation has run at ~3 tokens/s since 2026-10-19 (15–20 on 2026-10-18); cause not confirmed
  (likely CPU contention). Do not quote latency until re-measured.
- OCR runs in its own environment (`$VINFINX_ROOT/env-ocr`: Tesseract); run `ai.ocr.*` and `ai.extraction.evaluate`
  with `env-ocr/bin/python`. Tests that need Tesseract are skipped in the main env; run them in `env-ocr`.
- Load Qwen with `device_map="auto"`: an explicit single device (`{"": 0}`) runs ~7× slower.
- Run Python from the VinFinx environment (`/mnt/disk1/backup_user/24chuong.ta/vinfinx/env`); tests run
  from `prototype/vinfinx` with `python -m unittest discover -s ai/tests -t .`.
- Phase 1 processed data and the `final_scorer.joblib` model are gitignored, so Phase 1
  scripts will not run from a fresh clone.

## Working style
- Don't modify Phase 1–3 code or data unless asked.
- New AI-layer code goes under `prototype/vinfinx/`; new plans go in `prototype/vinfinx/docs/`.
- Record decisions as a dated entry in `prototype/AI_ENGINEER_FINDINGS.md`.
- Ask before choosing anything listed under "Open" in `docs/PIPELINE.md`.
- Never tune the verifier or prompts on the adversarial set's `holdout` cases, and never edit a frozen
  case set; changes go into a new version.
