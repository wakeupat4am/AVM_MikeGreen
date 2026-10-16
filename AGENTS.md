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

Status: design only. Nothing in `docs/PIPELINE.md` is built yet.

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
  OCR is a pluggable service (PaddleOCR by default, Qwen as second reading; see
  "OCR service" in `docs/PIPELINE.md`). Customer documents use local engines only.
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
- `serve-qwen.sh` runs `vinfinx_qwen_api:app`, but the file is `qwen_api.py`.
- `qwen_api.py` hardcodes a 21 GiB per-GPU limit and the demo context in its system prompt.
- Phase 1 processed data and the `final_scorer.joblib` model are gitignored, so Phase 1
  scripts will not run from a fresh clone.

## Working style
- Don't modify Phase 1–3 code or data unless asked.
- New AI-layer code goes under `prototype/vinfinx/`; new plans go in `prototype/vinfinx/docs/`.
- Record decisions as a dated entry in `prototype/AI_ENGINEER_FINDINGS.md`.
- Ask before choosing anything listed under "Open" in `docs/PIPELINE.md`.
