"""Verifier v0: deterministic rule checks on LLM outputs (handoff §6). No model is involved."""

VERIFIER_VERSION = "verifier.v0.1.1"
# v0.1 (2026-10-19): general reason codes need support.
# v0.1.1 (2026-10-19): evidence IDs are removed before reading numbers even when not cited.
