# Internal 10-user pilot and live inference

2026-11-02. Owner requested implementation, live inference and internal VPN/SSH delivery on cecs-server1. Ten people across all three roles; proposed allocation6 Customer,2 Advisor,2 Authenticator. Preserve the existing local demonstration and all Phase1–3 sources/data.

## Scope and acceptance

1. Same-origin web/iPhone UI named **VinFinx x Vietcombank**; research-only/bank-issuer disclaimers stay visible.
2. Authenticated live AVM inference from actual inputs, optional persisted valuation. Frozen LightGBM plus existing research residual calibration only; no promotion of pooled structural research. Each output records input digest, all loaded artifact digests, time and version.
3. Model catalogue/readiness distinguishes loaded/warmed AVM, unavailable Qwen and OCR engine. An unavailable service never masquerades as a successful inference.
4. Single-process bounded HTTP serving, WAL SQLite, closed connections, timeouts, bounded inference/OCR/chat concurrency, rate limits and duplicate-upload protection. No real customer data; this is an internal synthetic-data pilot, not a bank production deployment.
5. Ten distinct accounts/profiles and advisor assignments in a separate pilot state directory. No reset/copy of the owner's local DB or uploads. Account credentials remain owner-only, ignored and out of HTTP/logs.
6. Functional/security tests,10 simultaneous clients with real inference, artifact parity, input sensitivity, per-account isolation, simultaneous review/upload protection and restart/persistence checks. Report measured latency/errors, not a throughput guarantee.
7. Versioned release bundle and runbook for loopback-only service under supervision, SQLite+upload backup/restore verification, dependency/artifact manifests and rollback. Users connect through existing authorized VPN+SSH forwarding. No new SSH users, firewall changes, public listeners, cloud hosting or credentials committed.

## Existing blocker

cecs-server1 survey on2026-11-02: SSH works, GPU NVML fails, no known running tmux Qwen service. Do not repair shared drivers or load another model silently. Qwen live acceptance requires an actually healthy compatible endpoint and a verified request. AVM CPU deployment can proceed independently; label the remaining AI gate explicitly.

## Boundaries

Macro research console remains separate until its service, corpus and approved mappings are installed on the chosen host. Never invent macro signals or use unvalidated LLM output to change prices/eligibility. Tesseract/native PDF OCR is local-only, human review still required. Any environment installation is isolated and pinned; no edits to other server projects/environments.
