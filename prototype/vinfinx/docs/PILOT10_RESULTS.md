# Internal pilot acceptance — 2026-11-02

**AVM + local document workflow accepted for a10-person synthetic internal demonstration. Not all AI models are live.** Qwen is blocked by the GPU/NVML issue; macro research and the pooled structural hierarchy are not deployed into customer decisions. This is not production-bank/security/compliance acceptance.

## Delivered

VinFinx x Vietcombank: English web/iPhone presentation, themes/drafts/activity; isolated cecs-server1 loopback4175 service,6 Customer /2 Advisor /2 Authenticator accounts and assigned profiles; six personal synthetic PDF+PNG payslips. The older Mac demo DB was not copied/reset.

Actual frozen LightGBM CPU inference, native SHAP and optional **Phase1 research hierarchy residual calibration** (not the pooled structural hierarchy or a promoted upgrade). Non-persisting Preview and saved/pending valuations are separate. Outputs include version, time, elapsed time, inference ID and input/artifact SHA256. No retraining or Phase1–3 source/data edits.

Local Tesseract5.5.3/PDF extraction, deterministic checks and explicit human income review. Waitress16 threads/64 connections, SQLite WAL/closed connections, bounded inference/OCR/chat, timeouts/rate limits, same-origin/Host authorization, atomic idempotency, duplicate-upload/review protection, supervisor and private rotating logs. No public listener, new SSH user or GPU repair.

## Verification

| Gate | Final result |
| --- | --- |
| UI unit checks |16/16: theme, iPhone shell, account drafts, navigation/task races and copy |
| Mac backend/model/OCR checks |9/9, temporary SQLite; real Vision scan OCR and parity with20 frozen Phase1 predictions |
| Server pilot checks |5/5, temporary10-account DB and actual Waitress/CPU inference; invalid JSON/NaN/Origin/Host, role/customer isolation, input sensitivity and both model methods |
| Income review |Matching personal PDF changes no income on upload; Authenticator approval sets43M; wrong-person sample cannot verify income |
| Concurrent writes |Duplicate uploads return one document; competing review200/409; idempotency replays one valuation, changed body409 |
| Actual deployed service |10 actual accounts,74 calls including12 real previews, no saved property/profile changes |
| Server extraction |12/12 personal fixtures:6 PDF text +6 actual Tesseract PNG, correct43M net and validation |
| Backup/restore |Live backup14 files; new-directory restore, session invalidation and separate-process verification; reverified after final deployment |
| Supervisor recovery |Validated child SIGTERM → healthy in6.27s, persistent records/credential file unchanged |
| Browser smoke |Server Customer login → live preview, trace/native SHAP; Light/Dark. Other roles exercised through server HTTP acceptance, not a new full browser walkthrough |

The first cross-process restore exposed transient WAL/SHM files in the manifest. Fixed by committing/checkpointing/closing SQLite before hashing, with subprocess verification regression. Failed scratch output was preserved privately; live state was never overwritten.

## Final concurrency measurement

Shared cecs-server1 CPU, Python3.12.16 / LightGBM4.6.0 / scikit-learn1.9.1; actual inference/native SHAP, Qwen excluded. Eight rounds per user,10 concurrent clients split6/2/2:

| Measurement | Result |
| --- | --- |
| Calls / actual inferences |182 /64 |
| Unexpected failures / expected403 denials |0 /16 |
| Elapsed |34.04s |
| Successful HTTP p50 / p95 |187.91 /4,128.68ms |
| Inference HTTP p95 |4,133.30ms |

Separate actual-service check:12 previews, p95 **3,127.35ms**,74 total calls;12.57s including12 extractions. Timings are server-local HTTP, not VPN/browser round trips or an SLA/throughput guarantee. A nearby observation showed host load3.99/2.66/1.78 and803GiB free on shared `/home/ubuntu`.

Comparison only: the same local-Mac concurrent test completed182 calls/64 inferences in7.60s, inference p95 **950ms**. Do not quote local speed as server performance. No long-duration soak, log-volume stress, hostile-file sandbox test, degraded/Vietnamese customer-document benchmark or physical-iPhone Safari test was performed.

## Reproducibility and boundaries

Private deployed release: `20261102T115956Z-abfe122fb0`; archive SHA256 `a7cef0fcc6705919d30da31632904b3ba9a370e5208d2eb3b9182e5c7d1c66a9`.

Model digests: [`avm-pilot.sha256`](../requirements/avm-pilot.sha256). Server dependency snapshot: [`pilot-server.lock`](../requirements/pilot-server.lock). OCR: official `tessdata_best@e12c65a915945e4c28e237a9b52bc4a8f39a0cec`, [`hash list`](../requirements/tessdata_best-e12c65a.sha256). English salary schema measured; installed Vietnamese language data is not a validated customer-document feature. Artifact verification and real AVM/OCR warm-up run before serving.

Weights, archives, datasets/street snapshot, generated payslips, SQLite/uploads, credentials, raw private test outputs and screenshots are **not committed**. Reports remain under server `current/local_app/pilot-results/`; owner copies are ignored `pilot-private/`.

## Open gates — not complete

- **Qwen:** NVML still returns `Failed to initialize NVML: Unknown Error`; no compatible8061 service. Admin GPU recovery/resource allocation is needed before pinned private Qwen setup and a real verified generation test. Evidence mode is not an LLM result.
- **Macro / structural research:** separate corpus/pipeline/mapping services and approvals are not deployed here. No silent LLM price, policy or eligibility changes. Semantic conversation verification and Qwen OCR second reading remain research work.
- **Bank/production:** no bank SSO/API/real transactions, signed lending policy, actual bank-user percentile, real-person OCR evaluation, compliance/security certification, host-boot service, HA, scheduled/off-host encrypted backups or extended-load acceptance.

Operation, rollback and AI-engineer/admin next steps: [PILOT10_RUNBOOK.md](PILOT10_RUNBOOK.md). Hallucination-control requirements/questions remain in [AI_ENGINEERING_HANDOFF.md](../AI_ENGINEERING_HANDOFF.md).
