# VinFinx x Vietcombank — internal pilot

Owner-approved target: **10 synthetic-data testers on cecs-server1**, through existing VPN + authorized SSH only. This is not internet banking, a production appraisal, or a credit-decision system. See [results](PILOT10_RESULTS.md) for measured gates and remaining AI blockers.

## Open and sign in

On each tester's computer, connect the approved VPN, configure their own authorized `cecs-server1` SSH alias, then run:

```bash
bash prototype/vinfinx/connect-pilot.sh
```

Keep that terminal open. Visit `http://127.0.0.1:4175/` or `http://127.0.0.1:4175/?demo=ios`. The latter is an iPhone-style web presentation, not a native iOS app. No physical-iPhone/VPN browser delivery has been configured.

Six Customers: `customer01@vinfinx.demo` through `customer06@vinfinx.demo`. Two Advisors: `advisor01@vinfinx.demo`, `advisor02@vinfinx.demo`. Two Authenticators: `authenticator01@vinfinx.demo`, `authenticator02@vinfinx.demo`.

Unique passwords live only in `/home/ubuntu/vinfinx-pilot/state/pilot_accounts.json` (mode600). The owner has a private local copy at `prototype/vinfinx/pilot-private/server_accounts.json`. Never commit it, put it in a URL, or send the whole list to every tester. Distribute only that tester's account through an approved private channel. Do not share the operator's SSH keys. Provisioning does not create SSH/VPN users.

Use separate browser profiles for concurrent roles. Sign out before switching accounts. Cookies are shared across tabs and ports on the same hostname; local4173/4176 and server4175 have separate databases, so logging into one can invalidate the browser's session in another.

## Demonstration journey

1. Customer enters financial inputs, then Property value. **Preview live inference** executes the real model without saving a valuation or changing the score. **Calculate estimate** saves a pending valuation; each saved result includes its computation timestamp, model/input SHA256 and inference ID. SHAP describes associations, not causality.
2. Try LightGBM and **GBM + hierarchy residual calibration (research)**. This is not the pooled structural hierarchy and not a promoted production upgrade. Saved asking-price bands are not appraisals or bank-approved collateral.
3. Documents offers that customer's labelled synthetic PDF/PNG. Upload it and inspect extraction/checks; income remains unverified until an Authenticator explicitly reviews it. All six samples use a fictional employer and net43M VND. This is a test fixture, not evidence of real income.
4. Authenticator reviews/rejects the synthetic document or property; review is single-use, conflict-safe and audited. Only valid, matching synthetic payslips can update demo income. The app cannot approve real lending.
5. Advisor01 sees Customers01–03; Advisor02 sees04–06. Customers cannot read each other's profiles/documents. Advisors cannot upload or approve salaries. Each customer has their own report and assigned-advisor appointment flow.
6. Check services reports AVM/OCR/Qwen separately. Chat explicitly returns Evidence mode when Qwen is unavailable or a generated answer fails the limited numeric/citation checks. No LLM output silently changes prices, health policy or product eligibility.

Health rules, transaction charts, peer comparisons and bank-product assumptions remain **synthetic/provisional**. Maps use historical asking-price aggregates with labelled approximate district-center markers, not live OSM prices or exact property geolocation. Leaflet is vendored; public OSM tiles still require network access.

## Runtime layout and supervision

Everything new is under `/home/ubuntu/vinfinx-pilot` (mode700); no shared project/environment or GPU driver was changed.

```text
env/                       isolated Python3.12 + Tesseract5.5.3
releases/<version>/         allowlisted app, trusted model, synthetic fixtures
current -> releases/...    active release pointer
state/                     SQLite WAL, uploads, samples, private credentials/logs
incoming/                  owner-only private release archives
tessdata_best-e12c65a/      pinned official eng/vie/osd data + hashes
backups/                   consistent DB + referenced uploads + samples + account file
```

Scoped tmux session `vinfinx-pilot` runs `run-pilot-server.sh`, whose supervisor owns one Waitress child. Binding is **127.0.0.1:4175**,16 HTTP threads,64 connections, bounded CPU inference/OCR/chat, request deadlines/rate limits and same-origin/Host checks. One process is intentional: do not start multiple workers against this coordination design. The prediction lock serializes CPU model work and native SHAP; a saturated queue returns503 instead of a fabricated price.

The supervisor restarts a failed child after3s and stops after five failures within five minutes. It captures bounded16KiB output chunks and rotates private child logs at5MiB, retaining one previous log (`service.log.1`), including during a long-lived child. Inspect disk/log size during an extended pilot. **No boot-time system service is installed**: after a host reboot, an authorized operator starts tmux again. This is a pilot, not an HA/SLA deployment.

```bash
tmux ls
tail -n 40 /home/ubuntu/vinfinx-pilot/state/service.log
tmux new-session -d -s vinfinx-pilot 'bash /home/ubuntu/vinfinx-pilot/run-pilot-server.sh'
```

Do not start a duplicate session. To stop, use the current release's `local_app/stop_pilot.py` with the isolated Python; it validates `/proc`, ownership and the recorded supervisor before sending SIGTERM. Never kill unrelated Python/tmux/GPU processes.

## Tests and backups

On cecs-server1:

```bash
bash /home/ubuntu/vinfinx-pilot/run-pilot-server.sh test
bash /home/ubuntu/vinfinx-pilot/run-pilot-server.sh check
```

`test` provisions a temporary independent10-account DB: real HTTP inference, RBAC/CSRF, input sensitivity, idempotency, duplicate uploads, simultaneous review and cross-process backup verification. It never approves documents in the live pilot. `check` uses actual pilot accounts and previews only; it also extracts all12 per-customer PDF/PNG fixtures. Its only live writes are session/audit sign-in/sign-out records. Reports are private `current/local_app/pilot-results/*.json`.

`restart-check` deliberately stops only the validated pilot child and measures supervisor recovery plus unchanged records/credentials. Run it during a quiet acceptance window, not during a user's upload.

For backup use **new absolute private directories**:

```bash
bash /home/ubuntu/vinfinx-pilot/run-pilot-server.sh backup /home/ubuntu/vinfinx-pilot/backups/NEW-NAME
bash /home/ubuntu/vinfinx-pilot/run-pilot-server.sh verify /home/ubuntu/vinfinx-pilot/backups/NEW-NAME
bash /home/ubuntu/vinfinx-pilot/run-pilot-server.sh restore /home/ubuntu/vinfinx-pilot/backups/NEW-NAME /home/ubuntu/vinfinx-pilot/NEW-RESTORE
bash /home/ubuntu/vinfinx-pilot/run-pilot-server.sh verify /home/ubuntu/vinfinx-pilot/NEW-RESTORE
```

Restore refuses existing destinations, rebases upload paths, checkpoints/closes SQLite before hashing, and invalidates copied sessions. It does **not** switch or overwrite live state. Operator inspection/approval precedes a real state switch. Backups include sensitive credential files: retain them privately, with restricted permissions and an owner-defined retention/encryption policy. No recurring backup scheduler, off-host encrypted archive or automatic deletion is configured.

## Rebuild and deploy

Git includes source, scripts, dependency/model hashes and these instructions—not weights, raw data, generated street statistics, DBs, credentials or payslips. A fresh clone alone cannot run the real AVM: obtain the four trusted artifacts from the authorized server/operator and verify `requirements/avm-pilot.sha256`. Never deserialize an untrusted joblib/pickle. For the full local20-row parity test, the original private Phase1 feature/prediction parquet files are also required.

`pilot-setup.sh /home/ubuntu/vinfinx-pilot` installs only the dedicated conda environment/cache. Direct Python packages are pinned in `requirements/pilot.txt`; the measured complete freeze is `requirements/pilot-server.lock`. Tesseract is conda-forge5.5.3; `prepare_tessdata.py` downloads only the previously approved official commit and verifies the committed SHA256 list. The app's pilot salary schema is **English-only** and separately measured; this is not a claim that all Vietnamese customer documents are supported or that Qwen second reading is wired.

Build a private release using `local_app/build_release.py` in the prepared local environment. It includes only app source, three legacy research frontend files, trusted models, historical aggregate snapshot and synthetic fixtures. Upload the ignored archive to the dedicated `incoming/` directory. Run `install_release.py ARCHIVE SHA256` with the isolated server Python: extraction rejects links, absolute/traversal paths and unexpected top-level members, checks source hashes and atomically updates `current`. It never resets state.

Stop the validated old supervisor, start the scoped tmux command, then run the tests above. Startup warms the real AVM and requires actual sample image OCR; missing/corrupt artifacts fail explicitly. For rollback, stop the pilot, point `current` at a previously verified release under this root, and restart. Keep the previous release/backup until acceptance. Inspect database schema compatibility before a future migration; no downgrade/overwrite is automated.

## Remaining AI engineer/admin gates

- **Qwen:** NVML currently fails and no compatible8061 service is listening on cecs-server1. A server admin must restore GPU health/resource availability first. Then follow [Qwen setup](SERVER_SETUP_PLAN.md), select an actually free GPU, use the pinned Qwen3.5-9B bf16 configuration, bind127.0.0.1 only, and verify both health and a real schema-constrained conversation. Do not silently substitute another model, external API or precision. `VINFINX_QWEN_URL` accepts only loopbackHTTP, disables proxies and redirects. `VINFINX_REQUIRE_QWEN=1` can make startup fail closed after that gate is ready; it is currently0 so AVM/OCR remain usable.
- **Macro:** the historical research console/services/corpus/mappings on vishc-server-1 are distinct from this cecs pilot. The frontend can be preserved without those services being deployed. Do not claim macro decisions live, copy secrets or transfer a research corpus silently. Follow the approved handoff/build plans and domain-owner sign-offs; validated versioned deterministic mappings—not ad-hoc LLM opinions—control effects.
- **Hallucination questions:** How do we prevent unsupported macro direction/causality from changing a recommendation? How do we enforce semantic grounding (not only numeric checks) in conversation? How do OCR misreads, forged/cropped documents or a Qwen second reading remain untrusted until deterministic validation and human review? The requirements and pending evaluations are in [AI_ENGINEERING_HANDOFF.md](../AI_ENGINEERING_HANDOFF.md). This pilot does not resolve those research gates.
- Real bank APIs/SSO/MFA, legal appraisals, security/compliance review, malware isolation, at-rest encryption, real-customer OCR evaluation, risk-owner policy/thresholds, physical-iPhone testing and long-duration capacity testing remain outside this synthetic internal acceptance.
