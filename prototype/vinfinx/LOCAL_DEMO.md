# VinFinx local application

**2026-11-02 update:** the new10-account server pilot uses loopback4175 through `connect-pilot.sh`; see [PILOT10_RUNBOOK.md](docs/PILOT10_RUNBOOK.md) and [results](docs/PILOT10_RESULTS.md). This document describes the separate older three-account Mac4173 state. The app is now co-branded **VinFinx x Vietcombank**, adds non-persisting live previews, complete inference traces, service readiness and bounded Waitress serving for the pilot. The server uses local Tesseract; the Mac keeps Vision. Existing `.local-demo` records are not copied/reset.

Runs on **http://127.0.0.1:4173** only. This is a functional local research demo, not bank internet banking or a production credit platform. All customer records, payslips and peer comparisons are synthetic. Brand use was authorized by the project owner.

## iPhone presentation demo

Open **http://127.0.0.1:4173/?demo=ios**. The standard web app also has an “Open iPhone demo” link; “Web view” returns to it. This is the same authenticated application and backend, not a second dataset or native iOS build. Cookies are shared with other local tabs.

The presentation has an adaptive phone frame, illustrative 9:41 status bar/Dynamic Island, role-aware bottom navigation, internally scrolling pages and phone-bounded dialogs. The clock/status icons are visual props, not device telemetry. Uses system sans-serif for body copy and `ui-rounded` / SF Pro Rounded where available for headings; platform fallbacks keep it usable without external font requests. Default/Larger text is an in-memory preview preference; Light/Dark/System retains its existing persistence.

At narrow browser widths, presentation side panels disappear and the phone frame resizes without shrinking text into a screenshot. Use the **Aa** button below the frame to toggle text size. Overview keeps the headline score prominent; full score components remain in Health. Menus, tabs, document flows, map, assistant, report and simulator retain existing authorization and financial/model boundaries.

Verification: 16 UI unit checks (`node --test test_ui.cjs test_ios.cjs`) and 9 isolated backend/model/OCR checks pass. Browser checked default desktop/narrow layouts plus393×852, dark/larger text, no horizontal overflow, navigation draft retention, menu closing, assistant containment and simulator modal bounds. Not tested on a physical iPhone or its software keyboard; this is not Xcode Simulator, TestFlight, an App Store package or a complete Safari accessibility audit.

The service remains loopback-only. A physical iPhone cannot reach the Mac via `127.0.0.1`; real-device access needs a separately approved secure hosting/network setup. Do not expose the local demo or SSH model ports publicly for this preview.

## Start / stop

```bash
cd /Users/dienmayhaituyet/Documents/AVM/prototype/vinfinx
bash start-local.sh
```

Keep the terminal open; Ctrl-C stops that run. The agent-launched preview may already be listening: do not start a second copy. Its PID is `.local-demo/server.pid`; validate that PID's command before stopping it. Never kill an unrelated listener.

The installed `.venv-local` uses Python3.12, LightGBM4.6.0 and scikit-learn1.9.1 (matching the saved encoder). For a new environment, use Python3.12, create a dedicated venv and install `requirements/local.txt`. Model artifacts are gitignored and must exist at `Phase 4/phase1/results/group_baseline/`; model loading fails explicitly if they are missing. Research residual calibration additionally needs the frozen Phase1 `group_selected.json` under `Phase 4/combined_20261030/hybrid_calibration/local_phase1_not_combined/results/`.

## Test accounts

One account each: `customer@vinfinx.demo`, `advisor@vinfinx.demo`, `authenticator@vinfinx.demo`. The generated passwords are in **`.local-demo/demo_accounts.json`**, chmod600, not served over HTTP and excluded from Git. They are also handed to the owner in chat. Passwords are PBKDF2-hashed in SQLite; accounts are not regenerated on normal restarts.

Use Sign out before switching accounts in the same browser; cookie-based sessions are shared across tabs of the same browser profile. For concurrent roles use separate browser profiles. Do not reuse these demo credentials anywhere else.

## A complete test journey

1. **Customer**: view score and explicit synthetic percentile; edit financial inputs; calculate a property value with baseline or research hierarchical residual calibration; inspect actual native SHAP model associations.
2. Documents → download synthetic PDF or scan PNG → upload it → inspect extracted salary, arithmetic checks, provenance and pending status. Income must **not** change yet.
3. Sign out → **Authenticator** → Review queue → inspect original and extraction → approve the valid synthetic payslip (net43,000,000 VND). Invalid/mismatched/missing fields cannot be approved to verify income. Reject needs a note. Double review is refused.
4. Sign out → **Customer** → refresh profile: income is now43M and score recalculates. Export the HTML financial-health report (printable from a browser), try the borrowing-budget simulator, request a conversation, ask the evidence assistant.
5. **Advisor**: inspect three assigned synthetic personas, select a customer, read that customer's evidence, value a property, request a conversation, export a report. No salary-upload/verification permission.
6. **Authenticator**: acknowledge/reject property estimates; inspect audit trail, service-status page and evidence/source registry.

Local SQLite and uploads persist under `.local-demo/`. Test automation uses a **temporary independent DB**, so it does not pre-approve the owner's payslip or consume their test journey.

## Real vs simulated

### Appearance and multitasking (2026-10-30)

- Header and sign-in appearance controls support **Light / Dark / System**. The preference is saved locally, applied before paint, and System follows OS changes. Theme storage contains no customer data.
- Visited workspace tabs allow quick switching. Property/financial-form drafts and reviewer notes survive navigation and reload within the browser tab, partitioned by account, role and customer. Drafts use session storage and are cleared on sign-out or session-expiry handling; do not use real customer data. Closing a browser can be subject to browser session restoration, so explicitly sign out on shared devices.
- File selections stay in memory across internal navigation only. Reselect after reload. Chat transcripts and activity history are also session-memory only, not a durable job queue.
- Finx is a non-modal assistant panel: continue working, minimize, reopen, or press Escape. Customer-specific conversations do not mix when an Advisor changes customer. No new AI endpoint or inference capability is introduced.
- Calculations, OCR, chat and saves appear in **Activity** and may finish while you change pages. Keep the browser tab open until completion; reload/closing can interrupt response delivery. Duplicate submits are disabled while a matching task runs. Late responses after logout or superseded navigation cannot repaint the previous workspace.
- Copy uses sentence-case labels, readable SHAP field names, display-only Vietnamese place capitalization and time-aware greetings. Model inputs, price calculations and verification rules are unchanged. Responsive layouts, focus indicators, hover transitions and reduced-motion support apply to both themes.

UI unit checks: `cd local_app && node --test test_ui.cjs` (10 checks). Backend/OCR/model checks remain `../.venv-local/bin/python run_checks.py` (9 checks, isolated temporary DB). Browser checked theme persistence, draft round-trip, non-modal chat while navigating and activity completion at the available narrow viewport; this is not a full cross-browser/desktop accessibility audit.

| Feature | Actual implementation | Boundary |
| --- | --- | --- |
| Login / roles | Server-side sessions and per-endpoint authorization | Local-only demo, not bank SSO/MFA |
| AVM form | Frozen LightGBM artifact, real inference, native SHAP, model hash | Asking-price research model, not an appraisal |
| Research model option | Frozen hierarchical residual offsets applied to GBM | Not the structural-model blend or production promotion |
| Financial health | Versioned deterministic rules and arithmetic | Provisional demo weights, not a bank credit score |
| Peer percentile | Comparison with100 seeded synthetic scores | Not actual bank customers |
| Borrowing simulator | Affordability + collateral arithmetic | Assumed10% rate,20years,40%DSR,60%LTV; not official product terms or eligibility |
| Payslip PDF | Native text + page/word coordinates, local validation | Unverified until human review |
| Scanned image/PDF | macOS Apple Vision OCR, English, local only | Not Tesseract from Chuntran's server; original server OCR code unchanged |
| Review | SQLite transaction, immutable single review, audit | Synthetic evidence review only, no bank decision |
| Area exploration | Historical Phase1 asking-price aggregates, public OSM basemap; road geometry where available | Not OSM-supplied prices; fallback markers are explicitly district centers |
| Assistant | Authorized current-customer facts, optional Qwen, numeric/citation checks | Labelled deterministic Evidence mode if Qwen unavailable or output blocked; no guarantee of full semantic verification |
| Monthly chart | Synthetic illustration scaled to current profile | Not imported transactions |
| Research console | Preserved legacy UI under Authenticator-only route | Requires separate server API/artifacts; not production SSO for those services |

No bank API integration, malware-scanning claim, legal ownership verification, causal SHAP claim, approval of credit or automatic signed research mapping has been added. Full production compliance, encryption-at-rest beyond OS/file protections, real customer use and risk-owner thresholds remain out of scope.

## Qwen / cecs-server1

```bash
bash connect-cecs-local.sh
```

This only opens loopback SSH forwarding for8061/8062, never starts/stops a GPU process. Default Qwen URL is `http://127.0.0.1:8061/vinfinx-api`; override `VINFINX_QWEN_URL` when starting the app if a compatible service uses a different path.

**Verified at implementation:** SSH connects, but NVML reports `Failed to initialize NVML: Unknown Error` and neither8061 nor8050 health endpoint responded on the server. No Qwen service was started or driver repaired. Therefore the current app must be handed off as **Evidence mode**, not live Qwen. Restore the server GPU/service before claiming live LLM integration. Nothing is sent to OpenAI or another external AI API by this local app.

## Artifacts / verification

Synthetic salary files: repository `output/pdf/vinfinx-sample-payslip.pdf` and `.png`; fictional employer, watermark and explicit non-bank-issuer notice. They contain no signature, identity number, real bank account or real salary.

```bash
cd local_app
../.venv-local/bin/python run_checks.py
```

Tests cover auth/RBAC/CSRF, unauthorized customer access, real model sensitivity, valid/invalid upload, native PDF + actual Vision image OCR, income update only after review, report/audit/chat/appointments, financial arithmetic and parity against20 frozen benchmark predictions. The machine-readable outcome is `local_app/test-results/checks.json` (ignored).

Official partner image source: [Vietcombank](https://www.vietcombank.com.vn/), asset `/-/media/Project/VCB-Sites/VCB/Home-page/VCB-Logo/Logo-VCB-no-60.png?ts=20240605031450`. Saved without redesigning the logo. Leaflet/OSM need network; statistics and other local workflows work without map tiles. `prepare_data.py` creates only aggregate location/price data and never exports listing contacts. Optional `prepare_map_geometry.py` requests public road geometry only.

Current map snapshot:363 street aggregates across12 Hanoi districts. Both public geometry requests timed out during this build, so the delivered map uses explicitly labelled district-center markers and a searchable street-statistics list, **not exact street polylines or parcel geolocation**. The optional geometry loader and UI support are prepared for a later successful fetch. The currently running SSH tunnel can be stopped independently; it does not own/start a Qwen service.

The original2026-10-30 local implementation did not include commit/push/deployment or Phase1–3 edits. The owner separately authorized the internal server pilot and GitHub push on2026-11-02; only scoped source/docs/dependency hashes are published, never runtime records, credentials, models or generated datasets.
