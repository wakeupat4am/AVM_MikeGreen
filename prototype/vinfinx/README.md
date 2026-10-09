# VinFinx prototype

VinFinx is a browser prototype for three roles:

- customers reviewing a personalized financial-health profile and property estimate;
- advisors exploring customer opportunities; and
- authenticators reviewing suspicious valuation updates and market evidence.

The repository contains only the prototype and the small inference adapter. Model weights, credentials, bank data, identity documents, deployment IDs, and generated datasets are intentionally excluded.

## Files

- `index.html` — single-page interactive prototype, including the OpenStreetMap/Leaflet street-price layer.
- `qwen_api.py` — constrained FastAPI adapter for Qwen 3.5 9B.
- `serve-qwen.sh` — server-side launcher for `cecs-server1`.
- `start-qwen-bridge.sh` — client-side SSH bridge to the private Qwen endpoint.
- `AI_ENGINEERING_HANDOFF.md` — architecture, safety contracts, evaluation requirements, and open research questions.

## Local UI

```bash
python3 -m http.server 4173
```

Then open `http://127.0.0.1:4173`.

## Qwen service

Copy `qwen_api.py` and `serve-qwen.sh` into the root of `llms-conformity-rq1` on `cecs-server1`, then run:

```bash
chmod +x serve-qwen.sh
./serve-qwen.sh
```

On a client that has VPN and SSH access to `cecs-server1`:

```bash
./start-qwen-bridge.sh
```

The browser calls `http://127.0.0.1:8061/vinfinx-api`. The inference service remains bound to loopback on the remote server; no SSH credential is shipped to the browser.

## Current data boundary

All customer, percentile, eligibility, property-price and street-price values in the interface are synthetic demonstration data. OpenStreetMap supplies only the basemap. A production implementation must replace the demo layer with versioned VinFinx estimates and display their observation count, confidence interval, timestamp and provenance.
