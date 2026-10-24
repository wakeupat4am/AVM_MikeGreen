"""End-to-end runner CLI (RUNNER_PLAN.md).

    python -m ai.runner documents [--set docs-v0.1-demo-20261023]
    python -m ai.runner run --document LBEO-BUL-2026-10-clean-en-s42 --customer demo-minh [--set ...]

Needs Qwen on 127.0.0.1:8061 for the decision stage (the run stops there with a message otherwise) and the bundle
signing key (VINFINX_BUNDLE_KEY / VINFINX_BUNDLE_KEY_ID, or ~/.config/vinfinx/bundle.env).
"""
from __future__ import annotations

import argparse
import json

from . import DEMO_SET, documents, run_pipeline


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("documents")
    d.add_argument("--set", default=DEMO_SET)
    r = sub.add_parser("run")
    r.add_argument("--document", required=True)
    r.add_argument("--customer", default="demo-minh")
    r.add_argument("--set", default=DEMO_SET)
    a = p.parse_args()
    if a.cmd == "documents":
        for doc in documents(a.set):
            print(doc["document_id"], doc["variant"], doc["language"])
        return
    rec = run_pipeline(a.document, a.customer, set_name=a.set,
                       on_update=lambda rec: print(f"  {rec['stages'][-1]['name']}: {rec['stages'][-1]['status']}",
                                                   flush=True) if rec["stages"] else None)
    print(json.dumps({"run_id": rec["run_id"], "status": rec["status"], "verdict": rec.get("verdict"),
                      "stages": {s["name"]: [s["status"], s.get("seconds"), s.get("error")] for s in rec["stages"]}},
                     indent=1))


if __name__ == "__main__":
    main()
