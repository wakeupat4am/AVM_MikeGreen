"""Mapping agent CLI (docs/MAPPING_AGENT_PLAN.md).

    python -m ai.mapping run --corpus corpus-v0.1     # Qwen on 127.0.0.1:8061 and the OpenAI key are needed
    python -m ai.mapping sets                         # the registry, with the runner's choice
"""
from __future__ import annotations

import argparse

from . import registry_sets, run, select_mapping


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--corpus", required=True)
    sub.add_parser("sets")
    a = p.parse_args()
    if a.cmd == "run":
        run(a.corpus)
        return
    from ..catalog import load_catalogue
    from ..policy import load_policy_config

    for s in registry_sets():
        print(s.mapping_id, s.status, len(s.entries), "entries")
    mset, source, why = select_mapping(load_policy_config(catalogue=load_catalogue()))
    print(f"runner uses {mset.mapping_id} ({source}): {why}")


if __name__ == "__main__":
    main()
