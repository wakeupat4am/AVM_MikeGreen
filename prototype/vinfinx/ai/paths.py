"""Where VinFinx keeps its files on a server: one place, so no module carries one server's paths (handoff, 2026-10-24).

Values come from the environment, else from the private file ~/.config/vinfinx/paths.env (KEY=value lines; never
committed), else from defaults under the user's home:

    VINFINX_ROOT           environments, model weights, generated bulletins, caches      default ~/vinfinx
    VINFINX_DATA           large data: research corpus, pipeline runs, logs (may be a second disk)   default VINFINX_ROOT
    VINFINX_LOG_DIR        logs                                                           default VINFINX_DATA/logs
    VINFINX_PIPELINE_RUNS  end-to-end runner records                                      default VINFINX_DATA/pipeline_runs

Standard library only: this module is also imported in the OCR environment.
"""
from __future__ import annotations

import os
from pathlib import Path

PATHS_FILE = Path(os.path.expanduser("~/.config/vinfinx/paths.env"))


def _file_values(path: Path = PATHS_FILE) -> dict[str, str]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip().strip("'\"")
    return out


_FILE = _file_values()


def setting(name: str, default: str | Path) -> Path:
    """The environment first, then paths.env, then the default; `~` is expanded."""
    return Path(os.path.expanduser(os.environ.get(name) or _FILE.get(name) or str(default)))


ROOT = setting("VINFINX_ROOT", "~/vinfinx")
DATA = setting("VINFINX_DATA", ROOT)
CORPUS = DATA / "corpus"
LOGS = setting("VINFINX_LOG_DIR", DATA / "logs")
PIPELINE_RUNS = setting("VINFINX_PIPELINE_RUNS", DATA / "pipeline_runs")
