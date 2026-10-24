"""Research corpus: extraction, chunking, embeddings, index and retrieval (docs/CORPUS_INDEX_PLAN.md).
Kept free of heavy imports: `ocr_images` runs in env-ocr, which has no pypdfium2 or requests."""
from __future__ import annotations

import os
from pathlib import Path

from ..paths import CORPUS, LOGS, ROOT  # one place for server paths: ai/paths.py

VERSIONS = CORPUS / "versions"
OCR_PYTHON = ROOT / "env-ocr" / "bin" / "python"
