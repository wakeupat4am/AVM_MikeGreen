"""Research corpus: extraction, chunking, embeddings, index and retrieval (docs/CORPUS_INDEX_PLAN.md).
Kept free of heavy imports: `ocr_images` runs in env-ocr, which has no pypdfium2 or requests."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("VINFINX_ROOT", "/mnt/disk1/backup_user/24chuong.ta/vinfinx"))
CORPUS = ROOT / "corpus"  # symlink to /mnt/sdb/shared/chuong/vinfinx/corpus
VERSIONS = CORPUS / "versions"
LOGS = Path(os.environ.get("VINFINX_LOG_DIR", "/mnt/sdb/shared/chuong/vinfinx/logs"))
OCR_PYTHON = ROOT / "env-ocr" / "bin" / "python"
