"""Frozen research corpus and its index (docs/CORPUS_INDEX_PLAN.md, I1-I7). `CorpusChunk` lives in `mapping.py`,
where the mapping verifier uses it, and is re-exported here."""
from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, Field

from .common import Sha256, StrictModel
from .mapping import CorpusChunk

__all__ = ["CorpusChunk", "CorpusPage", "CorpusDocument", "CorpusManifest", "IndexManifest"]


class CorpusPage(StrictModel):
    """One page's text exactly as stored; chunk offsets point into it."""

    source_id: str
    page: int = Field(ge=1)
    text: str
    text_hash: Sha256
    extractor: str  # "pdfium:<version>" or "tesseract:<model>"
    ocr_mean_confidence: float | None = None


class CorpusDocument(StrictModel):
    source_id: str
    title: str
    publisher: str
    publication_date: str
    language: Literal["vi", "en"]
    geography: str
    original_sha256: Sha256
    pages: int
    pages_indexed: int
    pages_ocr: int
    chunks: int


class CorpusManifest(StrictModel):
    schema_version: Literal["corpus_manifest/0.1"] = "corpus_manifest/0.1"
    corpus_version: str  # "corpus-v0.1", or "corpus-dev" (never frozen, never cited)
    frozen: bool
    source_list_version: str
    source_list_hash: Sha256
    built_at: AwareDatetime
    extractor: str
    chunker: str
    documents: list[CorpusDocument]
    pages_hash: Sha256  # over every page's text_hash, in order
    chunks_hash: Sha256  # over every chunk's text_hash and offsets, in order
    code_commit: str | None


class IndexManifest(StrictModel):
    schema_version: Literal["index_manifest/0.1"] = "index_manifest/0.1"
    index_version: str
    corpus_version: str
    corpus_manifest_hash: Sha256
    frozen: bool
    built_at: AwareDatetime
    embedding_model: str  # as returned by the API
    embedding_dimensions: int
    embedding_input: str  # what was embedded per chunk
    embedding_tokens: int
    chunks: int
    keyword_index: str  # e.g. "sqlite fts5 unicode61 remove_diacritics=0"
    embeddings_sha256: Sha256
    sqlite_sha256: Sha256
    code_commit: str | None
