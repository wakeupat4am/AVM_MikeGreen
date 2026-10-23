"""Build the corpus and its index, and search it (docs/CORPUS_INDEX_PLAN.md).

    python -m ai.corpus.index build --list <source list .json> --version corpus-v0.1      # signed list, frozen
    python -m ai.corpus.index build --list <draft .json> --version corpus-dev --dev       # development, never frozen
    python -m ai.corpus.index search --version corpus-dev "refinancing rate mortgage lending" [--geography vn]

A frozen version is never overwritten. Only the signed source list's public documents are read, so nothing else can
reach OpenAI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..schemas.common import content_hash
from ..schemas.corpus import CorpusChunk, CorpusDocument, CorpusManifest, CorpusPage, IndexManifest
from ..schemas.source import Decision, Geography, SourceList, SourceRecord
from ..sources.checks import detect_language
from . import CORPUS, VERSIONS
from .chunk import CHUNKER_VERSION, spans
from .extract import extract_pages, pdfium_version

RRF_K = 60
POOL = 50  # candidates taken from each method before fusion
KEYWORD_INDEX = "sqlite fts5 unicode61 remove_diacritics=0"
EMBEDDING_INPUT = "document title, newline, chunk text"
DEV_PREFIX = "corpus-dev"


def _sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def _commit() -> str | None:
    from ..evaluation.runner import git_commit

    return git_commit()


# ---------------------------------------------------------------- corpus


def select_records(source_list: SourceList, dev: bool) -> list[SourceRecord]:
    """Frozen: what the reviewer accepted. Dev: what the agent accepted or left unclear, if stored."""
    if dev:
        return [r for r in source_list.records if r.decision in (Decision.ACCEPTED, Decision.UNCLEAR) and r.sha256]
    if source_list.status != "signed_off":
        raise SystemExit("a frozen corpus needs a signed-off source list (python -m ai.sources.review)")
    return source_list.accepted()


def geography_of(r: SourceRecord) -> str:
    """I6: 'vn' for a document about Viet Nam, 'em' for a cross-country or comparable emerging-market study."""
    return "vn" if r.screening is not None and r.screening.geography is Geography.VN else "em"


def target_dir(version: str, dev: bool) -> Path:
    if dev != version.startswith(DEV_PREFIX):
        raise SystemExit(f"dev builds are named {DEV_PREFIX}*, frozen ones are not (got {version!r})")
    out = VERSIONS / version
    if out.exists():
        if not dev:
            raise SystemExit(f"{out} exists; a frozen version is never overwritten")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    return out


def build_corpus(list_path: Path, version: str, dev: bool, ocr: bool = True, log=print) -> Path:
    source_list = SourceList.model_validate_json(list_path.read_text(encoding="utf-8"))
    records = select_records(source_list, dev)
    out = target_dir(version, dev)
    originals = (CORPUS / "originals").resolve()
    pages_out, chunks_out, docs = [], [], []
    for r in sorted(records, key=lambda r: r.source_id):
        pdf = (originals / (r.sha256.split(":")[1] + ".pdf")).resolve()
        if pdf.parent != originals or not pdf.exists() or _file_sha(pdf) != r.sha256:
            raise SystemExit(f"{r.source_id}: stored original missing or changed ({pdf.name})")
        pages = extract_pages(pdf, ocr=ocr)
        language = r.language or detect_language(" ".join(p.text for p in pages[:5])) or "en"
        geography = geography_of(r)
        n_chunks = 0
        for p in pages:
            pages_out.append(CorpusPage(source_id=r.source_id, page=p.page, text=p.text, text_hash=p.text_hash,
                                        extractor=p.extractor, ocr_mean_confidence=p.ocr_mean_confidence))
            for i, (s, e) in enumerate(spans(p.text), 1):
                text = p.text[s:e]
                chunks_out.append(CorpusChunk(
                    chunk_id=f"{r.source_id}.p{p.page}.c{i}", source_id=r.source_id, document_hash=r.sha256,
                    page=p.page, text=text, publisher=r.publisher or "", publication_date=r.publication_date,
                    geography=geography, language=language, corpus_version=version, page_char_start=s,
                    page_char_end=e, text_hash=_sha(text), title=r.title))
                n_chunks += 1
        docs.append(CorpusDocument(
            source_id=r.source_id, title=r.title, publisher=r.publisher or "",
            publication_date=r.publication_date.isoformat(), language=language, geography=geography,
            original_sha256=r.sha256, pages=_page_count(pdf), pages_indexed=len(pages),
            pages_ocr=sum(p.extractor.startswith("tesseract") for p in pages), chunks=n_chunks))
        log(f"  {r.source_id}: {len(pages)} pages, {n_chunks} chunks ({geography}, {language}) {r.title[:60]}")
    with open(out / "pages.jsonl", "w", encoding="utf-8") as f:
        f.writelines(p.model_dump_json() + "\n" for p in pages_out)
    with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
        f.writelines(c.model_dump_json() + "\n" for c in chunks_out)
    manifest = CorpusManifest(
        corpus_version=version, frozen=not dev, source_list_version=source_list.version,
        source_list_hash=source_list.list_hash, built_at=datetime.now(timezone.utc),
        extractor=f"{pdfium_version()}; scanned pages: Tesseract via env-ocr", chunker=CHUNKER_VERSION,
        documents=docs, pages_hash=_sha("\n".join(p.text_hash for p in pages_out)),
        chunks_hash=_sha("\n".join(f"{c.chunk_id}:{c.page_char_start}:{c.page_char_end}:{c.text_hash}"
                                   for c in chunks_out)), code_commit=_commit())
    (out / "manifest.json").write_text(manifest.model_dump_json(indent=1), encoding="utf-8")
    log(f"corpus {version}: {len(docs)} documents, {len(pages_out)} pages, {len(chunks_out)} chunks -> {out}")
    return out


def _page_count(pdf: Path) -> int:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(pdf))
    try:
        return len(doc)
    finally:
        doc.close()


def load_chunks(corpus_dir: Path) -> list[CorpusChunk]:
    with open(corpus_dir / "chunks.jsonl", encoding="utf-8") as f:
        return [CorpusChunk.model_validate_json(line) for line in f]


# ---------------------------------------------------------------- embeddings cache and index


class EmbeddingCache:
    """Vectors by (model, sha256 of the input), so rebuilding a version from the same text does not pay twice."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("create table if not exists v (model text, input_hash text, vector blob, "
                        "primary key (model, input_hash))")

    def get(self, model: str, hashes: list[str]) -> dict[str, np.ndarray]:
        out = {}
        for i in range(0, len(hashes), 500):
            part = hashes[i:i + 500]
            q = f"select input_hash, vector from v where model=? and input_hash in ({','.join('?' * len(part))})"
            out.update({h: np.frombuffer(b, dtype=np.float32) for h, b in self.db.execute(q, [model, *part])})
        return out

    def put(self, model: str, rows: list[tuple[str, np.ndarray]]) -> None:
        self.db.executemany("insert or replace into v values (?, ?, ?)",
                            [(model, h, v.astype(np.float32).tobytes()) for h, v in rows])
        self.db.commit()


def embedding_input(c: CorpusChunk) -> str:
    return f"{c.title or ''}\n{c.text}"


def build_index(version: str, client=None, log=print) -> Path:
    from .openai_client import EMBEDDING_MODEL, Client

    corpus_dir = VERSIONS / version
    manifest = CorpusManifest.model_validate_json((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    chunks = load_chunks(corpus_dir)
    index_dir = corpus_dir / "index"
    if index_dir.exists():
        if manifest.frozen:
            raise SystemExit(f"{index_dir} exists; a frozen index is never overwritten")
        shutil.rmtree(index_dir)
    index_dir.mkdir()
    client = client or Client("corpus_index")
    inputs = [embedding_input(c) for c in chunks]
    hashes = [_sha(t) for t in inputs]
    cache = EmbeddingCache(CORPUS / "cache" / "embeddings.sqlite")
    known = cache.get(EMBEDDING_MODEL, hashes)
    todo = [i for i, h in enumerate(hashes) if h not in known]
    tokens, model = 0, EMBEDDING_MODEL
    if todo:
        log(f"embedding {len(todo)} of {len(chunks)} chunks with {EMBEDDING_MODEL} ({len(known)} cached)")
        vectors, model, tokens = client.embed([inputs[i] for i in todo])
        cache.put(EMBEDDING_MODEL, [(hashes[i], v) for i, v in zip(todo, vectors)])
        known.update({hashes[i]: v for i, v in zip(todo, vectors)})
    matrix = np.vstack([known[h] for h in hashes]).astype(np.float32)
    np.save(index_dir / "embeddings.npy", matrix)
    db = sqlite3.connect(index_dir / "index.sqlite")
    db.executescript("""
        create table chunks (row integer primary key, chunk_id text unique, source_id text, page integer,
            page_char_start integer, page_char_end integer, text text, text_hash text, title text, publisher text,
            publication_date text, language text, geography text);
        create virtual table chunks_fts using fts5(text, title, content='chunks', content_rowid='row',
            tokenize='unicode61 remove_diacritics 0');
    """)
    db.executemany("insert into chunks values (?,?,?,?,?,?,?,?,?,?,?,?,?)", [
        (i, c.chunk_id, c.source_id, c.page, c.page_char_start, c.page_char_end, c.text, c.text_hash, c.title,
         c.publisher, c.publication_date.isoformat(), c.language, c.geography) for i, c in enumerate(chunks)])
    db.execute("insert into chunks_fts(chunks_fts) values ('rebuild')")
    db.commit()
    db.close()
    im = IndexManifest(
        index_version=f"index-{version.removeprefix('corpus-')}", corpus_version=version,
        corpus_manifest_hash=content_hash(manifest), frozen=manifest.frozen, built_at=datetime.now(timezone.utc),
        embedding_model=model, embedding_dimensions=int(matrix.shape[1]), embedding_input=EMBEDDING_INPUT,
        embedding_tokens=tokens, chunks=len(chunks), keyword_index=KEYWORD_INDEX,
        embeddings_sha256=_file_sha(index_dir / "embeddings.npy"),
        sqlite_sha256=_file_sha(index_dir / "index.sqlite"), code_commit=_commit())
    (index_dir / "manifest.json").write_text(im.model_dump_json(indent=1), encoding="utf-8")
    log(f"index {im.index_version}: {len(chunks)} chunks, {im.embedding_dimensions} dims, {tokens} new tokens "
        f"-> {index_dir}")
    return index_dir


# ---------------------------------------------------------------- search


FILTERS = ("geography", "language", "publisher", "source_id")


class Index:
    def __init__(self, version: str, client=None, base: Path = VERSIONS) -> None:
        self.dir = base / version / "index"
        self.manifest = IndexManifest.model_validate_json((self.dir / "manifest.json").read_text(encoding="utf-8"))
        self.db = sqlite3.connect(self.dir / "index.sqlite")
        self.db.row_factory = sqlite3.Row
        self.matrix = np.load(self.dir / "embeddings.npy")
        self.meta = [dict(r) for r in self.db.execute("select * from chunks order by row")]
        self._client = client
        self._queries: dict[str, np.ndarray] = {}

    @property
    def client(self):
        if self._client is None:
            from .openai_client import Client

            self._client = Client("retrieval")
        return self._client

    def _allowed(self, row: dict, filters: dict) -> bool:
        for k, v in filters.items():
            if k in FILTERS and v is not None and row[k] != v:
                return False
        if filters.get("published_after") and row["publication_date"] < filters["published_after"]:
            return False
        return True

    def keyword(self, query: str, filters: dict, limit: int = POOL) -> list[int]:
        terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 1]
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in dict.fromkeys(terms))
        rows = self.db.execute("select rowid from chunks_fts where chunks_fts match ? order by bm25(chunks_fts) "
                               "limit ?", (match, limit * 4)).fetchall()
        return [r[0] for r in rows if self._allowed(self.meta[r[0]], filters)][:limit]

    def vector(self, query: str, filters: dict, limit: int = POOL) -> list[int]:
        if query not in self._queries:  # one embedding call per distinct query
            self._queries[query] = self.client.embed([query], model=self.manifest.embedding_model)[0][0]
        scores = self.matrix @ self._queries[query]
        order = np.argsort(-scores)
        return [int(i) for i in order if self._allowed(self.meta[i], filters)][:limit]

    def search(self, query: str, k: int = 8, mode: str = "hybrid", **filters) -> list[dict]:
        ranked = {}
        lists = {"keyword": self.keyword(query, filters) if mode in ("hybrid", "keyword") else [],
                 "vector": self.vector(query, filters) if mode in ("hybrid", "vector") else []}
        for name, rows in lists.items():
            for rank, row in enumerate(rows, 1):
                hit = ranked.setdefault(row, {"score": 0.0, "keyword_rank": None, "vector_rank": None})
                hit["score"] += 1 / (RRF_K + rank)
                hit[f"{name}_rank"] = rank
        top = sorted(ranked.items(), key=lambda kv: -kv[1]["score"])[:k]
        return [{**{key: self.meta[row][key] for key in ("chunk_id", "source_id", "page", "page_char_start",
                                                         "page_char_end", "title", "publisher", "publication_date",
                                                         "language", "geography", "text")},
                 **hit, "score": round(hit["score"], 5)} for row, hit in top]


# ---------------------------------------------------------------- CLI


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--list", type=Path, required=True)
    b.add_argument("--version", required=True)
    b.add_argument("--dev", action="store_true")
    b.add_argument("--no-ocr", action="store_true")
    b.add_argument("--no-embed", action="store_true", help="build the corpus only")
    s = sub.add_parser("search")
    s.add_argument("--version", required=True)
    s.add_argument("--k", type=int, default=8)
    s.add_argument("--mode", default="hybrid", choices=["hybrid", "keyword", "vector"])
    for f in FILTERS:
        s.add_argument(f"--{f.replace('_', '-')}")
    s.add_argument("query")
    a = p.parse_args()
    if a.cmd == "build":
        if shutil.disk_usage(CORPUS.resolve()).free < 2_000_000_000:
            sys.exit(f"less than 2 GB free under {CORPUS.resolve()}")
        build_corpus(a.list, a.version, a.dev, ocr=not a.no_ocr)
        if not a.no_embed:
            build_index(a.version)
    else:
        idx = Index(a.version)
        hits = idx.search(a.query, a.k, a.mode, **{f: getattr(a, f) for f in FILTERS})
        for h in hits:
            print(f"{h['score']:.4f} kw={h['keyword_rank']} vec={h['vector_rank']} {h['chunk_id']} "
                  f"[{h['geography']},{h['language']}] {h['title'][:50]}")
            print("     ", " ".join(h["text"].split())[:220])


if __name__ == "__main__":
    main()
