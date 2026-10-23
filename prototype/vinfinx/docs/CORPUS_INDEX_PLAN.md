# Corpus ingestion and index plan

*Status: **I1–I7 approved 2026-10-22** (AI engineer, jimmytrn154); being built. Offline research steps 2–6 in [`PIPELINE.md`](PIPELINE.md) ("Ingestion and indexing (RAG)"): extract, chunk, embed, index, freeze, and measure retrieval. It starts from the signed-off source list ([`SOURCE_AGENT_PLAN.md`](SOURCE_AGENT_PLAN.md)) and ends with a frozen corpus and index that the mapping agent retrieves from.*

## What it does and why

The mapping agent may only cite the frozen corpus, and the mapping verifier checks every quote word for word against the chunk it came from. So this step has three jobs:

1. **Exact text with exact positions.** Every chunk is an exact substring of one page's text, with page number and character offsets, so any quote traces back to a visible span of the original PDF.
2. **Retrieval that finds the right passages,** in English and Vietnamese, measured against gold passages rather than assumed.
3. **A frozen, hashed version,** so every mapping (and every decision that uses it) can be replayed against exactly the text the agent saw.

No model writes anything in this step except the embeddings (vectors, not text) and the gold-label drafts, which you review.

## Flow

```
signed source list (source_list/v0.1, hash)
   │  originals already stored and hashed by the source agent (corpus/originals/<sha256>.pdf)
   ▼
1. Extract   page text with pypdfium2; a page with no text layer is rendered and read by Tesseract (vie+eng)
   ▼
2. Chunk     windows inside one page, at paragraph or sentence breaks, with page and character offsets
   ▼
3. Embed     OpenAI embeddings for every chunk (public research text only)
   ▼
4. Index     SQLite: chunk table + FTS5 keyword index; embeddings in a NumPy file
   ▼
5. Freeze    corpus/v0.1 and index/v0.1, each with a manifest of hashes; never overwritten
   ▼
6. Measure   gold passages (OpenAI drafts, you review ~20%) → recall@k for keyword, vector and hybrid search
```

## Design choices (I1–I7), for approval

| # | Choice | Proposal |
|---|---|---|
| I1 | Text extraction | **pypdfium2** page by page (installed 2026-10-21). A page with fewer than 50 characters of text but an image is treated as scanned: rendered at 300 dpi in the main env, read by the existing Tesseract adapter in `env-ocr` (`vie+eng`, orientation detection). Each page records which extractor produced it. Text is NFC-normalised; page text is stored as extracted, so offsets stay valid |
| I2 | Chunking | Target **~1,200 characters** (about 300–400 tokens in English, more in Vietnamese), **200-character overlap**, cut at a paragraph break, else a sentence end, else a space. **A chunk never crosses a page**, so a quote always has one page. Pages under 50 characters (covers, blanks) are skipped. `chunk_id` = `<source_id>.p<page>.c<n>`; each chunk carries the hash of its text |
| I3 | Embedding model | **OpenAI `text-embedding-3-large`**, full 3,072 dimensions. It is multilingual (the corpus mixes Vietnamese and English) and the corpus is small (a few thousand chunks ≈ 40 MB of vectors). The exact model ID and its price are checked with the key before the build and recorded in the index manifest. Estimated cost: well under $1 for a v0.1 corpus of 20–40 documents. Called through the REST API with `requests` (the `openai` package cannot be installed while `/mnt/disk1` is full), batched, with retries |
| I4 | Index storage | **SQLite** (`index.sqlite`): a chunk table with metadata (source, page, offsets, publisher, publication date, language, geography) and an **FTS5** keyword index (Unicode tokenizer, diacritics kept, so "lãi suất" is matched exactly). Vectors in `embeddings.npy` (float32), searched by exact cosine similarity. No FAISS or vector database: exact search over a few thousand vectors takes milliseconds, and nothing new is installed |
| I5 | Retrieval | **Hybrid:** keyword (BM25) and vector results merged by reciprocal-rank fusion (k = 60), with metadata filters (geography, publication date, publisher, language). Default 8 chunks per query. The retriever logs the query, the filters and the chunk IDs returned, so a mapping proposal can show what the agent saw |
| I6 | Geography per chunk | From the signed record: `vn` for a document about Viet Nam; `em` for a cross-country or comparable emerging-market study. The mapping verifier today requires the chunk's geography to equal the proposal's (`vn`), so `em` evidence would be refused; whether it may support a Viet Nam mapping as weaker evidence is a decision for the mapping step, not this one |
| I7 | Freeze and replay | `corpus/versions/corpus-v0.1/` holds `manifest.json` (source-list version and hash, each original's hash, extractor versions, chunking parameters, the hash of every page text and chunk) plus `pages.jsonl` and `chunks.jsonl`. `index-v0.1/` holds `index.sqlite`, `embeddings.npy` and a manifest (embedding model, dimensions, date, chunk-set hash). A new document, a changed extractor, new chunking or a re-embedding creates a new version; an existing version is never overwritten |

## Retrieval evaluation (gold passages)

PIPELINE.md: "OpenAI drafts the labels; a human reviews a random ~20% and records agreement before they are frozen. The labelling model or prompt differs from the support check's."

1. **Questions:** one per (metric, target) pair in the query file, in English and Vietnamese (about 24). Example: "How does a change in the refinancing rate affect household borrowing capacity in Viet Nam?"
2. **Candidate pool:** for each question, the union of the top 20 from keyword search, vector search and hybrid search, so the labels do not only cover what one method finds.
3. **Draft labels:** an OpenAI chat model marks each pooled chunk `supports`, `related` or `irrelevant`, quoting the sentence it relied on. Code verifies that quote word for word in the chunk, as everywhere else. The exact model ID is chosen when the key is in place and recorded; the support check (mapping step) will use a different model or prompt.
4. **Your review:** a random 20% of the labels in a review table; agreement is recorded. If it is high, the set is frozen as `gold/v0.1`. If not, the prompt is fixed and the drafts are redone.
5. **Measures:** recall@5, recall@8 and MRR for keyword, vector and hybrid search, overall and for Vietnamese questions. Questions with no supporting chunk at all are reported: they show gaps in the corpus, not in retrieval (today, for example, the rate → borrowing-capacity pairs).

## Data boundary and cost

- **Only public research text goes to OpenAI:** chunks of signed-off corpus documents, the gold questions, and nothing else. The corpus code reads only from the signed source list, never from customer data. The key is read from `~/.config/vinfinx/openai.env` (mode 600), never logged or committed.
- **Every OpenAI call is logged** (handoff §9): service, model, tokens, latency, price-table version, outcome. This is the first piece of the cost accounting.

## Build

| Part | Where |
|---|---|
| Schemas: `CorpusPage`, `CorpusChunk` 0.2 (adds page offsets, text hash, extractor), corpus and index manifests | `ai/schemas/corpus.py` (`CorpusChunk` moves there; `ai/schemas/mapping.py` re-exports it) |
| Extraction (pypdfium2; scanned pages to Tesseract in `env-ocr`) | `ai/corpus/extract.py` |
| Chunker | `ai/corpus/chunk.py` |
| OpenAI client (REST, batching, retries, usage log) | `ai/corpus/openai_client.py` |
| Index build, freeze, hybrid search | `ai/corpus/index.py` |
| Gold-label drafts, review import, retrieval metrics | `ai/corpus/evaluate.py` |
| Tests: offsets are exact, no chunk crosses a page, chunk text equals the page slice, versions are never overwritten, fusion order, data-boundary guard (fake embedder, no network) | `ai/tests/test_corpus.py` |

**Before sign-off:** the code can be built and tried on the PDFs the first source run already stored, as a **development** corpus (`corpus-dev`, never frozen, never cited by a mapping). The frozen v0.1 is built only from the signed list.

**Acceptance:** tests pass; `corpus-v0.1` and `index-v0.1` built from the signed list with manifests; every chunk's text equals the slice of its page text; gold set reviewed and frozen; retrieval results reported.

## For the demo website

The demo is `index.html` (AGENTS.md, "Demo website"). This step writes everything the site needs as JSON: the corpus and index manifests (versions, hashes, document and chunk counts), the retrieval log of each query (query, filters, chunk IDs with page and offsets), and the retrieval results. The Authenticator's **Evidence sources** view can then show which documents are in the frozen corpus and open the exact passage behind any quote. The source-list review and sign-off moves from the CSV into that same view (a separate small plan), served by a loopback API through the existing SSH tunnel.

## Needed from you

- Approval of I1–I7 (in particular **I3, the embedding model**, and **I6**, how cross-country studies are labelled).
- The OpenAI key on the server, in `~/.config/vinfinx/openai.env` (mode 600).
- The signed-off source list, for the frozen build. The development build can start without it.
- Later: reviewing about 20% of the gold labels.

## Not in this step

- The mapping agent, the support check and the mapping registry (next step).
- A reranker, or a second embedding model for comparison. Added only if the measured recall is too low.

## As built (2026-10-22)

Code: `ai/corpus/` (`extract.py`, `ocr_images.py` run in env-ocr, `chunk.py`, `openai_client.py`, `index.py`), `ai/schemas/corpus.py`; `CorpusChunk` stays in `ai/schemas/mapping.py` with four optional fields added (page offsets, text hash, title), so hand-written test chunks and the mapping verifier are unchanged. Tests `ai/tests/test_corpus.py` (11). Choices made while building:

- **Broken text layers go to OCR.** One World Bank Vietnamese PDF (src-0013) uses a legacy Vietnamese font with no Unicode map: its text layer reads "Kinh t tip t!c t\"ng tr…". A page with more than 10 symbols inside words per 1,000 characters is now read by Tesseract instead (src-0013: 21 on average, every other document 0–6). OCR then reads "Kinh tế tiếp tục tăng trưởng mạnh".
- **The index sits inside its corpus version** (`versions/corpus-dev/index/`), with its own manifest, rather than a separate `index-…` folder.
- **Embedding cache** (`corpus/cache/embeddings.sqlite`) keyed by model and input hash: a rebuild re-embedded only the 87 changed chunks; the frozen v0.1 will reuse the dev vectors for unchanged text.
- **Embedding input** is the document title, a newline and the chunk text.
- **Dev builds** are named `corpus-dev*` and may be rebuilt; frozen names may not, and need a signed list.

### Development build `corpus-dev` (draft list `20261022-sources-1`, accepted and unclear documents with a stored PDF)

| | |
|---|---|
| Documents | 17 (14 accepted, 3 unclear); 16 `vn`, 2 `em`; 7 Vietnamese, 10 English |
| Pages | 527 indexed of 543 (16 empty); 31 read by Tesseract (30 of them src-0013) |
| Chunks | 1,633 |
| Embeddings | `text-embedding-3-large`, 3,072 dims; 536,867 tokens on the first build, 47,068 on the rebuild; 29 calls, all first-attempt |

Spot checks: a Vietnamese query on home loans returns the World Bank's Vietnamese passages on mortgage lending; "credit growth house prices Vietnam" returns the VECM study (src-0002). A refinancing-rate query finds only the 1995 World Bank review: the corpus lacks recent rate-transmission evidence (a corpus gap, to address in source run 2).

**Still to do in this step:** the gold-label retrieval evaluation (needs the labelling model chosen) and, after sign-off, the frozen `corpus-v0.1`.
