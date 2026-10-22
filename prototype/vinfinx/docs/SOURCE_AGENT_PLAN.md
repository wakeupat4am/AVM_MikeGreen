# Source agent plan

*Status: **S1–S7 approved 2026-10-22** (AI engineer, jimmytrn154); being built. First component of the offline research half (`PIPELINE.md`, "Offline research", step 1). Criteria: [`CORPUS_SOURCE_CRITERIA.md`](CORPUS_SOURCE_CRITERIA.md) (approved 2026-10-17). Decided already: free scholarly and institutional sources only (2026-10-21); the source list is signed off by jimmytrn154; access facts in [`SERVER_SETUP_PLAN.md`](SERVER_SETUP_PLAN.md), "Source access check".*

## What it does

It produces a **draft source list**: one source record per candidate document, accepted or rejected, each with `qualifies_because`, the checks it passed or failed, and verified quotes. The AI engineer reviews every record and signs the list off. Only then is it frozen as the corpus's source list, and ingestion (fetch, extract, chunk, index) runs from it.

The risk is the usual one: a model asked for sources invents titles, URLs and licences. So, as in the rest of the pipeline, **the model proposes, code verifies, a human approves**, and the model is given as little room as possible:

| Step | Done by | Why |
|---|---|---|
| 1. Queries | Code, from a versioned query file built from the metric catalogue | Replayable; the model cannot steer what is searched |
| 2. Discovery | Code: API and feed adapters | Every candidate comes from a real index, so the model **cannot invent a source** |
| 3. Mechanical criteria: C1 publisher type, C2 licence, C3 language, C6 retrievability, C7 date | Code, from metadata and the downloaded file | Facts, not judgement |
| 4. Relevance (C4) and geography (C5) | **Qwen**, constrained by a JSON schema, quoting the text it relied on | Judgement; quotes are checked word for word by code |
| 5. Decision | Code: a fixed rule over steps 3 and 4 | The model's output is one input, not the decision |
| 6. Sign-off | The AI engineer, every record | The agent choosing what it later cites is checking its own work (criteria, "Process") |

## Design choices (S1–S7), for approval

| # | Choice | Proposal |
|---|---|---|
| S1 | Queries | A versioned file `ai/sources/queries_v0.yaml`: for each (metric, target) pair the policy uses, plus close proxies (mortgage affordability, house prices, credit supply), short queries in English and Vietnamese, e.g. "monetary policy house prices Vietnam", "lãi suất cho vay mua nhà". Qwen does not write queries in v0 (it could suggest extra terms later, reviewed like any change) |
| S2 | Discovery adapters | **OpenAlex** (`is_oa:true`, English or Vietnamese, articles, reports, preprints); the **IMF** (DOI prefix `10.5089`) and **ADB** (`10.22617`) through OpenAlex, because their sites block scripts; **World Bank** Documents & Reports API, limited to analytical document types (working papers, economic updates, country reports; not project paperwork); **BIS** RSS feeds (working papers, BIS papers, quarterly review); **SBV** and the **statistics office** (`nso.gov.vn`): their publication list pages, keeping only linked PDFs. Every raw API response is stored with its hash, so a run can be replayed |
| S3 | C1 publisher type (code) | An allowlist mapping each candidate's publisher to a `publisher_type`: central bank, statistics office, IMF, World Bank, ADB, BIS, OECD, university working-paper series. **Journal articles count as peer-reviewed only if the journal is in DOAJ or is an OpenAlex "core" source**; otherwise rejected as "journal not verifiable" (the first test search returned journals of doubtful standing) |
| S4 | C2 licence (code) | Accepted when OpenAlex reports a Creative Commons licence, or when the publisher is a public body on the allowlist whose terms permit storage and short quotation (one terms link per publisher, checked once by a person and stored in the allowlist). "Free to read" without a licence from any other publisher is rejected under C2 |
| S5 | Qwen screening | Input: title, abstract and the text of the first 3 pages (≤ 6,000 characters, extracted with pypdfium2). Output schema: `relevant` (yes/no/unclear), `metrics` and `targets` (enums from the catalogue), `geography` (`vn`, `cross_country_incl_vn`, `comparable_emerging`, `other`), up to 2 `quotes`, `qualifies_because`. Code checks that each quote appears word for word in the text shown (whitespace may differ; at least 20 characters). An unverified quote makes the record "unclear", which a person decides |
| S6 | Volume | At most 150 candidates reach Qwen per run, at most 15 per (metric, target) group, so coverage spreads across pairs; Viet Nam-specific documents first, then newer first. At today's ~3 tokens/s that is about 2–3 hours, run in tmux. Target: 20–40 accepted sources |
| S7 | IMF PDFs | IMF candidates are screened on their OpenAlex metadata and abstract. For the accepted ones, the run writes a download list; a person saves the PDFs into `corpus/inbox/`; code matches them by DOI, hashes them, records `retrieved_by: person`, and screens the full text then |

**Decision rule (code):** `accepted` only if C1, C2, C3, C6 and C7 pass, Qwen says `relevant: yes` with at least one verified quote, and the geography is `vn`, `cross_country_incl_vn` or `comparable_emerging`. Otherwise `rejected` with the first failed criterion, or `unclear` for a person. Every record follows the source-record fields in the criteria, with `proposed_by: source_agent` plus model, prompt version, query-file hash and run ID.

## Review and sign-off

The run writes the draft list to `corpus/source_lists/<version>.json` and a review table (CSV: one row per record, accepted and rejected, with the reasons, quotes and links). The AI engineer marks each row `accept` or `reject` (with a reason when overruling the agent). Code imports the table, records `signed_off_by: jimmytrn154` and `signed_off_at`, and freezes the list as `source_list/v0.1` with its hash. Changes after that create a new version.

## Measures

- **Fabricated sources and quotes:** 0 by construction (candidates come from indexes; quotes are checked). Any unverified quote that reaches "accepted" is a bug.
- **Agent against reviewer:** the share of the agent's `accepted` that you also accept, the share of its `rejected` you overrule, and how often `unclear` was needed.
- **Coverage:** accepted sources per (metric, target) pair the policy uses. Pairs with none are reported, because the mapping agent cannot support them.
- **Optional check on misses:** if you name a few documents you would expect (for example the latest IMF Article IV for Viet Nam), the run reports whether it found them. This is a check only; those documents still pass the same criteria.
- **Cost (handoff §9):** prompt and completion tokens and latency for every Qwen call, in the run report.

## Build

| Part | Where |
|---|---|
| `SourceRecord` schema (`source_record/0.1`, the criteria's fields plus check results and verified quotes) | `ai/schemas/source.py` |
| Polite HTTP client: User-Agent `VinFinx-research/0.1 (mailto:jimmytrn154@gmail.com)`, `robots.txt` check, ≤ 1 request per second per host, retries, raw responses stored and hashed | `ai/sources/http.py` |
| Query file, publisher allowlist | `ai/sources/queries_v0.yaml`, `ai/sources/publishers_v0.yaml` |
| Adapters: OpenAlex, World Bank, BIS, SBV, statistics office | `ai/sources/discover/` |
| Checks, Qwen screening, decision rule, run, review import and freeze | `ai/sources/` |
| Tests on recorded responses (no network in tests) | `ai/tests/test_sources.py` |

Runs in the main environment (`requests`, `pypdfium2`, `pydantic`), calling Qwen through the adapter. Storage: `$VINFINX_ROOT/corpus/` on `/mnt/sdb`: `runs/<run_id>/` (raw responses, report), `originals/<sha256>.pdf`, `inbox/`, `source_lists/`.

**Acceptance:** tests pass on recorded responses; one live run produces a draft list and review table; no accepted record lacks a verified quote or a passed C1–C7 check.

## Not in this step

- Ingestion of the signed-off list: full-text extraction (OCR for scanned PDFs), chunking, embeddings and index. That is the next step.
- The deliberately bad documents (criteria, "Deliberately bad documents"). They are a separate, labelled set, built after the real list.
- Qwen-written queries.

## As built (2026-10-22)

Code: `ai/schemas/source.py` (`source_record/0.1`, `source_list/0.1`), `ai/sources/` (`http.py`, `discover.py` as one module, `checks.py`, `screen.py`, `run.py`, `review.py`, `queries_v0.yaml`, `publishers_v0.yaml`), tests `ai/tests/test_sources.py` (27). Choices made while building:

- **OpenAlex costs money per request:** without a key, every client on this server's IP shares **$0.10 a day** ($0.001 per list request). A run uses 48 requests ($0.048) and stops at a budget (`--openalex-budget`, default $0.08). A free key (`OPENALEX_API_KEY`, $1/day) would need the AI engineer's account.
- **Routes as found:** BIS working papers through OpenAlex by author affiliation, and the BIS RSS feeds filtered to housing-finance topics (the feeds list only the latest papers on any topic; the BIS site sometimes serves a bot challenge). SBV: the annual-report list. Statistics office: the English publication list (the Vietnamese one redirects to it). PDFs are found on landing pages by `citation_pdf_url` or a same-host PDF link.
- **Prompt:** the adapter caps each message at 4,000 characters, so the document is split over several messages (prompt 0.2). Prompt 0.3 spells out the target proxies and judges relevance apart from geography, after 0.2 rejected a Viet Nam credit → house-price study as "not valuation range". The earlier screenings are kept in the run folder.
- **Review table:** rows rejected by a failed check come pre-filled `reject` (change to overrule, with a note); accepted and unclear rows are blank and must be filled in.

### First run: `20261022-sources-1` (prompt 0.3)

| Stage | Result |
|---|---|
| Discovery | 454 found, 342 after de-duplication; OpenAlex $0.048 |
| C1, C2, C3, C7 | 147 failed C1 (mostly journals neither in DOAJ nor OpenAlex core), 60 failed C2 (no licence) |
| Selected (caps S6) | 110; downloads 79 stored, 10 failed, 23 need a person (IMF, or a site blocking scripts) |
| Screening | 100 Qwen calls, 0 errors, 622 s (~22 tokens/s); 158 quotes verified, 36 given but not found in the text (never used as evidence) |
| Decisions | **2 accepted, 21 unclear, 294 rejected** (C4 71, C5 6, C6 10) |

- **Why so few accepted:** 13 of the 21 unclear are relevant public-body documents (World Bank, IMF, BIS, SBV) waiting only for their publisher's terms to be checked once by a person (S4); 7 need a manual download. Most rejections are off-topic search results.
- **Coverage:** accepted sources support only credit growth → valuation range and CPI → valuation range. The rate → borrowing-capacity pairs the policy relies on have none yet.
