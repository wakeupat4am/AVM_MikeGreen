# Research-corpus source criteria

*Status: approved 2026-10-17 (decision D9 in [`SPECS_AND_SCHEMAS_PLAN.md`](SPECS_AND_SCHEMAS_PLAN.md), Part C). Applies to the offline research half of [`PIPELINE.md`](PIPELINE.md): the source agent proposes sources against these criteria, and a human signs off the list before the corpus is frozen. Who signs off is still open.*

## Purpose

The mapping agent may only cite the frozen corpus, so the corpus decides what the system can "know" about how a macro change affects a target. These criteria keep that corpus credible, legally storable and relevant, and make every inclusion decision reviewable.

A citation proves that a quote exists, not that the source is credible or that it supports the claim (2026-10-14 notes). Credibility is handled here, at selection time; support is handled by the mapping verifier's support check.

## Acceptance criteria

A document enters the corpus only if it meets **all** of these:

| # | Criterion | Accepted | Not accepted |
|---|---|---|---|
| C1 | Publisher type | Central bank (including its research series); national statistics office; IMF, World Bank, ADB, BIS, OECD; peer-reviewed journal article; working paper from a university or one of the institutions above | News articles, broker or developer marketing, blogs, social media, anonymous documents, LLM-generated text |
| C2 | Licence | Public, and storage plus short verbatim quotation allowed (open-access licence, public-sector publication, or terms that permit it) | Paywalled or subscription reports; terms that forbid storage or reuse |
| C3 | Language | Vietnamese or English | Others |
| C4 | Relevance | Addresses the effect of a catalogue metric (rates, inflation, credit growth, unemployment, inventory, transactions, prices) on a mapping target (borrowing capacity, valuation range), or on a close proxy (mortgage affordability, house prices, credit supply) | Background material with no stated relationship |
| C5 | Geography | About Viet Nam (preferred), or a cross-country study that includes Viet Nam or comparable emerging markets | Single-country studies of unrelated markets, except as labelled bad documents (below) |
| C6 | Retrievability | A stable URL or DOI; the file can be downloaded and hashed | Pages that change in place without versions |
| C7 | Date | Publication date stated. No hard cut-off; age is recorded and the mapping verifier's freshness rule decides whether it may support a mapping | Undated documents |

Preferred, not required: empirical results with stated methods and data; Vietnamese-language sources from Vietnamese institutions (they exercise Vietnamese retrieval).

## Source record

Every source on the list, accepted or rejected, has a record:

| Field | Notes |
|---|---|
| `source_id` | Stable short ID, e.g. `src-0007` |
| `title`, `publisher`, `publisher_type` | `publisher_type` from C1 |
| `country`, `language` | |
| `publication_date` | As stated in the document |
| `url` (and `doi` if any) | |
| `licence` | The licence or terms that satisfy C2, with a link |
| `retrieved_at`, `sha256` | Of the stored original file |
| `qualifies_because` | One or two sentences, against C1–C7 |
| `proposed_by` | `source_agent` (with model, prompt version, run ID) or a person |
| `decision` | `accepted`, `rejected` (with the failed criterion), or `bad_document` (see below) |
| `signed_off_by`, `signed_off_at` | Required before the corpus is frozen |

The list is versioned; a change to the list creates a new corpus version (PIPELINE.md, ingestion step 6).

## Deliberately bad documents

The corpus also holds a small, **separately labelled** set of documents chosen to fail, so the mapping verifier and the agent are tested on them (2026-10-14 notes):

| Type | What it tests |
|---|---|
| Wrong-country paper | Geography check: its findings must not be mapped to Viet Nam |
| Two conflicting papers | Conflict flagging between sources |
| Outdated study | Freshness rule |
| Correlation-only claim | Support check: correlation is not a direction of effect |
| Page with injected instructions | Prompt injection through retrieved text |
| Fabricated-quote case | A proposal quoting text that is not in the chunk; the verbatim check must catch it |

Real documents are used where they exist (wrong-country, conflicting, outdated, correlation-only). The injected page and the fabricated quote are **synthetic** and labelled as such, per the hard rule on synthetic data. Bad documents carry `decision: bad_document` and are excluded from any "real corpus" statistics.

## Process

1. The source agent (Qwen) proposes candidates with a filled source record and `qualifies_because`.
2. Code checks what it can: URL resolves, file downloads, hash recorded, date and language present.
3. A human reviews every record against C1–C7 and signs off. The agent choosing the sources it later cites is checking its own work, so this step is not skipped.
4. The signed-off list is frozen as a corpus version; ingestion runs from it.

## Open

- Who signs off the source list (and reviews the gold-label sample).
