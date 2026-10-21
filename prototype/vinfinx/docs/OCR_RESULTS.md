# OCR results

Measurements for choosing the OCR engine (`PIPELINE.md`, "Choosing the engine"). The engines are scored against the synthetic document generator's ground truth: each printed field's box is compared with the words the engine read inside it.

- **Code:** `ai/ocr/`, with the scoring in `match.py` and the runner in `evaluate.py`.
- **Reports:** `$VINFINX_ROOT/evaluation/ocr/<run>/report.json` (not committed).
- **Field outcomes:**
  - **correct:** the text read equals the printed text, after Unicode NFC, all dash characters treated as a hyphen, and single spaces.
  - **wrong:** something different was read.
  - **missing:** nothing was read in the box.
- **The dangerous case:** a **wrong reading with confidence ≥ 0.9**, which a confidence threshold alone would accept.

## 2026-10-21: PaddleOCR 3.7.0 (CPU), smoke test only

PP-OCRv6 medium detector and recognizer, oneDNN off (it crashes in Paddle 3.3.1); page 1 of the clean documents only.

- **English:** 134/145 fields. The misses are minus signs and one underscore.
- **Vietnamese:** 94/145.
- **Speed:** about 200 s per page.
- **Not usable for Vietnamese:** the recognizer's character set lacks **44 of the 67 Vietnamese lowercase letters** (all with a dot below, a hook above or stacked marks), so "Việt" is read "Vit". The older Latin PP-OCRv5 recognizer lacks 45. Details: `SERVER_SETUP_PLAN.md`, "OCR environment".

## 2026-10-21: Tesseract 5.5.3, full draft set

**Run:** `20261021T004223Z-tesseract-psm3`.
- **Engine and data:** `tessdata_best` at commit `e12c65a`; `eng` for English documents, `vie+eng` for Vietnamese; page segmentation mode 3.
- **Set:** `docs-v0.1-draft`, 62 documents (186 pages), regenerated after the document-ID fix.
- **Speed:** 8 workers, 116 s in total, **4–6 s per page** on one core.

Accuracy on all fields and on value fields, and value fields read wrong with confidence ≥ 0.9:

| Setting | All EN | All VI | Values EN | Values VI | Confident wrong values EN/VI |
|---|---|---|---|---|---|
| clean (and every content variant) | 0.97 | 0.97 | 1.00 | 0.99 | 0/0 |
| blur σ 1 | 0.94 | 0.95 | 0.95 | 0.95 | 0/0 |
| blur σ 2 | 0.89 | 0.87 | 0.85 | 0.86 | 1/5 |
| blur σ 3 | 0.32 | 0.39 | 0.32 | 0.48 | 4/7 |
| crop 5% / 15% | 0.97 / 0.96 | 0.97 / 0.97 | 1.00 / 1.00 | 1.00 / 1.00 | 0/0 |
| JPEG q50 / q20 | 0.97 / 0.97 | 0.95 / 0.92 | 1.00 / 1.00 | 0.98 / 0.98 | 0/0 |
| 150 dpi | 0.76 | 0.88 | 0.79 | 0.95 | 0/1 |
| 100 dpi | 0.42 | 0.44 | 0.34 | 0.40 | 7/1 |
| noise + shadow | 0.95 | 0.95 | 0.96 | 0.96 | 0/1 |
| rotate ±1° | 0.93–0.94 | 0.93–0.94 | 0.93–0.94 | 0.97–0.99 | 0–1 |
| rotate ±3° | 0.89–0.90 | 0.92–0.95 | 0.91–0.95 | 0.94–0.98 | 0–1 |
| rotate 90° | 0.03 | 0.03 | 0.05 | 0.05 | 0/7 |

**Findings:**
- **Vietnamese is read correctly, diacritics included**, as well as English.
- **Clean pages:**
  - Every value field is correct in English. In Vietnamese, one is wrong: `−57` read as `=57`.
  - The remaining errors are labels: "y/y" read as "yly", and "±" read as "+" in every uncertainty field with confidence 0.92. The last is a confident error, but `+0.1 pp (90%)` does not parse as an uncertainty interval, so extraction can reject it.
- **Robust to:** JPEG, noise and shadow, cropping, rotation up to 3°, a blacked-out or smudged cell, and hidden text.
- **Fails on:**
  - strong blur (σ 3);
  - 100 dpi scans;
  - pages rotated 90°. Tesseract's orientation detection (`osd`) is available but was not used in this run.
- **Confident wrong values appear on degraded pages.** The main pattern is a lost or swapped decimal separator (`4.8` → `48`, `3.6` → `36`, `143,8` → `143.8`), plus rotated dates read backwards (`06/2026` → `9202/90`). OCR confidence alone does not protect against them. Extraction and validation must:
  - parse numbers with a strict grammar for the document's language (`143.8` is not a valid Vietnamese number);
  - check the catalogue's valid ranges;
  - check that Change = Value − Baseline;
  - use the Qwen second reading.

  The degraded variants are there to measure whether those controls leave any value wrong **and trusted**.

## 2026-10-21: decision and orientation

- **Decision (AI engineer):** **Tesseract is the OCR engine**, and PaddleOCR is removed. `env-ocr` was rebuilt with only Tesseract, `pytesseract` and the main environment's versions of pillow, pydantic, numpy and PyYAML (760 MB). The PaddleOCR models were deleted.
- **Orientation:**
  - Before reading, Tesseract's orientation detection (`osd`, about 1 s per page) turns a page rotated 90, 180 or 270 degrees upright.
  - Word boxes are mapped back onto the page as received, so every field still points at the original image.
  - The rotation group went from 62% to 91–92% of fields read correctly; the 90° pages now read like clean ones.
  - Small tilts are corrected in the layout step instead.

## 2026-10-21: end to end, OCR → extraction → validation ("wrong and trusted")

**Code:** `ai/extraction/`:
- `parse.py`: strict parsers per language;
- `layout.py`: rows anchored on metric IDs, small skew removed, and the template's columns placed using whichever header labels were read;
- `macro_document.py`: builds `EvidenceItem`s and runs every check;
- `evaluate.py`: the end-to-end measure.

**An item is trusted only if every check passes:**
- the metric ID is in the catalogue;
- the period, value, baseline and change parse strictly for the document's language;
- Change = Value − Baseline;
- the unit, measure, geography and segment agree with the catalogue (labels may differ by one misread letter);
- the value is in range, and whole for counts;
- the observation period ends before publication;
- no figure in the commentary contradicts the table.

The Qwen second reading is **not wired in yet**.

**Classification of each (document, metric) item against the ground truth:**
- ok: trusted and correct;
- WRONG_TRUSTED: trusted but wrong, or with no printed value;
- MISSED_FLAG: trusted although the ground truth says it must be flagged;
- caught: untrusted, and it was wrong or had to be flagged;
- lost: untrusted although correct;
- not extracted.

| Set | Items | ok | **WRONG_TRUSTED** | **MISSED_FLAG** | caught | lost | not extracted |
|---|---|---|---|---|---|---|---|
| `docs-v0.1-draft` (seed 42, 2026-10-05), used while building | 868 | 590 (68%) | **0** | **0** | 79 | 136 | 63 |
| `docs-v0.1-holdout-s7` (seed 7, 2026-11-06), never looked at before this run | 868 | 579 (67%) | **0** | **0** | 68 | 158 | 63 |

**Findings:**
- **No item came out wrong and trusted, and no item that had to be flagged was trusted, on either set.** The held-out set has new values and dates and was scored once with the unchanged chain; the result is the same.
- **A minimum OCR confidence does not catch anything extra; it only loses good items.** On the held-out set, trusted-and-correct items drop from 579 (no minimum) to 427 at 0.8 and 305 at 0.9, with 0 wrong and trusted at every level. The format checks catch the misreadings first. The field-review thresholds are an open question in the handoff (§5 Q1), so the threshold is the AI engineer's decision.
- **Expected limit:** `tampered_consistent` was accepted (2 items per set: a changed value with a matching Change). The spec expects this to be undetectable from one document; only a second source can catch it.
- **Where coverage is lost** (correct items left untrusted, or not extracted): a change that does not parse (a minus read as "="), a row whose metric ID was not read, a measure misread ("y/y" as "yly"), a unit not read. By variant:
  - cropped pages: 0 trusted, by design, because the Change column is cut and the value cannot be cross-checked;
  - 100 dpi and blur σ 3: few rows read;
  - JPEG q20 in Vietnamese.
- **Clean pages:** 13 of 14 items trusted on the draft set in both languages. On the held-out set, 11 of 14 in English and 14 of 14 in Vietnamese.
- **Caveats:**
  - The documents are synthetic and generated by the same team as the extractor, from one template.
  - Values are the only fields compared. Commentary injection is not measured here, because the commentary is never extracted as evidence.

## Next

1. ~~Field extraction and validation~~: built 2026-10-21 (above).
2. ~~Orientation detection~~: built 2026-10-21 (above).
3. ~~Decide whether to keep PaddleOCR~~: removed 2026-10-21; Tesseract chosen.
4. Wire in the Qwen second reading and measure what it adds (coverage or catches) on both sets.
5. Freeze a document set (draft or a new one) with its manifest committed, so later changes are measured on fixed documents.
