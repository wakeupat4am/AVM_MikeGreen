# OCR confidence threshold: proposal

*Status: **proposal for the AI engineer and the risk owner** (2026-10-21). Answers handoff §5 Q1 ("What per-field thresholds should trigger review rather than auto-acceptance?") for the synthetic macro bulletin only. Customer documents need their own measurement.*

## The question

OCR returns a confidence score for each word. A **minimum confidence** sends low-confidence fields to human review instead of accepting them. A higher minimum stops more misreadings, but it also rejects more correct readings, and each of those costs a review.

A threshold only matters for a misreading that **gets past every other check**. On the bulletin, almost every value has independent checks:
- the number must parse in the document's own format;
- Change = Value − Baseline;
- the value must be within the catalogue's range;
- the unit, measure and district must agree with the catalogue.

A misread digit usually breaks one of these and is caught whatever the confidence.

## What we measured

Two sets of synthetic bulletins, 62 documents each (clean, tampered and degraded variants, English and Vietnamese): the draft set, and a held-out set generated after the extractor was written. Tesseract 5.5.3. Details: [`OCR_RESULTS.md`](OCR_RESULTS.md).

### 1. Misreadings that get past the other checks

These are fields that are wrong inside an item that ends up trusted, the only errors a threshold could still stop.

| Field | Correct readings in trusted items | Wrong readings in trusted items |
|---|---|---|
| Value, baseline, change, periods, unit, measure, district, metric ID | 1,169 each | **0** |
| Source, revision status, data quality, previous value | 4–1,133 | **0** |
| Uncertainty ("±0.1 pp") | 0 | **0**: Tesseract always reads "±" as "+", so the extractor drops it (390 of 390) |

**No misreading survived the other checks**, so on these documents a threshold has nothing left to stop.

### 2. What a threshold would cost

Items trusted and correct, by minimum confidence (wrong-and-trusted items: 0 at every level):

| Minimum confidence | none | 0.5 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|
| Draft set (868 items) | 590 | 531 | 463 | 379 | 301 |
| Held-out set (868 items) | 579 | 539 | 459 | 427 | 305 |

At 0.9, about half of the correct items would go to human review, and no error would be stopped.

### 3. Is Tesseract's confidence trustworthy?

The table shows the share of fields read exactly right in each confidence band (number of fields in brackets):

| Confidence | Values, clean pages | Values, degraded pages | Labels, degraded pages |
|---|---|---|---|
| below 0.5 | 65% (80) | 28% (422) | 4% (456) |
| 0.5–0.7 | 100% (158) | 64% (291) | 46% (277) |
| 0.7–0.8 | 88% (126) | 66% (220) | 52% (235) |
| 0.8–0.9 | 100% (391) | 85% (653) | 75% (524) |
| 0.9–0.95 | 100% (1,797) | **95% (1,481)** | **77% (1,354)** |
| 0.95 and above | 100% (3,726) | 99.6% (2,647) | 98% (3,271) |

- **On clean pages, confidence barely matters:** values above 0.5 are essentially always right.
- **On degraded pages Tesseract is overconfident:** at 0.90–0.95 it is wrong on 1 value in 20 (lost or swapped decimal separators, for example `4.8` read as `48`). A threshold of 0.9 would let those through, and only 0.95 would stop most of them. That would reject two thirds of all readings, because many correct fields score just under 0.95.

So confidence is a weak signal on bad scans. The format and arithmetic checks are what catch these errors.

### 4. How sure can we be about "zero"?

Zero wrong out of 1,169 trusted items (both sets) means the true rate is, with 95% confidence, **below about 3 in 1,169 (0.26%)**: the "rule of three". Proving a tighter rate needs more documents. The errors on our synthetic pages are also easier than real scans: one template, a clean font, and known degradations.

## Proposal

1. **Bulletin value fields: no minimum confidence** (provisional). The cross-checks catch the misreadings, and a threshold only adds review work. Recorded in a versioned config, so a later change is replayable.
2. **Bulletin fields without a cross-check** (source, revision status, data quality, previous value, uncertainty) **never drive a decision on their own.** A field that does not parse is dropped or marked "unknown", never guessed. When the desk on page 2 is not read, the item now carries a caveat instead of silently naming the publisher.
3. **Customer documents: no threshold yet.** They have no Change column to check against, so a threshold (or a second reading) will matter there. Set it by the same method, on labelled representative documents, before any customer document is processed.
4. **Second reading before thresholds:** for fields without a cross-check, disagreement between Tesseract and the Qwen second reading is likely a stronger signal than Tesseract's confidence. Measure it once it is wired in.

## Decisions needed from the risk owner

- **Acceptable rate of wrong-and-trusted fields,** per document type (for example "at most 1 in 1,000 value fields"). This sets how many test documents are needed to show it: about 3,000 error-free items for 1 in 1,000.
- **The relative cost of a human review against a wrong accepted value.** This decides how much review work is worth one stopped error, wherever a threshold does matter (customer documents, fields without cross-checks).
- **Agreement with points 1–3 above,** as provisional values until real documents are measured.

## Reproduce

```bash
cd prototype/vinfinx
python -m ai.extraction.evaluate --set docs-v0.1-draft --ocr-run 20261021T144959Z-tesseract-psm3
python -m ai.extraction.evaluate --set docs-v0.1-holdout-s7 --ocr-run 20261021T145228Z-tesseract-psm3
python -m ai.extraction.confidence --run docs-v0.1-draft=20261021T144959Z-tesseract-psm3 \
                                   --run docs-v0.1-holdout-s7=20261021T145228Z-tesseract-psm3
```

Reports: `$VINFINX_ROOT/evaluation/extraction/` (`20261021T145741Z.json`, `20261021T145753Z.json`, `confidence-20261021T145724Z.json`).
