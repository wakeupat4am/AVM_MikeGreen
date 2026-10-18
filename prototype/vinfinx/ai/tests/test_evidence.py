"""Tests for EvidenceItem. Run from prototype/vinfinx: python -m unittest discover -s ai/tests -t ."""
import copy
import unittest
from datetime import datetime, timezone
from decimal import Decimal

from pydantic import ValidationError

from ai.catalog import load_catalogue
from ai.schemas.evidence import EvidenceItem, catalogue_violations

CAT = load_catalogue()
H = "sha256:" + "a" * 64
OCR = {"engine": "paddleocr", "engine_version": "3.0.0", "model": "PP-OCRv5"}


def field(attribute, text, value, x0=0.1, y0=0.2, presence="present", confidence=0.97):
    return {
        "attribute": attribute,
        "presence": presence,
        "page": 1,
        "bbox": {"x0": x0, "y0": y0, "x1": x0 + 0.08, "y1": y0 + 0.02},
        "ocr_text": text,
        "normalized_value": value,
        "confidence": confidence,
        "extractor": OCR,
    }


def ocr_item() -> dict:
    """The handoff's worked example: refinancing rate 4.0% for 2026-09, read from the synthetic document."""
    end = datetime(2026, 9, 30, 23, 59, 59, tzinfo=timezone.utc)
    return {
        "evidence_id": "macro.vn.refi_rate.2026-09",
        "metric_id": "macro.vn.refi_rate",
        "geography": "vn",
        "source": "Lotus Bay Economic Observatory",
        "source_document_id": "lbeo-bulletin-2026-10",
        "observation_period": {"label": "2026-09", "start": "2026-09-01T00:00:00Z", "end": end.isoformat()},
        "publication_date": "2026-10-05T00:00:00Z",
        "retrieved_at": "2026-10-06T08:00:00Z",
        "value": "4.0",
        "unit": "pct",
        "transformation": "level",
        "baseline": {"value": "4.5", "unit": "pct", "period_label": "2026-06"},
        "revision_status": "final",
        "data_quality": "ok",
        "freshness": {
            "deadline": CAT.freshness_deadline("macro.vn.refi_rate", end).isoformat(),
            "catalogue_version": CAT.version,
            "catalogue_hash": CAT.catalogue_hash,
        },
        "presence": "present",
        "trust": "trusted",
        "synthetic": True,
        "provenance": {
            "kind": "ocr",
            "document_id": "lbeo-bulletin-2026-10",
            "document_hash": H,
            "page_hashes": {1: H, 2: H},
            "fields": {
                "value": field("value", "4.0%", "4.0"),
                "unit": field("unit", "%", "pct", x0=0.2),
                "observation_period": field("observation_period", "Sep 2026", "2026-09", x0=0.3),
            },
        },
    }


class EvidenceItemTest(unittest.TestCase):
    def test_valid_ocr_item(self):
        item = EvidenceItem.model_validate(ocr_item())
        self.assertEqual(item.value, Decimal("4.0"))
        self.assertEqual(catalogue_violations(item, CAT), [])

    def test_valid_system_item(self):
        raw = ocr_item()
        raw["provenance"] = {
            "kind": "system",
            "system": "macro.fixture",
            "record_id": "r-1",
            "export_hash": H,
            "exported_at": "2026-10-06T08:00:00Z",
        }
        EvidenceItem.model_validate(raw)

    def test_hash_stable_through_json_round_trip(self):
        item = EvidenceItem.model_validate(ocr_item())
        again = EvidenceItem.model_validate_json(item.model_dump_json())
        self.assertEqual(item.item_hash, again.item_hash)
        changed = copy.deepcopy(ocr_item())
        changed["value"] = "6.0"  # tampered
        self.assertNotEqual(item.item_hash, EvidenceItem.model_validate(changed).item_hash)

    def assertRejected(self, raw):
        with self.assertRaises(ValidationError):
            EvidenceItem.model_validate(raw)

    def test_rejects_mismatched_ids(self):
        raw = ocr_item()
        raw["evidence_id"] = "macro.vn.refi_rate.2026-08"
        self.assertRejected(raw)
        raw = ocr_item()
        raw["geography"] = "hn-dong-da"
        self.assertRejected(raw)

    def test_rejects_naive_datetime_and_unknown_field(self):
        raw = ocr_item()
        raw["retrieved_at"] = "2026-10-06T08:00:00"
        self.assertRejected(raw)
        raw = ocr_item()
        raw["llm_note"] = "trust me"
        self.assertRejected(raw)

    def test_absent_and_unreadable_values(self):
        raw = ocr_item()
        raw["value"] = None
        raw["presence"] = "unreadable"
        self.assertRejected(raw)  # cannot stay trusted
        raw["trust"] = "untrusted"
        raw["trust_reasons"] = ["VALUE_UNREADABLE"]
        raw["provenance"]["fields"]["value"] = field("value", None, None, presence="unreadable", confidence=None)
        EvidenceItem.model_validate(raw)  # unreadable, blacked out: a bbox, no text, no normalized value
        raw["provenance"]["fields"]["value"]["ocr_text"] = "4.?%"
        EvidenceItem.model_validate(raw)  # unreadable, smudged: a bbox and partial raw text
        bad = copy.deepcopy(raw)
        bad["provenance"]["fields"]["value"]["normalized_value"] = "4.0"
        self.assertRejected(bad)  # an unreadable field cannot carry a normalized value
        bad["provenance"]["fields"]["value"]["normalized_value"] = None
        bad["provenance"]["fields"]["value"]["bbox"] = None
        self.assertRejected(bad)  # unreadable means printed, so it must have a location on the page

        absent = copy.deepcopy(raw)
        absent["presence"] = "absent"
        absent["trust_reasons"] = ["VALUE_ABSENT"]
        absent["provenance"]["fields"]["value"] = {
            "attribute": "value", "presence": "absent", "page": 1, "bbox": None,
            "ocr_text": None, "normalized_value": None, "confidence": None, "extractor": OCR,
        }
        EvidenceItem.model_validate(absent)

    def test_present_value_requires_value(self):
        raw = ocr_item()
        raw["value"] = None
        self.assertRejected(raw)

    def test_untrusted_needs_reasons(self):
        raw = ocr_item()
        raw["trust"] = "untrusted"
        self.assertRejected(raw)

    def test_provenance_field_key_must_match(self):
        raw = ocr_item()
        raw["provenance"]["fields"]["unit"] = field("value", "4.0%", "4.0")
        self.assertRejected(raw)

    def test_bbox_ordered(self):
        raw = ocr_item()
        raw["provenance"]["fields"]["value"]["bbox"] = {"x0": 0.5, "y0": 0.2, "x1": 0.4, "y1": 0.3}
        self.assertRejected(raw)

    def test_catalogue_violations(self):
        raw = ocr_item()
        raw["value"] = "60"  # e.g. "6.0" misread as "60"
        raw["unit"] = "bps"
        raw["freshness"]["deadline"] = "2027-12-31T00:00:00Z"  # a document trying to extend its own life
        found = catalogue_violations(EvidenceItem.model_validate(raw), CAT)
        self.assertIn("OUT_OF_RANGE:60", found)
        self.assertIn("UNIT_MISMATCH:bps!=pct", found)
        self.assertIn("FRESHNESS_NOT_FROM_CATALOGUE", found)


if __name__ == "__main__":
    unittest.main()
