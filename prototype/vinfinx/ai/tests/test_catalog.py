"""Tests for the metric catalogue. Run from prototype/vinfinx: python -m unittest discover -s ai/tests -t ."""
import copy
import unittest
from datetime import datetime, timezone

import yaml
from pydantic import ValidationError

from ai.catalog import CATALOGUE_PATH, METRIC_ID_RE, Catalogue, load_catalogue


def raw_catalogue() -> dict:
    with open(CATALOGUE_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


class CatalogueTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = load_catalogue()

    def test_metric_ids(self):
        ids = set(self.cat.instances)
        # 5 national macro metrics + 3 market metrics x 3 Ha Noi districts
        self.assertEqual(len(ids), 14)
        self.assertIn("macro.vn.refi_rate", ids)
        self.assertIn("market.hn-dong-da.listing_inventory", ids)
        self.assertNotIn("market.vn.listing_inventory", ids)
        for metric_id in ids:
            self.assertRegex(metric_id, METRIC_ID_RE)

    def test_districts_match_phase3_names(self):
        names = {g.phase3_district_name for g in self.cat.geographies if g.level == "district"}
        self.assertEqual(names, {"Q. Đống Đa", "Q. Hai Bà Trưng", "Q. Ba Đình"})

    def test_evidence_id_period_format(self):
        self.assertEqual(self.cat.evidence_id("macro.vn.refi_rate", "2026-09"), "macro.vn.refi_rate.2026-09")
        self.assertEqual(
            self.cat.evidence_id("market.hn-ba-dinh.listing_inventory", "2026-10-01"),
            "market.hn-ba-dinh.listing_inventory.2026-10-01",
        )
        self.assertEqual(
            self.cat.evidence_id("macro.vn.unemployment_rate", "2026-Q3"), "macro.vn.unemployment_rate.2026-Q3"
        )
        with self.assertRaises(ValueError):
            self.cat.evidence_id("macro.vn.refi_rate", "2026-09-30")  # monthly metric, daily label
        with self.assertRaises(KeyError):
            self.cat.evidence_id("macro.vn.gdp", "2026-09")

    def test_reason_codes(self):
        codes = self.cat.reason_codes()
        self.assertIn("REFI_RATE_DROP", codes)
        self.assertIn("INVENTORY_RISE", codes)
        self.assertIn("CONFLICTING_SIGNALS", codes)
        self.assertEqual(len(codes), len(set(codes)))
        self.assertEqual(len(codes), 8 * 2 + 3)

    def test_freshness_deadline(self):
        end = datetime(2026, 9, 30, 23, 59, 59, tzinfo=timezone.utc)
        self.assertEqual(
            self.cat.freshness_deadline("macro.vn.refi_rate", end), datetime(2026, 11, 29, 23, 59, 59, tzinfo=timezone.utc)
        )
        with self.assertRaises(ValueError):
            self.cat.freshness_deadline("macro.vn.refi_rate", datetime(2026, 9, 30))  # naive datetime

    def test_hash_is_stable(self):
        self.assertTrue(self.cat.catalogue_hash.startswith("sha256:"))
        self.assertEqual(self.cat.catalogue_hash, load_catalogue().catalogue_hash)

    def test_rejects_unknown_field(self):
        raw = raw_catalogue()
        raw["metrics"][0]["comment"] = "not allowed"
        with self.assertRaises(ValidationError):
            Catalogue.model_validate(raw)

    def test_rejects_unknown_unit(self):
        raw = raw_catalogue()
        raw["metrics"][0]["unit"] = "percent"
        with self.assertRaises(ValidationError):
            Catalogue.model_validate(raw)

    def test_rejects_bad_range_and_duplicates(self):
        raw = raw_catalogue()
        bad = copy.deepcopy(raw)
        bad["metrics"][0]["valid_range"] = ["20", "0"]
        with self.assertRaises(ValidationError):
            Catalogue.model_validate(bad)
        dup = copy.deepcopy(raw)
        dup["metrics"].append(copy.deepcopy(dup["metrics"][0]))
        with self.assertRaises(ValidationError):
            Catalogue.model_validate(dup)

    def test_frozen(self):
        with self.assertRaises(ValidationError):
            self.cat.version = "9.9.9"


if __name__ == "__main__":
    unittest.main()
