"""The synthetic macro-status document generator (MACRO_DOCUMENT_SPEC.md): ground truth is exact, consistent and
reproducible, every document is labelled synthetic, and each variant changes what it says it changes."""
import json
import tempfile
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from ai.catalog import load_catalogue
from ai.documents.generate import generate_document
from ai.evaluation.cases import period_bounds
from ai.schemas.evidence import Presence

AS_OF = datetime(2026, 10, 5, tzinfo=timezone.utc)
CAT = load_catalogue()


class DocumentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.docs = {}
        for variant, params, lang in [
            ("clean", {}, "en"), ("clean", {}, "vi"),
            ("tampered_inconsistent", {"metric_id": "macro.vn.refi_rate", "printed_value": "6.0"}, "en"),
            ("tampered_consistent", {"metric_id": "macro.vn.refi_rate", "printed_value": "6.0"}, "en"),
            ("absent", {"metric_id": "macro.vn.cpi", "mode": "na"}, "en"),
            ("unreadable", {"metric_id": "macro.vn.cpi", "mode": "black"}, "vi"),
            ("stale", {"months": 6}, "en"),
            ("degraded_crop", {"right_fraction": 0.15}, "en"),
            ("degraded_rotate", {"angle": 90}, "en"),
        ]:
            truth = generate_document(variant, params, catalogue=CAT, as_of=AS_OF, seed=42, lang=lang,
                                      out_root=cls.root)
            cls.docs[(variant, lang)] = truth

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def values(self, truth, attribute):
        return {f.metric_id: f for f in truth.fields if f.attribute == attribute}

    def test_clean_covers_every_metric_consistently(self):
        truth = self.docs[("clean", "en")]
        self.assertTrue(truth.synthetic)
        self.assertEqual(truth.changed_fields, [])
        values, baselines = self.values(truth, "value"), self.values(truth, "baseline_value")
        changes, periods = self.values(truth, "change_vs_baseline"), self.values(truth, "observation_period")
        self.assertEqual(set(values), set(CAT.instances))  # 14 rows, one per metric ID
        for metric_id, v in values.items():
            with self.subTest(metric_id):
                self.assertEqual(v.presence, Presence.PRESENT)
                value, baseline = Decimal(v.true_value), Decimal(baselines[metric_id].true_value)
                self.assertEqual(Decimal(changes[metric_id].true_value), value - baseline)
                lo, hi = CAT.get(metric_id).definition.valid_range
                self.assertTrue(lo <= value <= hi)
                CAT.evidence_id(metric_id, periods[metric_id].true_value)  # catalogue period format
                end = period_bounds(periods[metric_id].true_value, CAT.get(metric_id).definition.frequency)[1]
                self.assertLess(end, AS_OF)

    def test_both_languages_carry_the_same_values(self):
        en, vi = self.docs[("clean", "en")], self.docs[("clean", "vi")]
        strip = lambda t: {f.field_id: f.true_value for f in t.fields if f.attribute not in ("title",)}
        self.assertEqual(strip(en), strip(vi))
        printed = {f.field_id: f.printed_text for f in vi.fields}
        self.assertIn(",", printed["macro.vn.refi_rate/value"])  # 4,9
        change = printed["macro.vn.refi_rate/change_vs_baseline"]
        self.assertTrue(change.startswith(("\u2212", "+")) and change.endswith("đpt"))

    def test_every_spec_column_is_printed(self):
        """2026-10-21: an edit commented out page 2's Source column. Every Table 1 and Table 2 attribute of the spec
        must reach the ground truth for every metric."""
        truth = self.docs[("clean", "en")]
        attributes = {f.attribute for f in truth.fields if f.metric_id == "macro.vn.refi_rate"}
        spec = {"metric_id", "geography", "segment", "transformation", "observation_period", "value", "unit",
                "baseline_value", "baseline_period", "change_vs_baseline", "source", "revision_status",
                "previous_value", "data_quality", "uncertainty", "next_update"}
        self.assertEqual(attributes, spec)

    def test_every_setting_has_its_own_document_id(self):
        """2026-10-21: rotations of -1 and 1 degrees got the same ID and overwrote each other."""
        from ai.documents.generate import ALL_VARIANTS, document_id

        ids = [document_id("2026-10", v, params, lang, 42) for v, ps in ALL_VARIANTS.items() for params in ps
               for lang in ("en", "vi")]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("LBEO-BUL-2026-10-degraded_rotate-neg3-en-s42", ids)

    def test_reproducible(self):
        again = generate_document("clean", {}, catalogue=CAT, as_of=AS_OF, seed=42, lang="en",
                                  out_root=self.root / "again")
        first = self.docs[("clean", "en")]
        self.assertEqual([p.sha256 for p in again.pages], [p.sha256 for p in first.pages])
        self.assertEqual(again.pdf.sha256, first.pdf.sha256)
        other = generate_document("clean", {}, catalogue=CAT, as_of=AS_OF, seed=7, lang="en",
                                  out_root=self.root / "other")
        self.assertNotEqual(self.values(other, "value"), self.values(first, "value"))

    def test_labelled_synthetic_and_image_only(self):
        truth = self.docs[("clean", "en")]
        pdf = (self.root / truth.document_id / "document.pdf").read_bytes()
        from PIL import PdfParser

        info = {k.name: v.decode("utf-16") for k, v in PdfParser.PdfParser(buf=pdf).info.items()
                if v.startswith(b"\xfe\xff")}  # metadata strings are UTF-16 in the file
        self.assertIn("SYNTHETIC TEST DATA", info[b"Title"])
        self.assertIn("synthetic, test data", info[b"Keywords"])
        self.assertNotIn(b"/Font", pdf)  # no text layer: reading it needs OCR
        printed = {f.printed_text for f in truth.fields}
        self.assertIn("Lotus Bay Economic Observatory", printed)
        on_disk = json.loads((self.root / truth.document_id / "ground_truth.json").read_text())
        self.assertIs(on_disk["synthetic"], True)

    def test_tampering(self):
        clean = self.values(self.docs[("clean", "en")], "change_vs_baseline")["macro.vn.refi_rate"]
        for variant, consistent in (("tampered_inconsistent", False), ("tampered_consistent", True)):
            truth = self.docs[(variant, "en")]
            value = self.values(truth, "value")["macro.vn.refi_rate"]
            change = self.values(truth, "change_vs_baseline")["macro.vn.refi_rate"]
            baseline = self.values(truth, "baseline_value")["macro.vn.refi_rate"]
            self.assertEqual(value.true_value, "6.0")
            self.assertEqual(Decimal(change.true_value) == Decimal("6.0") - Decimal(baseline.true_value), consistent)
            self.assertEqual(change.true_value != clean.true_value, consistent)
            self.assertIn("macro.vn.refi_rate/value", truth.changed_fields)

    def test_absent_and_unreadable(self):
        absent = self.values(self.docs[("absent", "en")], "value")["macro.vn.cpi"]
        self.assertEqual((absent.presence, absent.printed_text, absent.true_value), (Presence.ABSENT, "n/a", None))
        unreadable = self.values(self.docs[("unreadable", "vi")], "value")["macro.vn.cpi"]
        self.assertEqual((unreadable.presence, unreadable.true_value), (Presence.UNREADABLE, None))
        self.assertIsNotNone(unreadable.bbox)

    def test_stale_is_stale_when_evaluated(self):
        truth = self.docs[("stale", "en")]
        self.assertEqual((truth.as_of.month, truth.evaluate_at), (4, AS_OF))
        for metric_id, p in self.values(truth, "observation_period").items():
            with self.subTest(metric_id):
                end = period_bounds(p.true_value, CAT.get(metric_id).definition.frequency)[1]
                self.assertLess(CAT.freshness_deadline(metric_id, end), truth.evaluate_at)
        self.assertEqual(len(truth.expected), len(CAT.instances))

    def test_crop_removes_the_right_columns(self):
        truth = self.docs[("degraded_crop", "en")]
        changes = self.values(truth, "change_vs_baseline")
        self.assertTrue(all(c.presence != Presence.PRESENT for c in changes.values()))
        self.assertTrue(all(f.presence == Presence.PRESENT for f in self.values(truth, "metric_id").values()))
        clean = {f.field_id: f.presence for f in self.docs[("clean", "en")].fields}
        lost = {f.field_id for f in truth.fields if f.presence != clean[f.field_id]}
        self.assertEqual(set(truth.changed_fields), lost)
        self.assertLess(truth.pages[0].width_px, 3508)

    def test_rotation_moves_the_boxes(self):
        clean = self.values(self.docs[("clean", "en")], "value")["macro.vn.refi_rate"].bbox
        rotated = self.docs[("degraded_rotate", "en")]
        box = self.values(rotated, "value")["macro.vn.refi_rate"].bbox
        self.assertEqual((rotated.pages[0].width_px, rotated.pages[0].height_px), (2480, 3508))
        # 90 degrees counter-clockwise: a field near the top-middle moves to the left edge, near the middle
        self.assertAlmostEqual(box.x0, clean.y0, places=2)
        self.assertAlmostEqual(1 - box.y1, clean.x0, places=2)


if __name__ == "__main__":
    unittest.main()
