"""Field extraction and validation (ai/extraction/): strict parsers per language, and end to end on generated
documents (that part needs Tesseract, so it runs in env-ocr and is skipped elsewhere)."""
import importlib.util
import tempfile
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from ai.extraction import parse
from ai.schemas.common import Frequency, Unit

HAS_TESSERACT = importlib.util.find_spec("pytesseract") is not None


class ParseTest(unittest.TestCase):
    def test_numbers_are_strict_per_language(self):
        self.assertEqual(parse.number("2,262", "en"), Decimal("2262"))
        self.assertEqual(parse.number("1.250", "vi"), Decimal("1250"))
        self.assertEqual(parse.number("4,9", "vi"), Decimal("4.9"))
        self.assertEqual(parse.number("−0.5", "en"), Decimal("-0.5"))
        self.assertIsNone(parse.number("143.8", "vi"))  # a swapped separator is rejected, not misread as 1438
        self.assertIsNone(parse.number("2,26", "en"))
        self.assertIsNone(parse.number("4.9%", "en"))

    def test_change_needs_its_sign_and_suffix(self):
        self.assertEqual(parse.change("−0.5 pp", "en", Unit.PCT), Decimal("-0.5"))
        self.assertEqual(parse.change("+1,2 đpt", "vi", Unit.PCT), Decimal("1.2"))
        self.assertEqual(parse.change("—57", "en", Unit.COUNT), Decimal("-57"))  # em dash read for a minus
        self.assertIsNone(parse.change("=57", "vi", Unit.COUNT))
        self.assertIsNone(parse.change("0.5 pp", "en", Unit.PCT))  # unsigned non-zero change
        self.assertIsNone(parse.change("-0.5 pts", "en", Unit.PCT))  # suffix of another unit

    def test_periods_and_dates(self):
        self.assertEqual(parse.period("Sep 2026", "en", Frequency.MONTHLY), "2026-09")
        self.assertEqual(parse.period("09/2026", "vi", Frequency.MONTHLY), "2026-09")
        self.assertEqual(parse.period("Quý 3/2026", "vi", Frequency.QUARTERLY), "2026-Q3")
        self.assertEqual(parse.period("1 Oct 2026", "en", Frequency.SNAPSHOT), "2026-10-01")
        self.assertIsNone(parse.period("9202/90", "vi", Frequency.MONTHLY))  # read backwards on a rotated page
        self.assertIsNone(parse.period("13/2026", "vi", Frequency.MONTHLY))

    def test_units_measures_uncertainty(self):
        self.assertEqual(parse.unit("giao dịch"), Unit.COUNT)
        self.assertIsNone(parse.unit("thousand listings"))
        self.assertEqual(parse.measure("So với cùng kỳ").value, "yoy")
        self.assertIsNone(parse.measure("yly"))
        self.assertEqual(parse.uncertainty("±0,1 đpt (90%)", "vi", Decimal("2.3"), Unit.PCT),
                         (Decimal("2.2"), Decimal("2.4"), Decimal("0.9")))
        self.assertIsNone(parse.uncertainty("+0.1 pp (90%)", "en", Decimal("2.3"), Unit.PCT))

    def test_labels_tolerate_a_misread_letter_but_not_another_district(self):
        others = ["Dong Da District, Ha Noi", "Hai Ba Trung District, Ha Noi", "Viet Nam"]
        self.assertTrue(parse.label_matches("Ba Dinh District, Ha Noj", "Ba Dinh District, Ha Noi", others))
        self.assertFalse(parse.label_matches("Dong Da District, Ha Noi", "Ba Dinh District, Ha Noi", others))


@unittest.skipUnless(HAS_TESSERACT, "needs Tesseract: run in $VINFINX_ROOT/env-ocr")
class EndToEndTest(unittest.TestCase):
    """Generate, OCR, extract: a clean page gives trusted correct items; tampering ends up untrusted."""

    @classmethod
    def setUpClass(cls):
        from ai.catalog import load_catalogue
        from ai.documents.generate import generate_document
        from ai.extraction.evaluate import classify
        from ai.extraction.macro_document import extract
        from ai.ocr.tesseract import TesseractEngine

        cls.tmp = tempfile.TemporaryDirectory()
        cat = load_catalogue()
        engine = TesseractEngine()
        cls.results = {}
        for variant, params in (("clean", {}), ("tampered_inconsistent",
                                                {"metric_id": "macro.vn.refi_rate", "printed_value": "6.0"})):
            truth = generate_document(variant, params, catalogue=cat, as_of=datetime(2026, 10, 5, tzinfo=timezone.utc),
                                      seed=42, lang="en", out_root=Path(cls.tmp.name))
            d = Path(cls.tmp.name) / truth.document_id
            pages = {p.page: engine.read(str(d / p.file), p.page, "eng") for p in truth.pages}
            ex = extract(pages, lang="en", catalogue=cat, document_id=truth.document_id, document_hash=truth.pdf.sha256,
                         retrieved_at=truth.evaluate_at, synthetic=True)
            cls.results[variant] = (ex, classify(truth, ex.items, cat))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_clean_page(self):
        ex, classes = self.results["clean"]
        self.assertNotIn("WRONG_TRUSTED", classes.values())
        self.assertGreaterEqual(list(classes.values()).count("ok"), 12)
        self.assertTrue(all(i.synthetic for i in ex.items))
        refi = next(i for i in ex.items if i.metric_id == "macro.vn.refi_rate")
        self.assertEqual(refi.provenance.kind, "ocr")
        self.assertIsNotNone(refi.provenance.fields["value"].bbox)

    def test_inconsistent_tampering_is_untrusted(self):
        ex, classes = self.results["tampered_inconsistent"]
        refi = next(i for i in ex.items if i.metric_id == "macro.vn.refi_rate")
        self.assertEqual((refi.value, refi.trust), (Decimal("6.0"), "untrusted"))
        self.assertTrue(any(r.startswith("CHANGE_MATCHES_BASELINE") for r in refi.trust_reasons))
        self.assertEqual(classes["macro.vn.refi_rate"], "caught")


if __name__ == "__main__":
    unittest.main()
