"""OCR scoring (ai/ocr/match.py): fields are matched to the words drawn inside their boxes, and the dangerous case,
a wrong reading with high confidence, is counted. Engine-independent: no OCR engine is needed."""
import unittest

from ai.documents.ground_truth import FieldTruth
from ai.ocr import OcrPage, OcrWord
from ai.ocr.match import normalize, score_field, summarize


def field(field_id="macro.vn.refi_rate/value", text="4.9", presence="present", bbox=(0.40, 0.20, 0.43, 0.22)):
    metric, attribute = field_id.split("/")
    return FieldTruth(field_id=field_id, metric_id=metric, attribute=attribute, page=1,
                      bbox=dict(zip(("x0", "y0", "x1", "y1"), bbox)) if bbox else None, printed_text=text,
                      true_value=text if presence == "present" else None, presence=presence)


def page(*words):
    return {1: OcrPage(engine="t", engine_version="0", model="m", page_sha256="sha256:" + "0" * 64, page=1, seconds=1,
                       width_px=3508, height_px=2480,
                       words=[OcrWord(text=t, confidence=c, line_id=0,
                                      bbox=dict(zip(("x0", "y0", "x1", "y1"), b))) for t, c, b in words])}


class MatchTest(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize("−0.5  pp"), "-0.5 pp")
        self.assertEqual(normalize("—0.5 pp"), "-0.5 pp")  # em dash read for a printed minus
        self.assertEqual(normalize("Việt"), "Việt")  # decomposed diacritics compose (NFC)

    def test_outcomes(self):
        inside = (0.405, 0.202, 0.425, 0.218)
        self.assertEqual(score_field(field(), page(("4.9", 0.95, inside))).outcome, "correct")
        wrong = score_field(field(), page(("4.8", 0.97, inside)))
        self.assertEqual((wrong.outcome, wrong.confidence), ("wrong", 0.97))
        self.assertEqual(score_field(field(), page(("4.9", 0.95, (0.6, 0.5, 0.62, 0.52)))).outcome, "missing")
        # a smudged value read as a number anyway
        self.assertEqual(score_field(field(presence="unreadable", text=None), page(("2.3", 0.9, inside))).outcome,
                         "read")

    def test_words_join_in_reading_order(self):
        r = score_field(field(text="-0.5 pp"), page(("pp", 0.9, (0.415, 0.202, 0.428, 0.218)),
                                                   ("−0.5", 0.9, (0.401, 0.202, 0.413, 0.218))))
        self.assertEqual((r.read, r.outcome, r.confidence), ("-0.5 pp", "correct", 0.9))

    def test_summary_counts_confident_errors(self):
        inside = (0.405, 0.202, 0.425, 0.218)
        results = [score_field(field(), page(("4.9", 0.99, inside))), score_field(field(), page(("4.8", 0.93, inside))),
                   score_field(field(), page(("4.8", 0.5, inside)))]
        s = summarize(results)["all_fields"]
        self.assertEqual((s["correct"], s["wrong"], s["wrong_conf_ge_0.9"], s["wrong_conf_ge_0.95"]), (1, 2, 1, 0))


if __name__ == "__main__":
    unittest.main()
