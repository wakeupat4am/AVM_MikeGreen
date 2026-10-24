"""Verifier v0 tests: the four worked cases from AI_ENGINEER_FINDINGS.md (2026-10-11) plus one test per rule.
These are unit tests of the rules; they do not use the adversarial case set."""
import json
import unittest
from datetime import date, timedelta
from decimal import Decimal

from ai.schemas.bundle import EvidenceBundle, sign_bundle
from ai.schemas.mapping import CorpusChunk, MappingEntry, SupportCheck
from ai.tests.fixtures import BUNDLE_X, BUNDLE_Y, CAT, H, INVENTORY, MAPPING, NOW, REFI, item, policy_x
from ai.verifier.mapping import find_conflicts, to_verification, verify_proposal
from ai.verifier.numbers import find_numbers
from ai.verifier.recommendation import verify_recommendation

KEY = bytes(range(32))
KEYS = {"k1": KEY}
SX = sign_bundle(BUNDLE_X, KEY, "k1")
SY = sign_bundle(BUNDLE_Y, KEY, "k1")


def rec(**overrides) -> str:
    out = {
        "recommended_action": "offer_review",
        "magnitude": "low",
        "reason_codes": ["REFI_RATE_DROP"],
        "evidence_ids": [REFI.evidence_id],
        "mapping_id": MAPPING.mapping_id,
        "customer_impacts": ["borrowing_capacity"],
        "uncertainty": 0.2,
        "requires_human_review": False,
        "explanation": "The refinancing rate is 4.0%, so a borrowing-capacity review may be worthwhile.",
    }
    out.update(overrides)
    return json.dumps(out, ensure_ascii=False)


def verify(raw, bundle=SX, now=NOW):
    return verify_recommendation(raw, bundle, now=now, keys=KEYS, catalogue=CAT)


def updated(model, **changes):
    """A validated copy (model_copy(update=...) skips validation)."""
    return type(model).model_validate({**model.model_dump(), **changes})


def codes(result):
    return {f.code.value for f in result.findings}


def signed(items, policy=None, expires=timedelta(hours=1)):
    b = EvidenceBundle(bundle_id="b", case_id="case-1", created_at=NOW, expires_at=NOW + expires,
                       items=items, policy=policy or policy_x())
    return sign_bundle(b, KEY, "k1")


class WorkedCasesTest(unittest.TestCase):
    def test_1_clean_output_passes(self):
        r = verify(rec())
        self.assertEqual(r.decision, "PASS", r.findings)

    def test_2_cherry_picking_is_blocked(self):
        r = verify(rec(), bundle=SY)
        self.assertEqual(r.decision, "BLOCK")
        self.assertIn("CONFLICTING_SIGNALS_MUST_ABSTAIN", codes(r))

    def test_3_invented_number_is_blocked(self):
        r = verify(rec(explanation="The rate is 4.0%, so your district will appreciate 15% next year."))
        self.assertEqual(r.decision, "BLOCK")
        self.assertIn("UNSUPPORTED_NUMBER", codes(r))
        self.assertIn("15", " ".join(f.detail for f in r.findings))

    def test_4_unknown_citation_is_blocked(self):
        r = verify(rec(evidence_ids=["macro.vn.cpi.2026-09"], reason_codes=[], explanation="Inflation is falling."))
        self.assertEqual(r.decision, "BLOCK")
        self.assertIn("UNKNOWN_EVIDENCE", codes(r))


class RecommendationRulesTest(unittest.TestCase):
    def test_abstaining_on_conflict_routes_to_human(self):
        r = verify(rec(recommended_action="abstain", magnitude="none", reason_codes=["CONFLICTING_SIGNALS"],
                       evidence_ids=[REFI.evidence_id, INVENTORY.evidence_id],
                       explanation="The rate is 4.0% but 18 more listings appeared, so the signals conflict."),
                   bundle=SY)
        self.assertEqual(r.decision, "ROUTE_TO_HUMAN", r.findings)
        self.assertEqual(codes(r), {"ABSTAINED"})

    def test_general_reason_codes_need_support(self):
        """Verifier v0.1 (2026-10-19): Qwen used CONFLICTING_SIGNALS as filler where nothing conflicted."""
        r = verify(rec(reason_codes=["CONFLICTING_SIGNALS"]))
        self.assertEqual(r.decision, "BLOCK")
        self.assertIn("UNSUPPORTED_REASON_CODE", codes(r))
        self.assertIn("UNSUPPORTED_REASON_CODE", codes(verify(rec(reason_codes=["STALE_EVIDENCE"]))))
        self.assertIn("UNSUPPORTED_REASON_CODE", codes(verify(rec(reason_codes=["INSUFFICIENT_EVIDENCE"]))))
        # supported: a stale item in the bundle, an untrusted item
        old = item("macro.vn.refi_rate", "2026-03", REFI.observation_period.start.replace(month=3),
                   REFI.observation_period.end.replace(month=3, day=31), "4.0", "pct", "level")
        r = verify(rec(reason_codes=["STALE_EVIDENCE"]), bundle=signed([REFI, old]))
        self.assertNotIn("UNSUPPORTED_REASON_CODE", codes(r))
        bad = updated(REFI, trust="untrusted", trust_reasons=["OCR_LOW_CONFIDENCE"])
        r = verify(rec(recommended_action="abstain", magnitude="none", evidence_ids=[],
                       reason_codes=["INSUFFICIENT_EVIDENCE"], explanation="The rate cannot be read."),
                   bundle=signed([bad]))
        self.assertEqual(codes(r), {"ABSTAINED"})

    def test_uncited_evidence_id_is_not_a_number(self):
        """v0.1.1: ADV-014 abstained, cited nothing and named the unreadable item's ID; "09" was blocked."""
        bad = updated(REFI, trust="untrusted", trust_reasons=["OCR_LOW_CONFIDENCE"])
        r = verify(rec(recommended_action="abstain", magnitude="none", evidence_ids=[], reason_codes=[],
                       explanation=f"The entry {REFI.evidence_id} is unreadable, so no figure can be given."),
                   bundle=signed([bad]))
        self.assertEqual(codes(r), {"ABSTAINED"})
        # its figures are still unsupported when it is not cited
        r = verify(rec(recommended_action="abstain", magnitude="none", evidence_ids=[], reason_codes=[],
                       explanation=f"The entry {REFI.evidence_id} showed 4.0%, but it is untrusted."),
                   bundle=signed([bad]))
        self.assertIn("UNSUPPORTED_NUMBER", codes(r))

    def test_bundle_must_be_signed_unchanged_and_fresh(self):
        self.assertIn("BUNDLE_SIGNATURE_INVALID", codes(verify(rec(), bundle=BUNDLE_X)))
        raw = SX.model_dump()
        raw["items"][0]["value"] = Decimal("6.0")
        self.assertIn("BUNDLE_SIGNATURE_INVALID", codes(verify(rec(), bundle=EvidenceBundle.model_validate(raw))))
        self.assertIn("BUNDLE_EXPIRED", codes(verify(rec(), now=NOW + timedelta(hours=2))))

    def test_unknown_action_fails_schema(self):
        r = verify(rec(recommended_action="approve_loan"))
        self.assertEqual((r.decision, codes(r)), ("BLOCK", {"SCHEMA_INVALID"}))
        self.assertEqual(verify("not json").decision, "BLOCK")

    def test_action_mapping_and_impacts_come_from_policy(self):
        self.assertIn("ACTION_NOT_ALLOWED", codes(verify(rec(recommended_action="hold"))))
        self.assertIn("MAPPING_MISMATCH", codes(verify(rec(mapping_id="mapping.v9.000000000000"))))
        self.assertIn("IMPACT_NOT_IN_POLICY", codes(verify(rec(customer_impacts=["valuation_range"]))))

    def test_no_citations(self):
        # Already rejected by the Recommendation schema (only abstain may be uncited), so it blocks there.
        r = verify(rec(evidence_ids=[], reason_codes=[], explanation="Review."))
        self.assertEqual((r.decision, codes(r)), ("BLOCK", {"SCHEMA_INVALID"}))

    def test_magnitude_and_review_requests_route(self):
        r = verify(rec(magnitude="medium"))
        self.assertEqual((r.decision, codes(r)), ("ROUTE_TO_HUMAN", {"MAGNITUDE_ABOVE_CAP"}))
        r = verify(rec(requires_human_review=True))
        self.assertEqual((r.decision, codes(r)), ("ROUTE_TO_HUMAN", {"HUMAN_REVIEW_REQUESTED"}))

    def test_stale_and_untrusted_citations(self):
        old = item("macro.vn.refi_rate", "2026-03", REFI.observation_period.start.replace(month=3),
                   REFI.observation_period.end.replace(month=3, day=31), "4.0", "pct", "level")
        target = updated(policy_x().targets[0], signals=[], net_weight=Decimal(0))
        policy = updated(policy_x(), targets=[target.model_dump()])
        r = verify(rec(evidence_ids=[old.evidence_id], reason_codes=[]), bundle=signed([old], policy))
        self.assertIn("STALE_EVIDENCE", codes(r))
        bad = updated(REFI, trust="untrusted", trust_reasons=["OCR_LOW_CONFIDENCE"])
        self.assertIn("UNTRUSTED_EVIDENCE", codes(verify(rec(), bundle=signed([bad]))))

    def test_demo_context_number_is_blocked(self):
        r = verify(rec(explanation="The rate is 4.0%. A review could update your safe borrowing limit of VND 3.20B."))
        self.assertIn("UNSUPPORTED_NUMBER", codes(r))

    def test_vietnamese_number_format_passes(self):
        r = verify(rec(explanation="Lãi suất tái cấp vốn là 4,0% trong tháng 9/2026, nên có thể xem xét lại hạn mức vay."))
        self.assertEqual(r.decision, "PASS", r.findings)

    def test_wrong_unit_is_blocked(self):
        r = verify(rec(explanation="The refinancing rate is now 4.0 bps."))
        self.assertIn("UNIT_MISMATCH", codes(r))

    def test_change_and_direction_against_baseline(self):
        with_baseline = updated(REFI, baseline={"value": Decimal("4.5"), "unit": "pct", "period_label": "2026-06"})
        bundle = signed([with_baseline])
        ok = verify(rec(explanation="The rate fell 0.5 percentage points from 4.5% to 4.0%."), bundle=bundle)
        self.assertEqual(ok.decision, "PASS", ok.findings)
        wrong = verify(rec(reason_codes=["REFI_RATE_RISE"]), bundle=bundle)  # observed in the live check
        self.assertIn("UNSUPPORTED_REASON_CODE", codes(wrong))

    def test_reason_code_for_metric_not_cited(self):
        r = verify(rec(reason_codes=["PRICE_INDEX_DROP"]))  # observed in the live check
        self.assertIn("UNSUPPORTED_REASON_CODE", codes(r))
        self.assertIn("UNSUPPORTED_REASON_CODE", codes(verify(rec(reason_codes=["MADE_UP"]))))

    def test_freshness_date_is_allowed(self):
        # found in the first runner check (2026-10-18): the deadline is in the evidence shown to the model
        d = REFI.freshness.deadline
        r = verify(rec(explanation=f"The rate is 4.0%, fresh until {d:%B} {d.day}, {d.year}."))
        self.assertEqual(r.decision, "PASS", r.findings)

    def test_ids_in_text_are_not_numbers(self):
        r = verify(rec(explanation=f"Per {REFI.evidence_id}, the rate is 4.0% (mapping {MAPPING.mapping_id})."))
        self.assertEqual(r.decision, "PASS", r.findings)


class NumbersTest(unittest.TestCase):
    def test_readings(self):
        (m,) = find_numbers("1.250")
        self.assertEqual(m.readings, {Decimal("1250"), Decimal("1.25")})
        (m,) = find_numbers("4,0%")
        self.assertIn(Decimal("4"), m.readings)
        self.assertEqual(m.unit_hint, "pct")
        (m,) = find_numbers("down -0.5 pp")
        self.assertIn(Decimal("-0.5"), m.readings)
        self.assertEqual(m.unit_hint, "pp")


CHUNK = CorpusChunk(
    chunk_id="syn-src1:c4", source_id="syn-src1", document_hash=H, page=3,
    text="In our sample, a reduction in the policy rate was followed within one to six months by an\n"
         "increase in the loan amounts households could service.",
    publisher="Example Monetary Research Institute (synthetic)", publication_date=date(2024, 5, 1),
    geography="vn", language="en", corpus_version="corpus.v0", synthetic=True,
)


def proposal(**overrides) -> str:
    out = {"metric_key": "refi_rate", "target": "borrowing_capacity", "direction": "negative", "lag_months": [1, 6],
           "size_range": "medium", "geography": "vn",
           "evidence": [{"chunk_id": CHUNK.chunk_id, "quote": "a reduction in the policy rate was followed within one to six months"}]}
    out.update(overrides)
    return json.dumps(out)


def check_proposal(raw, chunks=(CHUNK,), support=None):
    return verify_proposal(raw, list(chunks), now=NOW, catalogue=CAT, support_check=support)


class MappingVerifierTest(unittest.TestCase):
    def test_verbatim_quote_passes_and_spans_are_exact(self):
        r, p, rel = check_proposal(proposal())
        self.assertEqual(r.decision, "PASS", r.findings)
        self.assertEqual(CHUNK.text[rel[0].char_start:rel[0].char_end], rel[0].quote)
        entry = MappingEntry(proposal=p, verification=to_verification(r, p))
        self.assertEqual(entry.verification.status, "accepted")

    def test_line_break_in_source_still_matches(self):
        quote = "followed within one to six months by an increase in the loan amounts"
        r, _, rel = check_proposal(proposal(evidence=[{"chunk_id": CHUNK.chunk_id, "quote": quote}]))
        self.assertEqual(r.decision, "PASS", r.findings)
        self.assertIn("\n", rel[0].quote)  # stored as the chunk's own text

    def test_fabricated_quote_unknown_chunk_and_short_quote(self):
        fake = [{"chunk_id": CHUNK.chunk_id, "quote": "A 1 percentage point cut raises borrowing capacity by 12%."}]
        self.assertIn("QUOTE_NOT_VERBATIM", codes(check_proposal(proposal(evidence=fake))[0]))
        other = [{"chunk_id": "syn-src9:c1", "quote": "Lower rates raise borrowing capacity a lot."}]
        self.assertIn("UNKNOWN_EVIDENCE", codes(check_proposal(proposal(evidence=other))[0]))
        short = [{"chunk_id": CHUNK.chunk_id, "quote": "policy rate"}]
        self.assertIn("QUOTE_TOO_SHORT", codes(check_proposal(proposal(evidence=short))[0]))

    def test_geography_age_and_metric(self):
        us = updated(CHUNK, geography="us")
        self.assertIn("GEOGRAPHY_MISMATCH", codes(check_proposal(proposal(), chunks=[us])[0]))
        old = updated(CHUNK, publication_date=date(2001, 1, 1))
        self.assertIn("STALE_EVIDENCE", codes(check_proposal(proposal(), chunks=[old])[0]))
        self.assertIn("UNKNOWN_METRIC", codes(check_proposal(proposal(metric_key="gdp"))[0]))

    def test_support_check(self):
        no = SupportCheck(model="openai-x", verdict="contradicts", rationale="the quote says the opposite", checked_at=NOW)
        r, p, _ = check_proposal(proposal(direction="positive"), support=no)
        self.assertIn("SUPPORT_NOT_ESTABLISHED", codes(r))
        self.assertEqual(to_verification(r, p, no).status, "rejected")
        r, _, _ = check_proposal(proposal(direction="positive"))  # without a support check the rules pass
        self.assertEqual(r.decision, "PASS")
        self.assertIn("skipped", next(c.detail for c in r.checks if c.check == "support"))

    def test_conflicting_sources(self):
        a = MAPPING.entries[0].proposal
        b = updated(a, direction="positive")
        self.assertEqual([f.code.value for f in find_conflicts([a, b])], ["CONFLICTING_SOURCES"])
        self.assertEqual(find_conflicts([a]), [])


if __name__ == "__main__":
    unittest.main()


class SentenceChecksTest(unittest.TestCase):
    """Verifier v0.2: the sentences Qwen wrote in the 2026-10-23 runner measurement (findings log)."""

    def setUp(self):
        from datetime import datetime, timezone
        sep = (datetime(2026, 9, 1, tzinfo=timezone.utc), datetime(2026, 9, 30, 23, 59, 59, tzinfo=timezone.utc))
        self.volume = updated(item("market.hn-dong-da.transaction_volume", "2026-09", *sep, "255", "count", "level"),
                              baseline={"value": "217", "unit": "count", "period_label": "2026-06"})
        self.refi = updated(REFI, baseline={"value": "4.5", "unit": "pct", "period_label": "2026-06"})
        day = (datetime(2026, 7, 6, tzinfo=timezone.utc), datetime(2026, 7, 6, 23, 59, 59, tzinfo=timezone.utc))
        self.p50 = item("avm.hn-dong-da.value_p50", "2026-07-06", *day, "18.41", "vnd_bn", "level")
        self.bundle = signed([self.refi, self.volume, self.p50])

    def run_rec(self, explanation, cited=None):
        cited = cited or [self.refi.evidence_id, self.volume.evidence_id]
        return verify(rec(evidence_ids=cited, explanation=explanation), self.bundle)

    def test_wrong_direction_blocks(self):
        r = self.run_rec("However, transaction volume in your area (Dong Da) fell to 255 units (from 217), "
                         "indicating slower market activity.")
        self.assertEqual(r.decision, "BLOCK")
        self.assertIn("DIRECTION_MISMATCH", codes(r))
        # Vietnamese movement words are covered too
        self.assertIn("DIRECTION_MISMATCH", codes(self.run_rec("Số giao dịch giảm xuống 255.")))

    def test_right_direction_and_effect_words_pass(self):
        for text in ("Transaction volume rose to 255 from 217, and the refinancing rate fell to 4.0%.",
                     "Lower refinancing rates (4.0%) may increase borrowing capacity.",
                     "The refinancing rate dropped to 4.0% while transaction volume rose to 255."):
            r = self.run_rec(text)
            self.assertNotIn("DIRECTION_MISMATCH", codes(r), text)
            self.assertEqual(r.decision, "PASS", (text, r.findings))

    def test_calling_fresh_evidence_stale_blocks(self):
        r = self.run_rec("Property valuation review is not recommended as the only available AVM data for your "
                         "property location (Dong Da) is from July 2026, which is considered stale.")
        self.assertIn("STALENESS_MISMATCH", codes(r))
        self.assertEqual(r.decision, "BLOCK")

    def test_true_or_negated_staleness_passes(self):
        stale_refi = updated(self.refi, freshness={**self.refi.freshness.model_dump(),
                                                   "deadline": NOW - timedelta(days=1)})
        bundle = signed([stale_refi, self.volume, self.p50])
        r = verify(rec(evidence_ids=[self.volume.evidence_id], reason_codes=[],
                       explanation="The refinancing rate figure is stale, so it was not used."), bundle)
        self.assertNotIn("STALENESS_MISMATCH", codes(r))
        r = self.run_rec("The AVM band is not stale.")
        self.assertNotIn("STALENESS_MISMATCH", codes(r))

    def test_percentile_labels_are_not_numbers(self):
        r = self.run_rec("The P50 estimate is 18.41 billion VND.", cited=[self.refi.evidence_id, self.p50.evidence_id])
        self.assertNotIn("UNSUPPORTED_NUMBER", codes(r))
        r = self.run_rec("The estimate is 50 billion VND.", cited=[self.refi.evidence_id, self.p50.evidence_id])
        self.assertIn("UNSUPPORTED_NUMBER", codes(r))  # a bare 50 is still a number

