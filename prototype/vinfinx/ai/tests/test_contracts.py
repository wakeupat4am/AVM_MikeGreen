"""Tests for the mapping, policy, bundle, recommendation and verifier schemas.
Run from prototype/vinfinx: python -m unittest discover -s ai/tests -t ."""
import os
import unittest
from datetime import timedelta
from decimal import Decimal

from pydantic import ValidationError

from ai.schemas.bundle import EvidenceBundle, sign_bundle, verify_bundle
from ai.schemas.common import content_hash
from ai.schemas.mapping import MappingSet, MappingVerification, RelationshipEvidence, proposal_generation_schema
from ai.schemas.policy import PolicyResult
from ai.schemas.recommendation import (
    MAX_REASON_CODES, Recommendation, citable_evidence_ids, generation_schema, supported_reason_codes,
)
from ai.schemas.verifier import VerifierResult
from ai.tests.fixtures import (
    BUNDLE_X, BUNDLE_Y, CAT, H, INVENTORY, MAPPING, NOW, REFI, bundle, mapping_set, policy_x, policy_y,
)

KEY = bytes(range(32))


def updated_item(item, **changes):
    return type(item).model_validate({**item.model_dump(), **changes})


def with_baseline(item, value: str, period_label: str):
    data = item.model_dump()
    data["baseline"] = {"value": value, "unit": item.unit, "period_label": period_label}
    return type(item).model_validate(data)


# The fixtures carry no baselines; these do: the refinancing rate fell, Dong Da inventory rose.
REFI_FELL = with_baseline(REFI, "4.5", "2026-06")
INVENTORY_ROSE = with_baseline(INVENTORY, "15", "2026-07-01")
BUNDLE_X_MOVED = bundle(policy_x(), [REFI_FELL])
BUNDLE_Y_MOVED = bundle(policy_y(), [REFI_FELL, INVENTORY_ROSE])


def stale_only_bundle():
    """The refinancing rate's only item is excluded as stale (adversarial case ADV-012's shape)."""
    target = policy_x().targets[0].model_dump()
    target["signals"][0].update(material=False, excluded_reason="STALE")
    target.update(net_weight=Decimal(0), outcome="none", allowed_actions=["request_evidence", "no_change", "abstain"],
                  magnitude_cap="none")
    policy = PolicyResult.model_validate({**policy_x().model_dump(), "targets": [target]})
    return bundle(policy, [REFI_FELL])


def props(schema: dict, action: str) -> dict:
    """The properties of the schema branch that allows `action`."""
    for branch in schema.get("anyOf", [schema]):
        if action in branch["properties"]["recommended_action"]["enum"]:
            return branch["properties"]
    raise AssertionError(f"no branch allows {action}")


def recommendation(**overrides) -> dict:
    rec = {
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
    rec.update(overrides)
    return rec


class RelationshipEvidenceTest(unittest.TestCase):
    def test_span_must_match_quote(self):
        quote = "Lower policy rates ease mortgage affordability."
        base = dict(
            evidence_id="rel.src1.c4.0", source_id="src1", document_hash=H, chunk_id="src1:c4", page=3,
            char_start=10, char_end=10 + len(quote), quote=quote, publisher="Example Central Bank Research",
            publication_date="2024-05-01", geography="vn", language="en", corpus_version="corpus.v1",
        )
        RelationshipEvidence(**base)
        with self.assertRaises(ValidationError):
            RelationshipEvidence(**{**base, "char_end": base["char_end"] + 5})


class MappingTest(unittest.TestCase):
    def test_mapping_id_derived_from_content(self):
        self.assertRegex(MAPPING.mapping_id, r"^mapping\.v1\.[0-9a-f]{12}$")
        raw = MAPPING.model_dump()
        raw["corpus_version"] = "corpus.v2"  # content changed but ID kept: rejected
        with self.assertRaises(ValidationError):
            MappingSet.model_validate(raw)

    def test_status_change_keeps_id(self):
        self.assertEqual(mapping_set("validated").mapping_id, MAPPING.mapping_id)
        raw = MAPPING.model_dump()
        raw["status"] = "validated"  # validated without a reviewer: rejected
        with self.assertRaises(ValidationError):
            MappingSet.model_validate(raw)

    def test_only_accepted_and_supported_proposals(self):
        p = MAPPING.entries[0].proposal
        with self.assertRaises(ValidationError):
            MappingVerification(
                proposal_hash=content_hash(p), rule_checks=[{"check": "quote_verbatim", "passed": False}],
                status="accepted",
            )
        with self.assertRaises(ValidationError):
            MappingVerification(
                proposal_hash=content_hash(p), rule_checks=[{"check": "quote_verbatim", "passed": True}],
                support_check={"model": "openai-x", "verdict": "contradicts", "rationale": "...", "checked_at": NOW},
                status="accepted",
            )

    def test_lookup_and_lag(self):
        self.assertIsNotNone(MAPPING.lookup("refi_rate", "borrowing_capacity", "vn"))
        self.assertIsNone(MAPPING.lookup("cpi", "borrowing_capacity", "vn"))
        raw = MAPPING.entries[0].proposal.model_dump()
        raw["lag_months"] = (6, 1)
        with self.assertRaises(ValidationError):
            type(MAPPING.entries[0].proposal).model_validate(raw)


class PolicyResultTest(unittest.TestCase):
    def test_net_weight_must_match_signals(self):
        raw = policy_x().model_dump()
        raw["targets"][0]["net_weight"] = Decimal("2.0")
        with self.assertRaises(ValidationError):
            PolicyResult.model_validate(raw)

    def test_conflict_must_abstain(self):
        raw = BUNDLE_Y.policy.model_dump()
        raw["targets"][0]["must_abstain"] = False
        with self.assertRaises(ValidationError):
            PolicyResult.model_validate(raw)

    def test_unvalidated_mapping_is_low_impact_only(self):
        raw = policy_x().model_dump()
        raw["targets"][0]["allowed_actions"] = ["hold"]
        with self.assertRaises(ValidationError):
            PolicyResult.model_validate(raw)
        raw = policy_x().model_dump()
        raw["targets"][0]["magnitude_cap"] = "high"
        with self.assertRaises(ValidationError):
            PolicyResult.model_validate(raw)


class BundleTest(unittest.TestCase):
    def test_sign_verify_and_tamper(self):
        signed = sign_bundle(BUNDLE_X, KEY, "k1")
        self.assertTrue(verify_bundle(signed, {"k1": KEY}))
        self.assertFalse(verify_bundle(signed, {"k2": KEY}))  # unknown key ID
        self.assertFalse(verify_bundle(BUNDLE_X, {"k1": KEY}))  # unsigned
        raw = signed.model_dump()
        raw["items"][0]["value"] = Decimal("6.0")  # tampered after signing
        self.assertFalse(verify_bundle(EvidenceBundle.model_validate(raw), {"k1": KEY}))
        with self.assertRaises(ValueError):
            sign_bundle(BUNDLE_X, b"short", "k1")

    def test_expiry_and_consistency(self):
        self.assertFalse(BUNDLE_X.is_expired(NOW))
        self.assertTrue(BUNDLE_X.is_expired(NOW + timedelta(hours=2)))
        raw = BUNDLE_Y.model_dump()
        raw["items"] = [raw["items"][0]]  # policy cites the inventory item, which is missing
        with self.assertRaises(ValidationError):
            EvidenceBundle.model_validate(raw)
        raw = BUNDLE_X.model_dump()
        raw["items"] = [raw["items"][0], raw["items"][0]]
        with self.assertRaises(ValidationError):
            EvidenceBundle.model_validate(raw)


class RecommendationTest(unittest.TestCase):
    def test_valid_and_uncited(self):
        Recommendation.model_validate(recommendation())
        Recommendation.model_validate(recommendation(recommended_action="abstain", evidence_ids=[]))
        with self.assertRaises(ValidationError):
            Recommendation.model_validate(recommendation(evidence_ids=[]))
        with self.assertRaises(ValidationError):
            Recommendation.model_validate(recommendation(recommended_action="approve_loan"))

    def test_generation_schema_follows_the_bundle(self):
        sx, sy = generation_schema(BUNDLE_X, CAT), generation_schema(BUNDLE_Y, CAT)
        offer = props(sx, "offer_review")
        self.assertEqual(offer["evidence_ids"]["items"]["enum"], [REFI.evidence_id])
        self.assertEqual(offer["mapping_id"]["enum"], [MAPPING.mapping_id])
        self.assertEqual(offer["magnitude"]["enum"], ["none", "low"])
        self.assertNotIn("hold", offer["recommended_action"]["enum"])
        # Bundle Y is a conflict: the model cannot even generate offer_review (worked case 2, cherry-picking)
        self.assertEqual(props(sy, "request_evidence")["recommended_action"]["enum"], ["request_evidence"])
        self.assertEqual(props(sy, "abstain")["recommended_action"]["enum"], ["abstain"])
        self.assertEqual([b["properties"]["magnitude"]["enum"] for b in sy["anyOf"]], [["none"], ["none"]])
        self.assertIn(INVENTORY.evidence_id, props(sy, "abstain")["evidence_ids"]["items"]["enum"])

    def test_citation_required_unless_abstaining(self):
        sx = generation_schema(BUNDLE_X, CAT)
        self.assertEqual(props(sx, "offer_review")["evidence_ids"]["minItems"], 1)
        self.assertNotIn("minItems", props(sx, "abstain")["evidence_ids"])
        # the branches generate their fields in the same order
        self.assertEqual([list(b["properties"]) for b in sx["anyOf"]][0], list(props(sx, "abstain")))

    def test_excluded_evidence_leaves_only_abstain(self):
        """Approved 2026-10-19: with nothing citable, abstain is the only action the schema allows."""
        b = stale_only_bundle()
        self.assertEqual(citable_evidence_ids(b), [])
        schema = generation_schema(b, CAT)
        self.assertNotIn("anyOf", schema)
        self.assertEqual(schema["properties"]["recommended_action"]["enum"], ["abstain"])
        self.assertEqual(schema["properties"]["evidence_ids"], {"type": "array", "maxItems": 0})
        # no metric code for the excluded item; the general codes the bundle shows
        self.assertEqual(supported_reason_codes(b, CAT), ["STALE_EVIDENCE", "INSUFFICIENT_EVIDENCE"])
        # stale or untrusted items are not citable even if the policy result did not exclude them
        untrusted = updated_item(REFI, trust="untrusted", trust_reasons=["OCR_LOW_CONFIDENCE"])
        self.assertEqual(citable_evidence_ids(bundle(policy_x(), [untrusted])), [])
        self.assertEqual(citable_evidence_ids(BUNDLE_X, at=REFI.freshness.deadline + timedelta(days=1)), [])

    def test_reason_codes_follow_the_evidence(self):
        """Only metrics in the bundle, in the direction they moved (baseline 2026-10-18: 13 of 20 model
        outputs named metrics that were not in the bundle), and general codes only when the bundle shows
        them (first-fix run 2026-10-19: CONFLICTING_SIGNALS in 8 of 12 outputs with no conflict)."""
        self.assertEqual(supported_reason_codes(BUNDLE_X_MOVED, CAT), ["REFI_RATE_DROP"])
        self.assertEqual(supported_reason_codes(BUNDLE_Y_MOVED, CAT),
                         ["REFI_RATE_DROP", "INVENTORY_RISE", "CONFLICTING_SIGNALS"])
        # No baseline: the evidence does not show a direction, so no metric code; nothing else is supported
        self.assertEqual(supported_reason_codes(BUNDLE_X, CAT), [])
        self.assertEqual(props(generation_schema(BUNDLE_X, CAT), "abstain")["reason_codes"],
                         {"type": "array", "maxItems": 0})
        reason_codes = props(generation_schema(BUNDLE_X_MOVED, CAT), "offer_review")["reason_codes"]
        self.assertEqual(reason_codes["maxItems"], MAX_REASON_CODES)


class VerifierResultTest(unittest.TestCase):
    def result(self, decision, codes):
        return VerifierResult(
            decision=decision, findings=[{"code": c} for c in codes], verifier_version="verifier.v0",
            subject="recommendation", subject_hash=H, checked_at=NOW,
        )

    def test_decision_follows_findings(self):
        self.result("PASS", [])
        self.result("BLOCK", ["UNSUPPORTED_NUMBER"])  # worked case 3
        self.result("BLOCK", ["UNKNOWN_EVIDENCE"])  # worked case 4
        self.result("ROUTE_TO_HUMAN", ["MAGNITUDE_ABOVE_CAP"])
        self.result("BLOCK", ["MAGNITUDE_ABOVE_CAP", "STALE_EVIDENCE"])
        for decision, codes in [("PASS", ["STALE_EVIDENCE"]), ("ROUTE_TO_HUMAN", ["NO_CITATIONS"]), ("BLOCK", [])]:
            with self.assertRaises(ValidationError):
                self.result(decision, codes)


MODEL_PATH = os.environ.get(
    "VINFINX_QWEN_PATH", "/mnt/disk1/backup_user/24chuong.ta/vinfinx/models/Qwen3.5-9B"
)


@unittest.skipUnless(os.path.isdir(MODEL_PATH), "Qwen tokenizer not available on this machine")
class GrammarTest(unittest.TestCase):
    """The generation schemas compile in xgrammar with Qwen's tokenizer and reject invalid references."""

    @classmethod
    def setUpClass(cls):
        import xgrammar as xgr
        from transformers import AutoConfig, AutoTokenizer

        cls.xgr = xgr
        cls.tok = AutoTokenizer.from_pretrained(MODEL_PATH)
        vocab = AutoConfig.from_pretrained(MODEL_PATH).get_text_config().vocab_size
        cls.compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(cls.tok, vocab_size=vocab))

    def accepts(self, schema, text) -> bool:
        import json

        matcher = self.xgr.GrammarMatcher(self.compiler.compile_json_schema(json.dumps(schema)))
        return all(matcher.accept_token(t) for t in self.tok.encode(text, add_special_tokens=False))

    def test_recommendation_grammar(self):
        import json

        schema = generation_schema(BUNDLE_X_MOVED, CAT)
        self.assertTrue(self.accepts(schema, json.dumps(recommendation())))
        self.assertFalse(self.accepts(schema, json.dumps(recommendation(reason_codes=["PRICE_INDEX_RISE"]))))
        self.assertFalse(self.accepts(schema, json.dumps(recommendation(reason_codes=["REFI_RATE_RISE"]))))
        self.assertFalse(self.accepts(schema, json.dumps(recommendation(reason_codes=["REFI_RATE_DROP"] * 4))))
        self.assertFalse(self.accepts(schema, json.dumps(recommendation(reason_codes=["CONFLICTING_SIGNALS"]))))
        self.assertFalse(self.accepts(schema, json.dumps(recommendation(evidence_ids=[]))))  # uncited action
        self.assertTrue(self.accepts(schema, json.dumps(recommendation(
            recommended_action="abstain", magnitude="none", evidence_ids=[], reason_codes=[]))))
        stale = generation_schema(stale_only_bundle(), CAT)
        self.assertFalse(self.accepts(stale, json.dumps(recommendation(recommended_action="no_change", magnitude="none"))))
        self.assertFalse(self.accepts(schema, json.dumps(recommendation(evidence_ids=["macro.vn.cpi.2026-09"]))))
        self.assertFalse(self.accepts(schema, json.dumps(recommendation(recommended_action="hold"))))
        self.assertFalse(self.accepts(schema, json.dumps(recommendation(mapping_id="mapping.v9.000000000000"))))

    def test_proposal_grammar(self):
        import json

        schema = proposal_generation_schema(["refi_rate"], ["src1:c4"], ["vn"])
        good = MAPPING.entries[0].proposal.model_dump(mode="json")
        self.assertTrue(self.accepts(schema, json.dumps(good)))
        bad = {**good, "evidence": [{"chunk_id": "src9:c1", "quote": "invented"}]}
        self.assertFalse(self.accepts(schema, json.dumps(bad)))


if __name__ == "__main__":
    unittest.main()
