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
from ai.schemas.recommendation import Recommendation, generation_schema
from ai.schemas.verifier import VerifierResult
from ai.tests.fixtures import BUNDLE_X, BUNDLE_Y, CAT, H, INVENTORY, MAPPING, NOW, REFI, mapping_set, policy_x

KEY = bytes(range(32))


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
        self.assertEqual(sx["properties"]["evidence_ids"]["items"]["enum"], [REFI.evidence_id])
        self.assertEqual(sx["properties"]["mapping_id"]["enum"], [MAPPING.mapping_id])
        self.assertEqual(sx["properties"]["magnitude"]["enum"], ["none", "low"])
        self.assertNotIn("hold", sx["properties"]["recommended_action"]["enum"])
        # Bundle Y is a conflict: the model cannot even generate offer_review (worked case 2, cherry-picking)
        self.assertEqual(sy["properties"]["recommended_action"]["enum"], ["request_evidence", "abstain"])
        self.assertEqual(sy["properties"]["magnitude"]["enum"], ["none"])
        self.assertIn(INVENTORY.evidence_id, sy["properties"]["evidence_ids"]["items"]["enum"])


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

        schema = generation_schema(BUNDLE_X, CAT)
        self.assertTrue(self.accepts(schema, json.dumps(recommendation())))
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
