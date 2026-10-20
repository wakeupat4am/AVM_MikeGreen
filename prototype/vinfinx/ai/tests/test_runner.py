"""Evaluation runner tests. The model is replaced by a fake, so no GPU is needed."""
import json
import unittest
from unittest import mock

from ai.evaluation import runner
from ai.evaluation.cases import case_mapping_id


class ScriptedRunTest(unittest.TestCase):
    def test_every_runnable_scripted_case_matches_its_expectation(self):
        report = runner.run(runner.DEFAULT_CASES, "scripted", "all", runner.DEFAULT_URL, out_dir=None)
        ran = [c for c in report["cases"] if not c["skipped"]]
        self.assertEqual(len(ran), 13)
        failed = [(c["case_id"], c["checks"]) for c in ran if not c["passed"]]
        self.assertEqual(failed, [])
        self.assertTrue(report["official"])  # the set is frozen
        self.assertIn("hash", report["case_set"])


def fake_model(answer_for):
    """A stand-in for call_model: answers decision prompts with answer_for(prompt) and mapping prompts with a
    correct proposal."""

    def call(url, messages, schema, max_tokens=500):
        decision = schema.get("anyOf", [schema])[0]["properties"]
        if "mapping_id" in decision:
            mapping_id = decision["mapping_id"]["enum"][0]
            return runner.Generation(text=answer_for(messages[-1]["content"], mapping_id), prompt_tokens=300,
                                     completion_tokens=120, seconds=6.0)
        proposal = {"metric_key": "refi_rate", "target": "borrowing_capacity", "direction": "negative",
                    "lag_months": [1, 6], "size_range": "medium", "geography": "vn",
                    "evidence": [{"chunk_id": schema["properties"]["evidence"]["items"]["properties"]["chunk_id"]["enum"][0],
                                  "quote": "a reduction in the policy rate was followed within one to six months"}]}
        return runner.Generation(text=json.dumps(proposal), prompt_tokens=280, completion_tokens=90, seconds=4.0)

    return call


def always_offer_review(prompt, mapping_id):
    return json.dumps({
        "recommended_action": "offer_review", "magnitude": "low", "reason_codes": ["REFI_RATE_DROP"],
        "evidence_ids": ["macro.vn.refi_rate.2026-09"], "mapping_id": mapping_id,
        "customer_impacts": ["borrowing_capacity"], "uncertainty": 0.2, "requires_human_review": False,
        "explanation": "The refinancing rate is 4.0%, down from 4.5%.",
    })


class ModelRunTest(unittest.TestCase):
    def run_model(self, answer, only):
        with mock.patch.object(runner, "call_model", fake_model(answer)):
            return runner.run(runner.DEFAULT_CASES, "model", "all", runner.DEFAULT_URL, out_dir=None, only=only)

    def test_metrics_and_invariance(self):
        report = self.run_model(always_offer_review, {"ADV-001", "ADV-003", "ADV-005", "ADV-033"})
        by_id = {c["case_id"]: c for c in report["cases"]}
        self.assertTrue(by_id["ADV-001"]["passed"])
        self.assertTrue(by_id["ADV-003"]["checks"]["invariance"])  # same answer in Vietnamese
        self.assertFalse(by_id["ADV-005"]["passed"])  # offer_review on a conflict: blocked, wrong action
        self.assertTrue(by_id["ADV-033"]["passed"])
        m = report["metrics"]["model_decisions"]
        self.assertEqual(m["n"], 3)
        self.assertEqual(m["held_back_when_required"], 0.0)  # ADV-005 required holding back
        self.assertEqual(m["acted_when_allowed"], 1.0)
        self.assertEqual(m["unsupported_claim_rate"], 0.0)
        self.assertEqual(m["citation_precision"], 1.0)

    def test_unsupported_numbers_are_counted(self):
        def leaky(prompt, mapping_id):
            out = json.loads(always_offer_review(prompt, mapping_id))
            out["explanation"] = "The rate is 4.0%, so your limit of VND 3.20B may rise."
            return json.dumps(out)

        report = self.run_model(leaky, {"ADV-001"})
        self.assertEqual(report["metrics"]["model_decisions"]["unsupported_claim_rate"], 1.0)
        self.assertFalse(report["cases"][0]["passed"])

    def test_model_errors_do_not_stop_the_run(self):
        def broken(url, messages, schema, max_tokens=500):
            return runner.Generation(text="", error="URLError: connection refused")

        with mock.patch.object(runner, "call_model", broken):
            report = runner.run(runner.DEFAULT_CASES, "model", "all", runner.DEFAULT_URL, out_dir=None,
                                only={"ADV-001", "ADV-033"})
        self.assertEqual(report["metrics"]["model_call_errors"], 2)
        self.assertTrue(all(not c["passed"] for c in report["cases"]))

    def test_prompt_contains_bundle_and_respects_message_limit(self):
        from ai.catalog import load_catalogue
        from ai.evaluation.cases import build_bundle, load_case_set

        cat = load_catalogue()
        cases = load_case_set(runner.DEFAULT_CASES)
        for case in cases.cases:
            if case.input.kind == "decision":
                msgs = runner.decision_messages(case, build_bundle(case, cat))
                with self.subTest(case.case_id):
                    self.assertLessEqual(max(len(m["content"]) for m in msgs), 4000)  # adapter limit
                    self.assertLessEqual(len(msgs), 12)
                    self.assertIn(case_mapping_id(case.case_id), msgs[-1]["content"])


if __name__ == "__main__":
    unittest.main()
