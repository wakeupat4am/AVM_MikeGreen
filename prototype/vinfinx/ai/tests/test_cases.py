"""The adversarial case set is well formed. These tests check the cases' format only; they deliberately do
not run the verifier against the cases (the set is for the AI engineer to edit and freeze first)."""
import json
import unittest
from pathlib import Path

from ai.catalog import load_catalogue
from ai.evaluation.cases import (
    Case,
    CaseSet,
    DecisionInput,
    MappingInput,
    build_bundle,
    case_mapping_id,
    load_case_set,
    scripted_output,
)

CASES_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "cases" / "adversarial_v0.yaml"


class CaseSetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = load_catalogue()
        cls.cases = load_case_set(CASES_PATH)

    def test_loads_and_covers_every_category(self):
        from ai.evaluation.cases import Category

        self.assertEqual({c.category for c in self.cases.cases}, set(Category))
        self.assertTrue(self.cases.synthetic)
        self.assertEqual(self.cases.status, "frozen")

    def test_both_languages_present(self):
        self.assertEqual({c.language for c in self.cases.cases}, {"en", "vi"})

    def test_every_decision_case_builds_a_valid_bundle(self):
        for case in self.cases.cases:
            if isinstance(case.input, DecisionInput):
                with self.subTest(case.case_id):
                    bundle = build_bundle(case, self.cat)
                    self.assertEqual(bundle.mapping_id, case_mapping_id(case.case_id))
                    self.assertTrue(all(i.synthetic for i in bundle.items))

    def test_scripted_outputs_are_json_with_the_case_mapping_id(self):
        for case in self.cases.cases:
            if isinstance(case.input, DecisionInput) and case.input.scripted_output is not None:
                with self.subTest(case.case_id):
                    out = json.loads(scripted_output(case))
                    self.assertEqual(out["mapping_id"], case_mapping_id(case.case_id))
            if isinstance(case.input, MappingInput):
                for raw in case.input.scripted_outputs:
                    with self.subTest(case.case_id):
                        json.loads(raw)

    def test_scripted_cases_name_required_findings(self):
        for case in self.cases.cases:
            if case.scripted and case.expect.kind != "document":
                with self.subTest(case.case_id):
                    self.assertTrue(case.expect.required_findings)

    def test_invariance_pairs_are_decision_cases(self):
        by_id = {c.case_id: c for c in self.cases.cases}
        for case in self.cases.cases:
            if case.invariance is not None:
                with self.subTest(case.case_id):
                    other = by_id[case.invariance.same_as]
                    self.assertEqual(case.input.kind, other.input.kind)

    def test_stale_case_is_actually_stale(self):
        case = next(c for c in self.cases.cases if c.case_id == "ADV-012")
        bundle = build_bundle(case, self.cat)
        self.assertLess(bundle.items[0].freshness.deadline, case.input.as_of)

    def test_rejects_malformed_case(self):
        raw = self.cases.cases[0].model_dump(mode="json")
        raw["expect"] = {"kind": "mapping", "verdicts": ["accepted"]}  # decision input, mapping expectation
        with self.assertRaises(ValueError):
            Case.model_validate(raw)
        raw = self.cases.model_dump(mode="json")
        raw["frozen_by"] = None  # frozen without saying who froze it
        with self.assertRaises(ValueError):
            CaseSet.model_validate(raw)

    def test_set_hash_is_stable(self):
        self.assertEqual(self.cases.set_hash, load_case_set(CASES_PATH).set_hash)

    def test_frozen_files_match_the_manifest(self):
        """A frozen set must never change: both its content hash and its file hash are recorded."""
        import hashlib
        import re

        manifest = (CASES_PATH.parent / "MANIFEST.md").read_text(encoding="utf-8")
        row = next(line for line in manifest.splitlines() if line.startswith(f"| `{CASES_PATH.name}`"))
        content_hash, file_hash = re.findall(r"`(sha256:[0-9a-f]{64})`", row)
        self.assertEqual(self.cases.set_hash, content_hash)
        self.assertEqual("sha256:" + hashlib.sha256(CASES_PATH.read_bytes()).hexdigest(), file_hash)


if __name__ == "__main__":
    unittest.main()
