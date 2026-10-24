"""End-to-end runner and its API (docs/RUNNER_PLAN.md): every stage is recorded, a failing stage stops the run with
its error, the held-out set is never run, a routed run is reviewed once, and only passed or approved recommendations
reach the customer. No GPU, no Tesseract: OCR and the model are fakes; extraction is patched to fixed items."""
import csv
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from ai import runner as rn
from ai.api import runs
from ai.api.service import ApiError, Store
from ai.avm import avm_items
from ai.evaluation.runner import Generation
from ai.extraction.macro_document import Extraction
from ai.ocr import OcrPage
from ai.policy.provisional_mapping import provisional_mapping_set
from ai.tests.fixtures import CAT, NOW, REFI

PROPERTY = "7b99d5ef-3f40-fb25-4a5a-d984c54c5b41"  # demo-minh's property (customers_v0.yaml)
KEYS = {"test-key": b"k" * 32}


def fake_ocr(images):
    return [OcrPage(engine="fake", engine_version="0", model="fake", page_sha256="sha256:" + "b" * 64, page=n,
                    width_px=100, height_px=100, words=[], seconds=0.0) for n, _ in enumerate(images, 1)]


def model_that(review=False):
    """No change, citing the refinancing rate (an abstention would always be routed to a person)."""
    def model(messages, schema):
        assert all(len(m["content"]) <= 4000 for m in messages)  # the adapter's limit per message
        bundle = json.loads("".join(m["content"].split(":\n", 1)[1] for m in messages
                                    if m["content"].startswith("Input part")))
        return Generation(text=json.dumps({
            "recommended_action": "no_change", "magnitude": "none", "reason_codes": [],
            "evidence_ids": [REFI.evidence_id], "mapping_id": bundle["policy_result"]["mapping_id"],
            "customer_impacts": [], "uncertainty": 0.5, "requires_human_review": review,
            "explanation": "The refinancing rate gives no reason to change your borrowing capacity."}), seconds=0.1)
    return model


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.fixtures, self.runs_dir = root / "fixtures", root / "runs"
        doc = self.fixtures / "docs-demo" / "DOC-1"
        doc.mkdir(parents=True)
        for name in ("page-1.png", "page-2.png", "document.pdf"):
            (doc / name).write_bytes(name.encode())
        (self.fixtures / "docs-demo" / "MANIFEST.json").write_text(json.dumps({"documents": [
            {"document_id": "DOC-1", "variant": "clean", "language": "en",
             "files": {"page-1.png": "x", "page-2.png": "x", "document.pdf": "x"}}]}))
        self.avm = root / "phase3.csv"
        with open(self.avm, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["property_id", "district_name", "ward_name", "total_area", "number_of_floors",
                        "last_modified_date", "asking_B", "P10_B", "P50_B", "P90_B", "over_P90_pct", "flag",
                        "time_adjusted"])
            w.writerow([PROPERTY, "Q. Đống Đa", "P. Quang Trung", "52.3", "5.0", "2026-07-06 04:53:18", "33.0",
                        "8.36", "18.41", "38.54", "-14.4", "fair", "True"])
        self.store = Store(corpus=root / "corpus", logs=root / "logs", now=lambda: NOW)
        self.patch = mock.patch.object(rn, "extract", lambda pages, **kw: Extraction(items=[REFI]))
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def run_one(self, **kw):
        args = dict(set_name="docs-demo", fixtures=self.fixtures, runs_dir=self.runs_dir, ocr_fn=fake_ocr,
                    model_fn=model_that(), keys=KEYS, catalogue=CAT, avm_path=self.avm, now=lambda: NOW,
                    mapping_fn=lambda config: (provisional_mapping_set(), "provisional", "test"))
        return rn.run_pipeline("DOC-1", "demo-minh", **{**args, **kw})

    def test_avm_items_are_context_only(self):
        items = avm_items(PROPERTY, CAT, NOW, self.avm)
        self.assertEqual([i.metric_id.split(".")[-1] for i in items], ["value_p10", "value_p50", "value_p90",
                                                                      "asking_price"])
        self.assertTrue(all(i.trust == "trusted" and i.geography == "hn-dong-da" for i in items))
        self.assertEqual(str(items[1].uncertainty.lower), "8.36")
        self.assertEqual(len(CAT.reason_codes()), 19)  # no reason codes for AVM metrics
        self.assertNotIn("avm.hn-dong-da.value_p50", CAT.instances)  # not listed with the signal metrics

    def test_a_run_records_every_stage(self):
        rec = self.run_one()
        self.assertEqual((rec["status"], rec["verdict"]), ("done", "PASS"))
        self.assertEqual([s["name"] for s in rec["stages"]], list(rn.STAGES))
        self.assertTrue(all(s["status"] == "ok" for s in rec["stages"]))
        for key in ("catalogue_hash", "policy_hash", "mapping_id", "bundle_hash", "code_commit", "prompt"):
            self.assertIn(key, rec["versions"])
        self.assertEqual(rec["versions"]["mapping_status"], "unvalidated")
        saved = json.loads((self.runs_dir / rec["run_id"] / "run.json").read_text())
        self.assertEqual(saved["verdict"], "PASS")
        bundle_ids = {i["evidence_id"] for i in rec["stages"][5]["bundle"]["items"]}
        self.assertIn("avm.hn-dong-da.value_p50.2026-07-06", bundle_ids)
        with self.assertRaises(FileExistsError):  # a run directory is never reused
            self.run_one()

    def test_prompt_withholds_values_it_may_not_cite(self):
        from ai.tests.fixtures import INVENTORY
        untrusted = INVENTORY.model_copy(update={"value": __import__("decimal").Decimal("477"), "trust": "untrusted",
                                                 "trust_reasons": ["CHANGE_PARSES:change_vs_baseline"]})
        seen = {}

        def model(messages, schema):
            seen["text"] = " ".join(m["content"] for m in messages)
            return model_that()(messages, schema)

        with mock.patch.object(rn, "extract", lambda pages, **kw: Extraction(items=[REFI, untrusted])):
            self.run_one(model_fn=model)
        self.assertIn("withheld (untrusted)", seen["text"])
        self.assertNotIn("477", seen["text"])
        self.assertIn('"role": "context: the customer\'s property"', seen["text"])
        self.assertIn("Refinancing rate", seen["text"])  # items carry their catalogue name

    def test_lists_are_capped_so_output_always_closes(self):
        from ai.schemas.recommendation import MAX_EVIDENCE_IDS, generation_schema
        from ai.tests.fixtures import BUNDLE_X
        branch = generation_schema(BUNDLE_X, CAT)["anyOf"][0]["properties"]
        self.assertEqual(branch["evidence_ids"]["maxItems"], 1)  # one citable item
        self.assertEqual(branch["customer_impacts"]["maxItems"], 1)  # one target
        self.assertEqual(MAX_EVIDENCE_IDS, 8)

    def test_a_failing_stage_stops_the_run(self):
        rec = self.run_one(model_fn=lambda m, s: Generation(text="", error="Qwen is not reachable"))
        self.assertEqual(rec["status"], "failed")
        self.assertEqual(rec["stages"][-1]["name"], "decision")
        self.assertIn("not reachable", rec["stages"][-1]["error"])
        self.assertNotIn("verdict", rec)

    def test_held_out_set_and_unknown_inputs_are_refused(self):
        with self.assertRaises(rn.RunError):
            rn.documents("docs-v0.1-holdout-s7", self.fixtures)
        with self.assertRaises(rn.RunError):
            self.run_one(set_name="docs-v0.1-holdout-s7")
        with self.assertRaises(rn.RunError):
            rn.run_pipeline("DOC-9", "demo-minh", set_name="docs-demo", fixtures=self.fixtures,
                            runs_dir=self.runs_dir, keys=KEYS)

    def test_review_and_release_to_the_customer(self):
        passed = self.run_one()
        self.assertEqual(runs.customer_latest(self.runs_dir, "demo-minh")["run_id"], passed["run_id"])
        with self.assertRaises(ApiError):  # a passed run is not reviewable
            runs.review_run(self.store, self.runs_dir, passed["run_id"], "jimmytrn154", "approve", None)
        later = lambda: NOW.replace(hour=10)
        routed = self.run_one(model_fn=model_that(review=True), now=later)
        self.assertEqual(routed["verdict"], "ROUTE_TO_HUMAN")
        self.assertEqual(runs.customer_latest(self.runs_dir, "demo-minh")["run_id"], passed["run_id"])  # not yet
        with self.assertRaises(ApiError):
            runs.review_run(self.store, self.runs_dir, routed["run_id"], "jimmytrn154", "reject", None)  # needs a note
        runs.review_run(self.store, self.runs_dir, routed["run_id"], "jimmytrn154", "approve", None)
        latest = runs.customer_latest(self.runs_dir, "demo-minh")
        self.assertEqual(latest["run_id"], routed["run_id"])
        self.assertEqual(latest["band"]["value_p50"]["value"], "18.41")
        with self.assertRaises(ApiError):  # reviewed once
            runs.review_run(self.store, self.runs_dir, routed["run_id"], "jimmytrn154", "reject", "changed my mind")
        self.assertEqual([r["review"]["action"] if r["review"] else None for r in runs.list_runs(self.runs_dir)],
                         ["approve", None])
        self.assertEqual(runs.page_image(self.runs_dir, routed["run_id"], 2, self.fixtures).name, "page-2.png")
        with self.assertRaises(ApiError):
            runs.page_image(self.runs_dir, routed["run_id"], 3, self.fixtures)
        actions = [a["action"] for a in runs.audit(self.store, self.runs_dir)["actions"]]
        self.assertEqual(actions, ["pipeline_run_reviewed"])

    def test_start_run_in_the_background_one_at_a_time(self):
        gate = mock.MagicMock()
        release = __import__("threading").Event()

        def slow(document_id, customer_id, runs_dir, on_update, **kw):
            on_update({"run_id": "R1"})
            release.wait(5)
            gate()

        out = runs.start_run(self.store, "jimmytrn154", "DOC-1", "demo-minh", self.runs_dir, run_fn=slow)
        self.assertEqual(out, {"run_id": "R1"})
        with self.assertRaises(ApiError) as cm:
            runs.start_run(self.store, "jimmytrn154", "DOC-1", "demo-minh", self.runs_dir, run_fn=slow)
        self.assertEqual(cm.exception.status, 409)
        release.set()
        for _ in range(50):
            if gate.called and not runs._current:
                break
            time.sleep(0.05)
        refused = lambda *a, **k: (_ for _ in ()).throw(rn.RunError("held out"))
        time.sleep(0.05)
        with self.assertRaises(ApiError) as cm:
            runs.start_run(self.store, "jimmytrn154", "DOC-1", "demo-minh", self.runs_dir, run_fn=refused)
        self.assertEqual(cm.exception.status, 400)


if __name__ == "__main__":
    unittest.main()
