"""Second source run (docs/SOURCE_RUN_2_PLAN.md): v0.2 keeps every v0.1 record unchanged, IDs never clash across runs,
documents already signed are skipped, and the shared numbering keeps run 1's IDs."""
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

from ai.schemas.source import Decision, Review, SourceList
from ai.sources import run as sr
from ai.sources.review import ReviewError, apply_review
from ai.tests.test_sources import AT, cand, record


def signed(records, version="source_list/v0.1", run_id="r1"):
    review = Review(decision="accept", reviewer="jimmytrn154", reviewed_at=AT)
    return SourceList(version=version, status="signed_off", run_id=run_id, created_at=AT, signed_off_by="jimmytrn154",
                      signed_off_at=AT, records=[r.model_copy(update={"review": review}) for r in records])


class SignOnTopTest(unittest.TestCase):
    def setUp(self):
        self.v01 = signed([record(sid="src-0001", title="Credit and housing in Viet Nam")])
        self.draft = SourceList(version="r2-draft", status="draft", run_id="r2", created_at=AT,
                                records=[record(sid="src-0318", title="Rate pass-through in Viet Nam")])
        self.rows = [{"source_id": "src-0318", "reviewer_decision": "accept", "reviewer_note": ""}]

    def test_base_records_are_kept_unchanged(self):
        v02 = apply_review(self.draft, self.rows, "jimmytrn154", "source_list/v0.2", AT, self.v01)
        self.assertEqual([r.source_id for r in v02.records], ["src-0001", "src-0318"])
        self.assertEqual(v02.records[0], self.v01.records[0])  # review and all, byte for byte
        self.assertEqual(len(v02.accepted()), 2)

    def test_clashes_and_same_version_are_refused(self):
        clash = self.draft.model_copy(update={"records": [record(sid="src-0001", title="Another document")]})
        rows = [{"source_id": "src-0001", "reviewer_decision": "accept", "reviewer_note": ""}]
        with self.assertRaises(ReviewError) as cm:
            apply_review(clash, rows, "jimmytrn154", "source_list/v0.1", AT, self.v01)
        self.assertIn("already in source_list/v0.1", str(cm.exception))
        self.assertIn("must differ from its base", str(cm.exception))


class KnownDocumentsTest(unittest.TestCase):
    def test_signed_documents_are_recognised_by_doi_or_title_and_year(self):
        r = record(title="Monetary policy and house prices in Viet Nam", doi="10.1/x", url="https://doi.org/10.1/x",
                   publication_date=date(2020, 1, 1))
        keys = sr.record_keys(r)
        self.assertTrue(sr.candidate_keys(cand()) & keys)  # same DOI
        same_title = cand(key="url:https://elsewhere/x", doi=None, landing_url="https://elsewhere/x")
        self.assertTrue(sr.candidate_keys(same_title) & keys)  # same title and year
        other = cand(key="doi:10.1/y", doi="10.1/y", title="Something else entirely")
        self.assertFalse(sr.candidate_keys(other) & keys)

    def test_shared_ids_keep_run_one_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "runs" / "r1").mkdir(parents=True)
            (root / "runs" / "r1" / "ids.json").write_text(json.dumps({"doi:10.1/x": "src-0001",
                                                                       "doi:10.1/y": "src-0317"}))
            with mock.patch.object(sr, "CORPUS", root), mock.patch.object(sr, "SHARED_IDS", root / "source_ids.json"):
                path = sr.shared_ids_path()
                recs = [record(Decision.REJECTED, failed_criterion="C1", checks=[]), record(sid="src-0000")]
                out = sr.number(recs, ["doi:10.1/x", "doi:10.1/new"], path)
            self.assertEqual(sorted(r.source_id for r in out), ["src-0001", "src-0318"])
