"""Pipeline API service (docs/PIPELINE_API_PLAN.md): the site can sign and review only what the CLI would accept,
never overwrites a signed list or a reviewed gold set, serves only known originals, and refuses personal data in
search queries before anything leaves the server. No network: embeddings and labels come from fakes."""
import csv
import io
import json
import unittest
from pathlib import Path
from unittest import mock

from ai.api import service as sv
from ai.api.service import ApiError, Store
from ai.corpus import index as idx
from ai.schemas.source import SourceList
from ai.sources.review import apply_review
from ai.tests.test_corpus import CorpusBuildTest, FakeEmbedder, FakeLabeller
from ai.tests.test_sources import AT


class ApiTest(CorpusBuildTest):
    def setUp(self):
        super().setUp()
        self.root = Path(self.tmp.name)
        (self.root / "source_lists").mkdir()
        (self.root / "source_lists" / "r-draft.json").write_text(self.draft.model_dump_json())
        run = self.root / "runs" / "r"
        run.mkdir(parents=True)
        with open(run / "review.csv", "w", newline="") as f:
            w = csv.DictWriter(f, ["source_id", "agent_decision", "reviewer_decision", "reviewer_note", "title"])
            w.writeheader()
            w.writerow({"source_id": "src-0001", "agent_decision": "accepted", "title": "Credit"})
            w.writerow({"source_id": "src-0002", "agent_decision": "unclear", "title": "Other"})
        self.store = Store(corpus=self.root, logs=self.root / "logs", now=lambda: AT)

    def actions(self):
        return [json.loads(l) for l in (self.root / "logs" / "review_actions.jsonl").read_text().splitlines()]

    def error(self, fn, *args) -> ApiError:
        with self.assertRaises(ApiError) as cm:
            fn(*args)
        return cm.exception

    # ------------------------------------------------------------ source list

    def test_sign_runs_the_cli_rules_and_never_overwrites(self):
        save = lambda rows: sv.save_source_progress(self.store, "r", "jimmytrn154", rows)
        save([{"source_id": "src-0001", "decision": "reject"}])
        e = self.error(sv.sign_source_list, self.store, "r", "jimmytrn154", "source_list/v0.1")
        self.assertEqual(e.status, 422)
        self.assertTrue(any("needs a reviewer_note" in p for p in e.problems))  # overruling the agent
        self.assertTrue(any("src-0002" in p for p in e.problems))  # undecided
        self.assertFalse((self.root / "source_lists" / "source_list-v0.1.json").exists())
        save([{"source_id": "src-0001", "decision": "accept"}, {"source_id": "src-0002", "decision": "accept",
                                                                 "note": "Viet Nam housing data"}])
        out = sv.sign_source_list(self.store, "r", "jimmytrn154", "source_list/v0.1")
        self.assertEqual((out["accepted"], out["signed_off_by"]), (2, "jimmytrn154"))
        signed = SourceList.model_validate_json((self.root / "source_lists" / "source_list-v0.1.json").read_text())
        self.assertEqual(signed.list_hash, out["list_hash"])
        self.assertEqual(self.error(sv.sign_source_list, self.store, "r", "jimmytrn154", "source_list/v0.1").status,
                         409)
        self.assertEqual(self.error(save, [{"source_id": "src-0001", "decision": "reject"}]).status, 409)
        self.assertEqual([a["action"] for a in self.actions()][-1], "source_list_signed")
        self.assertEqual(sv.source_run(self.store, "r")["signed"][0]["version"], "source_list/v0.1")

    def test_writes_check_reviewer_rows_and_version(self):
        rows = [{"source_id": "src-0001", "decision": "accept"}]
        self.assertEqual(self.error(sv.save_source_progress, self.store, "r", "someone", rows).status, 403)
        bad = [{"source_id": "src-9999", "decision": "accept"}]
        self.assertEqual(self.error(sv.save_source_progress, self.store, "r", "jimmytrn154", bad).status, 400)
        maybe = [{"source_id": "src-0001", "decision": "maybe"}]
        self.assertEqual(self.error(sv.save_source_progress, self.store, "r", "jimmytrn154", maybe).status, 400)
        self.assertEqual(self.error(sv.sign_source_list, self.store, "r", "jimmytrn154", "v1").status, 400)
        self.assertEqual(self.error(sv.source_run, self.store, "../r").status, 400)

    def test_progress_round_trips_with_the_cli_csv(self):
        csv_path = self.root / "runs" / "r" / "review.csv"
        text = csv_path.read_text().replace("src-0002,unclear,,", "src-0002,unclear,reject,not about housing")
        text += "src-0003,rejected,reject,,Pre-filled by the source agent\r\n"
        csv_path.write_text(text)
        self.assertEqual(sv.source_run(self.store, "r")["records"][1]["progress"]["from"], "review.csv")
        self.assertNotIn("src-0003", sv._source_progress(self.store, "r"))  # the agent's default is not a decision
        sv.save_source_progress(self.store, "r", "jimmytrn154", [{"source_id": "src-0001", "decision": "accept"}])
        exported = [r for r in csv.DictReader(io.StringIO(sv.source_review_csv(self.store, "r")))
                    if r["source_id"] != "src-0003"]
        self.assertEqual([r["reviewer_decision"] for r in exported], ["accept", "reject"])
        signed = apply_review(self.draft, exported, "jimmytrn154", "source_list/v0.1", AT)  # what the CLI does
        self.assertEqual(len(signed.accepted()), 1)

    def test_parallel_saves_keep_every_decision(self):
        import threading

        errors = []

        def save(row):
            try:
                sv.save_source_progress(self.store, "r", "jimmytrn154", [row])
            except Exception as exc:  # a thread's exception would otherwise not reach the test
                errors.append(exc)

        rows = [{"source_id": sid, "decision": "accept"} for sid in ("src-0001", "src-0002")] * 10
        threads = [threading.Thread(target=save, args=(row,)) for row in rows]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(set(sv._source_progress(self.store, "r")), {"src-0001", "src-0002"})

    def test_only_known_originals_are_served(self):
        sha = self.draft.records[0].sha256.split(":")[1]
        self.assertTrue(sv.original_path(self.store, sha).exists())
        self.assertEqual(self.error(sv.original_path, self.store, "f" * 64).status, 404)
        self.assertEqual(self.error(sv.original_path, self.store, "../x").status, 400)

    # ------------------------------------------------------------ gold labels and search

    def build(self):
        from ai.corpus import evaluate as ev

        idx.build_corpus(self.list_path, "corpus-dev", dev=True, ocr=False, log=lambda *_: None)
        self.fake = FakeEmbedder()
        idx.build_index("corpus-dev", client=self.fake, log=lambda *_: None)
        self.index = idx.Index("corpus-dev", client=self.fake, base=self.root / "versions")
        with mock.patch.object(ev, "VERSIONS", self.root / "versions"):
            ev.draft("corpus-dev", client=FakeLabeller(), index=self.index, log=lambda *_: None)

    def test_gold_review_submit_freezes_and_remeasures(self):
        self.build()
        view = sv.gold_review(self.store, "corpus-dev")
        self.assertEqual((view["status"], view["decided"]), ("draft", 0))
        self.assertTrue(all(r["passage"] and r["page"] for r in view["rows"]))
        first = view["rows"][0]
        sv.save_gold_progress(self.store, "corpus-dev", "jimmytrn154",
                              [{"question_id": first["question_id"], "chunk_id": first["chunk_id"],
                                "label": "related"}])
        e = self.error(sv.submit_gold_review, self.store, "corpus-dev", "jimmytrn154", lambda v: self.index)
        self.assertEqual((e.status, len(e.problems)), (422, len(view["rows"]) - 1))
        rows = [{"question_id": r["question_id"], "chunk_id": r["chunk_id"],
                 "label": r["label"] if r["label"] != "unlabelled" else "irrelevant"} for r in view["rows"]]
        sv.save_gold_progress(self.store, "corpus-dev", "jimmytrn154", rows)
        out = sv.submit_gold_review(self.store, "corpus-dev", "jimmytrn154", lambda v: self.index)
        self.assertEqual((out["review"]["agreement"], out["metrics"]["labels_status"]), (1.0, "reviewed"))
        self.assertEqual(sv.gold_review(self.store, "corpus-dev")["status"], "reviewed")
        self.assertEqual(self.error(sv.submit_gold_review, self.store, "corpus-dev", "jimmytrn154",
                                    lambda v: self.index).status, 409)
        exported = list(csv.DictReader(io.StringIO(sv.gold_review_csv(self.store, "corpus-dev"))))
        self.assertTrue(all(r["reviewer_label"] for r in exported))

    def test_personal_data_is_refused_before_search(self):
        for q in ("khách hàng CCCD 001203004567", "call 0912 345 678", "send to minh@example.com",
                  "vay 3,2 tỷ được không", "income ₫18.6M", "lương 25 triệu"):
            self.assertEqual(self.error(sv.check_query, q).status, 400, q)
        for q in ("How does the refinancing rate affect house prices in Viet Nam?",
                  "tỷ lệ thất nghiệp và khả năng vay của hộ gia đình", "credit growth 2010-2020"):
            self.assertEqual(sv.check_query(q), q)
        self.assertEqual(self.error(sv.check_query, "x" * 201).status, 400)

    def test_search_logs_only_queries_sent_to_openai(self):
        self.build()
        log = self.root / "logs" / "search_queries.jsonl"
        out = sv.search(self.store, self.index, "house prices", mode="keyword")
        self.assertTrue(out["hits"] and out["hits"][0]["original_sha256"])
        self.assertFalse(log.exists())
        calls = self.fake.calls
        sv.search(self.store, self.index, "house prices", mode="hybrid")
        self.assertEqual(self.fake.calls, calls + 1)
        self.assertEqual(json.loads(log.read_text())["query"], "house prices")
        with self.assertRaises(ApiError):
            sv.search(self.store, self.index, "0912345678", mode="hybrid")
        self.assertEqual(self.fake.calls, calls + 1)  # a refused query is never embedded


if __name__ == "__main__":
    unittest.main()
