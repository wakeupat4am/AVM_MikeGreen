"""Mapping agent (docs/MAPPING_AGENT_PLAN.md): a supported proposal enters a content-addressed set, a contradicted or
fabricated one does not, "no mapping" is an answer, only frozen corpora are used, a set is reviewed once, and the
runner keeps the provisional mapping until a set covers every weighted pair. No GPU, no network: Qwen, the support
model and the embeddings are fakes."""
import json
from pathlib import Path
from unittest import mock

from ai import mapping as mp
from ai.corpus import index as idx
from ai.evaluation.runner import Generation
from ai.policy import load_policy_config
from ai.schemas.common import Target
from ai.schemas.source import Review
from ai.tests.fixtures import CAT, NOW
from ai.tests.test_corpus import AT, CorpusBuildTest, FakeEmbedder

QUOTE = "Credit growth raised house prices in Ha Noi and Ho Chi Minh City."


def agent(chunks, quote=QUOTE, direction="positive"):
    """Proposes credit_growth -> valuation_range from the retrieved chunk holding QUOTE; no mapping for other pairs."""
    def model(messages, schema):
        text = " ".join(m["content"] for m in messages)
        props = schema["anyOf"][0]["properties"]
        if props["metric_key"]["enum"] == ["credit_growth"] and "Target: valuation_range" in text:
            chunk = next(c for c in props["evidence"]["items"]["properties"]["chunk_id"]["enum"]
                         if QUOTE in chunks[c].text)
            return Generation(text=json.dumps({"metric_key": "credit_growth", "target": "valuation_range",
                                               "direction": direction, "lag_months": [1, 6], "size_range": "low",
                                               "geography": "vn", "evidence": [{"chunk_id": chunk, "quote": quote}]}))
        return Generation(text=json.dumps({"no_mapping": True, "reason": "the passages do not state this effect"}))
    return model


class FakeSupport:
    def __init__(self, verdict="supports"):
        self.verdict, self.calls = verdict, 0

    def chat_json(self, messages, schema, model=None, max_completion_tokens=None):
        self.calls += 1
        assert model == mp.SUPPORT_MODEL and "Quote:" in messages[1]["content"]
        return {"verdict": self.verdict, "rationale": "fake"}, {"prompt_tokens": 100, "completion_tokens": 10}


class MappingAgentTest(CorpusBuildTest):
    # this class builds corpus-v0.1 in setUp; the inherited corpus tests build it themselves
    test_dev_corpus_chunks_are_page_slices = test_frozen_needs_a_signed_list_and_is_never_overwritten = None
    test_index_search_and_cache = None

    def setUp(self):
        super().setUp()
        self.root = Path(self.tmp.name)
        review = Review(decision="accept", reviewer="jimmytrn154", reviewed_at=AT)
        signed = self.draft.model_copy(update={
            "status": "signed_off", "version": "source_list/v0.1", "signed_off_by": "jimmytrn154",
            "signed_off_at": AT, "records": [r.model_copy(update={"review": review}) for r in self.draft.records]})
        (self.root / "signed.json").write_text(signed.model_dump_json())
        idx.build_corpus(self.root / "signed.json", "corpus-v0.1", dev=False, ocr=False, log=lambda *_: None)
        self.fake = FakeEmbedder()
        idx.build_index("corpus-v0.1", client=self.fake, log=lambda *_: None)
        self.index = idx.Index("corpus-v0.1", client=self.fake, base=self.root / "versions")
        self.registry = self.root / "mappings"
        self.chunks = {c.chunk_id: c for c in idx.load_chunks(self.root / "versions" / "corpus-v0.1")}

    def run_agent(self, model=None, support=None):
        return mp.run("corpus-v0.1", model_fn=model or agent(self.chunks), support_client=support or FakeSupport(),
                      index=self.index, catalogue=CAT, registry=self.registry, base=self.root / "versions",
                      now=lambda: NOW, log=lambda *_: None)

    def test_supported_proposal_enters_a_set(self):
        rec = self.run_agent()
        self.assertEqual(rec["counts"], {"accepted": 1, "no_mapping": 11, "rejected_by_rules": 0,
                                         "rejected_by_support": 0, "error": 0})
        mset = mp.load_set(1, self.registry)
        self.assertEqual(mset.status, "unvalidated")
        entry = mset.entries[0]
        self.assertEqual((entry.proposal.metric_key, entry.proposal.target.value), ("credit_growth", "valuation_range"))
        self.assertEqual(entry.verification.support_check.quotes[0].verdict, "supports")
        pair = next(p for p in rec["pairs"] if p["outcome"] == "accepted")
        self.assertEqual(pair["retrieval"]["filters"], {"geography": "vn"})
        self.assertTrue((self.registry / "runs" / f"{rec['run_id']}.json").exists())
        self.assertGreater(rec["support_cost_usd"], 0)

    def test_both_prompts_name_the_same_proxies(self):
        agent_text = mp.agent_messages(CAT, "credit_growth", Target.VALUATION_RANGE, [])[0]["content"]
        support_text = mp.support_messages(CAT, "credit_growth", Target.VALUATION_RANGE, "positive", "q", "p")[1]["content"]
        for text in (agent_text, support_text):
            self.assertIn(mp.METRIC_PROXIES["credit_growth"], text)
            self.assertIn(mp.TARGET_PROXIES["valuation_range"], text)
        self.assertEqual(set(mp.METRIC_PROXIES), {m.key for m in CAT.metrics})  # every signal metric has proxies

    def test_contradicted_or_fabricated_proposals_are_rejected(self):
        support = FakeSupport("contradicts")
        rec = self.run_agent(support=support)
        self.assertEqual(rec["counts"]["rejected_by_support"], 1)
        self.assertNotIn("mapping_set", rec)
        self.assertFalse(list(self.registry.glob("mapping-v*.json")))
        support = FakeSupport()
        rec = mp.run("corpus-v0.1", model_fn=agent(self.chunks, quote="Credit growth always doubles house prices everywhere."),
                     support_client=support, index=self.index, catalogue=CAT, registry=self.root / "m2",
                     base=self.root / "versions", now=lambda: NOW.replace(hour=11), log=lambda *_: None)
        self.assertEqual(rec["counts"]["rejected_by_rules"], 1)
        self.assertEqual(support.calls, 0)  # a quote that is not in the text is never sent for a support check

    def test_only_frozen_corpora(self):
        idx.build_corpus(self.list_path, "corpus-dev", dev=True, ocr=False, log=lambda *_: None)
        idx.build_index("corpus-dev", client=self.fake, log=lambda *_: None)
        dev = idx.Index("corpus-dev", client=self.fake, base=self.root / "versions")
        with self.assertRaises(SystemExit):
            mp.run("corpus-dev", model_fn=agent(self.chunks), support_client=FakeSupport(), index=dev, catalogue=CAT,
                   registry=self.registry, base=self.root / "versions", now=lambda: NOW, log=lambda *_: None)

    def test_review_once_and_the_runner_choice(self):
        self.run_agent()
        config = load_policy_config(catalogue=CAT)
        mset, source, why = mp.select_mapping(config, self.registry)
        self.assertEqual(source, "provisional")  # the set does not cover refi_rate, mortgage_rate_avg, inventory
        self.assertIn("refi_rate->borrowing_capacity", why)
        covered = config.model_copy(update={"weights": {Target.VALUATION_RANGE: {"credit_growth": 1}}})
        self.assertEqual(mp.select_mapping(covered, self.registry)[1], "registry")
        with self.assertRaises(ValueError):
            mp.review_set(1, "jimmytrn154", "rejected", None, NOW, self.registry)  # rejecting needs a note
        mp.review_set(1, "jimmytrn154", "validated", None, NOW, self.registry)
        self.assertEqual(mp.load_set(1, self.registry).status, "validated")
        self.assertEqual(mp.load_set(1, self.registry).mapping_id, mp.registry_sets(self.registry)[0].mapping_id)
        with self.assertRaises(ValueError):
            mp.review_set(1, "jimmytrn154", "rejected", "changed my mind", NOW, self.registry)
