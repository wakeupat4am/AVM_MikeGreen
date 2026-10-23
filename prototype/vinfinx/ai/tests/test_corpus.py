"""Corpus and index (docs/CORPUS_INDEX_PLAN.md): chunks are exact page slices that never cross a page, frozen
versions need a signed list and are never overwritten, hybrid search fuses keyword and vector ranks, embeddings
are paid once, and the OpenAI client retries and never logs the key. No network: a fake embedder."""
import base64
import csv
import hashlib
import json
import re
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from unittest import mock

import numpy as np

from ai.corpus import chunk as chunker
from ai.corpus import index as idx
from ai.corpus.extract import clean, extract_pages
from ai.corpus.openai_client import Client, OpenAIError
from ai.schemas.source import Decision, Review, SourceList
from ai.tests.test_sources import AT, record

TEXT = ("Monetary policy in Viet Nam.\n\nA cut in the refinancing rate lowers lending rates. " * 30
        + "Credit growth raised house prices in Ha Noi. " * 40)


def pdf_bytes(pages: list[str]) -> bytes:
    """A minimal PDF with one line of Helvetica text per page."""
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [%s] /Count %d >>" % (b" ".join(b"%d 0 R" % (3 + 2 * i) for i in range(len(pages))),
                                                         len(pages))]
    for i, text in enumerate(pages):
        body = b"BT /F1 10 Tf 20 760 Td (" + text.encode("latin-1") + b") Tj ET"
        objs.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 2000 792] /Contents %d 0 R "
                    b"/Resources << /Font << /F1 %d 0 R >> >> >>" % (4 + 2 * i, 3 + 2 * len(pages)))
        objs.append(b"<< /Length %d >>\nstream\n" % len(body) + body + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out, offsets = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1) + b"".join(b"%010d 00000 n \n" % o for o in offsets)
    return out + b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)


class FakeEmbedder:
    """Bag-of-words vectors: deterministic, and similar texts get similar vectors."""

    def __init__(self):
        self.calls = 0

    def embed(self, texts, model="text-embedding-3-large"):
        self.calls += 1
        m = np.zeros((len(texts), 64), np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"\w+", t.lower()):
                m[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
        m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
        return m, model, sum(len(t.split()) for t in texts)


class ChunkTest(unittest.TestCase):
    def test_chunks_are_exact_slices_with_overlap(self):
        s = chunker.spans(TEXT)
        self.assertGreater(len(s), 2)
        for (a, b), (c, _) in zip(s, s[1:]):
            self.assertLess(c, b)  # overlap
            self.assertLessEqual(b - a, chunker.SIZE)
            self.assertFalse(TEXT[a].isspace() or TEXT[b - 1].isspace())
        covered = set()
        for a, b in s:
            covered.update(range(a, b))
        self.assertTrue(all(i in covered for i, ch in enumerate(TEXT) if not ch.isspace()))

    def test_cuts_prefer_paragraph_then_sentence(self):
        a, b = chunker.spans(TEXT)[0]
        self.assertTrue(TEXT[b:b + 2] == "\n\n" or TEXT[b - 1] in ".!?;:")

    def test_text_without_breaks_still_progresses(self):
        s = chunker.spans("x" * 5000)
        self.assertEqual((s[0], s[-1][1]), ((0, chunker.SIZE), 5000))

    def test_garbled_text_layer_is_detected(self):
        from ai.corpus.extract import garbled

        broken = 'MC LC PHN I TÌNH HÌNH PHÁT TRIN KINH T GN ÂY Kinh t tip t!c t"ng tr%&ng nhanh. ' * 5
        self.assertTrue(garbled(broken))
        self.assertFalse(garbled("Inflation rose 4.0% y/y in Q2/2025 (see Figure 3). " * 5))
        self.assertFalse(garbled("Kinh tế tiếp tục tăng trưởng nhanh trong quý II/2025. " * 5))

    def test_clean(self):
        self.assertEqual(clean("Việt\r\nNam\x00"), "Việt\nNam")  # decomposed letter composed (NFC)


class CorpusBuildTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.patches = [mock.patch.object(idx, "CORPUS", root), mock.patch.object(idx, "VERSIONS", root / "versions"),
                        mock.patch.object(idx, "_commit", lambda: None)]
        for p in self.patches:
            p.start()
        data = pdf_bytes([TEXT[:900], "short", "Credit growth raised house prices in Ha Noi and Ho Chi Minh City. " * 4])
        sha = "sha256:" + hashlib.sha256(data).hexdigest()
        (root / "originals").mkdir()
        (root / "originals" / f"{sha.split(':')[1]}.pdf").write_bytes(data)
        rec = record(sha256=sha, title="Credit and housing in Viet Nam", publication_date=date(2025, 2, 27))
        unclear = record(Decision.UNCLEAR, "src-0002", sha256=sha, title="Other", checks=[])
        self.draft = SourceList(version="r-draft", status="draft", run_id="r", created_at=AT, records=[rec, unclear])
        self.list_path = root / "draft.json"
        self.list_path.write_text(self.draft.model_dump_json())

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_dev_corpus_chunks_are_page_slices(self):
        out = idx.build_corpus(self.list_path, "corpus-dev", dev=True, ocr=False, log=lambda *_: None)
        rows = [json.loads(l) for l in (out / "pages.jsonl").read_text().splitlines()]
        pages = {(r["source_id"], r["page"]): r["text"] for r in rows}
        chunks = idx.load_chunks(out)
        self.assertEqual({c.source_id for c in chunks}, {"src-0001", "src-0002"})
        self.assertNotIn(("src-0001", 2), pages)  # the "short" page is skipped
        for c in chunks:
            self.assertEqual(pages[(c.source_id, c.page)][c.page_char_start:c.page_char_end], c.text)
            self.assertEqual((c.geography, c.corpus_version), ("vn", "corpus-dev"))

    def test_frozen_needs_a_signed_list_and_is_never_overwritten(self):
        with self.assertRaises(SystemExit):
            idx.build_corpus(self.list_path, "corpus-v0.1", dev=False, ocr=False, log=lambda *_: None)
        review = Review(decision="accept", reviewer="jimmytrn154", reviewed_at=AT)
        signed = self.draft.model_copy(update={
            "status": "signed_off", "version": "source_list/v0.1", "signed_off_by": "jimmytrn154",
            "signed_off_at": AT, "records": [r.model_copy(update={"review": review}) for r in self.draft.records]})
        path = Path(self.tmp.name) / "signed.json"
        path.write_text(signed.model_dump_json())
        idx.build_corpus(path, "corpus-v0.1", dev=False, ocr=False, log=lambda *_: None)
        with self.assertRaises(SystemExit):
            idx.build_corpus(path, "corpus-v0.1", dev=False, ocr=False, log=lambda *_: None)
        with self.assertRaises(SystemExit):  # names keep dev and frozen apart
            idx.build_corpus(self.list_path, "corpus-v0.2", dev=True, ocr=False, log=lambda *_: None)

    def test_index_search_and_cache(self):
        idx.build_corpus(self.list_path, "corpus-dev", dev=True, ocr=False, log=lambda *_: None)
        fake = FakeEmbedder()
        idx.build_index("corpus-dev", client=fake, log=lambda *_: None)
        manifest = json.loads((Path(self.tmp.name) / "versions" / "corpus-dev" / "index" / "manifest.json").read_text())
        self.assertEqual((manifest["embedding_dimensions"], manifest["frozen"]), (64, False))
        idx.build_index("corpus-dev", client=fake, log=lambda *_: None)
        self.assertEqual(fake.calls, 1)  # the rebuild came from the cache
        index = idx.Index("corpus-dev", client=fake, base=Path(self.tmp.name) / "versions")
        hits = index.search("house prices credit growth", k=3)
        self.assertIn("house prices", hits[0]["text"].lower())
        self.assertIsNotNone(hits[0]["keyword_rank"])
        self.assertIsNotNone(hits[0]["vector_rank"])
        self.assertEqual({h["source_id"] for h in index.search("credit", k=20, source_id="src-0002")}, {"src-0002"})
        self.assertEqual(index.keyword("!!", {}), [])


class FakeLabeller:
    """Marks a passage 'supports' when it mentions house prices, quoting its first sentence; one quote is made up."""

    def __init__(self):
        self.calls = 0

    def chat_json(self, messages, schema, model=None):
        self.calls += 1
        ids = schema["properties"]["labels"]["required"]
        assert ids == list(schema["properties"]["labels"]["properties"])  # every passage must be labelled
        text = messages[1]["content"]
        labels = {}
        for cid in ids:
            passage = text.split(f"[{cid}]\n", 1)[1].split("\n\n[", 1)[0]
            if "house prices" in passage and self.calls == 1:
                labels[cid] = {"label": "supports", "quote": passage.split(". ")[0] + "."}
            elif "house prices" in passage:
                labels[cid] = {"label": "supports", "quote": "House prices always fall."}
            else:
                labels[cid] = {"label": "irrelevant", "quote": ""}
        return {"labels": labels}, {"prompt_tokens": 100, "completion_tokens": 20}


class GoldEvaluationTest(CorpusBuildTest):
    def test_draft_metrics_and_review(self):
        from ai.corpus import evaluate as ev

        idx.build_corpus(self.list_path, "corpus-dev", dev=True, ocr=False, log=lambda *_: None)
        fake = FakeEmbedder()
        idx.build_index("corpus-dev", client=fake, log=lambda *_: None)
        index = idx.Index("corpus-dev", client=fake, base=Path(self.tmp.name) / "versions")
        self.assertEqual(len(ev.questions()), 24)
        with mock.patch.object(ev, "VERSIONS", Path(self.tmp.name) / "versions"):
            out = ev.draft("corpus-dev", client=FakeLabeller(), index=index, log=lambda *_: None)
        data = json.loads((out / "draft.json").read_text())
        made_up = [r for r in data["labels"] if r["label"] == "supports" and not r["quote_verified"]]
        self.assertTrue(made_up and not any(r["gold"] for r in made_up))  # an unverified quote is never gold
        self.assertTrue(any(r["gold"] for r in data["labels"]))
        m = ev.retrieval_metrics(index, data["labels"])
        self.assertEqual(m["questions_with_gold"], 1)
        self.assertEqual(m["hybrid"]["recall@8"], 1.0)
        with open(out / "review.csv", newline="") as f:
            sample = list(csv.DictReader(f))
        self.assertTrue(any(r["label"] == "supports" for r in sample))  # stratified: rare labels sampled
        for r in sample:
            r["reviewer_label"] = r["label"] if r["label"] != "unlabelled" else "irrelevant"
        gold, summary = ev.apply_review(data, sample, "jimmytrn154")
        self.assertEqual((summary["agreement"], gold["meta"]["status"]), (1.0, "reviewed"))
        with self.assertRaises(SystemExit):
            ev.apply_review(data, [{**sample[0], "reviewer_label": ""}], "jimmytrn154")


class PdfTextTest(unittest.TestCase):
    def test_pages_and_skipped_blank_page(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.pdf"
            p.write_bytes(pdf_bytes([TEXT[:300], "tiny"]))
            pages = extract_pages(p, ocr=False)
        self.assertEqual([pg.page for pg in pages], [1])
        self.assertTrue(pages[0].extractor.startswith("pdfium:"))


class FakeHttp:
    def __init__(self, responses):
        self.responses, self.calls = responses, []

    def post(self, url, json=None, timeout=None, headers=None):
        self.calls.append((url, json, headers))
        status, body = self.responses.pop(0)
        r = mock.Mock(status_code=status, headers={"content-type": "application/json"})
        r.json.return_value = body
        return r


def emb_reply(vectors):
    return {"model": "text-embedding-3-large", "usage": {"prompt_tokens": 7},
            "data": [{"index": i, "embedding": base64.b64encode(np.asarray(v, "<f4").tobytes()).decode()}
                     for i, v in enumerate(vectors)]}


class OpenAIClientTest(unittest.TestCase):
    def test_retries_then_returns_unit_vectors_and_never_logs_the_key(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "calls.jsonl"
            http = FakeHttp([(429, {"error": {"message": "slow down"}}), (200, emb_reply([[3, 4], [0, 2]]))])
            c = Client("test", key="sk-SECRET", log_path=log, session=http, sleep=lambda s: None)
            m, model, tokens = c.embed(["a", "b"])
            self.assertTrue(np.allclose(m, [[0.6, 0.8], [0, 1]]))
            self.assertEqual((model, tokens, len(http.calls)), ("text-embedding-3-large", 7, 2))
            self.assertEqual(http.calls[0][1]["encoding_format"], "base64")
            text = log.read_text()
            self.assertNotIn("SECRET", text)
            self.assertEqual(len(text.splitlines()), 2)

    def test_client_errors_are_not_retried(self):
        http = FakeHttp([(400, {"error": {"message": "bad input"}})])
        c = Client("test", key="k", log_path=Path(tempfile.mkdtemp()) / "l.jsonl", session=http, sleep=lambda s: None)
        with self.assertRaises(OpenAIError):
            c.embed(["a"])
        self.assertEqual(len(http.calls), 1)


if __name__ == "__main__":
    unittest.main()
