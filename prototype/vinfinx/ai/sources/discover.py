"""Discovery adapters (SOURCE_AGENT_PLAN.md, S2). Each turns a real index's response into Candidates. No model is
involved, so every candidate is a document some index actually lists."""
from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from datetime import date, datetime
from urllib.parse import urljoin

from . import Candidate, PublisherFile, QueryGroup
from .http import Client, Response, openalex_params

OPENALEX = "https://api.openalex.org/works"
OPENALEX_TYPES = "article|report|preprint|book|book-chapter"
OPENALEX_SELECT = ("id,doi,title,publication_date,language,type,open_access,best_oa_location,primary_location,"
                   "authorships,abstract_inverted_index")
WORLDBANK = "https://search.worldbank.org/api/v3/wds"
WORLDBANK_MAJOR_TYPES = "Publications & Research^Economic & Sector Work"  # not project paperwork
BIS_FEEDS = {"wp": "https://www.bis.org/doclist/wppubls.rss", "bispapers": "https://www.bis.org/doclist/bispapers.rss"}
# The feeds list only the latest papers on any topic; keep those whose title or summary names a housing-finance
# topic (topic search for BIS goes through OpenAlex, by author affiliation).
BIS_TOPIC = re.compile(r"hous|mortgage|real estate|property|credit|lending|monetary|interest rate|inflation", re.I)
SBV_PAGES = {"annual_reports": "https://sbv.gov.vn/vi/bao-cao-thuong-nien"}
NSO_PAGES = {"publications": "https://www.nso.gov.vn/en/publication/"}  # /publication/ redirects here
PER_PAGE = 25


def _doi(value: str | None) -> str | None:
    if not value:
        return None
    return value.lower().removeprefix("https://doi.org/").removeprefix("http://doi.org/")


def _date(value: str | None) -> date | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(value[:10], fmt).date()
        except ValueError:
            pass
    return None


def abstract_from_index(index: dict[str, list[int]] | None) -> str | None:
    """OpenAlex stores abstracts as word -> positions."""
    if not index:
        return None
    words = sorted((p, w) for w, ps in index.items() for p in ps)
    return " ".join(w for _, w in words)


# ---------------------------------------------------------------- OpenAlex


def openalex_candidate(w: dict, query_id: str, group: str, raw: str) -> Candidate | None:
    title = (w.get("title") or "").strip()
    if not title:
        return None
    loc = w.get("best_oa_location") or w.get("primary_location") or {}
    prim = w.get("primary_location") or {}
    src = prim.get("source") or loc.get("source") or {}
    doi = _doi(w.get("doi"))
    insts = [i for a in w.get("authorships") or [] for i in a.get("institutions") or []]
    return Candidate(
        key=f"doi:{doi}" if doi else f"url:{w['id']}", title=title,
        abstract=abstract_from_index(w.get("abstract_inverted_index")), adapter="openalex",
        query_ids=[query_id], groups=[group], raw_sha256=raw, doi=doi,
        landing_url=(f"https://doi.org/{doi}" if doi else loc.get("landing_page_url") or w["id"]),
        pdf_url=loc.get("pdf_url"), language=w.get("language"), publication_date=_date(w.get("publication_date")),
        work_type=w.get("type"), source_name=src.get("display_name"), source_type=src.get("type"),
        source_in_doaj=src.get("is_in_doaj"), source_is_core=src.get("is_core"),
        publisher_ids=[p.split("/")[-1] for p in src.get("host_organization_lineage") or []],
        institution_ids=sorted({i["id"].split("/")[-1] for i in insts if i.get("id")}),
        institution_types=sorted({i.get("type") for i in insts if i.get("type")}),
        licence=loc.get("license"), oa_status=(w.get("open_access") or {}).get("oa_status"))


class OpenAlexBudget:
    """OpenAlex charges each request against a daily budget shared by everyone on this IP without a key."""

    def __init__(self, limit_usd: float) -> None:
        self.limit, self.spent, self.requests = limit_usd, 0.0, 0

    def allows(self) -> bool:
        return self.spent < self.limit


def openalex_search(client: Client, budget: OpenAlexBudget, group: QueryGroup, mode: str, query: str,
                    extra_filter: str = "") -> tuple[list[Candidate], Response | None]:
    if not budget.allows():
        return [], None
    flt = f"title_and_abstract.search:{query.replace(',', ' ')},is_oa:true,language:en|vi,type:{OPENALEX_TYPES}"
    if extra_filter:
        flt += "," + extra_filter
    params = {"filter": flt, "per-page": PER_PAGE, "select": OPENALEX_SELECT, **openalex_params()}
    if not extra_filter:
        params["sort"] = "relevance_score:desc"
    resp = client.get(OPENALEX, params)
    data = resp.json()
    budget.spent += float(data.get("meta", {}).get("cost_usd") or 0.001)
    budget.requests += 1
    qid = f"{group.id}/openalex/{mode}"
    out = [c for w in data.get("results", []) if (c := openalex_candidate(w, qid, group.id, resp.sha256))]
    return out, resp


def openalex_group(client: Client, budget: OpenAlexBudget, group: QueryGroup,
                   publishers: PublisherFile) -> list[Candidate]:
    """Per group: each English and Vietnamese query on all open-access works, then the first English query
    restricted to the allowlisted institutions' authors, to IMF-published works and to ADB DOIs."""
    out: list[Candidate] = []
    for i, q in enumerate(group.en):
        out += openalex_search(client, budget, group, f"en{i}", q)[0]
    for i, q in enumerate(group.vi):
        out += openalex_search(client, budget, group, f"vi{i}", q)[0]
    q = group.en[0]
    inst = "|".join(publishers.institution_ids())
    out += openalex_search(client, budget, group, "institutions", q, f"authorships.institutions.lineage:{inst}")[0]
    imf = next(p for p in publishers.publishers if p.id == "imf")
    out += openalex_search(client, budget, group, "imf", q,
                           f"primary_location.source.host_organization_lineage:{imf.openalex_publishers[0]}")[0]
    adb = next(p for p in publishers.publishers if p.id == "adb")
    out += openalex_search(client, budget, group, "adb", q, f"doi_starts_with:{adb.doi_prefixes[0]}")[0]
    return out


# ---------------------------------------------------------------- World Bank


WB_LANG = {"English": "en", "Vietnamese": "vi"}


def worldbank_candidates(data: dict, query_id: str, group: str, raw: str) -> list[Candidate]:
    out = []
    for k, v in (data.get("documents") or {}).items():
        if k == "facets" or not isinstance(v, dict) or not v.get("pdfurl"):
            continue
        abstract = v.get("abstracts")
        if isinstance(abstract, dict):
            abstract = abstract.get("cdata!")
        out.append(Candidate(
            key=f"url:{v.get('url') or v['pdfurl']}", title=" ".join((v.get("display_title") or "").split()),
            abstract=abstract, adapter="worldbank", query_ids=[query_id], groups=[group], raw_sha256=raw,
            landing_url=v.get("url") or v["pdfurl"], pdf_url=v["pdfurl"].replace("http://", "https://", 1),
            language=WB_LANG.get((v.get("lang") or "").strip(), (v.get("lang") or "").strip() or None),
            publication_date=_date(v.get("docdt")), work_type=v.get("docty"), source_name=v.get("majdocty"),
            site_publisher="world_bank"))
    return out


def worldbank_group(client: Client, group: QueryGroup) -> list[Candidate]:
    out: list[Candidate] = []
    for lang, queries in (("en", group.en[:1]), ("vi", group.vi)):
        for i, q in enumerate(queries):
            resp = client.get(WORLDBANK, {"format": "json", "qterm": q.replace(" Vietnam", ""),
                                          "count_exact": "Viet Nam", "majdocty_exact": WORLDBANK_MAJOR_TYPES,
                                          "rows": 20, "srt": "score", "order": "desc",
                                          "fl": "display_title,docdt,docty,majdocty,lang,pdfurl,url,abstracts"})
            out += worldbank_candidates(resp.json(), f"{group.id}/worldbank/{lang}{i}", group.id, resp.sha256)
    return out


# ---------------------------------------------------------------- BIS (RSS)


def bis_candidates(xml: bytes, feed: str, raw: str) -> list[Candidate]:
    root = ET.fromstring(xml)
    out = []
    for item in root.iter():
        if not item.tag.endswith("item"):
            continue
        fields = {child.tag.split("}")[-1]: (child.text or "").strip() for child in item}
        link = fields.get("link")
        summary = html.unescape(re.sub("<[^>]+>", " ", fields.get("description", "")))
        if not link or not fields.get("title") or not BIS_TOPIC.search(fields["title"] + " " + summary):
            continue
        out.append(Candidate(
            key=f"url:{link}", title=fields["title"], abstract=summary, adapter="bis",
            query_ids=[f"listing/bis/{feed}"], groups=[], raw_sha256=raw, landing_url=link,
            pdf_url=link if link.endswith(".pdf") else None, language="en",  # else found on the page
            publication_date=_date(fields.get("date")), work_type=feed, site_publisher="bis"))
    return out


# ---------------------------------------------------------------- SBV and the statistics office (listing pages)


SBV_ITEM = re.compile(r'<a href="(?P<page>[^"]+)" class="doc-item[^"]*">\s*<span[^>]*>\s*<b>(?P<title>[^<]+)</b>'
                      r"\s*</span>\s*\((?P<date>\d{2}/\d{2}/\d{4})\)\s*</a>\s*<a href=\"(?P<file>[^\"]+)\" "
                      r'class="doc-icon')
NSO_ITEM = re.compile(r'<a href="(?P<page>https://www\.nso\.gov\.vn/[^"]+)">\s*</p>\s*<section class="item">\s*'
                      r"<h3>(?P<title>[^<]+)</h3>\s*<p>(?P<desc>.*?)</p>.*?Date of issue:\s*(?P<date>\d{2}/\d{2}/\d{4})",
                      re.S)


def sbv_candidates(page_html: str, page_url: str, listing: str, raw: str) -> list[Candidate]:
    out = []
    for m in SBV_ITEM.finditer(page_html):
        file_url = urljoin(page_url, html.unescape(m["file"]))
        out.append(Candidate(
            key=f"url:{file_url}", title=html.unescape(m["title"]).strip(), adapter="sbv",
            query_ids=[f"listing/sbv/{listing}"], groups=[], raw_sha256=raw, landing_url=html.unescape(m["page"]),
            pdf_url=file_url, language=None, publication_date=_date(m["date"]), work_type=listing,
            site_publisher="sbv"))
    return out


def nso_candidates(page_html: str, listing: str, raw: str) -> list[Candidate]:
    out = []
    for m in NSO_ITEM.finditer(page_html):
        desc = html.unescape(re.sub("<[^>]+>", " ", m["desc"])).strip()
        out.append(Candidate(
            key=f"url:{m['page']}", title=html.unescape(m["title"].strip()), abstract=desc or None, adapter="nso",
            query_ids=[f"listing/nso/{listing}"], groups=[], raw_sha256=raw, landing_url=m["page"],
            item_page=m["page"], language="en", publication_date=_date(m["date"]), work_type=listing,
            site_publisher="nso"))
    return out


NSO_PDF = re.compile(r'href="(https://www\.nso\.gov\.vn/wp-content/uploads/[^"]+\.pdf)"', re.I)


def nso_pdf_link(item_html: str) -> str | None:
    m = NSO_PDF.search(item_html)
    return m.group(1) if m else None
