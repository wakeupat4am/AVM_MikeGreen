"""Deterministic checks of the corpus criteria (CORPUS_SOURCE_CRITERIA.md C1, C2, C3, C6, C7) and the decision
rule (SOURCE_AGENT_PLAN.md). C4 and C5 come from the screening, after its quotes are verified."""
from __future__ import annotations

import re
import unicodedata
from urllib.parse import urlsplit

from ..schemas.source import (ACCEPTED_GEOGRAPHIES, Check, Criterion, Decision, Outcome, PublisherType,
                              Screening)
from . import Candidate, Publisher, PublisherFile

OPEN_LICENCES = re.compile(r"^(cc-|public-domain$|cc0)")
UNIVERSITY = "education"


def _domain_matches(url: str | None, domains: list[str]) -> bool:
    if not url:
        return False
    host = urlsplit(url).netloc.lower().split(":")[0]
    return any(host == d or host.endswith("." + d) for d in domains)


def match_publisher(c: Candidate, publishers: PublisherFile) -> tuple[Publisher | None, str]:
    """The allowlisted body that *published* the document: by site, DOI prefix, OpenAlex publisher or domain.
    Authors' affiliations are handled separately (working papers)."""
    for p in publishers.publishers:
        if c.site_publisher == p.id:
            return p, f"listed on the {p.name} site"
        if c.doi and any(c.doi.startswith(pre + "/") for pre in p.doi_prefixes):
            return p, f"DOI prefix of {p.name}"
        if set(c.publisher_ids) & set(p.openalex_publishers):
            return p, f"OpenAlex publisher {p.name}"
        if _domain_matches(c.pdf_url, p.domains) or _domain_matches(c.landing_url, p.domains):
            return p, f"hosted by {p.name}"
    return None, ""


def check_c1(c: Candidate, publishers: PublisherFile) -> tuple[Check, Publisher | None, PublisherType | None]:
    p, how = match_publisher(c, publishers)
    if p is not None:
        return Check(criterion=Criterion.C1_PUBLISHER, outcome=Outcome.PASS, detail=how, by="code"), p, \
            p.publisher_type
    if c.source_type == "journal":
        if c.source_in_doaj or c.source_is_core:
            listed = "DOAJ" if c.source_in_doaj else "OpenAlex core sources"
            return Check(criterion=Criterion.C1_PUBLISHER, outcome=Outcome.PASS,
                         detail=f"journal '{c.source_name}' is in {listed}", by="code"), None, \
                PublisherType.PEER_REVIEWED_JOURNAL
        return Check(criterion=Criterion.C1_PUBLISHER, outcome=Outcome.FAIL,
                     detail=f"journal not verifiable: '{c.source_name}' is neither in DOAJ nor an OpenAlex core "
                            "source", by="code"), None, None
    staff = [p for p in publishers.publishers if set(c.institution_ids) & set(p.openalex_institutions)]
    if staff and c.work_type in ("preprint", "report", "article"):
        return Check(criterion=Criterion.C1_PUBLISHER, outcome=Outcome.PASS,
                     detail=f"working paper by {staff[0].name} staff (author affiliation)", by="code"), \
            staff[0], PublisherType.WORKING_PAPER
    if c.work_type in ("preprint", "report") and UNIVERSITY in c.institution_types:
        return Check(criterion=Criterion.C1_PUBLISHER, outcome=Outcome.PASS,
                     detail=f"university working paper ('{c.source_name}')", by="code"), None, \
            PublisherType.WORKING_PAPER
    return Check(criterion=Criterion.C1_PUBLISHER, outcome=Outcome.FAIL,
                 detail=f"publisher type not accepted or not verifiable (type {c.work_type}, source "
                        f"'{c.source_name}' {c.source_type})", by="code"), None, None


def check_c2(c: Candidate, publisher: Publisher | None) -> tuple[Check, str | None, str | None]:
    """Returns the check, the licence and its link."""
    if c.licence and OPEN_LICENCES.match(c.licence):
        return Check(criterion=Criterion.C2_LICENCE, outcome=Outcome.PASS, detail=f"open licence {c.licence}",
                     by="code"), c.licence, f"https://openalex.org/licenses/{c.licence}"
    if publisher is not None and publisher.terms_url:
        if publisher.terms_checked_by:
            return Check(criterion=Criterion.C2_LICENCE, outcome=Outcome.PASS,
                         detail=f"public-body publication; {publisher.name} terms checked by "
                                f"{publisher.terms_checked_by}", by="code"), "publisher terms", publisher.terms_url
        return Check(criterion=Criterion.C2_LICENCE, outcome=Outcome.UNCLEAR,
                     detail=f"public-body publication; {publisher.name} terms not yet checked by a person: "
                            f"{publisher.terms_url}", by="code"), "publisher terms (unchecked)", publisher.terms_url
    detail = f"no open licence (licence {c.licence or 'none'}, access {c.oa_status or 'unknown'})"
    return Check(criterion=Criterion.C2_LICENCE, outcome=Outcome.FAIL, detail=detail, by="code"), c.licence, None


def detect_language(text: str) -> str | None:
    """'vi' when Vietnamese-only letters are frequent, 'en' when common English words are; else None."""
    letters = [ch for ch in unicodedata.normalize("NFC", text) if ch.isalpha()]
    if len(letters) < 200:
        return None
    vi_only = sum(1 for ch in letters if ch in "ăâđêôơưĂÂĐÊÔƠƯ" or (ord(ch) > 0x1E00 and ord(ch) < 0x1EFF))
    if vi_only / len(letters) > 0.02:
        return "vi"
    words = re.findall(r"[a-z]+", text.lower())
    common = sum(1 for w in words if w in {"the", "of", "and", "in", "to", "is", "for", "that", "on", "with"})
    if words and common / len(words) > 0.08:
        return "en"
    return None


def check_c3(c: Candidate, text: str | None) -> tuple[Check, str | None]:
    detected = detect_language(text) if text else None
    stated = c.language if c.language in ("en", "vi") else None
    if detected:
        note = "" if stated in (None, detected) else f" (index said {c.language})"
        return Check(criterion=Criterion.C3_LANGUAGE, outcome=Outcome.PASS,
                     detail=f"text is {detected}{note}", by="code"), detected
    if stated:
        return Check(criterion=Criterion.C3_LANGUAGE, outcome=Outcome.PASS,
                     detail=f"index says {stated}", by="code"), stated
    if c.language:
        return Check(criterion=Criterion.C3_LANGUAGE, outcome=Outcome.FAIL,
                     detail=f"language {c.language}", by="code"), None
    return Check(criterion=Criterion.C3_LANGUAGE, outcome=Outcome.UNCLEAR, detail="language not known",
                 by="code"), None


def check_c7(c: Candidate) -> Check:
    if c.publication_date:
        return Check(criterion=Criterion.C7_DATE, outcome=Outcome.PASS,
                     detail=f"published {c.publication_date.isoformat()}", by="code")
    return Check(criterion=Criterion.C7_DATE, outcome=Outcome.FAIL, detail="no publication date", by="code")


def screening_checks(s: Screening) -> list[Check]:
    """C4 and C5 from the agent's judgement, counted only with a verified quote."""
    quoted = bool(s.quotes)
    if s.relevant == "yes" and quoted and s.metrics and s.targets:
        c4 = Check(criterion=Criterion.C4_RELEVANCE, outcome=Outcome.PASS,
                   detail=f"agent: relevant to {', '.join(s.metrics)} -> {', '.join(t.value for t in s.targets)}; "
                          f"{len(s.quotes)} verified quote(s)", by="source_agent")
    elif s.relevant == "no":
        c4 = Check(criterion=Criterion.C4_RELEVANCE, outcome=Outcome.FAIL, detail="agent: not relevant",
                   by="source_agent")
    else:
        why = "no verified quote" if not quoted else f"agent said {s.relevant}"
        c4 = Check(criterion=Criterion.C4_RELEVANCE, outcome=Outcome.UNCLEAR, detail=why, by="source_agent")
    if s.geography in ACCEPTED_GEOGRAPHIES:
        outcome = Outcome.PASS if quoted else Outcome.UNCLEAR
    else:
        outcome = Outcome.FAIL
    c5 = Check(criterion=Criterion.C5_GEOGRAPHY, outcome=outcome, detail=f"agent: {s.geography.value}",
               by="source_agent")
    return [c4, c5]


ORDER = list(Criterion)


def decide(checks: list[Check]) -> tuple[Decision, Criterion | None, str]:
    """The fixed rule: every criterion passed -> accepted; any failed -> rejected (first failure, C1..C7);
    otherwise unclear, for a person."""
    by_c = {k.criterion: k for k in checks}
    for c in ORDER:
        if c in by_c and by_c[c].outcome is Outcome.FAIL:
            return Decision.REJECTED, c, f"{c.value}: {by_c[c].detail}"
    missing = [c.value for c in ORDER if c not in by_c]
    unclear = [f"{k.criterion.value}: {k.detail}" for k in sorted(checks, key=lambda k: ORDER.index(k.criterion))
               if k.outcome is Outcome.UNCLEAR]
    if not missing and not unclear:
        return Decision.ACCEPTED, None, "every criterion passed"
    reasons = unclear + ([f"not checked: {', '.join(missing)}"] if missing else [])
    return Decision.UNCLEAR, None, "; ".join(reasons)
