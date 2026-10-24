"""Source agent: finds candidate research documents, checks them against the corpus criteria, and lets Qwen judge
relevance and geography with verified quotes. A person signs the list off (docs/SOURCE_AGENT_PLAN.md)."""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field

from ..schemas.common import StrictModel, Target, content_hash
from ..schemas.source import PublisherType

AGENT_VERSION = "source_agent/0.1.0"
HERE = Path(__file__).parent
QUERIES_PATH = HERE / "queries_v0.yaml"
PUBLISHERS_PATH = HERE / "publishers_v0.yaml"
from ..paths import CORPUS, ROOT  # noqa: E402  (one place for server paths: ai/paths.py)
CONTACT = "jimmytrn154@gmail.com"  # given by the AI engineer for OpenAlex and the User-Agent (2026-10-21)
USER_AGENT = f"VinFinx-research/0.1 (mailto:{CONTACT})"


class QueryGroup(StrictModel):
    id: str
    metrics: list[str] = Field(min_length=1)
    targets: list[Target] = Field(min_length=1)
    en: list[str] = Field(min_length=1)
    vi: list[str] = []


class QueryFile(StrictModel):
    version: str
    groups: list[QueryGroup] = Field(min_length=1)

    @property
    def file_hash(self) -> str:
        return content_hash(self)


class Publisher(StrictModel):
    id: str
    name: str
    publisher_type: PublisherType
    country: str | None
    openalex_publishers: list[str] = []
    openalex_institutions: list[str] = []
    doi_prefixes: list[str] = []
    domains: list[str] = []
    terms_url: str | None
    terms_checked_by: str | None
    manual_download: bool = False


class PublisherFile(StrictModel):
    version: str
    publishers: list[Publisher]

    @property
    def file_hash(self) -> str:
        return content_hash(self)

    def institution_ids(self) -> list[str]:
        return [i for p in self.publishers for i in p.openalex_institutions]


def load_queries(path: Path = QUERIES_PATH) -> QueryFile:
    return QueryFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def load_publishers(path: Path = PUBLISHERS_PATH) -> PublisherFile:
    return PublisherFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


class Candidate(StrictModel):
    """One document as an index or page described it, before any check. Built by code only."""

    key: str  # "doi:<doi>" or "url:<url>"
    title: str
    abstract: str | None = None
    adapter: Literal["openalex", "worldbank", "bis", "sbv", "nso"]
    query_ids: list[str]  # "<group>/<adapter>/<mode>" or "listing/<page>"
    groups: list[str]
    raw_sha256: str
    doi: str | None = None
    landing_url: str
    pdf_url: str | None = None
    item_page: str | None = None  # a page that links to the PDF (NSO)
    language: str | None = None  # "en", "vi" or what the index said
    publication_date: date | None = None
    work_type: str | None = None
    source_name: str | None = None
    source_type: str | None = None
    source_in_doaj: bool | None = None
    source_is_core: bool | None = None
    publisher_ids: list[str] = []  # OpenAlex publisher lineage of the source
    institution_ids: list[str] = []  # OpenAlex institutions of the authors
    institution_types: list[str] = []
    licence: str | None = None
    oa_status: str | None = None
    site_publisher: str | None = None  # set by the SBV, NSO, World Bank and BIS adapters

    def merged(self, other: Candidate) -> Candidate:
        """Same document found twice: keep the first description, record every query that found it."""
        ids = self.query_ids + [q for q in other.query_ids if q not in self.query_ids]
        groups = self.groups + [g for g in other.groups if g not in self.groups]
        update = {"query_ids": ids, "groups": groups}
        for field in ("abstract", "pdf_url", "licence"):
            if getattr(self, field) is None and getattr(other, field) is not None:
                update[field] = getattr(other, field)
        return self.model_copy(update=update)
