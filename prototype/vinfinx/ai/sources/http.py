"""A polite HTTP client: honours robots.txt, waits between requests to a host, retries transient errors, never
works around bot protection, and stores every response body with its hash so a run can be replayed."""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.robotparser
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urlsplit, urlunsplit, parse_qsl

import requests
import urllib3

from . import USER_AGENT

MIN_INTERVAL = 1.0  # seconds between requests to one host (SOURCE_AGENT_PLAN.md, "Build")
MAX_BYTES = 60_000_000
RETRY_STATUS = {429, 500, 502, 503, 504}
CHALLENGE_MARKERS = (b"cf-browser-verification", b"Just a moment...", b"<TITLE>Access Denied</TITLE>",
                     b"challenge-platform")
SECRET_PARAMS = {"api_key"}


class Blocked(Exception):
    """The site refused a script (bot protection) or robots.txt disallows the URL. Not retried."""


class FetchError(Exception):
    pass


@dataclass
class Response:
    url: str
    status: int
    content_type: str
    content: bytes
    sha256: str
    fetched_at: datetime

    def json(self):
        return json.loads(self.content)

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    @property
    def is_pdf(self) -> bool:
        return self.content[:5] == b"%PDF-"


def redact(url: str) -> str:
    parts = urlsplit(url)
    query = [(k, "***" if k in SECRET_PARAMS else v) for k, v in parse_qsl(parts.query, keep_blank_values=True)]
    return urlunsplit(parts._replace(query=urlencode(query)))


def sha256_hex(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


class Client:
    def __init__(self, raw_dir: Path | None, session: requests.Session | None = None,
                 min_interval: float = MIN_INTERVAL, sleep=time.sleep) -> None:
        self.raw_dir = raw_dir
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.min_interval = min_interval
        self.sleep = sleep
        self._last: dict[str, float] = {}
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        if raw_dir is not None:
            raw_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- politeness

    def _wait(self, host: str) -> None:
        last = self._last.get(host)
        if last is not None:
            gap = time.monotonic() - last
            if gap < self.min_interval:
                self.sleep(self.min_interval - gap)
        self._last[host] = time.monotonic()

    def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        base = f"{parts.scheme}://{parts.netloc}"
        if base not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                self._wait(parts.netloc)
                r = self.session.get(base + "/robots.txt", timeout=30)
                if r.status_code in (401, 403):
                    rp.disallow_all = True
                elif r.status_code >= 400:
                    rp.allow_all = True
                else:
                    rp.parse(r.text.splitlines())
            except requests.RequestException:
                rp.allow_all = True  # unreachable robots.txt: the request itself will fail or succeed
            self._robots[base] = rp
        return self._robots[base].can_fetch(USER_AGENT, url)

    # ---------------------------------------------------------------- fetch

    def get(self, url: str, params: dict | None = None, store: bool = True, retries: int = 3) -> Response:
        if params:
            url = url + ("&" if "?" in url else "?") + urlencode(params)
        if not self.allowed(url):
            raise Blocked(f"robots.txt disallows {redact(url)}")
        host = urlsplit(url).netloc
        delay = 2.0
        for attempt in range(retries + 1):
            self._wait(host)
            try:
                r = self.session.get(url, timeout=60, stream=True)
                content = r.raw.read(MAX_BYTES + 1, decode_content=True)
            # urllib3 errors come from reading the streamed body (a server that stalls mid-file: BIS, 2026-10-24); they
            # are not wrapped in requests' exceptions, so both are caught
            except (requests.RequestException, urllib3.exceptions.HTTPError) as exc:
                if attempt == retries:
                    raise FetchError(f"{redact(url)}: {type(exc).__name__}: {exc}") from None
                self.sleep(delay)
                delay *= 2
                continue
            if len(content) > MAX_BYTES:
                raise FetchError(f"{redact(url)}: larger than {MAX_BYTES} bytes")
            if r.status_code in RETRY_STATUS and attempt < retries:
                wait = r.headers.get("Retry-After")
                self.sleep(float(wait) if wait and wait.isdigit() else delay)
                delay *= 2
                continue
            if r.status_code == 403 or (r.status_code == 202 and not content) or (
                    r.status_code >= 400 and any(m in content[:20000] for m in CHALLENGE_MARKERS)):
                raise Blocked(f"{redact(url)}: HTTP {r.status_code}, the site blocks scripts")
            if r.status_code >= 400:
                raise FetchError(f"{redact(url)}: HTTP {r.status_code}")
            resp = Response(url=redact(r.url), status=r.status_code,
                            content_type=r.headers.get("Content-Type", ""), content=content,
                            sha256=sha256_hex(content), fetched_at=datetime.now(timezone.utc))
            if store and self.raw_dir is not None:
                self._store(resp)
            return resp
        raise FetchError(f"{redact(url)}: retries exhausted")

    def _store(self, resp: Response) -> None:
        name = resp.sha256.split(":")[1]
        path = self.raw_dir / name
        if not path.exists():
            path.write_bytes(resp.content)
        with open(self.raw_dir / "index.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"url": resp.url, "status": resp.status, "content_type": resp.content_type,
                                "sha256": resp.sha256, "bytes": len(resp.content),
                                "fetched_at": resp.fetched_at.isoformat()}) + "\n")


def openalex_params() -> dict[str, str]:
    """The contact address always; an API key only if one is set in the environment (never stored or logged)."""
    from . import CONTACT

    params = {"mailto": CONTACT}
    key = os.environ.get("OPENALEX_API_KEY")
    if key:
        params["api_key"] = key
    return params
