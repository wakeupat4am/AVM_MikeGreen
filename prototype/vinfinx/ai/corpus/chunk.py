"""Chunker (CORPUS_INDEX_PLAN.md, I2): windows of about SIZE characters inside one page, cut at a paragraph break,
else a sentence end, else a space, overlapping by about OVERLAP. A chunk is always page_text[start:end]."""
from __future__ import annotations

import re

SIZE = 1200
OVERLAP = 200
MIN_CUT = 0.6  # a cut is searched back to 60% of the window
CHUNKER_VERSION = f"chunker/0.1 size={SIZE} overlap={OVERLAP}"
SENTENCE_END = re.compile(r"[.!?;:…](?=\s)")


def _cut(text: str, start: int, end: int) -> int:
    """The best break in text[start+MIN_CUT*SIZE : end]: paragraph, then sentence end, then whitespace."""
    lo = start + int(MIN_CUT * (end - start))
    window = text[lo:end]
    para = window.rfind("\n\n")
    if para >= 0:
        return lo + para + 2
    ends = [m.end() for m in SENTENCE_END.finditer(window)]
    if ends:
        return lo + ends[-1]
    space = max(window.rfind(" "), window.rfind("\n"))
    if space >= 0:
        return lo + space + 1
    return end


def spans(text: str, size: int = SIZE, overlap: int = OVERLAP) -> list[tuple[int, int]]:
    """(start, end) of each chunk; leading and trailing whitespace trimmed; empty chunks dropped."""
    out: list[tuple[int, int]] = []
    start, n = 0, len(text)
    while start < n:
        end = n if n - start <= size else _cut(text, start, start + size)
        s, e = start, end
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            out.append((s, e))
        if end >= n:
            break
        nxt = max(end - overlap, start + 1)
        # start the overlap on a word boundary
        while nxt < end and not text[nxt - 1].isspace():
            nxt += 1
        start = nxt if nxt < end else end
    return out
