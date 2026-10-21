"""From OCR words to table cells, using what is printed plus the document type's template (the order and relative
widths of its columns):
- the page is laid out upright (undoing a 90/180/270 degree turn the OCR detected);
- the metric IDs anchor the rows; a small skew is measured from the metric-ID column and removed;
- the header labels that were read (two or more) fix the template's offset and scale on the page, so a header
  that OCR missed, a lower resolution or a cropped edge does not lose a column;
- each word goes to the column its centre falls in.
"""
from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass, field
from statistics import median

from ..ocr import OcrPage
from .parse import clean

ID_START = re.compile(r"^(macro|market)\.")


@dataclass
class Column:
    attribute: str | None  # None for a label column (the metric's name)
    en: str
    vi: str
    align: str  # "left" or "right"
    width: float  # relative width in the template


@dataclass
class Token:
    text: str
    conf: float
    x0: float  # upright, deskewed pixel coordinates
    y0: float
    x1: float
    y1: float
    box: tuple[float, float, float, float]  # normalized box on the page as received

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def h(self) -> float:
        return self.y1 - self.y0


@dataclass
class Cell:
    text: str
    conf: float
    box: tuple[float, float, float, float]  # normalized, union of its words


@dataclass
class Table:
    rows: dict[str, dict[str, Cell]] = field(default_factory=dict)  # metric_id -> attribute -> cell
    headers_found: list[str] = field(default_factory=list)
    skew_degrees: float = 0.0
    unmatched_ids: list[str] = field(default_factory=list)
    calibrated: bool = False


def _upright(page: OcrPage):
    """Word boxes in pixels on the upright page, with the original normalized box kept for provenance."""
    w, h = page.width_px, page.height_px
    r = page.rotation_applied % 360
    out = []
    for wd in page.words:
        b = wd.bbox
        corners = [(b.x0 * w, b.y0 * h), (b.x1 * w, b.y1 * h)]
        if r == 90:  # the page as received is the upright page turned 90 degrees counter-clockwise
            corners = [(h - y, x) for x, y in corners]
        elif r == 180:
            corners = [(w - x, h - y) for x, y in corners]
        elif r == 270:
            corners = [(y, w - x) for x, y in corners]
        xs, ys = [c[0] for c in corners], [c[1] for c in corners]
        out.append((wd.text, wd.confidence, (min(xs), min(ys), max(xs), max(ys)), (b.x0, b.y0, b.x1, b.y1)))
    size = (h, w) if r in (90, 270) else (w, h)
    return out, size


def _tokens(raw, size, angle: float) -> list[Token]:
    cx, cy = size[0] / 2, size[1] / 2
    c, s = math.cos(angle), math.sin(angle)
    out = []
    for text, conf, (x0, y0, x1, y1), box in raw:
        pts = [((x - cx) * c - (y - cy) * s + cx, (x - cx) * s + (y - cy) * c + cy)
               for x, y in ((x0, y0), (x1, y0), (x0, y1), (x1, y1))]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        out.append(Token(text, conf, min(xs), min(ys), max(xs), max(ys), box))
    return out


def _resolve_id(text: str, known: set[str]) -> str | None:
    """A printed metric ID, tolerating an underscore read as a space; exact, or one clearly closest match."""
    t = clean(text).replace(" ", "_").rstrip(".,;:")
    if t in known:
        return t
    scored = sorted(((difflib.SequenceMatcher(None, t, k).ratio(), k) for k in known), reverse=True)
    if scored[0][0] >= 0.92 and (len(scored) == 1 or scored[0][0] - scored[1][0] > 0.03):
        return scored[0][1]
    return None


def _anchors(tokens: list[Token], known: set[str]):
    """Metric-ID tokens, joined with following lowercase words on the same line when OCR split the ID."""
    found, unmatched, used = [], [], set()
    for i, t in enumerate(tokens):
        if i in used or not ID_START.match(clean(t.text)):
            continue
        parts = [t]
        while True:
            nxt = min((j for j, u in enumerate(tokens) if j not in used and j != i and u not in parts
                       and abs(u.cy - t.cy) < 0.5 * t.h and u.x0 > parts[-1].x0 and u.x0 - parts[-1].x1 < 2.0 * t.h
                       # a lost underscore leaves a gap; OCR boxes of the two halves may also overlap
                       and re.fullmatch(r"[a-z0-9_.-]+", clean(u.text))), key=lambda j: tokens[j].x0, default=None)
            if nxt is None:
                break
            parts.append(tokens[nxt])
            used.add(nxt)
        used.add(i)
        metric_id = _resolve_id(" ".join(p.text for p in parts), known)
        if metric_id is None:
            unmatched.append(" ".join(p.text for p in parts))
        else:
            found.append((metric_id, parts))
    return found, unmatched


def _skew(anchors) -> float:
    """Angle (radians) of the metric-ID column from vertical, by least squares of its left edges."""
    pts = [(p[0].x0, p[0].cy) for _, p in anchors]
    if len(pts) < 3:
        return 0.0
    mx, my = sum(x for x, _ in pts) / len(pts), sum(y for _, y in pts) / len(pts)
    var = sum((y - my) ** 2 for _, y in pts)
    return math.atan(sum((x - mx) * (y - my) for x, y in pts) / var) if var else 0.0


def _union(tokens: list[Token]) -> Cell:
    boxes = [t.box for t in tokens]
    return Cell(text=clean(" ".join(t.text for t in sorted(tokens, key=lambda t: t.x0))),
                conf=min(t.conf for t in tokens),
                box=(min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes),
                     max(b[3] for b in boxes)))


def _header_points(header: list[Token], template: list[Column], lang: str) -> list[tuple[float, float, str]]:
    """(template position, observed x, attribute) for each header label that was read."""
    words = sorted(header, key=lambda t: t.x0)
    starts, x = [], 0.0
    for col in template:
        starts.append(x)
        x += col.width
    points, used = [], set()
    for col, start in zip(template, starts):
        label = (col.en if lang == "en" else col.vi).lower()
        n, best = len(label.split()), None
        for i in range(len(words)):
            window = words[i:i + n]
            if len(window) < n or any(id(w) in used for w in window):
                continue
            score = difflib.SequenceMatcher(None, clean(" ".join(w.text for w in window)).lower(), label).ratio()
            if best is None or score > best[0]:
                best = (score, window)
        if best and best[0] >= 0.8:
            used.update(id(w) for w in best[1])
            if col.align == "left":  # labels are drawn with a 10-unit padding inside their column
                points.append((start + 10, best[1][0].x0, col.attribute or "label"))
            else:
                points.append((start + col.width - 10, best[1][-1].x1, col.attribute or "label"))
    return points


def _fit(points, total_width: float) -> tuple[float, float] | None:
    """x = a * template + b by least squares, dropping a point that disagrees by more than 1% of the table's width
    (a label matched to the wrong word) and refitting."""
    pts = list(points)
    while len(pts) >= 2:
        n = len(pts)
        mt, mx = sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n
        var = sum((p[0] - mt) ** 2 for p in pts)
        if var == 0:
            return None
        a = sum((p[0] - mt) * (p[1] - mx) for p in pts) / var
        b = mx - a * mt
        resid = [abs(a * p[0] + b - p[1]) for p in pts]
        worst = max(range(n), key=lambda i: resid[i])
        if a > 0 and resid[worst] <= 0.01 * a * total_width:
            return a, b
        pts.pop(worst)
    return None


def read_table(page: OcrPage, template: list[Column], lang: str, known_ids: set[str]) -> Table:
    raw, size = _upright(page)
    flat = _tokens(raw, size, 0.0)
    anchors, _ = _anchors(flat, known_ids)
    angle = _skew(anchors)
    tokens = _tokens(raw, size, angle) if angle else flat
    anchors, unmatched = _anchors(tokens, known_ids)
    table = Table(skew_degrees=round(math.degrees(angle), 2), unmatched_ids=unmatched)
    if not anchors:
        return table
    anchors.sort(key=lambda a: a[1][0].cy)
    ys = [a[1][0].cy for a in anchors]
    row_h = median(b - a for a, b in zip(ys, ys[1:])) if len(ys) > 1 else 4 * anchors[0][1][0].h
    anchor_ids = {id(t) for _, parts in anchors for t in parts}
    rest = [t for t in tokens if id(t) not in anchor_ids]
    # the header is searched in a band above the first row found, not exactly one row up: if OCR missed the first
    # row's metric ID, the row just above the first anchor is a data row, not the header
    header = [t for t in rest if ys[0] - 3.5 * row_h <= t.cy <= ys[0] - 0.4 * row_h]
    points = _header_points(header, template, lang)
    # the metric-ID column's left edge is a calibration point too
    id_index = next(i for i, c in enumerate(template) if c.attribute == "metric_id")
    id_start = sum(c.width for c in template[:id_index]) + 10
    points.append((id_start, median(p[0].x0 for _, p in anchors), "metric_id"))
    table.headers_found = [p[2] for p in points[:-1]]
    fit = _fit(points, sum(c.width for c in template))
    if fit is None:
        return table
    table.calibrated = True
    a, b = fit
    bounds, x = [], 0.0
    for col in template:
        bounds.append((col.attribute, a * x + b, a * (x + col.width) + b))
        x += col.width
    for metric_id, parts in anchors:
        if metric_id in table.rows:
            continue  # printed twice: keep the first (top) row
        y = parts[0].cy
        cells: dict[str, list[Token]] = {"metric_id": list(parts)}
        for t in rest:
            if abs(t.cy - y) >= 0.45 * row_h:
                continue
            for attribute, x0, x1 in bounds:
                if x0 <= t.cx < x1 and attribute not in (None, "metric_id"):
                    cells.setdefault(attribute, []).append(t)
                    break
        table.rows[metric_id] = {k: _union(v) for k, v in cells.items()}
    return table


def line_after(page: OcrPage, labels: list[str]) -> Cell | None:
    """The text printed to the right of a label such as "Published:" on the same line."""
    raw, size = _upright(page)
    tokens = _tokens(raw, size, 0.0)
    for label in labels:
        words = label.lower().split()
        for first in (t for t in tokens if clean(t.text).lower() == words[0]):
            chain = [first]
            for w in words[1:]:
                nxt = min((t for t in tokens if abs(t.cy - chain[-1].cy) < 0.5 * first.h and t.x0 > chain[-1].x1),
                          key=lambda t: t.x0, default=None)
                if nxt is None or clean(nxt.text).lower() != w:
                    break
                chain.append(nxt)
            if len(chain) != len(words):
                continue
            last, picked = chain[-1], []
            for t in sorted((t for t in tokens if abs(t.cy - last.cy) < 0.5 * last.h and t.x0 > last.x1),
                            key=lambda t: t.x0):
                if t.x0 - (picked[-1].x1 if picked else last.x1) > (2.0 if picked else 3.0) * last.h:
                    break
                picked.append(t)
            if picked:
                return _union(picked)
    return None


def full_text(page: OcrPage) -> str:
    return clean(" ".join(w.text for w in sorted(page.words, key=lambda w: (w.line_id, w.bbox.x0))))
