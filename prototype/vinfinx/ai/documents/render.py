"""Draws a Document as three A4-landscape greyscale pages at 300 dpi and records where each field was drawn.
Image-only by design: OCR is the only way to read it back."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .content import ANALYST, FOOTER, WATERMARK, Cell, Document

DPI = 300
WIDTH, HEIGHT = 3508, 2480  # A4 landscape at 300 dpi
MARGIN = 150
FONT_DIR = Path(os.environ.get("VINFINX_FONT_DIR", "/usr/share/fonts/truetype/dejavu"))
FONT_FILES = {"regular": "DejaVuSans.ttf", "bold": "DejaVuSans-Bold.ttf"}  # full Vietnamese diacritics

TABLE1 = [("Metric", "Chỉ tiêu", None, 540), ("Metric ID", "Mã chỉ tiêu", "metric_id", 680),
          ("Geography", "Địa bàn", "geography", 430), ("Segment", "Phân khúc", "segment", 230),
          ("Measure", "Thước đo", "transformation", 260), ("Period", "Kỳ", "observation_period", 200),
          ("Value", "Giá trị", "value", 140), ("Unit", "Đơn vị", "unit", 170),
          ("Baseline", "Kỳ gốc", "baseline_value", 140), ("Baseline period", "Kỳ so sánh", "baseline_period", 218),
          ("Change", "Thay đổi", "change_vs_baseline", 200)]
# The page-2 Metric ID column is the join key to Table 1: drawn, not a separate ground-truth field.
TABLE2 = [("Metric ID", "Mã chỉ tiêu", None, 800), ("Source", "Nguồn", "source", 600),
          ("Revision status", "Tình trạng điều chỉnh", "revision_status", 350),
          ("Previous value", "Giá trị trước", "previous_value", 330),
          ("Data quality", "Chất lượng dữ liệu", "data_quality", 300),
          ("Uncertainty", "Độ bất định", "uncertainty", 500), ("Next update", "Kỳ cập nhật tiếp", "next_update", 328)]
RIGHT_ALIGNED = {"value", "baseline_value", "change_vs_baseline", "previous_value"}


@dataclass
class Placed:
    """A drawn field: page (1-based) and pixel box (x0, y0, x1, y1) on the 300 dpi page."""

    field_id: str
    metric_id: str | None
    attribute: str
    page: int
    box: tuple[float, float, float, float] | None
    cell: Cell


def font_hashes() -> dict[str, str]:
    return {name: "sha256:" + hashlib.sha256((FONT_DIR / name).read_bytes()).hexdigest()
            for name in FONT_FILES.values()}


_fonts: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def font(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    key = (weight, size)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(str(FONT_DIR / FONT_FILES[weight]), size)
    return _fonts[key]


def _fit(draw: ImageDraw.ImageDraw, text: str, width: int, size: int, weight: str = "regular"):
    while size > 14 and draw.textlength(text, font=font(size, weight)) > width:
        size -= 1
    return font(size, weight)


class Page:
    def __init__(self, number: int, lang: str) -> None:
        self.number, self.lang = number, lang
        self.image = Image.new("L", (WIDTH, HEIGHT), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.placed: list[Placed] = []

    def text(self, xy, text, size, weight="regular", fill=0, anchor="la") -> tuple[float, float, float, float]:
        self.draw.text(xy, text, font=font(size, weight), fill=fill, anchor=anchor)
        return self.draw.textbbox(xy, text, font=font(size, weight), anchor=anchor)

    def field(self, field_id, metric_id, attribute, cell: Cell, x, y, width, size=30, right=False, weight="regular"):
        box = None
        if cell.printed is not None:
            f = _fit(self.draw, cell.printed, width - 20, size, weight)
            xy, anchor = ((x + width - 10, y), "ra") if right else ((x + 10, y), "la")
            self.draw.text(xy, cell.printed, font=f, fill=0, anchor=anchor)
            box = self.draw.textbbox(xy, cell.printed, font=f, anchor=anchor)
            if cell.obscure == "black":
                self.draw.rectangle((box[0] - 6, box[1] - 6, box[2] + 6, box[3] + 6), fill=0)
            elif cell.obscure == "smudge":
                region = (int(box[0]) - 12, int(box[1]) - 12, int(box[2]) + 12, int(box[3]) + 12)
                smeared = self.image.crop(region).filter(ImageFilter.GaussianBlur(9))
                smeared = smeared.point(lambda v: int(v * 0.55 + 70))  # a grey, blurred stain
                self.image.paste(smeared, region[:2])
        self.placed.append(Placed(field_id, metric_id, attribute, self.number, box, cell))

    def footer(self, total: int) -> None:
        self.text((MARGIN, HEIGHT - 110), FOOTER[self.lang], 28)
        page = f"Page {self.number} of {total}" if self.lang == "en" else f"Trang {self.number}/{total}"
        self.text((WIDTH - MARGIN, HEIGHT - 110), page, 28, anchor="ra")

    def watermark(self) -> None:
        layer = Image.new("L", (WIDTH, HEIGHT), 0)
        ImageDraw.Draw(layer).text((WIDTH / 2, HEIGHT / 2), WATERMARK, font=font(260, "bold"), fill=255,
                                   anchor="mm")
        layer = layer.rotate(28, resample=Image.Resampling.BICUBIC)
        self.image = Image.fromarray(_darken(self.image, layer))  # ~15% grey
        self.draw = ImageDraw.Draw(self.image)


def _darken(page: Image.Image, mask: Image.Image):
    import numpy as np

    p = np.asarray(page, dtype=np.float32)
    m = np.asarray(mask, dtype=np.float32) / 255.0
    return np.minimum(p, 255 * (1 - 0.15 * m)).astype(np.uint8)  # text under the watermark stays dark


def _table(page: Page, rows, columns, lang: str, top: int, row_h: int, label_col: bool) -> None:
    x = MARGIN
    for en, vi, attribute, w in columns:
        label = en if lang == "en" else vi
        f = _fit(page.draw, label, w - 20, 28, "bold")
        if attribute in RIGHT_ALIGNED:
            page.draw.text((x + w - 10, top + 20), label, font=f, fill=0, anchor="ra")
        else:
            page.draw.text((x + 10, top + 20), label, font=f, fill=0)
        x += w
    page.draw.line((MARGIN, top + row_h - 8, WIDTH - MARGIN, top + row_h - 8), fill=0, width=3)
    y = top + row_h
    for row in rows:
        x = MARGIN
        for _, _, attribute, w in columns:
            if attribute is None:
                label = row.label if label_col else row.metric_id
                f = _fit(page.draw, label, w - 20, 30)
                page.draw.text((x + 10, y + 30), label, font=f, fill=0)
            else:
                page.field(f"{row.metric_id}/{attribute}", row.metric_id, attribute, row.cells[attribute], x, y + 30,
                           w, right=attribute in RIGHT_ALIGNED)
            x += w
        page.draw.line((MARGIN, y + row_h - 8, WIDTH - MARGIN, y + row_h - 8), fill=170, width=1)
        y += row_h


def _wrap(draw: ImageDraw.ImageDraw, text: str, f, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=f) <= width:
            line = trial
        else:
            lines.append(line)
            line = word
    return lines + [line] if line else lines


def render(doc: Document) -> tuple[list[Image.Image], list[Placed]]:
    lang = doc.language
    pages = [Page(n, lang) for n in (1, 2, 3)]

    p1 = pages[0]
    p1.field("document/publisher", None, "publisher", doc.header["publisher"], MARGIN - 10, MARGIN, 2000, 56, weight="bold")
    p1.field("document/title", None, "title", doc.header["title"], MARGIN - 10, MARGIN + 80, 2600, 44)
    y = MARGIN + 160
    x = MARGIN
    for attribute, en, vi in (("issue", "Issue", "Số"), ("document_id", "Document ID", "Mã tài liệu"),
                              ("publication_date", "Published", "Ngày phát hành")):
        box = p1.text((x, y), f"{en if lang == 'en' else vi}:", 32, "bold")
        p1.field(f"document/{attribute}", None, attribute, doc.header[attribute], box[2] + 4, y, 600, 32)
        x += 900
    _table(p1, doc.table1, TABLE1, lang, top=MARGIN + 260, row_h=118, label_col=True)

    p2 = pages[1]
    p2.text((MARGIN, MARGIN), "Indicator metadata" if lang == "en" else "Siêu dữ liệu chỉ tiêu", 48, "bold")
    _table(p2, doc.table2, TABLE2, lang, top=MARGIN + 110, row_h=124, label_col=False)

    p3 = pages[2]
    p3.text((MARGIN, MARGIN), "Analyst commentary" if lang == "en" else "Nhận định của chuyên viên phân tích", 48, "bold")
    p3.text((MARGIN, MARGIN + 80), ANALYST[lang], 32)
    y = MARGIN + 190
    width = WIDTH - 2 * MARGIN
    for para in doc.commentary:
        size, fill = {"tiny": (17, 0), "low_contrast": (40, 246)}.get(para.style, (40, 0))
        f = font(size)
        for line in _wrap(p3.draw, para.text, f, width):
            p3.draw.text((MARGIN, y), line, font=f, fill=fill)
            y += int(size * 1.45)
        y += 40

    for p in pages:
        p.watermark()
        p.footer(len(pages))
    return [p.image for p in pages], [pl for p in pages for pl in p.placed]
