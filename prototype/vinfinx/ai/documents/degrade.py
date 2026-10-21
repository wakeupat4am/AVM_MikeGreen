"""Image degradations (MACRO_DOCUMENT_SPEC.md, `degraded_*` variants). Each transforms the rendered pages and the
field boxes with them, so the ground truth stays exact."""
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from ..schemas.evidence import Presence
from .render import DPI, Placed

IMAGE_DEFAULTS: dict[str, list[dict[str, Any]]] = {
    "degraded_rotate": [{"angle": a} for a in (1, -1, 3, -3, 90)],
    "degraded_blur": [{"sigma": s} for s in (1, 2, 3)],
    "degraded_lowres": [{"dpi": 150}, {"dpi": 100}],
    "degraded_jpeg": [{"quality": 50}, {"quality": 20}],
    "degraded_crop": [{"right_fraction": 0.05}, {"right_fraction": 0.15}],
    "degraded_noise": [{"noise_sigma": 12, "shadow": 0.25}],
}


@dataclass
class PageImage:
    image: Image.Image
    dpi: int
    encoded: bytes | None = None  # set for JPEG pages: the file is stored exactly as encoded
    ext: str = "png"


@dataclass
class FieldState:
    placed: Placed
    box: tuple[float, float, float, float] | None
    presence: Presence


def _rotate_box(box, angle, size, new_size):
    (w, h), (nw, nh) = size, new_size
    t = math.radians(angle)
    pts = [(box[0], box[1]), (box[2], box[1]), (box[0], box[3]), (box[2], box[3])]
    out = [(nw / 2 + (x - w / 2) * math.cos(t) + (y - h / 2) * math.sin(t),
            nh / 2 - (x - w / 2) * math.sin(t) + (y - h / 2) * math.cos(t)) for x, y in pts]
    xs, ys = [p[0] for p in out], [p[1] for p in out]
    return min(xs), min(ys), max(xs), max(ys)


def apply(variant: str, params: dict[str, Any], images: list[Image.Image], placed: list[Placed],
          seed: int) -> tuple[list[PageImage], list[FieldState], list[str]]:
    """Returns the degraded pages, every field's box and presence afterwards, and the IDs of fields it changed."""
    pages = [PageImage(im, DPI) for im in images]
    states = [FieldState(p, p.box, p.cell.presence) for p in placed]
    changed: list[str] = []
    if variant == "degraded_rotate":
        angle = params["angle"]
        for i, pg in enumerate(pages):
            size = pg.image.size
            pg.image = pg.image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=255)
            for s in states:
                if s.placed.page == i + 1 and s.box is not None:
                    s.box = _rotate_box(s.box, angle, size, pg.image.size)
    elif variant == "degraded_blur":
        for pg in pages:
            pg.image = pg.image.filter(ImageFilter.GaussianBlur(params["sigma"]))
    elif variant == "degraded_lowres":
        scale = params["dpi"] / DPI
        for pg in pages:
            w, h = pg.image.size
            pg.image = pg.image.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)
            pg.dpi = params["dpi"]
        for s in states:
            if s.box is not None:
                s.box = tuple(v * scale for v in s.box)
    elif variant == "degraded_jpeg":
        for pg in pages:
            buf = io.BytesIO()
            pg.image.save(buf, "JPEG", quality=params["quality"], optimize=False)
            pg.encoded, pg.ext = buf.getvalue(), "jpg"
            pg.image = Image.open(io.BytesIO(pg.encoded)).convert("L")
    elif variant == "degraded_crop":
        for i, pg in enumerate(pages):
            w, h = pg.image.size
            edge = round(w * (1 - params["right_fraction"]))
            pg.image = pg.image.crop((0, 0, edge, h))
            for s in states:
                if s.placed.page != i + 1 or s.box is None:
                    continue
                if s.box[0] >= edge:  # entirely cut off
                    s.box, s.presence = None, Presence.ABSENT
                    changed.append(s.placed.field_id)
                elif s.box[2] > edge:  # partly cut: printed, but cannot be read reliably
                    s.box, s.presence = (s.box[0], s.box[1], edge - 1, s.box[3]), Presence.UNREADABLE
                    changed.append(s.placed.field_id)
    elif variant == "degraded_noise":
        rng = np.random.default_rng(seed)
        for pg in pages:
            a = np.asarray(pg.image, dtype=np.float32)
            a = a + rng.normal(0, params["noise_sigma"], a.shape)
            shadow = np.linspace(1.0, 1.0 - params["shadow"], a.shape[1], dtype=np.float32)
            pg.image = Image.fromarray(np.clip(a * shadow[None, :], 0, 255).astype(np.uint8))
    else:
        raise ValueError(f"unknown image variant {variant}")
    return pages, states, changed


def unchanged(images: list[Image.Image], placed: list[Placed]) -> tuple[list[PageImage], list[FieldState], list[str]]:
    return [PageImage(im, DPI) for im in images], [FieldState(p, p.box, p.cell.presence) for p in placed], []
