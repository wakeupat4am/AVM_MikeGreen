"""Tesseract adapter (runs in env-ocr). The language data is pinned: TESSDATA_DIR points at tessdata_best at a fixed
commit, and its hashes are recorded in the model directory.

Before reading, the page's orientation is detected (Tesseract OSD); a page turned by 90, 180 or 270 degrees is turned
upright, read, and the word boxes are mapped back onto the page as received, so every field still points at the
original image."""
from __future__ import annotations

import hashlib
import math
import os
import sys
import time
from pathlib import Path

import pytesseract
from PIL import Image

from . import OcrPage, OcrWord

from ..paths import ROOT  # noqa: E402  (one place for server paths: ai/paths.py)
TESSDATA_DIR = Path(os.environ.get("VINFINX_TESSDATA", ROOT / "models" / "tesseract" / "tessdata_best-e12c65a"))
MODEL = "tessdata_best@e12c65a"
MIN_OSD_CONFIDENCE = 2.0  # below this, Tesseract's orientation guess is not acted on


def _unrotate_box(box, clockwise: int, upright_size, original_size):
    """Map a pixel box on the upright image back to the original image, which is the upright one turned
    `clockwise` degrees counter-clockwise."""
    (w, h), (ow, oh) = upright_size, original_size
    t = math.radians(clockwise)  # undo: turn counter-clockwise by the same angle
    pts = [(box[0], box[1]), (box[2], box[1]), (box[0], box[3]), (box[2], box[3])]
    out = [(ow / 2 + (x - w / 2) * math.cos(t) + (y - h / 2) * math.sin(t),
            oh / 2 - (x - w / 2) * math.sin(t) + (y - h / 2) * math.cos(t)) for x, y in pts]
    xs, ys = [p[0] for p in out], [p[1] for p in out]
    return min(xs), min(ys), max(xs), max(ys)


class TesseractEngine:
    name = "tesseract"

    def __init__(self, psm: int = 3, auto_orient: bool = True, tessdata_dir: Path = TESSDATA_DIR) -> None:
        # the env is not activated when called as $VINFINX_ROOT/env-ocr/bin/python, so point at its binary
        pytesseract.pytesseract.tesseract_cmd = str(Path(sys.prefix) / "bin" / "tesseract")
        self.psm, self.auto_orient, self.tessdata_dir = psm, auto_orient, tessdata_dir
        self.version = str(pytesseract.get_tesseract_version())

    def orientation(self, image: Image.Image) -> tuple[int, float | None]:
        """Degrees clockwise to turn the image upright, and Tesseract's confidence. (0, None) if OSD fails."""
        try:
            osd = pytesseract.image_to_osd(image, config=f'--psm 0 --tessdata-dir "{self.tessdata_dir}"',
                                           output_type=pytesseract.Output.DICT)
        except pytesseract.TesseractError:
            return 0, None
        return int(osd["rotate"]) % 360, float(osd["orientation_conf"])

    def read(self, image_path: str, page: int, languages: str = "vie+eng") -> OcrPage:
        data = Path(image_path).read_bytes()
        original = Image.open(image_path)
        ow, oh = original.size
        start = time.time()
        rotate, osd_conf = self.orientation(original) if self.auto_orient else (0, None)
        if rotate and (osd_conf or 0) < MIN_OSD_CONFIDENCE:
            rotate = 0
        upright = original.rotate(-rotate, expand=True) if rotate else original  # PIL turns counter-clockwise
        w, h = upright.size
        out = pytesseract.image_to_data(upright, lang=languages, output_type=pytesseract.Output.DICT,
                                        config=f'--psm {self.psm} --tessdata-dir "{self.tessdata_dir}"')
        seconds = time.time() - start
        words, lines = [], {}
        for i, text in enumerate(out["text"]):
            conf = float(out["conf"][i])
            x, y, bw, bh = out["left"][i], out["top"][i], out["width"][i], out["height"][i]
            if not text.strip() or conf < 0 or bw <= 0 or bh <= 0:
                continue
            key = (out["block_num"][i], out["par_num"][i], out["line_num"][i])
            line_id = lines.setdefault(key, len(lines))
            box = (x, y, x + bw, y + bh)
            if rotate:
                box = _unrotate_box(box, rotate, (w, h), (ow, oh))
            x0, y0, x1, y1 = (max(box[0] / ow, 0), max(box[1] / oh, 0), min(box[2] / ow, 1), min(box[3] / oh, 1))
            if x0 >= x1 or y0 >= y1:
                continue
            words.append(OcrWord(text=text, confidence=conf / 100, line_id=line_id,
                                 bbox={"x0": x0, "y0": y0, "x1": x1, "y1": y1}))
        return OcrPage(engine=self.name, engine_version=self.version, model=f"{MODEL}:{languages}:psm{self.psm}",
                       page_sha256="sha256:" + hashlib.sha256(data).hexdigest(), page=page, width_px=ow,
                       height_px=oh, rotation_applied=rotate, orientation_confidence=osd_conf, words=words,
                       seconds=round(seconds, 2))
