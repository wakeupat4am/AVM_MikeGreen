"""Reads page images with the Tesseract adapter and prints one JSON row per image. Runs in env-ocr:

    env-ocr/bin/python -m ai.corpus.ocr_images --lang vie+eng p3.png p7.png
"""
from __future__ import annotations

import argparse
import json
from itertools import groupby

from ..ocr.tesseract import TesseractEngine


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lang", default="vie+eng")
    p.add_argument("images", nargs="+")
    a = p.parse_args()
    engine = TesseractEngine()
    rows = []
    for n, image in enumerate(a.images, 1):
        page = engine.read(image, n, a.lang)
        lines = [" ".join(w.text for w in ws) for _, ws in groupby(page.words, key=lambda w: w.line_id)]
        conf = [w.confidence for w in page.words]
        rows.append({"image": image, "text": "\n".join(lines), "model": page.model,
                     "mean_confidence": round(sum(conf) / len(conf), 3) if conf else None})
    print(json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
