"""Reads a bulletin's page images with the Tesseract adapter and prints the OcrPage records as one JSON list. Runs in
env-ocr (the runner calls it as a subprocess):

    env-ocr/bin/python -m ai.runner.ocr_pages --lang vie+eng page-1.png page-2.png page-3.png
"""
from __future__ import annotations

import argparse
import json

from ..ocr.tesseract import TesseractEngine


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lang", default="vie+eng")
    p.add_argument("images", nargs="+")
    a = p.parse_args()
    engine = TesseractEngine()
    print(json.dumps([engine.read(image, n, a.lang).model_dump(mode="json") for n, image in enumerate(a.images, 1)],
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
