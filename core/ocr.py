from __future__ import annotations
import os
import cv2
from PIL import Image
import pytesseract
from rapidfuzz import fuzz

def ocr_nodes(img_bgr, nodes):
    lang = os.environ.get("TESS_LANG", "eng+rus")

    for n in nodes:
        x1, y1, x2, y2 = n["bbox"]
        pad = 6
        x1 = max(0, x1 + pad); y1 = max(0, y1 + pad)
        x2 = min(img_bgr.shape[1], x2 - pad); y2 = min(img_bgr.shape[0], y2 - pad)

        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            n["label"] = ""
            continue

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 3)

        pil = Image.fromarray(thr)
        try:
            txt = pytesseract.image_to_string(pil, lang=lang, config="--psm 6")
        except pytesseract.TesseractNotFoundError as e:
            raise RuntimeError("Tesseract not found. Install via: brew install tesseract tesseract-lang") from e

        txt = _clean(txt)
        n["label"] = txt

        if n["kind"] == "ellipse":
            low = txt.lower()
            if fuzz.partial_ratio(low, "start") > 80 or fuzz.partial_ratio(low, "нач") > 80:
                n["semantic"] = "start"
            elif fuzz.partial_ratio(low, "end") > 80 or fuzz.partial_ratio(low, "кон") > 80:
                n["semantic"] = "end"

    return nodes

def _clean(s: str) -> str:
    s = s.replace("\n", " ").replace("\r", " ").strip()
    s = " ".join(s.split())
    return "" if len(s) <= 1 else s
