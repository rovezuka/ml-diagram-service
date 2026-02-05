from __future__ import annotations
import re
from rapidfuzz import fuzz

_OCR_GARBAGE_RE = re.compile(r"^[\W_]+$")

def normalize_step_text(t: str) -> str:
    t = (t or "").replace("\t", " ").replace("\n", " ").replace("\r", " ")
    t = " ".join(t.split())
    return t.strip()

def is_good_step_text(t: str) -> bool:
    t = normalize_step_text(t)
    if len(t) < 4:
        return False
    if _OCR_GARBAGE_RE.match(t):
        return False

    alnum = sum(ch.isalnum() for ch in t)
    if alnum < 3:
        return False

    ratio = alnum / max(1, len(t))
    if ratio < 0.35:
        return False

    weird = sum((not ch.isalnum()) and (ch not in " -.,:;()/%+&") for ch in t)
    if weird / max(1, len(t)) > 0.15:
        return False

    return True

def dedupe_steps(texts: list[str], threshold: int = 92) -> list[str]:
    out: list[str] = []
    for t in texts:
        t = normalize_step_text(t)
        if not t:
            continue
        if any(fuzz.token_sort_ratio(t.lower(), x.lower()) >= threshold for x in out):
            continue
        out.append(t)
    return out
