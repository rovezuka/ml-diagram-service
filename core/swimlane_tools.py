from __future__ import annotations
import cv2
import numpy as np
import pytesseract
from core.yolo_blocks import DiagramBlock

class Swimlane:
    def __init__(self, id: int, y_top: int, y_bottom: int, x_left: int, x_right: int, name: str=""):
        self.id=id; self.y_top=y_top; self.y_bottom=y_bottom; self.x_left=x_left; self.x_right=x_right; self.name=name

def process_swimlanes(image: np.ndarray, blocks: list[DiagramBlock], vertical_threshold: int = 30, text_search_width: int = 220):
    swim_blocks = [b for b in blocks if b.type.lower() in ("swimline","swimlane","pool","lane")]
    if not swim_blocks:
        return []

    groups = _group_by_height(swim_blocks, vertical_threshold)
    swimlanes=[]
    h,w=image.shape[:2]
    for i, group in enumerate(groups):
        y_top = min(b.bbox[1] for b in group)
        y_bottom = max(b.bbox[3] for b in group)
        name = extract_swimlane_name(image, group, text_search_width)
        swimlanes.append(Swimlane(i, y_top, y_bottom, 0, w, name))

    swimlanes.sort(key=lambda s: s.y_top)
    for i,s in enumerate(swimlanes):
        s.id=i

    for b in blocks:
        cy = (b.bbox[1]+b.bbox[3])//2
        for s in swimlanes:
            if s.y_top <= cy <= s.y_bottom:
                b.swimlane = s.id
                break
    return swimlanes

def _group_by_height(blocks, threshold):
    blocks = sorted(blocks, key=lambda b: (b.bbox[1]+b.bbox[3])//2)
    groups=[]
    cur=[blocks[0]]
    avg=(blocks[0].bbox[1]+blocks[0].bbox[3])//2
    for b in blocks[1:]:
        cy=(b.bbox[1]+b.bbox[3])//2
        if abs(cy-avg) <= threshold:
            cur.append(b)
            avg = int(sum((x.bbox[1]+x.bbox[3])//2 for x in cur)/len(cur))
        else:
            groups.append(cur); cur=[b]; avg=cy
    groups.append(cur)
    return groups

def extract_swimlane_name(image: np.ndarray, swimline_group: list[DiagramBlock], text_search_width: int = 220) -> str:
    if not swimline_group:
        return ""
    y_top = min(b.bbox[1] for b in swimline_group)
    y_bottom = max(b.bbox[3] for b in swimline_group)

    x1=0
    x2=min(text_search_width, image.shape[1])
    roi=image[y_top:y_bottom, x1:x2]
    if roi.size==0:
        return ""

    gray=cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _,thr=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    rot=cv2.rotate(thr, cv2.ROTATE_90_CLOCKWISE)
    rot=cv2.resize(rot, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    try:
        txt=pytesseract.image_to_string(rot, lang="rus+eng", config="--psm 6")
    except Exception:
        return ""
    return " ".join((txt or "").replace("\n"," ").split()).strip()
