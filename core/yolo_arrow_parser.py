from __future__ import annotations
import cv2
import numpy as np
from core.yolo_blocks import DiagramBlock

class DiagramArrow:
    def __init__(self, from_box_idx, to_box_idx, start_point, end_point, path):
        self.from_box = from_box_idx
        self.to_box = to_box_idx
        self.from_point = start_point
        self.to_point = end_point
        self.line_points = path

def parse_arrows(image: np.ndarray, blocks: list[DiagramBlock], proximity_threshold=30) -> list[DiagramArrow]:
    return _find_box_connections(image, blocks, proximity_threshold)

def _find_box_connections(image: np.ndarray, blocks: list[DiagramBlock], proximity_threshold=30) -> list[DiagramArrow]:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    mask = bin_img.copy()
    for b in blocks:
        x1,y1,x2,y2 = b.bbox
        pad = 6
        x1=max(0,x1-pad); y1=max(0,y1-pad)
        x2=min(mask.shape[1]-1,x2+pad); y2=min(mask.shape[0]-1,y2+pad)
        cv2.rectangle(mask, (x1,y1), (x2,y2), 0, thickness=-1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    lines = cv2.HoughLinesP(mask, 1, np.pi/180, threshold=80, minLineLength=30, maxLineGap=15)
    if lines is None:
        return []

    segs = [(int(x1),int(y1),int(x2),int(y2)) for x1,y1,x2,y2 in lines[:,0]]

    centers = [((x1+x2)//2, (y1+y2)//2) for (x1,y1,x2,y2) in (b.bbox for b in blocks)]

    connections = []
    for (x1,y1,x2,y2) in segs:
        a = _nearest_box_idx((x1,y1), centers, max_dist=120)
        c = _nearest_box_idx((x2,y2), centers, max_dist=120)
        if a is None or c is None or a == c:
            continue
        if (y1, x1) <= (y2, x2):
            fr, to = a, c
            sp, ep = (x1,y1), (x2,y2)
        else:
            fr, to = c, a
            sp, ep = (x2,y2), (x1,y1)
        connections.append(DiagramArrow(fr, to, sp, ep, [(sp, ep)]))

    uniq = {}
    for con in connections:
        uniq[(con.from_box, con.to_box)] = con
    return list(uniq.values())

def _nearest_box_idx(pt, centers, max_dist=120):
    x,y = pt
    best_i=None
    best_d=1e18
    for i,(cx,cy) in enumerate(centers):
        d=(cx-x)**2+(cy-y)**2
        if d<best_d:
            best_d=d
            best_i=i
    if best_i is None:
        return None
    if (best_d**0.5) > max_dist:
        return None
    return best_i
