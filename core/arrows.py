from __future__ import annotations
import cv2
import numpy as np

def detect_arrows(img_bgr, bin_img, nodes):
    mask = bin_img.copy()
    for n in nodes:
        x1, y1, x2, y2 = n["bbox"]
        pad = 8
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(mask.shape[1]-1, x2 + pad); y2 = min(mask.shape[0]-1, y2 + pad)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 0, thickness=-1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    lines = cv2.HoughLinesP(mask, 1, np.pi/180, threshold=60, minLineLength=25, maxLineGap=12)
    if lines is None:
        return []

    node_boxes = [(n["id"], n["bbox"], n["center"]) for n in nodes]
    segs = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in lines[:, 0]]

    edges = []
    for (x1, y1, x2, y2) in segs:
        a = _nearest_node((x1, y1), node_boxes)
        b = _nearest_node((x2, y2), node_boxes)
        if a is None or b is None or a == b:
            continue
        src, dst = (a, b) if (y1, x1) <= (y2, x2) else (b, a)
        edges.append({"source": src, "target": dst, "kind": "sequence"})

    uniq = {}
    for e in edges:
        uniq[(e["source"], e["target"])] = e
    return list(uniq.values())

def _nearest_node(pt, node_boxes, max_dist=80):
    x, y = pt
    best = None
    best_d = 1e18
    for nid, bbox, center in node_boxes:
        cx, cy = center
        d = (cx - x) ** 2 + (cy - y) ** 2
        if d < best_d:
            best_d = d
            best = nid
    if best is None:
        return None
    if (best_d ** 0.5) > max_dist:
        return None
    return best
