from __future__ import annotations
import cv2

def detect_shapes(img_bgr, bin_img):
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    nodes = []
    h, w = bin_img.shape[:2]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 700:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw * bh > 0.95 * w * h:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        kind = "rectangle"
        if len(approx) == 4:
            rect = cv2.minAreaRect(cnt)
            (_, _), (_, _), angle = rect
            if 20 < abs(angle) < 70:
                kind = "diamond"
        elif len(approx) > 6:
            kind = "ellipse"

        center = (x + bw / 2.0, y + bh / 2.0)
        nodes.append({
            "id": "tmp",
            "kind": kind,
            "bbox": [int(x), int(y), int(x + bw), int(y + bh)],
            "center": [float(center[0]), float(center[1])],
        })

    nodes.sort(key=lambda n: (n["bbox"][1], n["bbox"][0]))
    for i, n in enumerate(nodes):
        n["id"] = f"n{i}"
    return nodes
