from __future__ import annotations
import time
from typing import List

import cv2
import numpy as np

from core.preprocess import preprocess
from core.ocr import ocr_nodes
from core.graph_build import build_graph
from core.algorithm import graph_to_algorithm

from core.yolo_blocks import DiagramBlock
from core.yolo_arrow_parser import parse_arrows
from core.swimlane_tools import process_swimlanes

class YOLOUnavailable(RuntimeError):
    pass

_KIND_MAP = {
    "Task": "rectangle",
    "Activity": "rectangle",
    "StartEvent": "ellipse",
    "EndEvent": "ellipse",
    "Gateway": "diamond",
    "ExclusiveGateway": "diamond",
    "ParallelGateway": "diamond",
    "Swimline": "rectangle",
    "Swimlane": "rectangle",
    "Pool": "rectangle",
    "Lane": "rectangle",
}

def parse_with_yolo_bpmn(image_bytes: bytes, hard_timeout_s: float) -> dict:
    t0 = time.time()
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise YOLOUnavailable("ultralytics is not installed. Install: pip install -r requirements-yolo.txt") from e

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Could not decode image.")

    img_p, _bin = preprocess(img); _check(t0, hard_timeout_s)

    model = YOLO("model/best.pt")
    res = model.predict(source=img_p)
    _check(t0, hard_timeout_s)

    blocks = _to_blocks(res, img_p.shape[:2])
    _check(t0, hard_timeout_s)

    swimlanes = process_swimlanes(img_p, blocks)
    _check(t0, hard_timeout_s)

    arrows = parse_arrows(img_p, blocks, proximity_threshold=30)
    _check(t0, hard_timeout_s)

    nodes=[]
    for i,b in enumerate(blocks):
        x1,y1,x2,y2=b.bbox
        cx=(x1+x2)/2.0; cy=(y1+y2)/2.0
        kind=_kind_from_type(b.type)
        semantic=""
        if b.type.lower() in ("startevent","start"):
            semantic="start"
        if b.type.lower() in ("endevent","end"):
            semantic="end"
        nodes.append({
            "id": f"n{i}",
            "kind": kind,
            "semantic": semantic,
            "label": "",
            "bbox": [int(x1),int(y1),int(x2),int(y2)],
            "center": [float(cx), float(cy)],
            "role": str(b.swimlane) if b.swimlane >= 0 else "",
        })

    nodes = ocr_nodes(img_p, nodes); _check(t0, hard_timeout_s)

    edges=[]
    for a in arrows:
        if a.from_box < len(nodes) and a.to_box < len(nodes):
            edges.append({"source": nodes[a.from_box]["id"], "target": nodes[a.to_box]["id"], "kind": "sequence"})

    graph = build_graph(nodes, edges); _check(t0, hard_timeout_s)
    algo = graph_to_algorithm(graph); _check(t0, hard_timeout_s)

    extras = {
        "swimlanes": [{"id": s.id, "name": s.name, "y_top": s.y_top, "y_bottom": s.y_bottom} for s in swimlanes],
        "engine_notes": "yolo_bpmn: blocks via model/best.pt; arrows via hough; swimlanes via yolo+ocr(left strip)"
    }
    return {
        "meta": {"engine": "yolo_bpmn + swimlane + arrow_parser", "hard_timeout_s": hard_timeout_s},
        "graph": graph,
        "algorithm": algo,
        "extras": extras
    }

def _kind_from_type(t: str) -> str:
    if not t:
        return "rectangle"
    return _KIND_MAP.get(t, _KIND_MAP.get(t.capitalize(), "rectangle"))

def _to_blocks(res, shape_hw) -> List[DiagramBlock]:
    blocks=[]
    if not res:
        return blocks
    r = res[0]
    names = getattr(r, "names", {}) or {}
    boxes = getattr(r, "boxes", None)
    if boxes is None:
        return blocks
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    h,w=shape_hw
    for (x1,y1,x2,y2), c in zip(xyxy, cls):
        label = names.get(int(c), str(int(c)))
        x1=max(0,int(x1)); y1=max(0,int(y1))
        x2=min(w-1,int(x2)); y2=min(h-1,int(y2))
        blocks.append(DiagramBlock(type=label, bbox=(x1,y1,x2,y2)))
    blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
    return blocks

def _check(t0: float, hard_timeout_s: float):
    if time.time() - t0 > hard_timeout_s:
        raise TimeoutError(f"Hard timeout exceeded ({hard_timeout_s}s).")
