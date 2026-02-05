from __future__ import annotations
from core.engines.cv_engine import parse_with_cv
from core.engines.yolo_engine import parse_with_yolo_bpmn, YOLOUnavailable

def parse_image_bytes(image_bytes: bytes, hard_timeout_s: float = 20.0, engine: str = "cv") -> dict:
    engine = (engine or "cv").lower().strip()
    if engine in ("cv","opencv","contours"):
        return parse_with_cv(image_bytes, hard_timeout_s)
    if engine in ("yolo","yolo_bpmn","bpmn"):
        return parse_with_yolo_bpmn(image_bytes, hard_timeout_s)
    raise ValueError(f"Unknown engine: {engine}")
