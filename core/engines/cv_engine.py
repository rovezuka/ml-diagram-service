from __future__ import annotations
import time
import cv2
import numpy as np

from core.preprocess import preprocess
from core.shapes import detect_shapes
from core.arrows import detect_arrows
from core.ocr import ocr_nodes
from core.graph_build import build_graph
from core.algorithm import graph_to_algorithm

def parse_with_cv(image_bytes: bytes, hard_timeout_s: float) -> dict:
    t0 = time.time()
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Could not decode image.")

    img_p, bin_img = preprocess(img); _check(t0, hard_timeout_s)
    nodes = detect_shapes(img_p, bin_img); _check(t0, hard_timeout_s)
    edges = detect_arrows(img_p, bin_img, nodes); _check(t0, hard_timeout_s)
    nodes = ocr_nodes(img_p, nodes); _check(t0, hard_timeout_s)
    graph = build_graph(nodes, edges); _check(t0, hard_timeout_s)
    algo = graph_to_algorithm(graph); _check(t0, hard_timeout_s)

    return {
        "meta": {"engine": "opencv+contours + tesseract-ocr + rules", "hard_timeout_s": hard_timeout_s},
        "graph": graph,
        "algorithm": algo,
        "extras": {}
    }

def _check(t0: float, hard_timeout_s: float):
    if time.time() - t0 > hard_timeout_s:
        raise TimeoutError(f"Hard timeout exceeded ({hard_timeout_s}s).")
