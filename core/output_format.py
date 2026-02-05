from __future__ import annotations
from typing import List
from core.text_utils import normalize_step_text, is_good_step_text, dedupe_steps

def build_output(graph: dict, algorithm: dict) -> dict:
    ordered = _ordered_step_texts(graph, algorithm)

    minimal_steps = [{"step": i + 1, "description": t} for i, t in enumerate(ordered)]
    bpmn_steps = [{"step": i + 1, "action": t, "role": ""} for i, t in enumerate(ordered)]

    return {
        "minimal": {"schema": "step_table_v1", "steps": minimal_steps},
        "bpmn": {"schema": "bpmn_table_v1", "steps": bpmn_steps},
    }

def _ordered_step_texts(graph: dict, algorithm: dict) -> List[str]:
    nodes = graph.get("nodes", []) or []
    edges = graph.get("edges", []) or []

    # 1) traversal-based
    steps = algorithm.get("steps") or []
    traversal: list[str] = []
    for s in steps:
        t = normalize_step_text(s.get("text") or "")
        if t and t.lower() not in ("rectangle", "ellipse", "diamond", "node"):
            if is_good_step_text(t):
                traversal.append(t)
    traversal = dedupe_steps(traversal)

    # 2) spatial fallback
    spatial = _spatial_order_texts(nodes)

    graph_weak = len(edges) == 0
    traversal_too_short = len(traversal) < max(3, min(8, (len(spatial) // 2) if spatial else 3))

    if graph_weak or traversal_too_short:
        return spatial if spatial else traversal
    return traversal if traversal else spatial

def _spatial_order_texts(nodes: list[dict]) -> List[str]:
    labeled = []
    for n in nodes:
        t = normalize_step_text(n.get("label") or "")
        if not t:
            continue
        if not is_good_step_text(t):
            continue
        labeled.append((n, t))

    if not labeled:
        return []

    def key(item):
        n, _t = item
        cx, cy = n.get("center", [0, 0])
        ybin = int(cy // 70)
        return (ybin, cx)

    labeled.sort(key=key)
    texts = [t for _, t in labeled]
    return dedupe_steps(texts)
