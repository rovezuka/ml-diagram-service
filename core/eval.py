from __future__ import annotations
from typing import Dict, List, Tuple
from rapidfuzz import fuzz
import re

def parse_ground_truth_txt(text: str) -> Dict[str, List[dict]]:
    lines = [l.rstrip("\n") for l in text.splitlines()]
    blocks: Dict[str, List[dict]] = {"__global__": []}
    current = "__global__"

    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("###"):
            current = s[3:].strip()
            blocks.setdefault(current, [])
            continue

        s = re.sub(r"^\s*\d+\s*[\.\)\-]*\s*", "", s)
        desc, role = _split_desc_role(s)
        if desc:
            blocks[current].append({"description": desc, "role": role})

    if blocks.get("__global__") == [] and len(blocks) > 1:
        blocks.pop("__global__", None)
    return blocks

def evaluate_predictions(preds_map: Dict[str, List[dict]], gt_map: Dict[str, List[dict]]) -> dict:
    per_file = []
    agg = {"tp":0,"fp":0,"fn":0,
           "sum_score":0.0,"sum_matched":0,
           "sum_order_ok":0,"sum_order_total":0,
           "sum_role_ok":0,"sum_role_total":0}

    for fname, pred_steps in preds_map.items():
        gt_steps = gt_map.get(fname) or gt_map.get("__global__") or []
        m = _eval_one(pred_steps, gt_steps)
        per_file.append({"file": fname, **m})

        agg["tp"] += m["tp"]; agg["fp"] += m["fp"]; agg["fn"] += m["fn"]
        agg["sum_score"] += m["avg_match_score"] * m["matched"]
        agg["sum_matched"] += m["matched"]
        agg["sum_order_ok"] += m["order_ok"]; agg["sum_order_total"] += m["order_total"]
        agg["sum_role_ok"] += m["role_ok"]; agg["sum_role_total"] += m["role_total"]

    precision = agg["tp"]/(agg["tp"]+agg["fp"]) if (agg["tp"]+agg["fp"]) else 0.0
    recall = agg["tp"]/(agg["tp"]+agg["fn"]) if (agg["tp"]+agg["fn"]) else 0.0
    f1 = (2*precision*recall)/(precision+recall) if (precision+recall) else 0.0
    avg_score = (agg["sum_score"]/agg["sum_matched"]) if agg["sum_matched"] else 0.0
    order_acc = (agg["sum_order_ok"]/agg["sum_order_total"]) if agg["sum_order_total"] else 0.0
    role_acc = (agg["sum_role_ok"]/agg["sum_role_total"]) if agg["sum_role_total"] else 0.0

    return {"meta":{"threshold":70},
            "summary":{"step_precision":round(precision,4),
                       "step_recall":round(recall,4),
                       "step_f1@70":round(f1,4),
                       "avg_match_score":round(avg_score,2),
                       "order_accuracy":round(order_acc,4),
                       "role_accuracy@70":round(role_acc,4)},
            "per_file": per_file}

def _eval_one(pred: List[dict], gt: List[dict]) -> dict:
    pred_desc = [_norm(p.get("action") or p.get("description") or "") for p in pred]
    gt_desc = [_norm(g.get("description") or "") for g in gt]

    used_gt = set()
    matches: List[Tuple[int,int,int]] = []

    for pi, p in enumerate(pred_desc):
        best_gi, best_score = -1, -1
        for gi, g in enumerate(gt_desc):
            if gi in used_gt:
                continue
            score = fuzz.token_sort_ratio(p, g)
            if score > best_score:
                best_score, best_gi = score, gi
        if best_score >= 70 and best_gi != -1:
            used_gt.add(best_gi)
            matches.append((pi, best_gi, int(best_score)))

    tp = len(matches)
    fp = max(0, len(pred_desc) - tp)
    fn = max(0, len(gt_desc) - tp)
    avg_score = (sum(s for _,_,s in matches) / tp) if tp else 0.0

    m = min(len(pred_desc), len(gt_desc))
    order_ok = sum(1 for i in range(m) if pred_desc[i] == gt_desc[i])

    role_ok = 0
    role_total = 0
    for pi, gi, _ in matches:
        gt_role = (gt[gi].get("role") or "").strip()
        if not gt_role:
            continue
        role_total += 1
        pred_role = (pred[pi].get("role") or "").strip()
        if fuzz.token_sort_ratio(_norm(pred_role), _norm(gt_role)) >= 70:
            role_ok += 1

    return {"tp":tp,"fp":fp,"fn":fn,
            "matched":tp,"avg_match_score":round(avg_score,2),
            "order_ok":order_ok,"order_total":m,
            "role_ok":role_ok,"role_total":role_total,
            "pred_len":len(pred_desc),"gt_len":len(gt_desc)}

def _split_desc_role(s: str) -> tuple[str,str]:
    if "|" in s:
        parts = [p.strip() for p in s.split("|") if p.strip()]
        if len(parts) == 1:
            return parts[0], ""
        return " | ".join(parts[:-1]).strip(), parts[-1].strip()
    if "\t" in s:
        parts = [p.strip() for p in s.split("\t") if p.strip()]
        if len(parts) == 1:
            return parts[0], ""
        return parts[0], parts[1]
    return s.strip(), ""

def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    return " ".join(s.split())
