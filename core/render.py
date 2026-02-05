from __future__ import annotations
import re

def text_to_mermaid(text: str) -> str:
    text = text.strip()
    if not text:
        return "flowchart TD\n  A[Empty]"

    parts = [p.strip() for p in text.split(";") if p.strip()]
    nodes, edges = {}, []

    def nid(label):
        key = re.sub(r"[^a-zA-Z0-9_]+","_",label.strip())[:30].strip("_") or "N"
        if key not in nodes:
            nodes[key] = label.strip()
        return key

    for p in parts:
        seq = [s.strip() for s in p.split("->") if s.strip()]
        for a, b in zip(seq, seq[1:]):
            cond = ""
            m = re.match(r"^\(([^)]+)\)\s*(.*)$", b)
            if m:
                cond = m.group(1).strip()
                b = m.group(2).strip()
            edges.append((nid(a), nid(b), cond))

    out = ["flowchart TD"]
    for k, v in nodes.items():
        out.append(f"  {k}[{_esc(v)}]")
    for a, b, c in edges:
        out.append(f"  {a} -- {_esc(c)} --> {b}" if c else f"  {a} --> {b}")
    return "\n".join(out)

def _esc(s: str) -> str:
    return s.replace("[","(").replace("]",")")
