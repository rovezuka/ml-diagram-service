from __future__ import annotations
import networkx as nx

def graph_to_algorithm(graph_json: dict) -> dict:
    G = nx.DiGraph()
    for n in graph_json["nodes"]:
        G.add_node(n["id"], **n)
    for e in graph_json["edges"]:
        G.add_edge(e["source"], e["target"], **e)

    starts = [n for n, d in G.nodes(data=True) if d.get("semantic") == "start"]
    if not starts:
        starts = [n for n in G.nodes() if G.in_degree(n) == 0]
    if not starts and G.nodes:
        starts = [next(iter(G.nodes()))]

    visited, steps, lines = set(), [], []
    if not starts:
        return {"steps": [], "pseudocode": "", "start": None, "unvisited": []}

    def label(nid):
        d = G.nodes[nid]
        t = (d.get("label") or "").strip()
        return t if t else d.get("kind","node")

    def dfs(nid, indent=0):
        if nid in visited:
            return
        visited.add(nid)
        kind = G.nodes[nid].get("kind","rectangle")
        text = label(nid)

        if kind == "diamond":
            outs = list(G.successors(nid))
            steps.append({"type":"decision","id":nid,"text":text,"next":outs})
            lines.append(" " * indent + f"IF {text}:")
            if outs:
                dfs(outs[0], indent+2)
                if len(outs) > 1:
                    lines.append(" " * indent + "ELSE:")
                    dfs(outs[1], indent+2)
            lines.append(" " * indent + "END_IF")
        else:
            steps.append({"type":"step","id":nid,"text":text})
            lines.append(" " * indent + f"- {text}")
            for nxt in list(G.successors(nid)):
                dfs(nxt, indent)

    dfs(starts[0], 0)
    return {
        "steps": steps,
        "pseudocode": "\n".join(lines),
        "start": starts[0],
        "unvisited": [n for n in G.nodes() if n not in visited]
    }
