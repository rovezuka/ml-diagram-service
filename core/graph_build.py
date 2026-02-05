from __future__ import annotations
import networkx as nx

def build_graph(nodes, edges):
    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n["id"], **n)
    for e in edges:
        if e["source"] in G and e["target"] in G:
            G.add_edge(e["source"], e["target"], **e)

    out_nodes = []
    for nid, data in G.nodes(data=True):
        out_nodes.append({
            "id": nid,
            "kind": data.get("kind","rectangle"),
            "semantic": data.get("semantic",""),
            "label": data.get("label",""),
            "bbox": data.get("bbox"),
            "center": data.get("center"),
        })

    out_edges = []
    for u, v, data in G.edges(data=True):
        out_edges.append({"source": u, "target": v, "kind": data.get("kind","sequence")})

    return {"nodes": out_nodes, "edges": out_edges}
