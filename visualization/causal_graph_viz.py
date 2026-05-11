"""
visualization/causal_graph_viz.py

Side-by-side comparison of ground-truth vs Kan-predicted causal graphs.
Color coding:
    green  = true positive edges  (in both pred and truth)
    red    = false positive edges (in pred only)
    gray   = false negative edges (in truth only)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx


def draw_comparison(
    query: str,
    gt_graph: nx.DiGraph,
    predicted: Dict[str, nx.DiGraph],
    out_path: Optional[Path] = None,
    max_nodes: int = 30,
) -> None:
    """
    Plot ground truth alongside all predicted graphs.
    predicted = {"left_kan": G, "right_kan": G, "naive_rag": G}
    """
    methods = list(predicted.keys())
    n_cols  = len(methods) + 1   # +1 for ground truth
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 7))

    # Draw ground truth
    _draw_graph(axes[0], gt_graph, title="Ground Truth\n(F_econ)",
                node_color="#4a90d9", max_nodes=max_nodes)

    # Draw each predicted graph with TP/FP coloring
    for ax, method in zip(axes[1:], methods):
        pred_graph = predicted[method]
        _draw_with_tpfp(ax, pred_graph, gt_graph,
                        title=_method_label(method), max_nodes=max_nodes)

    legend_elements = [
        mpatches.Patch(color="#2ecc71", label="True positive"),
        mpatches.Patch(color="#e74c3c", label="False positive"),
        mpatches.Patch(color="#bdc3c7", label="False negative"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=3, fontsize=10, frameon=False)

    title = f"Causal Transfer: {query[:80]}"
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[Viz] Saved → {out_path}")
    plt.close(fig)


def _draw_graph(ax, G: nx.DiGraph, title: str, node_color: str, max_nodes: int):
    if len(G.nodes()) > max_nodes:
        # Keep highest-degree nodes
        top_nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:max_nodes]
        G = G.subgraph(top_nodes).copy()

    pos = nx.spring_layout(G, seed=42, k=2.0)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_color,
                           node_size=600, alpha=0.85)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=6)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#555555",
                           arrows=True, arrowsize=12,
                           connectionstyle="arc3,rad=0.1")

    edge_labels = {(u, v): d.get("relation", "")[:4]
                   for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 ax=ax, font_size=5)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")


def _draw_with_tpfp(ax, pred: nx.DiGraph, truth: nx.DiGraph, title: str, max_nodes: int):
    # Merge nodes for layout
    all_nodes = set(pred.nodes()) | set(truth.nodes())
    if len(all_nodes) > max_nodes:
        # Prefer TP nodes
        tp_nodes = set(pred.nodes()) & set(truth.nodes())
        remaining = list((set(pred.nodes()) | set(truth.nodes())) - tp_nodes)
        all_nodes = tp_nodes | set(remaining[: max_nodes - len(tp_nodes)])

    merged = nx.DiGraph()
    merged.add_nodes_from(all_nodes)
    pos = nx.spring_layout(merged, seed=42, k=2.0)

    pred_edges  = set(pred.edges())
    truth_edges = set(truth.edges())

    tp_edges = [(u, v) for u, v in pred_edges  if (u, v) in truth_edges and u in all_nodes and v in all_nodes]
    fp_edges = [(u, v) for u, v in pred_edges  if (u, v) not in truth_edges and u in all_nodes and v in all_nodes]
    fn_edges = [(u, v) for u, v in truth_edges if (u, v) not in pred_edges and u in all_nodes and v in all_nodes]

    nx.draw_networkx_nodes(merged, pos, ax=ax, node_color="#f0f0f0",
                           node_size=500, alpha=0.9)
    nx.draw_networkx_labels(merged, pos, ax=ax, font_size=6)

    for edges, color, style in [
        (tp_edges, "#2ecc71", "solid"),
        (fp_edges, "#e74c3c", "solid"),
        (fn_edges, "#bdc3c7", "dashed"),
    ]:
        if edges:
            nx.draw_networkx_edges(merged, pos, edgelist=edges, ax=ax,
                                   edge_color=color, style=style,
                                   arrows=True, arrowsize=12,
                                   connectionstyle="arc3,rad=0.1")

    tp = len(tp_edges); fp = len(fp_edges); fn = len(fn_edges)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0

    ax.set_title(f"{title}\nP={prec:.2f} R={rec:.2f} F1={f1:.2f}",
                 fontsize=10, fontweight="bold")
    ax.axis("off")


def _method_label(method: str) -> str:
    return {
        "left_kan":  "Left Kan (Lan)\nColimit / Optimistic",
        "right_kan": "Right Kan (Ran)\nLimit / Conservative",
        "naive_rag": "Naive RAG\nBaseline",
    }.get(method, method)
