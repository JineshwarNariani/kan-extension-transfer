"""
evaluation/metrics.py

Three-level evaluation of causal graph prediction quality.

Level 1 — Edge match (exact string):  (subj, obj) must match exactly
Level 2 — Relation match:             (subj, rel, obj) must match exactly
Level 3 — Soft node match:            nodes match if cosine_sim > threshold
           (handles paraphrase: "blood pressure" vs "arterial hypertension")
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np


def _f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate(
    predicted: nx.DiGraph,
    ground_truth: nx.DiGraph,
    embedder=None,
    semantic_threshold: float = 0.80,
) -> Dict[str, float]:
    """
    Evaluate predicted causal graph against ground truth.

    Returns a dict with:
        edge_precision, edge_recall, edge_f1
        relation_precision, relation_recall, relation_f1
        node_jaccard
        semantic_node_recall  (None if embedder not provided)
        pred_size, true_size
    """
    pred_edges = set(predicted.edges())
    true_edges = set(ground_truth.edges())

    # ── Level 1: Edge F1 ──────────────────────────────────────────────────────
    tp_e = len(pred_edges & true_edges)
    fp_e = len(pred_edges - true_edges)
    fn_e = len(true_edges - pred_edges)
    edge_p, edge_r, edge_f1 = _f1(tp_e, fp_e, fn_e)

    # ── Level 2: Relation-typed edge F1 ──────────────────────────────────────
    pred_rel = {
        (u, v, d.get("relation", ""))
        for u, v, d in predicted.edges(data=True)
    }
    true_rel = {
        (u, v, d.get("relation", ""))
        for u, v, d in ground_truth.edges(data=True)
    }
    tp_r = len(pred_rel & true_rel)
    fp_r = len(pred_rel - true_rel)
    fn_r = len(true_rel - pred_rel)
    rel_p, rel_r, rel_f1 = _f1(tp_r, fp_r, fn_r)

    # ── Level 3: Node metrics ──────────────────────────────────────────────────
    pred_nodes = set(predicted.nodes())
    true_nodes = set(ground_truth.nodes())
    union_size = len(pred_nodes | true_nodes)
    node_jaccard = len(pred_nodes & true_nodes) / union_size if union_size > 0 else 0.0

    # Soft node recall: does each true node have a semantically close predicted node?
    semantic_recall: Optional[float] = None
    if embedder is not None and pred_nodes and true_nodes:
        semantic_recall = _soft_node_recall(
            pred_nodes, true_nodes, embedder, semantic_threshold
        )

    return {
        "edge_precision":        edge_p,
        "edge_recall":           edge_r,
        "edge_f1":               edge_f1,
        "relation_precision":    rel_p,
        "relation_recall":       rel_r,
        "relation_f1":           rel_f1,
        "node_jaccard":          node_jaccard,
        "semantic_node_recall":  semantic_recall,
        "pred_size":             len(pred_edges),
        "true_size":             len(true_edges),
        "pred_nodes":            len(pred_nodes),
        "true_nodes":            len(true_nodes),
    }


def _soft_node_recall(
    pred_nodes: Set[str],
    true_nodes: Set[str],
    embedder,
    threshold: float,
) -> float:
    """
    For each true node, check if any predicted node is within cosine distance threshold.
    Returns fraction of true nodes with a soft match.
    """
    pred_list = list(pred_nodes)
    true_list = list(true_nodes)

    pred_embs = embedder.encode(pred_list)   # (M, 384)
    true_embs = embedder.encode(true_list)   # (N, 384)

    # sim_matrix[i, j] = cos_sim(true[i], pred[j])
    sim_matrix = embedder.cosine_similarity_matrix(true_embs, pred_embs)  # (N, M)
    # A true node is "matched" if max sim to any pred node >= threshold
    max_sims = sim_matrix.max(axis=1)  # (N,)
    matched  = int((max_sims >= threshold).sum())
    return matched / len(true_list)


def summarise_results(rows: List[Dict]) -> Dict[str, float]:
    """Average metrics across multiple query results."""
    if not rows:
        return {}
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float)) and k not in ("pred_size", "true_size")]
    summary = {}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        summary[f"mean_{k}"] = float(np.mean(vals)) if vals else 0.0
        summary[f"std_{k}"]  = float(np.std(vals))  if vals else 0.0
    return summary
