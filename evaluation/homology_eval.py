"""
evaluation/homology_eval.py

Homology-Inspired Structural Evaluation — Parallel 2: Biological Homology.

Core insight:
    A human hand, bat wing, whale flipper, and mole's digging claw look nothing
    alike and serve completely different functions.  But strip away the surface
    features and the bone structure is identical: humerus → radius/ulna → carpals
    → five digits.  This is HOMOLOGY: same underlying structure, different surface.

    A Kan extension that transfers causal STRUCTURE across domains should produce
    a graph that is homologous to the ground truth — same "bone structure"
    (degree distribution, relation type mix, connectivity pattern) even if the
    entity labels differ entirely.

This evaluator strips ALL node labels and compares graphs on 12 structural features:
  Degree features:
    1. Mean out-degree (normalised by graph density)
    2. Max out-degree (normalised)
    3. Out-degree entropy (uniformity of causation)
    4. Mean in-degree (normalised)
    5. Max in-degree (normalised)
    6. In-degree entropy
  Graph topology:
    7.  Graph density (edges / (n * (n-1)))
    8.  Largest weakly connected component fraction
    9.  Number of weakly connected components (normalised)
    10. Reciprocity (fraction of edges that have a reverse edge)
  Relation type distribution (5 dims, sum to 1):
    11-15. P(causes), P(reduces), P(increases), P(influences), P(leads_to)

Score: cosine similarity of these feature vectors.
Two structurally homologous graphs score ≈ 1.0 regardless of entity labels.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd

from functor.causal_functor import CausalFunctor


RELATION_TYPES = ["causes", "reduces", "increases", "influences", "leads_to", "affects"]


# ── Feature extraction ─────────────────────────────────────────────────────────

def _safe_entropy(counts: np.ndarray) -> float:
    """Shannon entropy of a count vector (returns 0 for empty/uniform-zero)."""
    total = counts.sum()
    if total < 1e-9:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p + 1e-12)))


def _extract_skeleton_features(G: nx.DiGraph) -> np.ndarray:
    """
    Extract a 16-dimensional structural feature vector, ignoring all node labels.
    Returns np.ndarray shape (16,).
    """
    feat = np.zeros(16, dtype=np.float32)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    if n_nodes == 0:
        return feat

    # ── Degree features ────────────────────────────────────────────────────────
    out_degs = np.array([d for _, d in G.out_degree()], dtype=np.float32)
    in_degs  = np.array([d for _, d in G.in_degree()],  dtype=np.float32)

    mean_out = float(out_degs.mean()) if len(out_degs) > 0 else 0.0
    max_out  = float(out_degs.max())  if len(out_degs) > 0 else 0.0
    mean_in  = float(in_degs.mean())  if len(in_degs)  > 0 else 0.0
    max_in   = float(in_degs.max())   if len(in_degs)  > 0 else 0.0

    # Normalise max degree by sqrt(n) to be size-invariant
    norm = max(np.sqrt(n_nodes), 1.0)
    feat[0] = mean_out / norm
    feat[1] = max_out  / norm
    feat[2] = _safe_entropy(out_degs)
    feat[3] = mean_in  / norm
    feat[4] = max_in   / norm
    feat[5] = _safe_entropy(in_degs)

    # ── Graph topology ─────────────────────────────────────────────────────────
    max_possible = n_nodes * (n_nodes - 1)
    feat[6] = n_edges / max_possible if max_possible > 0 else 0.0

    try:
        wcc     = list(nx.weakly_connected_components(G))
        largest = max(len(c) for c in wcc) / n_nodes
        feat[7]  = largest
        feat[8]  = len(wcc) / n_nodes
    except Exception:
        feat[7] = 1.0
        feat[8] = 1.0 / n_nodes

    # Reciprocity: fraction of edges (u,v) where (v,u) also exists
    if n_edges > 0:
        reciprocal = sum(
            1 for u, v in G.edges() if G.has_edge(v, u)
        )
        feat[9] = reciprocal / n_edges
    else:
        feat[9] = 0.0

    # ── Relation type distribution ─────────────────────────────────────────────
    rel_counts = np.zeros(len(RELATION_TYPES), dtype=np.float32)
    for _, _, d in G.edges(data=True):
        rel = d.get("relation", "affects").lower()
        if rel in RELATION_TYPES:
            rel_counts[RELATION_TYPES.index(rel)] += 1
        else:
            rel_counts[-1] += 1   # bucket into "affects"
    if rel_counts.sum() > 0:
        rel_counts /= rel_counts.sum()
    feat[10:16] = rel_counts

    return feat


def _skeleton_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity of skeleton feature vectors."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def homology_scores(
    predicted: nx.DiGraph,
    ground_truth: nx.DiGraph,
) -> Dict[str, float]:
    """
    Compute homology score: structural feature cosine similarity, ignoring node labels.

    A high score means the predicted graph has the same "bone structure" as GT
    even if all entity names differ.  This is the key insight: cross-domain
    Kan extension should be evaluated on structural preservation, not lexical match.
    """
    v_pred = _extract_skeleton_features(predicted)
    v_gt   = _extract_skeleton_features(ground_truth)
    sim    = _skeleton_similarity(v_pred, v_gt)

    return {
        "homology_score":   sim,
        "pred_density":     float(v_pred[6]),
        "gt_density":       float(v_gt[6]),
        "pred_rel_entropy": _safe_entropy(v_pred[10:16]),
        "gt_rel_entropy":   _safe_entropy(v_gt[10:16]),
        "pred_out_entropy": float(v_pred[2]),
        "gt_out_entropy":   float(v_gt[2]),
    }


# ── Batch evaluation ───────────────────────────────────────────────────────────

def run_homology_evaluation(
    target_queries: List[str],
    ground_truth_functor: CausalFunctor,
    method_preds: Dict[str, Dict[str, nx.DiGraph]],
    out_dir=None,
    _gt_cache: Optional[Dict[str, nx.DiGraph]] = None,
) -> pd.DataFrame:
    """
    Run homology evaluation across all methods and queries.
    Returns pd.DataFrame with one row per (method, query).
    """
    # Pre-compute GT graphs once — avoids re-encoding query 3× (once per method)
    if _gt_cache is None:
        _gt_cache = {q: ground_truth_functor(q) for q in target_queries}

    rows = []
    for method, preds in method_preds.items():
        for q in target_queries:
            gt   = _gt_cache[q]
            pred = preds.get(q, nx.DiGraph())
            scores = homology_scores(pred, gt)
            rows.append({"query": q, "method": method, **scores})

    df = pd.DataFrame(rows)

    print("\n" + "="*70)
    print("HOMOLOGY EVALUATION (Biological Structural Similarity)")
    print("="*70)
    if "homology_score" in df.columns:
        summary = df.groupby("method")["homology_score"].agg(["mean", "std"])
        print(summary.to_string())
    print("="*70 + "\n")

    if out_dir:
        from pathlib import Path
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "homology_evaluation.csv", index=False)
        print(f"[HomologyEval] Saved → {out_dir / 'homology_evaluation.csv'}")

    return df
