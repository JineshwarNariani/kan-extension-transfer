"""
evaluation/structural_motif_eval.py

Structural Motif Evaluation — Parallel 1: Gentner's Structure-Mapping Theory (1983).

Core insight:
    Gentner's experiments showed humans judge good analogies by RELATIONAL SIMILARITY
    (shared structure) rather than surface/object similarity.  A bat wing and a human
    arm are analogous not because they look alike, but because the relation "bone
    supports membrane for locomotion" maps between them.

    Applied to causal graphs: a Kan extension may transfer the WRONG entity labels
    but still preserve the correct relational pattern — a positive feedback loop,
    a cascade, a fan-in convergence.  This evaluator measures how well the motif
    "fingerprint" of the transferred graph matches the ground-truth graph.

Six motifs (Gentner's "systematicity principle" favours higher-order structure):
  1. Positive feedback cycle  — cycle where all edges amplify (causes/increases)
  2. Negative feedback cycle  — cycle with at least one dampening edge (reduces)
  3. Causal cascade           — directed chain of length ≥ 3
  4. Fan-out (divergence)     — node causes multiple independent effects (out-deg ≥ 2)
  5. Fan-in (convergence)     — multiple causes converge to one effect (in-deg ≥ 2)
  6. Bottleneck (amplifier)   — node with in-deg ≥ 2 AND out-deg ≥ 2

Score: cosine similarity of normalised motif count vectors.
A score of 1.0 means the predicted graph has the same structural fingerprint as GT.
A score of 0.0 means completely different structural patterns.

Relation taxonomy:
  AMPLIFYING = {"causes", "increases", "leads_to"}
  DAMPENING  = {"reduces"}
  NEUTRAL    = {"influences", "affects"}
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from functor.causal_functor import CausalFunctor
from functor.embedder import SharedEmbedder


AMPLIFYING = {"causes", "increases", "leads_to"}
DAMPENING  = {"reduces"}


# ── Motif extraction ───────────────────────────────────────────────────────────

def _extract_motif_vector(G: nx.DiGraph) -> np.ndarray:
    """
    Extract a normalised 6-dimensional motif count vector from a causal DiGraph.

    Returns np.ndarray of shape (6,), normalised by graph edge count so that
    graphs of different sizes are comparable.
    """
    n_edges = len(G.edges())
    if n_edges == 0:
        return np.zeros(6, dtype=np.float32)

    counts = np.zeros(6, dtype=np.float32)

    # --- Motif 1 & 2: Cycles ---
    # simple_cycles is exponential; cap at 100 cycles and 20 nodes to stay fast
    if G.number_of_nodes() <= 20:
        try:
            gen = nx.simple_cycles(G)
            cycles = []
            for cyc in gen:
                cycles.append(cyc)
                if len(cycles) >= 100:   # hard cap — prevent exponential blowup
                    break
        except Exception:
            cycles = []
        for cycle in cycles:
            if len(cycle) < 2:
                continue
            # Get the relations along the cycle edges
            rels = []
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                if G.has_edge(u, v):
                    rels.append(G[u][v].get("relation", "affects"))
            has_dampening = any(r in DAMPENING for r in rels)
            if has_dampening:
                counts[1] += 1   # Negative feedback cycle
            else:
                counts[0] += 1   # Positive feedback cycle

    # --- Motif 3: Causal cascade (longest simple path ≥ 3) ---
    # Sample up to 50 random pairs for path length; cap at 5 for speed
    try:
        nodes = list(G.nodes())
        longest = 0
        sample_size = min(50, len(nodes) ** 2)
        for _ in range(sample_size):
            u, v = random.choices(nodes, k=2)
            if u == v:
                continue
            try:
                p = nx.shortest_path_length(G, u, v)
                longest = max(longest, min(p, 5))   # cap at 5 to bound cost
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass
        counts[2] = float(longest)
    except Exception:
        counts[2] = 0.0

    # --- Motifs 4 & 5: Fan-out and Fan-in ---
    for node in G.nodes():
        out_deg = G.out_degree(node)
        in_deg  = G.in_degree(node)
        if out_deg >= 2:
            counts[3] += 1   # Fan-out
        if in_deg >= 2:
            counts[4] += 1   # Fan-in
        if in_deg >= 2 and out_deg >= 2:
            counts[5] += 1   # Bottleneck

    # Normalise by edge count so vectors are comparable across graph sizes
    normalised = counts / n_edges
    return normalised.astype(np.float32)


def _motif_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity between two motif vectors. Returns 0.0 if either is zero."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def motif_scores(
    predicted: nx.DiGraph,
    ground_truth: nx.DiGraph,
) -> Dict[str, float]:
    """
    Compute motif-level structural similarity between predicted and GT graphs.

    Returns dict with:
        motif_similarity      — cosine similarity of motif count vectors
        n_pos_feedback_pred   — positive feedback cycles in predicted
        n_neg_feedback_pred   — negative feedback cycles in predicted
        cascade_depth_pred    — longest causal cascade in predicted
        n_fanout_pred         — fan-out nodes in predicted
        n_fanin_pred          — fan-in nodes in predicted
        n_bottleneck_pred     — bottleneck nodes in predicted
        (same fields for _gt)
    """
    v_pred = _extract_motif_vector(predicted)
    v_gt   = _extract_motif_vector(ground_truth)

    sim = _motif_similarity(v_pred, v_gt)

    def _decode_vec(v: np.ndarray, n_edges: int) -> dict:
        raw = v * max(n_edges, 1)
        return {
            "n_pos_feedback": int(round(float(raw[0]))),
            "n_neg_feedback": int(round(float(raw[1]))),
            "cascade_depth":  int(round(float(raw[2]))),
            "n_fanout":       int(round(float(raw[3]))),
            "n_fanin":        int(round(float(raw[4]))),
            "n_bottleneck":   int(round(float(raw[5]))),
        }

    pred_counts = _decode_vec(v_pred, len(predicted.edges()))
    gt_counts   = _decode_vec(v_gt,   len(ground_truth.edges()))

    return {
        "motif_similarity": sim,
        **{f"{k}_pred": v for k, v in pred_counts.items()},
        **{f"{k}_gt":   v for k, v in gt_counts.items()},
    }


# ── Batch evaluation ───────────────────────────────────────────────────────────

def run_motif_evaluation(
    target_queries: List[str],
    ground_truth_functor: CausalFunctor,
    method_preds: Dict[str, Dict[str, nx.DiGraph]],
    out_dir=None,
    _gt_cache: Optional[Dict[str, nx.DiGraph]] = None,
) -> pd.DataFrame:
    """
    Run motif evaluation across all methods and queries.

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
            scores = motif_scores(pred, gt)
            rows.append({"query": q, "method": method, **scores})

    df = pd.DataFrame(rows)

    print("\n" + "="*70)
    print("STRUCTURAL MOTIF EVALUATION (Gentner Structure-Mapping)")
    print("="*70)
    if "motif_similarity" in df.columns:
        summary = df.groupby("method")["motif_similarity"].agg(["mean", "std"])
        print(summary.to_string())
    print("="*70 + "\n")

    if out_dir:
        from pathlib import Path
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "motif_evaluation.csv", index=False)
        print(f"[MotifEval] Saved → {out_dir / 'motif_evaluation.csv'}")

    return df
