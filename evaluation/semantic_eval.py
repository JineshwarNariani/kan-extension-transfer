"""
evaluation/semantic_eval.py

Semantic Soft-F1 for cross-domain causal graph evaluation.

Motivation (Parallel 1 / BERTScore insight):
    Exact-match F1 = 0 is guaranteed for cross-domain transfer because
    economic ground-truth triples use economic vocabulary and source-domain
    Kan predictions use source-domain vocabulary.  Zero string overlap does
    not mean zero semantic overlap.

    This module replaces the exact-match judge with an embedding-similarity
    judge: each (subj, rel, obj) triple is encoded as a sentence, and
    precision / recall / F1 are computed via maximum cosine similarity
    rather than exact set intersection.

Two variants:
    Greedy soft-F1 (BERTScore-style):
        - soft_precision  = mean over predicted triples of max_sim(pred→GT)
        - soft_recall     = mean over GT triples of max_sim(GT→pred)
        - soft_f1         = harmonic mean of precision and recall
        - Each triple may "match" multiple counterparts (no one-to-one constraint).

    Hungarian soft-F1 (strict):
        - Uses scipy.optimize.linear_sum_assignment to find the globally
          optimal one-to-one assignment maximising total similarity.
        - Stricter: each predicted triple is matched to at most one GT triple.

Both return 0.0 gracefully for empty graphs.  A similarity threshold
(default 0.4) gates partial credit: pairs below threshold contribute 0.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from functor.embedder import SharedEmbedder


# ── Triple helpers ─────────────────────────────────────────────────────────────

def _graph_to_triple_texts(G: nx.DiGraph) -> List[str]:
    """Return one sentence per edge: 'subj rel obj'."""
    texts = []
    for u, v, d in G.edges(data=True):
        rel  = d.get("relation", "affects")
        texts.append(f"{u} {rel} {v}")
    return texts


def _graph_to_triples(G: nx.DiGraph) -> List[Tuple[str, str, str]]:
    """Return list of (subj, rel, obj) tuples."""
    return [
        (u, v, d.get("relation", "affects"))
        for u, v, d in G.edges(data=True)
    ]


# ── Core metric functions ──────────────────────────────────────────────────────

def soft_edge_f1_greedy(
    predicted: nx.DiGraph,
    ground_truth: nx.DiGraph,
    embedder: Optional[SharedEmbedder] = None,
    sim_threshold: float = 0.40,
) -> Dict[str, float]:
    """
    BERTScore-style soft F1: max-similarity matching, no one-to-one constraint.

    Each predicted triple is matched to the nearest GT triple;
    each GT triple is matched to the nearest predicted triple.
    Similarity below sim_threshold contributes 0 (no partial credit for unrelated triples).
    """
    if embedder is None:
        embedder = SharedEmbedder.get()

    pred_texts = _graph_to_triple_texts(predicted)
    gt_texts   = _graph_to_triple_texts(ground_truth)

    if not pred_texts and not gt_texts:
        return _zero_metrics()
    if not pred_texts:
        return {"soft_precision": 0.0, "soft_recall": 0.0, "soft_f1": 0.0,
                "pred_size": 0, "true_size": len(gt_texts)}
    if not gt_texts:
        return {"soft_precision": 0.0, "soft_recall": 0.0, "soft_f1": 0.0,
                "pred_size": len(pred_texts), "true_size": 0}

    pred_embs = embedder.encode(pred_texts)   # (P, 384)
    gt_embs   = embedder.encode(gt_texts)     # (G, 384)

    # sim_matrix[i, j] = cos_sim(pred[i], gt[j])
    sim_matrix = embedder.cosine_similarity_matrix(pred_embs, gt_embs)   # (P, G)

    # Soft precision: for each predicted triple, how similar is its best GT match?
    max_sim_pred_to_gt = sim_matrix.max(axis=1)          # (P,)
    thresholded_p      = np.maximum(0.0, max_sim_pred_to_gt - sim_threshold) / (1 - sim_threshold)
    soft_precision     = float(thresholded_p.mean())

    # Soft recall: for each GT triple, how similar is its best predicted match?
    max_sim_gt_to_pred = sim_matrix.max(axis=0)          # (G,)
    thresholded_r      = np.maximum(0.0, max_sim_gt_to_pred - sim_threshold) / (1 - sim_threshold)
    soft_recall        = float(thresholded_r.mean())

    soft_f1 = _harmonic_mean(soft_precision, soft_recall)

    return {
        "soft_precision": soft_precision,
        "soft_recall":    soft_recall,
        "soft_f1":        soft_f1,
        "pred_size":      len(pred_texts),
        "true_size":      len(gt_texts),
    }


def soft_edge_f1_hungarian(
    predicted: nx.DiGraph,
    ground_truth: nx.DiGraph,
    embedder: Optional[SharedEmbedder] = None,
    sim_threshold: float = 0.40,
) -> Dict[str, float]:
    """
    Hungarian-matched soft F1: globally optimal one-to-one assignment.

    Stricter than greedy: each predicted triple is matched to at most one GT triple.
    Uses scipy.optimize.linear_sum_assignment (Kuhn-Munkres algorithm).
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        # Fallback: greedy matching
        return soft_edge_f1_greedy(predicted, ground_truth, embedder, sim_threshold)

    if embedder is None:
        embedder = SharedEmbedder.get()

    pred_texts = _graph_to_triple_texts(predicted)
    gt_texts   = _graph_to_triple_texts(ground_truth)

    if not pred_texts and not gt_texts:
        return _zero_metrics()
    if not pred_texts:
        return {"hungarian_precision": 0.0, "hungarian_recall": 0.0,
                "hungarian_f1": 0.0, "pred_size": 0, "true_size": len(gt_texts)}
    if not gt_texts:
        return {"hungarian_precision": 0.0, "hungarian_recall": 0.0,
                "hungarian_f1": 0.0, "pred_size": len(pred_texts), "true_size": 0}

    pred_embs = embedder.encode(pred_texts)
    gt_embs   = embedder.encode(gt_texts)
    sim_matrix = embedder.cosine_similarity_matrix(pred_embs, gt_embs)   # (P, G)

    # Solve assignment on the smaller axis (cost = 1 - similarity)
    P, G = sim_matrix.shape
    if P <= G:
        row_ind, col_ind = linear_sum_assignment(1.0 - sim_matrix)
        matched_sims = sim_matrix[row_ind, col_ind]
        # Precision: matched predicted triples / total predicted
        above = matched_sims[matched_sims >= sim_threshold]
        hungarian_precision = float(above.mean()) if len(above) > 0 else 0.0
        # Recall: matched GT triples / total GT (unmatched GT triples = 0 recall)
        above_r = np.zeros(G)
        above_r[col_ind] = np.where(matched_sims >= sim_threshold, matched_sims, 0.0)
        hungarian_recall = float(above_r.mean())
    else:
        # More GT than predicted: transpose
        col_ind, row_ind = linear_sum_assignment(1.0 - sim_matrix.T)
        matched_sims = sim_matrix[row_ind, col_ind]
        above = matched_sims[matched_sims >= sim_threshold]
        hungarian_recall    = float(above.mean()) if len(above) > 0 else 0.0
        above_p = np.zeros(P)
        above_p[row_ind] = np.where(matched_sims >= sim_threshold, matched_sims, 0.0)
        hungarian_precision = float(above_p.mean())

    hungarian_f1 = _harmonic_mean(hungarian_precision, hungarian_recall)

    return {
        "hungarian_precision": hungarian_precision,
        "hungarian_recall":    hungarian_recall,
        "hungarian_f1":        hungarian_f1,
        "pred_size":           len(pred_texts),
        "true_size":           len(gt_texts),
    }


def evaluate_semantic(
    predicted: nx.DiGraph,
    ground_truth: nx.DiGraph,
    embedder: Optional[SharedEmbedder] = None,
    sim_threshold: float = 0.40,
    _pred_embs: Optional[np.ndarray] = None,
    _gt_embs:   Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Combined semantic evaluation: both greedy and Hungarian soft-F1.
    Encodes predicted and GT triple texts ONCE and passes embeddings to both variants.
    """
    if embedder is None:
        embedder = SharedEmbedder.get()

    pred_texts = _graph_to_triple_texts(predicted)
    gt_texts   = _graph_to_triple_texts(ground_truth)

    if not pred_texts and not gt_texts:
        return _zero_metrics()

    # Encode once, reuse in both greedy and Hungarian
    if _pred_embs is None and pred_texts:
        _pred_embs = embedder.encode(pred_texts)
    if _gt_embs is None and gt_texts:
        _gt_embs = embedder.encode(gt_texts)

    result = {}

    # ── Greedy soft-F1 ─────────────────────────────────────────────────────────
    if not pred_texts:
        result.update({"soft_precision": 0.0, "soft_recall": 0.0, "soft_f1": 0.0})
    elif not gt_texts:
        result.update({"soft_precision": 0.0, "soft_recall": 0.0, "soft_f1": 0.0})
    else:
        sim_matrix = embedder.cosine_similarity_matrix(_pred_embs, _gt_embs)
        max_sim_p = sim_matrix.max(axis=1)
        max_sim_r = sim_matrix.max(axis=0)
        thr_p = np.maximum(0.0, max_sim_p - sim_threshold) / (1 - sim_threshold)
        thr_r = np.maximum(0.0, max_sim_r - sim_threshold) / (1 - sim_threshold)
        sp = float(thr_p.mean())
        sr = float(thr_r.mean())
        result["soft_precision"] = sp
        result["soft_recall"]    = sr
        result["soft_f1"]        = _harmonic_mean(sp, sr)

    # ── Hungarian soft-F1 ──────────────────────────────────────────────────────
    if not pred_texts or not gt_texts:
        result.update({"hungarian_precision": 0.0, "hungarian_recall": 0.0,
                        "hungarian_f1": 0.0})
    else:
        try:
            from scipy.optimize import linear_sum_assignment
            sim_matrix = embedder.cosine_similarity_matrix(_pred_embs, _gt_embs)
            P, G = sim_matrix.shape
            if P <= G:
                row_ind, col_ind = linear_sum_assignment(1.0 - sim_matrix)
                matched = sim_matrix[row_ind, col_ind]
                above   = matched[matched >= sim_threshold]
                hp      = float(above.mean()) if len(above) > 0 else 0.0
                above_r = np.zeros(G)
                above_r[col_ind] = np.where(matched >= sim_threshold, matched, 0.0)
                hr = float(above_r.mean())
            else:
                col_ind, row_ind = linear_sum_assignment(1.0 - sim_matrix.T)
                matched = sim_matrix[row_ind, col_ind]
                above   = matched[matched >= sim_threshold]
                hr      = float(above.mean()) if len(above) > 0 else 0.0
                above_p = np.zeros(P)
                above_p[row_ind] = np.where(matched >= sim_threshold, matched, 0.0)
                hp = float(above_p.mean())
            result["hungarian_precision"] = hp
            result["hungarian_recall"]    = hr
            result["hungarian_f1"]        = _harmonic_mean(hp, hr)
        except ImportError:
            result.update({"hungarian_precision": result.get("soft_precision", 0.0),
                            "hungarian_recall":    result.get("soft_recall",    0.0),
                            "hungarian_f1":        result.get("soft_f1",        0.0)})

    result["pred_size"] = len(pred_texts)
    result["true_size"] = len(gt_texts)
    return result


# ── Batch evaluation ───────────────────────────────────────────────────────────

def run_semantic_evaluation(
    target_queries: List[str],
    ground_truth_functor,
    method_preds: Dict[str, Dict[str, nx.DiGraph]],
    embedder: Optional[SharedEmbedder] = None,
    sim_threshold: float = 0.40,
    out_dir=None,
    _gt_cache: Optional[Dict[str, nx.DiGraph]] = None,
) -> "pd.DataFrame":
    """
    Run semantic soft-F1 evaluation across all methods and queries.

    Encodes all triple texts in two bulk calls per method (all GT texts concatenated
    in one call, all pred texts per method in another), slices the result matrix
    by query.  This avoids the ~10s per-call overhead on CPU for small batches.

    Args:
        target_queries:       held-out economic query strings
        ground_truth_functor: F_econ (CausalFunctor)
        method_preds:         {"left_kan": {q→G}, ...}
        embedder:             SharedEmbedder (loaded once, reused)
        sim_threshold:        pairs below this contribute 0 to soft F1
        out_dir:              optional save directory
        _gt_cache:            pre-computed GT graphs {q → DiGraph}; avoids re-encoding

    Returns:
        pd.DataFrame with one row per (method, query)
    """
    import json
    import pandas as pd

    if embedder is None:
        embedder = SharedEmbedder.get()

    # ── Pre-cache GT graphs ────────────────────────────────────────────────────
    gt_texts_per_q: Dict[str, List[str]] = {}
    if _gt_cache is not None:
        for q in target_queries:
            gt_texts_per_q[q] = _graph_to_triple_texts(_gt_cache[q])
    else:
        _gt_cache = {}
        for q in target_queries:
            gt = ground_truth_functor(q)
            _gt_cache[q] = gt
            gt_texts_per_q[q] = _graph_to_triple_texts(gt)

    # Bulk-encode all GT texts in ONE call
    all_gt_texts: List[str] = []
    gt_offsets: Dict[str, Tuple[int, int]] = {}
    offset = 0
    for q in target_queries:
        texts = gt_texts_per_q[q]
        gt_offsets[q] = (offset, offset + len(texts))
        all_gt_texts.extend(texts)
        offset += len(texts)

    gt_bulk_embs: Optional[np.ndarray] = embedder.encode(all_gt_texts) if all_gt_texts else None

    def _slice_gt(q: str) -> Optional[np.ndarray]:
        """Return pre-encoded GT embeddings for query q."""
        if gt_bulk_embs is None:
            return None
        s, e = gt_offsets[q]
        if s == e:
            return None
        return gt_bulk_embs[s:e]

    rows = []
    for method, preds in method_preds.items():
        # Bulk-encode all pred texts for this method in ONE call
        pred_texts_per_q: Dict[str, List[str]] = {}
        all_pred_texts: List[str] = []
        pred_offsets: Dict[str, Tuple[int, int]] = {}
        p_offset = 0
        for q in target_queries:
            pred = preds.get(q, nx.DiGraph())
            ptexts = _graph_to_triple_texts(pred)
            pred_texts_per_q[q] = ptexts
            pred_offsets[q] = (p_offset, p_offset + len(ptexts))
            all_pred_texts.extend(ptexts)
            p_offset += len(ptexts)

        pred_bulk_embs: Optional[np.ndarray] = (
            embedder.encode(all_pred_texts) if all_pred_texts else None
        )

        for q in target_queries:
            gt   = _gt_cache[q]
            pred = preds.get(q, nx.DiGraph())

            ps, pe = pred_offsets[q]
            pred_embs = pred_bulk_embs[ps:pe] if (pred_bulk_embs is not None and ps < pe) else None
            gt_embs   = _slice_gt(q)

            metrics = evaluate_semantic(pred, gt, embedder, sim_threshold,
                                        _pred_embs=pred_embs, _gt_embs=gt_embs)
            rows.append({"query": q, "method": method, **metrics})

    df = pd.DataFrame(rows)

    print("\n" + "="*70)
    print("SEMANTIC SOFT-F1 SUMMARY")
    print("="*70)
    cols = ["soft_f1", "soft_precision", "soft_recall", "hungarian_f1"]
    available = [c for c in cols if c in df.columns]
    summary = df.groupby("method")[available].agg(["mean", "std"])
    print(summary.to_string())
    print("="*70 + "\n")

    if out_dir:
        from pathlib import Path
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "semantic_evaluation_results.csv", index=False)

        summary_dict = {}
        for method in df["method"].unique():
            sub = df[df["method"] == method]
            summary_dict[method] = {
                c: {"mean": float(sub[c].mean()), "std": float(sub[c].std())}
                for c in available
            }
        (out_dir / "semantic_evaluation_summary.json").write_text(
            json.dumps(summary_dict, indent=2), encoding="utf-8"
        )
        print(f"[SemanticEval] Saved → {out_dir}")

    return df


# ── Helpers ────────────────────────────────────────────────────────────────────

def _harmonic_mean(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _zero_metrics() -> Dict[str, float]:
    return {
        "soft_precision": 0.0, "soft_recall": 0.0, "soft_f1": 0.0,
        "hungarian_precision": 0.0, "hungarian_recall": 0.0, "hungarian_f1": 0.0,
        "pred_size": 0, "true_size": 0,
    }
