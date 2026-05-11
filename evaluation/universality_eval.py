"""
evaluation/universality_eval.py

Universality Class Evaluation — Parallel 3: Renormalization Group Theory.

Core insight:
    Kenneth Wilson's Renormalization Group (1971) explained why completely different
    physical systems (boiling water, magnets, alloy transitions) obey identical
    mathematical laws near their critical points.  The microscopic details "renormalize
    away"; only a few structural properties determine the universality class.

    Analogously, causal mechanisms across domains may belong to universal structural
    classes regardless of the specific entities involved:

    CLASS 1 — Regulatory/Enforcement:
        Pattern: "authoritative body applies rule → compliance/behaviour changes"
        Legal:   "SEC enforcement → firms disclose material information"
        Medical: "FDA approval requirement → drug development changes"
        Economic:"Fed rate policy → bank lending behaviour changes"

    CLASS 2 — Amplification/Cascade:
        Pattern: "initial cause → downstream amplification → large effect"
        Medical: "inflammation → cytokine storm → organ failure"
        Economic:"bank failure → credit freeze → recession"
        Legal:   "fraud discovery → investor panic → market collapse"

    CLASS 3 — Contagion/Spread:
        Pattern: "risk/failure propagates from one entity to connected others"
        Medical: "pathogen → transmission → epidemic spread"
        Economic:"sovereign default → contagion → financial crisis"
        Legal:   "court precedent → cited in other jurisdictions → law change"

    CLASS 4 — Information/Signaling:
        Pattern: "information disclosure/announcement → expectation change → behaviour"
        Medical: "clinical trial results → physician prescribing → patient outcomes"
        Economic:"FOMC statement → market expectations → asset prices"
        Legal:   "regulatory guidance → compliance expectations → firm behaviour"

    CLASS 5 — Resource/Constraint:
        Pattern: "resource availability/restriction → activity is enabled/limited"
        Medical: "drug shortage → alternative prescription → patient adherence"
        Economic:"credit restriction → investment reduction → employment decline"
        Legal:   "capital requirement → bank lending capacity → credit availability"

    CLASS 6 — Threshold/Tipping:
        Pattern: "cumulative pressure builds until critical threshold → sudden change"
        Medical: "chronic inflammation → critical damage → organ failure"
        Economic:"debt accumulation → debt ceiling → fiscal crisis"
        Legal:   "repeated violations → pattern established → class action"

Scoring:
    1. Embed each triple as "subj rel obj"
    2. Assign to nearest universality class (by cosine similarity to class seeds)
    3. Build class distribution for predicted and GT graphs
    4. Score = 1 - Jensen-Shannon divergence(P_pred, P_gt)
       JSD = 0 → identical distributions → perfect universality preservation
       JSD = 1 → maximally different distributions

JSD is symmetric, bounded [0,1], and well-defined even when distributions have
zero-probability classes (unlike raw KL divergence).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from functor.causal_functor import CausalFunctor
from functor.embedder import SharedEmbedder


# ── Universality class definitions ─────────────────────────────────────────────

UNIVERSALITY_CLASSES = {
    "regulatory_enforcement": (
        "authoritative regulatory body enforces compliance rule "
        "market participants change behaviour obligation penalty"
    ),
    "amplification_cascade": (
        "small initial cause amplifies propagates accumulates "
        "large downstream systemic effect chain reaction"
    ),
    "contagion_spread": (
        "risk failure spreads propagates transmits from one entity "
        "to connected others network contagion epidemic"
    ),
    "information_signaling": (
        "information disclosure announcement signal changes "
        "expectations beliefs perception behaviour decision"
    ),
    "resource_constraint": (
        "resource availability restriction scarcity limits reduces "
        "economic activity output capacity investment employment"
    ),
    "threshold_tipping": (
        "cumulative pressure buildup reaches critical threshold "
        "tipping point triggers sudden nonlinear phase transition"
    ),
}

CLASS_NAMES = list(UNIVERSALITY_CLASSES.keys())
_CLASS_SEED_EMBEDS: Optional[np.ndarray] = None   # (6, 384), cached after first call


def _get_class_embeddings(embedder: SharedEmbedder) -> np.ndarray:
    """Load and cache class seed embeddings."""
    global _CLASS_SEED_EMBEDS
    if _CLASS_SEED_EMBEDS is None:
        seeds = [UNIVERSALITY_CLASSES[c] for c in CLASS_NAMES]
        _CLASS_SEED_EMBEDS = embedder.encode(seeds)   # (6, 384)
    return _CLASS_SEED_EMBEDS


def _classify_triples(
    G: nx.DiGraph,
    embedder: SharedEmbedder,
) -> np.ndarray:
    """
    Classify each edge in G into a universality class.
    Returns a probability distribution over CLASS_NAMES, shape (6,).
    """
    if G.number_of_edges() == 0:
        return np.ones(len(CLASS_NAMES), dtype=np.float32) / len(CLASS_NAMES)

    class_embeds = _get_class_embeddings(embedder)   # (6, 384)

    texts = []
    for u, v, d in G.edges(data=True):
        rel = d.get("relation", "affects")
        texts.append(f"{u} {rel} {v}")

    triple_embeds = embedder.encode(texts)                                   # (E, 384)
    sim_matrix    = embedder.cosine_similarity_matrix(triple_embeds, class_embeds)  # (E, 6)

    # Each triple → class with highest similarity
    assignments = np.argmax(sim_matrix, axis=1)   # (E,)
    counts = np.bincount(assignments, minlength=len(CLASS_NAMES)).astype(np.float32)
    if counts.sum() > 0:
        counts /= counts.sum()
    return counts


def _jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """JSD(P || Q) ∈ [0, 1].  Symmetric, well-defined for zero-probability events."""
    p = np.asarray(p, dtype=np.float64) + 1e-12
    q = np.asarray(q, dtype=np.float64) + 1e-12
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(np.clip(0.5 * (kl_pm + kl_qm), 0.0, 1.0))


def universality_scores(
    predicted: nx.DiGraph,
    ground_truth: nx.DiGraph,
    embedder: Optional[SharedEmbedder] = None,
) -> Dict[str, float]:
    """
    Compute universality class preservation score.

    universality_score = 1 - JSD(P_pred, P_gt)
      → 1.0 means predicted graph has same class distribution as GT
      → 0.0 means maximally different universality class distribution

    Also returns per-class fractions for both graphs.
    """
    if embedder is None:
        embedder = SharedEmbedder.get()

    dist_pred = _classify_triples(predicted,    embedder)
    dist_gt   = _classify_triples(ground_truth, embedder)

    jsd   = _jensen_shannon_divergence(dist_pred, dist_gt)
    score = 1.0 - jsd

    result: Dict[str, float] = {"universality_score": score, "jsd": jsd}
    for i, cls in enumerate(CLASS_NAMES):
        result[f"pred_{cls}"] = float(dist_pred[i])
        result[f"gt_{cls}"]   = float(dist_gt[i])
    return result


# ── Batch evaluation ───────────────────────────────────────────────────────────

def run_universality_evaluation(
    target_queries: List[str],
    ground_truth_functor: CausalFunctor,
    method_preds: Dict[str, Dict[str, nx.DiGraph]],
    embedder: Optional[SharedEmbedder] = None,
    out_dir=None,
    _gt_cache: Optional[Dict[str, nx.DiGraph]] = None,
) -> pd.DataFrame:
    """
    Run universality class evaluation across all methods and queries.
    Returns pd.DataFrame with one row per (method, query).
    """
    if embedder is None:
        embedder = SharedEmbedder.get()

    # Pre-warm class seed embeddings once
    class_embeds = _get_class_embeddings(embedder)   # (6, 384)

    # Bulk-encode ALL GT triple texts in ONE call, then classify per query
    if _gt_cache is None:
        _gt_cache = {q: ground_truth_functor(q) for q in target_queries}
    gt_graph_cache = _gt_cache

    gt_texts_per_q: Dict[str, List[str]] = {}
    gt_offsets_u: Dict[str, Tuple[int, int]] = {}
    all_gt_texts_u: List[str] = []
    g_offset = 0
    for q in target_queries:
        gt = gt_graph_cache[q]
        gtexts = [f"{u} {d.get('relation','affects')} {v}"
                  for u, v, d in gt.edges(data=True)]
        gt_texts_per_q[q]   = gtexts
        gt_offsets_u[q]     = (g_offset, g_offset + len(gtexts))
        all_gt_texts_u.extend(gtexts)
        g_offset += len(gtexts)

    gt_bulk_u = embedder.encode(all_gt_texts_u) if all_gt_texts_u else None

    gt_dist_cache: Dict[str, np.ndarray] = {}
    for q in target_queries:
        gs, ge = gt_offsets_u[q]
        if gt_bulk_u is not None and gs < ge:
            slice_embs  = gt_bulk_u[gs:ge]
            sim_matrix  = embedder.cosine_similarity_matrix(slice_embs, class_embeds)
            assignments = np.argmax(sim_matrix, axis=1)
            counts = np.bincount(assignments, minlength=len(CLASS_NAMES)).astype(np.float32)
            if counts.sum() > 0:
                counts /= counts.sum()
            gt_dist_cache[q] = counts
        else:
            gt_dist_cache[q] = np.ones(len(CLASS_NAMES), dtype=np.float32) / len(CLASS_NAMES)

    rows = []
    for method, preds in method_preds.items():
        # Bulk-encode all pred triple texts for this method in ONE call
        all_pred_texts: List[str] = []
        pred_offsets: Dict[str, Tuple[int, int]] = {}
        p_offset = 0
        for q in target_queries:
            pred   = preds.get(q, nx.DiGraph())
            ptexts = [f"{u} {d.get('relation','affects')} {v}"
                      for u, v, d in pred.edges(data=True)]
            pred_offsets[q] = (p_offset, p_offset + len(ptexts))
            all_pred_texts.extend(ptexts)
            p_offset += len(ptexts)

        pred_bulk_embs: Optional[np.ndarray] = (
            embedder.encode(all_pred_texts) if all_pred_texts else None
        )
        class_embeds = _get_class_embeddings(embedder)

        for q in target_queries:
            gt      = gt_graph_cache[q]
            dist_gt = gt_dist_cache[q]

            ps, pe = pred_offsets[q]
            if pred_bulk_embs is not None and ps < pe:
                slice_embs = pred_bulk_embs[ps:pe]
                sim_matrix = embedder.cosine_similarity_matrix(slice_embs, class_embeds)
                assignments = np.argmax(sim_matrix, axis=1)
                counts = np.bincount(assignments, minlength=len(CLASS_NAMES)).astype(np.float32)
                if counts.sum() > 0:
                    counts /= counts.sum()
                dist_pred = counts
            else:
                # empty predicted graph → uniform distribution
                dist_pred = np.ones(len(CLASS_NAMES), dtype=np.float32) / len(CLASS_NAMES)

            jsd   = _jensen_shannon_divergence(dist_pred, dist_gt)
            score = 1.0 - jsd

            result: Dict[str, float] = {"universality_score": score, "jsd": jsd}
            for i, cls in enumerate(CLASS_NAMES):
                result[f"pred_{cls}"] = float(dist_pred[i])
                result[f"gt_{cls}"]   = float(dist_gt[i])

            rows.append({"query": q, "method": method, **result})

    df = pd.DataFrame(rows)

    print("\n" + "="*70)
    print("UNIVERSALITY CLASS EVALUATION (RG-Theory)")
    print("="*70)
    if "universality_score" in df.columns:
        summary = df.groupby("method")[["universality_score", "jsd"]].agg(["mean", "std"])
        print(summary.to_string())

        # Show dominant universality class per method
        print("\nDominant universality class per method (predicted):")
        for method in df["method"].unique():
            sub = df[df["method"] == method]
            class_cols = [f"pred_{c}" for c in CLASS_NAMES]
            class_means = sub[class_cols].mean()
            dominant = class_means.idxmax().replace("pred_", "")
            print(f"  {method}: {dominant} ({class_means.max():.3f})")
    print("="*70 + "\n")

    if out_dir:
        from pathlib import Path
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "universality_evaluation.csv", index=False)
        print(f"[UniversalityEval] Saved → {out_dir / 'universality_evaluation.csv'}")

    return df
