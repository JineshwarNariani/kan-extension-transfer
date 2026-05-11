"""
kan/soft_coend.py — Soft Left Kan Extension via Causal Density Weighting

Mathematical basis — Parallel 5: Causal Density Function (Mahadevan, Categories for AGI).

The standard left Kan extension uses a HARD threshold:
    include source triple iff sim(source_topic, target_query) > τ

This replaces the hard gate with the CAUSAL DENSITY FUNCTION ρ, implementing
the Radon-Nikodym derivative from the textbook's RN-Kan-Sheaf diagram:

    ρ_i(x) = dP_do(X_i) / dP_obs(x)

In our discrete setting:
    P_obs(c)   = freq(c)        = |triples in topic c| / |total source triples|
                                  (how common this topic is in the SOURCE corpus)
    P_do(c|d)  = sim(c, d)      (relevance of source topic c to target query d,
                                  approximating the interventional density)
    ρ(c, d)    = sim(c, d) / freq(c)^α

where α ∈ [0, 1] is a smoothing exponent:
    α = 0  → equal weighting (ignores source frequency; equivalent to soft threshold)
    α = 0.5→ square-root smoothing (default; balanced upweighting)
    α = 1  → full RN reweighting (maximum upweighting of rare cross-domain topics)

WHY THIS MATTERS (the key insight):
    A rare topic in the source corpus that is nonetheless highly relevant to the
    target query gets a HIGH ρ weight under the RN reweighting.  This is the
    interventional distribution "amplifying" signals that would be underrepresented
    under the observational distribution.

    Example:
        Source (legal): topic "healthcare market competition" appears in only 2/43
        triples (freq = 0.046), but has high similarity to economic query
        "market concentration affects pricing".
        Standard Kan: includes this topic only if sim > τ (hard gate)
        Soft Kan:     weight = sim(0.62) / (0.046)^0.5 = 0.62 / 0.215 = 2.88 → HIGH

    This is the "density ratio" upweighting rare but relevant cross-domain structure.

Edge weight in the output graph:
    weight = ρ(c, d) = sim(c, d) / freq(c)^α
    (normalised across all contributing source topics)

No threshold is applied — all source topics contribute, weighted by ρ.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import networkx as nx
import numpy as np

from functor.causal_functor import CausalFunctor, _normalise_entity


def soft_left_kan_extension(
    source_functor: CausalFunctor,
    target_queries: List[str],
    k: int = 10,
    alpha: float = 0.5,
    min_rho: float = 0.05,
) -> Dict[str, nx.DiGraph]:
    """
    Compute the soft left Kan extension using causal density weighting.

    Args:
        source_functor:  source-domain causal functor
        target_queries:  target-domain query strings
        k:               number of source topics to consider per query
        alpha:           RN-derivative smoothing exponent (0=uniform, 1=full RN)
        min_rho:         minimum ρ weight to include a source topic (soft floor)

    Returns:
        dict mapping each target query → predicted nx.DiGraph
        Edge weights = normalised ρ(c, d) values (not raw cosine similarities)
    """
    embedder       = source_functor.embedder
    source_topics  = source_functor.topics
    source_embeds  = source_functor.topic_embeds   # (N_src, 384)
    total_triples  = max(len(source_functor.triples), 1)

    # Precompute topic frequencies: freq(c) = |triples in c| / total
    topic_freq = np.array([
        len(source_functor.topic_to_triples[c]) / total_triples
        for c in source_topics
    ], dtype=np.float32)   # (N_src,)

    # Frequency denominator: freq(c)^alpha (smoothed)
    freq_denom = np.power(np.maximum(topic_freq, 1e-6), alpha)   # (N_src,)

    target_embeds = embedder.encode(target_queries)   # (N_tgt, 384)
    hom_matrix    = embedder.cosine_similarity_matrix(target_embeds, source_embeds)
    # shape: (N_tgt, N_src)

    results: Dict[str, nx.DiGraph] = {}

    for i, d in enumerate(target_queries):
        sim_row = hom_matrix[i]   # (N_src,) — P_do(c|d) ≈ sim(c, d)

        # Compute ρ(c, d) = sim(c, d) / freq(c)^alpha
        rho = sim_row / freq_denom   # (N_src,)

        # Take top-k by ρ (not by raw similarity)
        sorted_idx = np.argsort(rho)[::-1][:k]
        top_idx    = [idx for idx in sorted_idx if float(rho[idx]) >= min_rho]

        # Normalise ρ weights so they sum to 1 across contributing topics
        if top_idx:
            rho_vals   = np.array([float(rho[idx]) for idx in top_idx])
            rho_normed = rho_vals / (rho_vals.sum() + 1e-12)
        else:
            rho_normed = np.array([])

        G_lan = nx.DiGraph()

        for j, src_idx in enumerate(top_idx):
            w_rho     = float(rho_normed[j])
            w_raw     = float(sim_row[src_idx])   # raw similarity for reference
            src_topic = source_topics[src_idx]

            for triple in source_functor.topic_to_triples[src_topic]:
                subj = _normalise_entity(triple.get("subj", ""))
                obj  = _normalise_entity(triple.get("obj",  ""))
                rel  = triple.get("rel", "affects")
                if not subj or not obj:
                    continue

                if G_lan.has_edge(subj, obj):
                    # COLIMIT: accumulate ρ weights (soft union)
                    G_lan[subj][obj]["weight"]    += w_rho
                    G_lan[subj][obj]["raw_sim"]    = max(G_lan[subj][obj]["raw_sim"], w_raw)
                    G_lan[subj][obj]["sources"].add(src_topic)
                else:
                    G_lan.add_edge(
                        subj, obj,
                        relation=rel,
                        weight=w_rho,
                        raw_sim=w_raw,
                        rho=w_rho,
                        sources={src_topic},
                        method="soft_left_kan",
                    )

        results[d] = G_lan

    _print_stats("soft_left_kan", target_queries, results)
    return results


def _print_stats(method: str, queries: List[str], results: Dict[str, nx.DiGraph]) -> None:
    sizes = [len(g.edges()) for g in results.values()]
    if sizes:
        print(f"[{method}] {len(queries)} queries → "
              f"avg {np.mean(sizes):.1f} edges/graph "
              f"(min {min(sizes)}, max {max(sizes)})")
    else:
        print(f"[{method}] No results.")
