"""
kan/soft_end.py — Soft Right Kan Extension via Causal Density Weighting

Mathematical basis — Parallel 5: Causal Density Function (Mahadevan, Categories for AGI).

The standard right Kan extension uses count-based consensus:
    include edge (u,v) iff it appears in >consensus_frac of top-k source topics

This replaces count-based consensus with ρ-WEIGHTED consensus, implementing
the limit/end as a density-weighted intersection:

    Ran_soft(d) = ∫_c F_source(c)^{ρ(c,d)}

An edge (u,v) passes the limit filter iff:
    Σ_{c: (u,v) ∈ F(c)} ρ(c, d)  ≥  consensus_frac × Σ_c ρ(c, d)

where ρ(c, d) = sim(c, d) / freq(c)^α is the causal density weight.

WHY THE DIFFERENCE FROM LEFT KAN:
    Left Kan  (coend/colimit): include if ANY source supports it (weighted union)
    Right Kan (end/limit):     include only if MOST sources support it (weighted intersection)

    With density weighting:
    - A rare but highly-relevant source topic (high ρ) counts MORE toward consensus
    - A common but marginally-relevant topic (low ρ) counts less
    - This means the right Kan includes only edges supported by dense (relevant) sources,
      not by high-frequency (common) sources that happen to appear in k nearest topics

This is the "consistent intervention" semantics: an edge is included only if the
causal density function assigns it high weight across multiple relevant source topics.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import networkx as nx
import numpy as np

from functor.causal_functor import CausalFunctor, _normalise_entity


def soft_right_kan_extension(
    source_functor: CausalFunctor,
    target_queries: List[str],
    k: int = 10,
    alpha: float = 0.5,
    consensus_frac: float = 0.60,
    min_rho: float = 0.05,
) -> Dict[str, nx.DiGraph]:
    """
    Compute the soft right Kan extension using causal density weighting.

    Args:
        source_functor:  source-domain causal functor
        target_queries:  target-domain query strings
        k:               number of source topics to consider per query
        alpha:           RN-derivative smoothing exponent (0=uniform, 1=full RN)
        consensus_frac:  fraction of total ρ-weight a claim must accumulate
                         to pass the limit filter
        min_rho:         minimum ρ weight to include a source topic

    Returns:
        dict mapping each target query → predicted nx.DiGraph
        Edge weights = fraction of ρ-weight supporting this edge (∈ [0,1])
    """
    embedder       = source_functor.embedder
    source_topics  = source_functor.topics
    source_embeds  = source_functor.topic_embeds
    total_triples  = max(len(source_functor.triples), 1)

    # Precompute topic frequencies
    topic_freq = np.array([
        len(source_functor.topic_to_triples[c]) / total_triples
        for c in source_topics
    ], dtype=np.float32)
    freq_denom = np.power(np.maximum(topic_freq, 1e-6), alpha)

    target_embeds = embedder.encode(target_queries)
    hom_matrix    = embedder.cosine_similarity_matrix(target_embeds, source_embeds)

    results: Dict[str, nx.DiGraph] = {}

    for i, d in enumerate(target_queries):
        sim_row = hom_matrix[i]
        rho     = sim_row / freq_denom   # (N_src,) — causal density weights

        sorted_idx = np.argsort(rho)[::-1][:k]
        top_idx    = [idx for idx in sorted_idx if float(rho[idx]) >= min_rho]

        if not top_idx:
            results[d] = nx.DiGraph()
            continue

        # Accumulate ρ-weighted support for each candidate edge
        edge_rho_support: Dict[tuple, float] = defaultdict(float)
        edge_relations:   Dict[tuple, str]   = {}
        total_rho = sum(float(rho[idx]) for idx in top_idx)

        for src_idx in top_idx:
            w_rho     = float(rho[src_idx])
            src_topic = source_topics[src_idx]

            for triple in source_functor.topic_to_triples[src_topic]:
                subj = _normalise_entity(triple.get("subj", ""))
                obj  = _normalise_entity(triple.get("obj",  ""))
                rel  = triple.get("rel", "affects")
                if not subj or not obj:
                    continue
                key = (subj, obj)
                edge_rho_support[key]  += w_rho
                edge_relations[key]     = rel

        # LIMIT/END filter: include only edges with ρ-weighted consensus
        min_support = consensus_frac * total_rho
        G_ran = nx.DiGraph()

        for (subj, obj), support in edge_rho_support.items():
            if support >= min_support:
                G_ran.add_edge(
                    subj, obj,
                    relation=edge_relations[(subj, obj)],
                    weight=support / total_rho,       # normalised ρ-consensus
                    rho_consensus=support / total_rho,
                    method="soft_right_kan",
                )

        results[d] = G_ran

    _print_stats("soft_right_kan", target_queries, results)
    return results


def _print_stats(method: str, queries: List[str], results: Dict[str, nx.DiGraph]) -> None:
    sizes = [len(g.edges()) for g in results.values()]
    if sizes:
        print(f"[{method}] {len(queries)} queries → "
              f"avg {np.mean(sizes):.1f} edges/graph "
              f"(min {min(sizes)}, max {max(sizes)})")
    else:
        print(f"[{method}] No results.")
