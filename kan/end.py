"""
kan/end.py — Right Kan Extension via End Approximation

Mathematical basis:
    Ran_J(F_med)(d) = ∫_c F_med(c)^{Hom(d, J(c))}

In Set the end is a LIMIT (greatest lower bound / intersection), dual to the
coend (colimit / union).  In our enriched setting:

    Ran_J(F_med)(d) ≈ intersection of F_med(c) for highly similar source topics c

LIMIT SEMANTICS:
    A causal edge (u, v) is included in Ran(d) only if it appears in a
    super-majority of the top-k source topics.
    Edge weight = normalised aggregate similarity across supporting sources.
    This is the conservative / cautious generalization.

The LEFT–RIGHT KAN CROSSOVER EXPERIMENT tests whether:
    - Left Kan > Right Kan at low domain proximity (distant domains)
    - Right Kan > Left Kan at high domain proximity (close domains)
because at low proximity the intersection (end) becomes near-empty,
while at high proximity the union (coend) becomes noisy.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import networkx as nx
import numpy as np

from functor.causal_functor import CausalFunctor, _normalise_entity


def right_kan_extension(
    source_functor: CausalFunctor,
    target_queries: List[str],
    k: int = 10,
    sim_threshold: float = 0.25,
    consensus_frac: float = 0.60,
) -> Dict[str, nx.DiGraph]:
    """
    Compute the right Kan extension Ran_J(F_source) for each target query.

    Args:
        source_functor:  F_med (or F_legal) — source-domain causal functor
        target_queries:  target-domain query strings
        k:               number of source topics to use per target query
        sim_threshold:   minimum cosine similarity to include a source topic
        consensus_frac:  fraction of total similarity weight a claim must
                         accumulate to pass the limit filter.
                         consensus_frac=1.0 → strict intersection (all agree)
                         consensus_frac=0.0 → equivalent to left Kan

    Returns:
        dict mapping each target query → predicted nx.DiGraph
    """
    embedder      = source_functor.embedder
    source_topics = source_functor.topics
    source_embeds = source_functor.topic_embeds   # (N_src, 384)

    target_embeds = embedder.encode(target_queries)   # (N_tgt, 384)
    hom_matrix    = embedder.cosine_similarity_matrix(target_embeds, source_embeds)
    # shape: (N_tgt, N_src)

    results: Dict[str, nx.DiGraph] = {}

    for i, d in enumerate(target_queries):
        hom_row = hom_matrix[i]

        sorted_idx = np.argsort(hom_row)[::-1]
        top_idx = [
            idx for idx in sorted_idx[:k]
            if float(hom_row[idx]) >= sim_threshold
        ]

        if not top_idx:
            results[d] = nx.DiGraph()
            continue

        # Accumulate per-edge support weights
        # edge_support[(u,v)] = sum of hom weights for top-k sources that contain (u,v)
        edge_support: Dict[tuple, float]  = defaultdict(float)
        edge_relations: Dict[tuple, str]  = {}
        total_weight = sum(float(hom_row[idx]) for idx in top_idx)

        for src_idx in top_idx:
            w         = float(hom_row[src_idx])
            src_topic = source_topics[src_idx]

            for triple in source_functor.topic_to_triples[src_topic]:
                subj = _normalise_entity(triple.get("subj", ""))
                obj  = _normalise_entity(triple.get("obj",  ""))
                rel  = triple.get("rel", "affects")
                if not subj or not obj:
                    continue
                key = (subj, obj)
                edge_support[key]   += w
                edge_relations[key]  = rel

        # END / LIMIT filter: include only edges above the consensus threshold
        min_support = consensus_frac * total_weight
        G_ran = nx.DiGraph()

        for (subj, obj), support in edge_support.items():
            if support >= min_support:
                G_ran.add_edge(
                    subj, obj,
                    relation=edge_relations[(subj, obj)],
                    weight=support / total_weight,
                    consensus=support / total_weight,
                    method="right_kan",
                )

        results[d] = G_ran

    _print_stats("right_kan", target_queries, results)
    return results


def _print_stats(method: str, queries: List[str], results: Dict[str, nx.DiGraph]) -> None:
    sizes = [len(g.edges()) for g in results.values()]
    if sizes:
        print(f"[{method}] {len(queries)} queries → "
              f"avg {np.mean(sizes):.1f} edges/graph "
              f"(min {min(sizes)}, max {max(sizes)})")
