"""
kan/coend.py — Left Kan Extension via Coend Approximation

Mathematical basis:
    Lan_J(F_med)(d) = ∫^c Hom_C(J(c), d) ⊗ F_med(c)

In our enriched setting:
    - Hom(J(c), d) ≈ cosine_similarity(embed(c), embed(d))   ∈ [0, 1]
    - ⊗ is tensor product in the symmetric monoidal category of causal graphs,
      interpreted as weighted union (colimit completion)
    - ∫^c (coend) = colimit over all source topics c, weighted by Hom

COLIMIT SEMANTICS:
    A causal edge (u, v) is included in Lan(d) if ANY source topic c
    with Hom(J(c), d) above threshold supports it.
    Edge weight = max similarity across contributing source topics.
    This is the optimistic / free generalization.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import networkx as nx
import numpy as np

from functor.causal_functor import CausalFunctor, _normalise_entity


def left_kan_extension(
    source_functor: CausalFunctor,
    target_queries: List[str],
    k: int = 10,
    sim_threshold: float = 0.25,
) -> Dict[str, nx.DiGraph]:
    """
    Compute the left Kan extension Lan_J(F_source) for each target query.

    Args:
        source_functor:  F_med (or F_legal) — the source-domain causal functor
        target_queries:  list of target-domain query strings (d ∈ C_econ)
        k:               number of source topics to use per target query
        sim_threshold:   minimum cosine similarity to include a source topic

    Returns:
        dict mapping each target query string → predicted nx.DiGraph
    """
    embedder       = source_functor.embedder
    source_topics  = source_functor.topics
    source_embeds  = source_functor.topic_embeds   # (N_src, 384)

    target_embeds  = embedder.encode(target_queries)  # (N_tgt, 384)
    # Compute full Hom matrix: Hom(J(c), d) for all (c, d) pairs
    hom_matrix = embedder.cosine_similarity_matrix(target_embeds, source_embeds)
    # shape: (N_tgt, N_src)

    results: Dict[str, nx.DiGraph] = {}

    for i, d in enumerate(target_queries):
        hom_row = hom_matrix[i]   # (N_src,)

        # Coend: integrate over all source topics, weighted by Hom
        sorted_idx = np.argsort(hom_row)[::-1]
        top_idx = [
            idx for idx in sorted_idx[:k]
            if float(hom_row[idx]) >= sim_threshold
        ]

        G_lan = nx.DiGraph()

        for src_idx in top_idx:
            w         = float(hom_row[src_idx])
            src_topic = source_topics[src_idx]

            for triple in source_functor.topic_to_triples[src_topic]:
                subj = _normalise_entity(triple.get("subj", ""))
                obj  = _normalise_entity(triple.get("obj",  ""))
                rel  = triple.get("rel", "affects")
                if not subj or not obj:
                    continue

                if G_lan.has_edge(subj, obj):
                    # COLIMIT: take max weight (union — include if any source supports it)
                    prev_w = G_lan[subj][obj]["weight"]
                    G_lan[subj][obj]["weight"] = max(prev_w, w)
                    G_lan[subj][obj]["sources"].add(src_topic)
                else:
                    G_lan.add_edge(
                        subj, obj,
                        relation=rel,
                        weight=w,
                        sources={src_topic},
                        method="left_kan",
                    )

        results[d] = G_lan

    _print_stats("left_kan", target_queries, results)
    return results


def _print_stats(method: str, queries: List[str], results: Dict[str, nx.DiGraph]) -> None:
    sizes = [len(g.edges()) for g in results.values()]
    if sizes:
        print(f"[{method}] {len(queries)} queries → "
              f"avg {np.mean(sizes):.1f} edges/graph "
              f"(min {min(sizes)}, max {max(sizes)})")
    else:
        print(f"[{method}] No results.")
