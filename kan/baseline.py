"""
kan/baseline.py — Naive RAG Baseline

The comparison point. No functor structure, no categorical composition.
For each target query d, retrieves the top-k most similar source triples
directly by embedding similarity and returns them as a graph.

This is what a non-categorical approach would do: treat all triples as
a flat retrieval corpus and fetch by cosine similarity.
"""

from __future__ import annotations

from typing import Dict, List

import networkx as nx
import numpy as np

from functor.causal_functor import CausalFunctor, _normalise_entity


def naive_rag_baseline(
    source_functor: CausalFunctor,
    target_queries: List[str],
    k: int = 20,
) -> Dict[str, nx.DiGraph]:
    """
    Naive RAG: embed each target query, retrieve top-k source triples
    by cosine similarity to their text representation, return as a graph.

    No Kan extension, no functor composition, no topic hierarchy.

    Args:
        source_functor:  provides triples and embedder
        target_queries:  target-domain queries
        k:               how many triples to retrieve per query

    Returns:
        dict mapping each target query → nx.DiGraph
    """
    embedder = source_functor.embedder
    triples  = source_functor.triples

    # Embed all source triples as "subj rel obj" text
    triple_texts = [
        f"{t.get('subj','')  } {t.get('rel','affects')} {t.get('obj','')}"
        for t in triples
    ]
    triple_embeds  = embedder.encode(triple_texts, show_progress_bar=True)  # (N_triples, 384)
    target_embeds  = embedder.encode(target_queries)                         # (N_tgt, 384)

    # Similarity: (N_tgt, N_triples)
    sim_matrix = embedder.cosine_similarity_matrix(target_embeds, triple_embeds)

    results: Dict[str, nx.DiGraph] = {}

    for i, d in enumerate(target_queries):
        sims    = sim_matrix[i]
        top_idx = np.argsort(sims)[::-1][:k]

        G = nx.DiGraph()
        for idx in top_idx:
            t    = triples[idx]
            subj = _normalise_entity(t.get("subj", ""))
            obj  = _normalise_entity(t.get("obj",  ""))
            rel  = t.get("rel", "affects")
            w    = float(sims[idx])
            if not subj or not obj:
                continue
            if G.has_edge(subj, obj):
                G[subj][obj]["weight"] = max(G[subj][obj]["weight"], w)
            else:
                G.add_edge(subj, obj, relation=rel, weight=w, method="naive_rag")

        results[d] = G

    sizes = [len(g.edges()) for g in results.values()]
    if sizes:
        print(f"[naive_rag] {len(target_queries)} queries → "
              f"avg {np.mean(sizes):.1f} edges/graph")
    return results
