"""
evaluation/domain_proximity.py

Measures semantic proximity between two causal domains.
Used in the ablation to test whether transfer quality correlates
with domain proximity — the empirical sheaf-structure test.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from functor.causal_functor import CausalFunctor


def domain_proximity(
    source_functor: CausalFunctor,
    target_functor: CausalFunctor,
) -> float:
    """
    Cosine similarity between domain centroid embeddings.
    Range [−1, 1]; typically [0.1, 0.9] for related domains.
    """
    src_centroid = source_functor.domain_centroid()
    tgt_centroid = target_functor.domain_centroid()
    norm_src = np.linalg.norm(src_centroid)
    norm_tgt = np.linalg.norm(tgt_centroid)
    if norm_src < 1e-9 or norm_tgt < 1e-9:
        return 0.0
    return float(np.dot(src_centroid, tgt_centroid) / (norm_src * norm_tgt))


def pairwise_domain_proximities(
    source_functors: Dict[str, CausalFunctor],
    target_functor: CausalFunctor,
) -> Dict[str, float]:
    """Compute proximity from each source domain to the target domain."""
    return {
        name: domain_proximity(f, target_functor)
        for name, f in source_functors.items()
    }


def topic_overlap(
    source_functor: CausalFunctor,
    target_functor: CausalFunctor,
    sim_threshold: float = 0.70,
) -> float:
    """
    Fraction of target topics that have a similar source topic (cos_sim >= threshold).
    Coarser but more interpretable than centroid distance.
    """
    embedder       = source_functor.embedder
    source_embeds  = source_functor.topic_embeds   # (N_src, 384)
    target_embeds  = target_functor.topic_embeds   # (N_tgt, 384)

    # For each target topic, max similarity to any source topic
    sim_matrix = embedder.cosine_similarity_matrix(target_embeds, source_embeds)  # (N_tgt, N_src)
    max_sims   = sim_matrix.max(axis=1)   # (N_tgt,)

    return float((max_sims >= sim_threshold).mean())
