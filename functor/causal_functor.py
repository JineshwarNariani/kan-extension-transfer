"""
causal_functor.py

Represents F: C_queries → C_causal_graphs

Objects of C_queries  = topic strings discovered by Democritus
Objects of C_causal   = NetworkX DiGraphs with relation-typed edges
Morphisms in C_queries = query refinements (more specific → less specific)

F is implemented as a fuzzy lookup: F(q) returns the weighted union of causal
graphs for the K nearest topics in embedding space.

This file is the central data structure — both the Kan modules and evaluation
consume CausalFunctor instances.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import networkx as nx
import numpy as np

from functor.embedder import SharedEmbedder


def _load_jsonl(path: Path) -> List[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


class CausalFunctor:
    """
    F: C_queries → C_causal_graphs

    Constructed from a relational_triples.jsonl file produced by Democritus.
    Each triple has shape: {subj, rel, obj, topic, domain, ...}
    """

    def __init__(
        self,
        triples_path: Path,
        domain_name: str,
        embedder: Optional[SharedEmbedder] = None,
        fuzzy_k: int = 5,
        sim_threshold: float = 0.10,
    ) -> None:
        self.domain_name   = domain_name
        self.fuzzy_k       = fuzzy_k
        self.sim_threshold = sim_threshold
        self.embedder      = embedder or SharedEmbedder.get()

        self.triples: List[dict] = _load_jsonl(triples_path)
        if not self.triples:
            raise RuntimeError(f"No triples loaded from {triples_path}")

        # Index: topic → list of triples
        self.topic_to_triples: Dict[str, List[dict]] = defaultdict(list)
        for t in self.triples:
            topic = t.get("topic") or t.get("domain") or "unknown"
            t["_topic"] = topic  # normalised key
            self.topic_to_triples[topic].append(t)

        self.topics: List[str] = list(self.topic_to_triples.keys())

        print(f"[Functor:{domain_name}] {len(self.triples)} triples, "
              f"{len(self.topics)} topics.")

        # Pre-compute topic embeddings — shape (N_topics, 384)
        self.topic_embeds: np.ndarray = self.embedder.encode(
            self.topics, show_progress_bar=True
        )

    # ── Functor evaluation ────────────────────────────────────────────────────

    def __call__(self, query: str) -> nx.DiGraph:
        """
        F(query) → nx.DiGraph

        Returns a weighted union of causal graphs for the top-fuzzy_k nearest
        topics.  Edge weight = cosine similarity of the query to source topic.
        """
        q_embed = self.embedder.encode([query])       # (1, 384)
        sims    = self.embedder.cosine_similarity_matrix(q_embed, self.topic_embeds)[0]
        # sims shape: (N_topics,)

        top_idx = np.argsort(sims)[::-1][: self.fuzzy_k]

        G = nx.DiGraph()
        for idx in top_idx:
            w = float(sims[idx])
            if w < self.sim_threshold:
                continue
            topic = self.topics[idx]
            for triple in self.topic_to_triples[topic]:
                subj = _normalise_entity(triple.get("subj", ""))
                obj  = _normalise_entity(triple.get("obj",  ""))
                rel  = triple.get("rel", "affects")
                if not subj or not obj:
                    continue
                if G.has_edge(subj, obj):
                    G[subj][obj]["weight"] = max(G[subj][obj]["weight"], w)
                else:
                    G.add_edge(subj, obj, relation=rel, weight=w, source_topic=topic)
        return G

    def morphism_on_query_refinement(self, q_general: str, q_specific: str) -> nx.DiGraph:
        """
        Apply F to a morphism f: q_general → q_specific.
        Returns the subgraph of F(q_general) that is also in F(q_specific),
        representing the causal knowledge shared after refinement.
        """
        G_gen  = self(q_general)
        G_spec = self(q_specific)
        common_edges = [(u, v) for u, v in G_gen.edges() if G_spec.has_edge(u, v)]
        return G_gen.edge_subgraph(common_edges).copy()

    # ── Inspection helpers ────────────────────────────────────────────────────

    def domain_centroid(self) -> np.ndarray:
        """Mean embedding over all topics — represents the domain in embed space."""
        return self.topic_embeds.mean(axis=0)

    def get_all_triples_as_graph(self) -> nx.DiGraph:
        """Return all triples as one large DiGraph (used for naive RAG baseline)."""
        G = nx.DiGraph()
        for t in self.triples:
            subj = _normalise_entity(t.get("subj", ""))
            obj  = _normalise_entity(t.get("obj",  ""))
            rel  = t.get("rel", "affects")
            if subj and obj:
                G.add_edge(subj, obj, relation=rel,
                           weight=1.0, source_topic=t.get("_topic", ""))
        return G

    def topic_summary(self) -> dict:
        return {
            "domain":        self.domain_name,
            "n_triples":     len(self.triples),
            "n_topics":      len(self.topics),
            "n_nodes":       len({e for t in self.triples
                                  for e in [t.get("subj",""), t.get("obj","")]}),
            "relation_types": list({t.get("rel","") for t in self.triples}),
        }


def _normalise_entity(s: str) -> str:
    """Lowercase, strip whitespace, remove short junk."""
    s = s.strip().lower()
    if len(s) < 2:
        return ""
    # Remove common filler prefixes LLMs sometimes output
    for prefix in ("the ", "a ", "an "):
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s.strip()
