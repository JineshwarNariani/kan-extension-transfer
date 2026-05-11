"""
evaluation/back_translation_eval.py

Back-Translation Consistency Test — Parallel 4: Nida's Dynamic Equivalence.

Core insight:
    Eugene Nida's dynamic equivalence (1964): a good translation preserves MEANING,
    not words.  "Il pleut des cordes" (French, "it's raining ropes") translates
    "It's raining cats and dogs" correctly — the words differ but the meaning is
    preserved.  Evaluating such a translation by word overlap gives zero score;
    evaluating by meaning gives a perfect score.

    Back-translation is the standard test: translate A→B, then B→A.  If the
    final result resembles the original A, the translation preserved meaning.

    Applied to cross-domain causal transfer:
      source_triple → [Kan Extension → economic vocabulary] → economic_triple
                    → [LLM Back-Translate → source vocabulary] → back_triple
      fidelity = cosine_similarity(embed(back_triple), embed(source_triple))

    A high fidelity score means the Kan extension transferred genuine structural
    meaning that round-trips back to the source.  A low score means the transferred
    triple is structurally disconnected from its supposed source — the analogy broke.

    Crucially, this test requires NO GROUND TRUTH — it is a self-consistency check.
    We can evaluate even predictions for queries where the GT is unknown.

LLM prompt strategy:
    We ask the LLM to preserve the CAUSAL STRUCTURE (which entity causes what)
    but re-express it in source-domain vocabulary.  The relation type (causes,
    reduces, etc.) should be preserved as-is.  We then embed the back-translated
    text and compute cosine similarity to the original source triples.

Aggregation:
    For each predicted triple, find the maximum cosine similarity to any triple
    in the original source corpus.  The back-translation fidelity for a method
    is the mean of these max similarities across all predicted triples.
"""

from __future__ import annotations

import os
import re
import time
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from functor.causal_functor import CausalFunctor
from functor.embedder import SharedEmbedder


# ── LLM call helper ────────────────────────────────────────────────────────────

def _call_llm_safe(prompt: str, max_tokens: int = 128) -> Optional[str]:
    """
    Call the LiteLLM proxy (same credentials as extraction pipeline).
    Returns None on failure so that back-translation gracefully degrades.
    """
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key  = os.getenv("OPENAI_API_KEY", ""),
            base_url = os.getenv("DEMOC_LLM_BASE_URL", "https://api.openai.com/v1"),
        )
        model = os.getenv("KAN_PRIMARY_MODEL", "claude-sonnet-4-6")
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        print(f"  [BackTranslate] LLM call failed: {str(exc)[:80]}")
        return None


def _back_translate_triple(
    subj: str,
    rel: str,
    obj: str,
    source_domain: str,
    batch_delay: float = 0.5,
) -> Optional[str]:
    """
    Ask LLM to translate an economic causal triple back into source-domain language.
    Returns a single sentence "subj rel obj" in source-domain vocabulary, or None.
    """
    prompt = (
        f"You are a {source_domain} domain expert reviewing an economic causal claim "
        f"that was derived via structural analogy from {source_domain} knowledge.\n\n"
        f"Economic claim: \"{subj} {rel} {obj}\"\n\n"
        f"Re-express this causal relationship in {source_domain} domain language, "
        f"preserving the causal direction and structure but replacing economic terms "
        f"with {source_domain}-appropriate concepts.\n"
        f"Output format (one line only): SUBJECT | RELATION | OBJECT\n"
        f"where RELATION is one of: causes, increases, reduces, influences, leads_to, affects\n"
        f"Do not explain. Output only the single line."
    )
    time.sleep(batch_delay)
    response = _call_llm_safe(prompt, max_tokens=100)
    return response


def _parse_back_translation(response: Optional[str]) -> Optional[str]:
    """Parse 'SUBJECT | RELATION | OBJECT' response into a single string."""
    if not response:
        return None
    if "|" in response:
        parts = [p.strip() for p in response.split("|")]
        if len(parts) >= 3:
            return f"{parts[0]} {parts[1]} {parts[2]}"
    # Fallback: return full response as-is
    return response.strip()


# ── Core computation ───────────────────────────────────────────────────────────

def back_translation_fidelity(
    predicted: nx.DiGraph,
    source_functor: CausalFunctor,
    source_domain: str,
    embedder: Optional[SharedEmbedder] = None,
    max_triples: int = 15,
    batch_delay: float = 0.4,
) -> Dict[str, float]:
    """
    Compute back-translation fidelity for a single predicted graph.

    Steps:
      1. Sample up to max_triples edges from predicted graph
      2. For each edge, LLM back-translates to source domain
      3. Embed back-translated text
      4. Find max cosine similarity to any source corpus triple
      5. Return mean max-similarity = "fidelity"

    Returns:
        fidelity              — mean max-similarity (higher = better round-trip)
        n_back_translated     — number of triples successfully back-translated
        n_predicted           — total edges in predicted graph
    """
    if embedder is None:
        embedder = SharedEmbedder.get()

    pred_edges = list(predicted.edges(data=True))
    if not pred_edges:
        return {"fidelity": 0.0, "n_back_translated": 0, "n_predicted": 0}

    # Sample if too many (to limit LLM calls)
    if len(pred_edges) > max_triples:
        import random
        random.seed(42)
        pred_edges = random.sample(pred_edges, max_triples)

    # Build source corpus triple embeddings (embed all source triples once)
    source_texts = [
        f"{t.get('subj','')} {t.get('rel','affects')} {t.get('obj','')}"
        for t in source_functor.triples
        if t.get('subj') and t.get('obj')
    ]
    if not source_texts:
        return {"fidelity": 0.0, "n_back_translated": 0, "n_predicted": len(pred_edges)}

    source_embeds = embedder.encode(source_texts)   # (S, 384)

    # Back-translate each predicted triple
    fidelities = []
    n_ok = 0
    for u, v, d in pred_edges:
        rel  = d.get("relation", "affects")
        resp = _back_translate_triple(u, rel, v, source_domain, batch_delay)
        back_text = _parse_back_translation(resp)
        if not back_text:
            continue
        n_ok += 1
        back_embed = embedder.encode([back_text])            # (1, 384)
        sims       = embedder.cosine_similarity_matrix(back_embed, source_embeds)[0]  # (S,)
        fidelities.append(float(sims.max()))

    if not fidelities:
        return {"fidelity": 0.0, "n_back_translated": 0, "n_predicted": len(pred_edges)}

    return {
        "fidelity":          float(np.mean(fidelities)),
        "fidelity_max":      float(np.max(fidelities)),
        "fidelity_min":      float(np.min(fidelities)),
        "n_back_translated": n_ok,
        "n_predicted":       len(pred_edges),
    }


# ── Batch evaluation ───────────────────────────────────────────────────────────

def run_back_translation_evaluation(
    target_queries: List[str],
    source_functor: CausalFunctor,
    source_domain: str,
    method_preds: Dict[str, Dict[str, nx.DiGraph]],
    embedder: Optional[SharedEmbedder] = None,
    max_triples_per_query: int = 5,
    max_queries: int = 10,
    out_dir=None,
) -> pd.DataFrame:
    """
    Run back-translation evaluation across methods.

    Limits LLM calls to max_queries * max_triples_per_query per method
    (default: 10 queries × 5 triples = 50 LLM calls per method).

    Args:
        max_queries:            evaluate on first N queries (to limit API cost)
        max_triples_per_query:  max triples to back-translate per predicted graph
    """
    if embedder is None:
        embedder = SharedEmbedder.get()

    queries_to_eval = target_queries[:max_queries]
    rows = []

    for method, preds in method_preds.items():
        print(f"\n[BackTranslate] Evaluating {method} on {len(queries_to_eval)} queries…")
        for q in queries_to_eval:
            pred = preds.get(q, nx.DiGraph())
            scores = back_translation_fidelity(
                pred, source_functor, source_domain,
                embedder=embedder,
                max_triples=max_triples_per_query,
            )
            rows.append({"query": q, "method": method, **scores})
            print(f"  {method} | {q[:40]}: fidelity={scores.get('fidelity', 0):.3f} "
                  f"({scores.get('n_back_translated', 0)} triples)")

    df = pd.DataFrame(rows)

    print("\n" + "="*70)
    print("BACK-TRANSLATION FIDELITY (Dynamic Equivalence Test)")
    print("="*70)
    if "fidelity" in df.columns:
        summary = df.groupby("method")["fidelity"].agg(["mean", "std"])
        print(summary.to_string())
    print("="*70 + "\n")

    if out_dir:
        from pathlib import Path
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "back_translation_evaluation.csv", index=False)
        print(f"[BackTranslate] Saved → {out_dir / 'back_translation_evaluation.csv'}")

    return df
