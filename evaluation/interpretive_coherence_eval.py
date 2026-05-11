"""
evaluation/interpretive_coherence_eval.py

Interpretive Coherence Evaluation — Parallel 6: Gadamer's Fusion of Horizons.

Core insight:
    Hans-Georg Gadamer (Truth and Method, 1960) argued that genuine understanding
    requires a "fusion of horizons" (Horizontverschmelzung): the reader's current
    understanding (their "horizon") meets the text's original context, producing
    new meaning that neither possessed alone.

    In cross-domain causal transfer:
    - The source domain (legal/medical) has its own "horizon" of concepts
    - The target domain (economics) has its own "horizon"
    - A successful Kan extension performs the fusion: the economic claim it produces
      should be interpretable and coherent FROM THE PERSPECTIVE OF AN ECONOMIST,
      even though it arose from legal/medical structural patterns

    This is NOT asking "is this economically factually correct?" — it is asking
    "does this make sense AS an economic causal claim?"  An answer can be novel,
    uncertain, or even wrong, but still be COHERENT (Gadamer's criterion).

    The opposite of coherence is incoherence: a claim like
    "cytokine storm reduces federal funds rate" is incoherent — not because it is
    false, but because the concepts do not belong in the same causal domain.

Implementation:
    Use an LLM as a Gadamerian "ideal reader" — an agent who inhabits both
    the economic horizon (as an economist) and can evaluate whether the transferred
    claim has undergone a successful fusion.

    Scoring rubric (0–5):
    5 — Valid: a well-known or plausible economic causal relationship
    4 — Plausible: not a textbook fact but economically sensible
    3 — Uncertain: could be interpreted economically with effort
    2 — Strained: requires heavy reinterpretation; terminology clash
    1 — Confused: the entities do not belong in economics
    0 — Incoherent: economically meaningless

    The interpretive coherence score = mean score / 5 (normalised to [0, 1]).
    A score of 0.6 (= 3.0/5) means the transferred claims are "interpretable
    with effort" — a meaningful Gadamerian fusion has partially occurred.
    A score of 0.8+ means the fusion is largely successful.

Comparison with back-translation:
    Back-translation (Parallel 4) tests STRUCTURAL round-trip fidelity:
        did the transfer preserve the causal skeleton?
    Interpretive coherence tests TARGET-DOMAIN VALIDITY:
        does the transferred claim make sense IN THE TARGET domain?
    These are complementary: high back-translation + low coherence = transferred
    structure but wrong vocabulary; high coherence + low back-translation = lucky
    surface match without structural preservation.
"""

from __future__ import annotations

import os
import re
import time
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd

from functor.causal_functor import CausalFunctor


# ── LLM judge helper ───────────────────────────────────────────────────────────

def _call_llm_judge(prompt: str, max_tokens: int = 64) -> Optional[str]:
    """Call LLM judge via LiteLLM proxy. Returns None on failure."""
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
        print(f"  [CoherenceJudge] LLM call failed: {str(exc)[:80]}")
        return None


def _parse_score(response: Optional[str]) -> Optional[float]:
    """Extract integer 0–5 from LLM response."""
    if not response:
        return None
    # Look for a digit 0-5 in the response
    matches = re.findall(r'\b([0-5])\b', response)
    if matches:
        return float(matches[0])
    # Look for decimal like "3.5"
    matches = re.findall(r'\b([0-5](?:\.[0-9])?)\b', response)
    if matches:
        return float(matches[0])
    return None


def _judge_triple_coherence(
    subj: str,
    rel: str,
    obj: str,
    source_domain: str,
    batch_delay: float = 0.5,
) -> Optional[float]:
    """
    Ask LLM to rate the economic coherence of a transferred triple.
    Returns a score 0–5 or None on failure.
    """
    prompt = (
        f"You are an expert economist evaluating causal claims.\n\n"
        f"The following causal statement was derived from {source_domain} knowledge "
        f"via structural analogy and applied to economics:\n\n"
        f"  \"{subj} {rel} {obj}\"\n\n"
        f"Rate this as an ECONOMIC causal claim on this scale:\n"
        f"  5 — Economically valid: a well-known or clearly plausible economic relationship\n"
        f"  4 — Economically plausible: not standard but makes economic sense\n"
        f"  3 — Uncertain: could be interpreted economically with effort\n"
        f"  2 — Strained: terminology conflicts; requires heavy reinterpretation\n"
        f"  1 — Confused: entities do not belong in economic analysis\n"
        f"  0 — Incoherent: economically meaningless\n\n"
        f"Respond with ONLY a single integer 0, 1, 2, 3, 4, or 5."
    )
    time.sleep(batch_delay)
    response = _call_llm_judge(prompt)
    return _parse_score(response)


# ── Core computation ───────────────────────────────────────────────────────────

def interpretive_coherence(
    predicted: nx.DiGraph,
    source_domain: str,
    max_triples: int = 10,
    batch_delay: float = 0.4,
) -> Dict[str, float]:
    """
    Compute interpretive coherence score for a single predicted graph.

    Returns:
        coherence_score  — mean LLM rating / 5 (normalised to [0,1])
        mean_raw_score   — mean LLM rating (0–5 scale)
        n_judged         — number of triples successfully judged
        n_predicted      — total edges in graph
    """
    pred_edges = list(predicted.edges(data=True))
    if not pred_edges:
        return {
            "coherence_score": 0.0, "mean_raw_score": 0.0,
            "n_judged": 0, "n_predicted": 0,
        }

    # Sample if too many
    if len(pred_edges) > max_triples:
        import random
        random.seed(42)
        pred_edges = random.sample(pred_edges, max_triples)

    scores = []
    for u, v, d in pred_edges:
        rel  = d.get("relation", "affects")
        s    = _judge_triple_coherence(u, rel, v, source_domain, batch_delay)
        if s is not None:
            scores.append(s)

    if not scores:
        return {
            "coherence_score": 0.0, "mean_raw_score": 0.0,
            "n_judged": 0, "n_predicted": len(pred_edges),
        }

    mean_raw = float(np.mean(scores))
    return {
        "coherence_score": mean_raw / 5.0,
        "mean_raw_score":  mean_raw,
        "n_judged":        len(scores),
        "n_predicted":     len(pred_edges),
    }


# ── Batch evaluation ───────────────────────────────────────────────────────────

def run_coherence_evaluation(
    target_queries: List[str],
    source_domain: str,
    method_preds: Dict[str, Dict[str, nx.DiGraph]],
    max_triples_per_query: int = 5,
    max_queries: int = 10,
    batch_delay: float = 0.5,
    out_dir=None,
) -> pd.DataFrame:
    """
    Run interpretive coherence evaluation across methods.

    Limits LLM calls to max_queries * max_triples_per_query per method.

    The LLM acts as a "Gadamerian ideal reader" — an economist who evaluates
    whether the transferred claims have successfully undergone horizon-fusion.
    """
    queries_to_eval = target_queries[:max_queries]
    rows = []

    for method, preds in method_preds.items():
        print(f"\n[CoherenceJudge] Evaluating {method} on {len(queries_to_eval)} queries…")
        for q in queries_to_eval:
            pred = preds.get(q, nx.DiGraph())
            scores = interpretive_coherence(
                pred, source_domain,
                max_triples=max_triples_per_query,
                batch_delay=batch_delay,
            )
            rows.append({"query": q, "method": method, **scores})
            print(f"  {method} | {q[:40]}: coherence={scores.get('coherence_score', 0):.3f} "
                  f"({scores.get('n_judged', 0)} triples, "
                  f"raw={scores.get('mean_raw_score', 0):.2f}/5)")

    df = pd.DataFrame(rows)

    print("\n" + "="*70)
    print("INTERPRETIVE COHERENCE EVALUATION (Gadamer Fusion-of-Horizons)")
    print("="*70)
    if "coherence_score" in df.columns:
        summary = df.groupby("method")[["coherence_score", "mean_raw_score"]].agg(["mean", "std"])
        print(summary.to_string())
        print("\nInterpretation:")
        print("  0.80+ → Successful fusion: claims are economically valid")
        print("  0.60–0.79 → Partial fusion: claims are interpretable with effort")
        print("  0.40–0.59 → Strained fusion: structural transfer but terminology clash")
        print("  <0.40 → Failed fusion: transferred structure is economically incoherent")
    print("="*70 + "\n")

    if out_dir:
        from pathlib import Path
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "coherence_evaluation.csv", index=False)
        print(f"[CoherenceJudge] Saved → {out_dir / 'coherence_evaluation.csv'}")

    return df
