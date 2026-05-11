"""
evaluation/naturality_test.py — Empirical Naturality Test

Tests whether the Kan extension behaves "naturally" under query refinements.

A natural transformation η: F_med ⟹ F_econ would satisfy the naturality square:
    F_econ(f) ∘ η_q = η_q' ∘ F_med(f)   for each query refinement f: q → q'

In practice, we test the computational version:
    Does evaluate(Lan(q'), F_econ(q')) > evaluate(Lan(q), F_econ(q'))?
    i.e., does refining the query improve the Kan extension's prediction quality?

If yes → the transfer is "natural" (responds coherently to query refinements)
If no  → the functor is not natural for this domain pair
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from evaluation.metrics import evaluate
from functor.causal_functor import CausalFunctor
from functor.embedder import SharedEmbedder


# Pre-defined query refinement pairs (general → specific)
# These are fixed before running any experiments.
REFINEMENT_PAIRS: List[Tuple[str, str]] = [
    (
        "monetary policy effects",
        "federal funds rate increases on unemployment rate",
    ),
    (
        "inflation causes",
        "supply chain disruptions causing consumer price inflation",
    ),
    (
        "economic growth factors",
        "business investment leading to GDP expansion",
    ),
    (
        "labor market dynamics",
        "wage growth effect on consumer spending",
    ),
    (
        "financial sector risks",
        "bank credit tightening reducing small business lending",
    ),
]


def run_naturality_test(
    source_functor: CausalFunctor,
    ground_truth_functor: CausalFunctor,
    kan_func,                       # left_kan_extension or right_kan_extension
    method_name: str,
    out_dir: Path = None,
) -> pd.DataFrame:
    """
    For each refinement pair (q_general, q_specific):
      1. Compute Lan(q_general) and evaluate against F_econ(q_specific)
      2. Compute Lan(q_specific) and evaluate against F_econ(q_specific)
      3. Test if F1(Lan(q_specific)) > F1(Lan(q_general))

    If the transfer is "natural", refinement should strictly improve prediction.
    """
    embedder = SharedEmbedder.get()
    rows = []

    for q_gen, q_spec in REFINEMENT_PAIRS:
        # Ground truth for the specific query
        gt = ground_truth_functor(q_spec)

        # Kan extension for both queries
        preds_gen  = kan_func(source_functor, [q_gen],  k=10, sim_threshold=0.25)
        preds_spec = kan_func(source_functor, [q_spec], k=10, sim_threshold=0.25)

        m_gen  = evaluate(preds_gen.get(q_gen,   nx.DiGraph()), gt, embedder=embedder)
        m_spec = evaluate(preds_spec.get(q_spec, nx.DiGraph()), gt, embedder=embedder)

        improvement = m_spec["edge_f1"] - m_gen["edge_f1"]
        is_natural  = improvement > 0

        row = {
            "method":        method_name,
            "q_general":     q_gen,
            "q_specific":    q_spec,
            "f1_general":    m_gen["edge_f1"],
            "f1_specific":   m_spec["edge_f1"],
            "improvement":   improvement,
            "natural":       is_natural,
        }
        rows.append(row)

        symbol = "✓" if is_natural else "✗"
        print(f"  [{method_name}] {symbol} {q_gen[:40]} → {q_spec[:40]}"
              f" | ΔF1={improvement:+.3f}")

    df = pd.DataFrame(rows)
    n_natural = df["natural"].sum()
    print(f"\n[Naturality] {method_name}: {n_natural}/{len(df)} pairs show improvement")
    print(f"  Mean ΔF1: {df['improvement'].mean():+.4f}")

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / f"naturality_{method_name}.csv", index=False)

    return df
