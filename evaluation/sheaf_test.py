"""
evaluation/sheaf_test.py — Empirical Sheaf Gluing Test

Tests whether causal knowledge satisfies the sheaf gluing axiom:
    "Local patches of causal knowledge from distinct source domains
     should produce a consistent global covering when combined."

Formally:
    If F_med and F_legal are two source functors covering different
    'neighbourhoods' of the Grothendieck topology on scientific domains,
    then their joint Kan extension F_{med+legal} should satisfy:

        F1(Lan_{med+legal}(d)) > max(F1(Lan_med(d)), F1(Lan_legal(d)))

    for most target queries d.

If this holds → causal knowledge is locally patchable (sheaf condition)
If not        → the domains' causal ontologies are incompatible at overlap
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import networkx as nx
import numpy as np
import pandas as pd

from evaluation.metrics import evaluate
from functor.causal_functor import CausalFunctor
from functor.embedder import SharedEmbedder
from kan.coend import left_kan_extension


def make_joint_functor(
    f1: CausalFunctor,
    f2: CausalFunctor,
    joint_name: str = "joint",
) -> CausalFunctor:
    """
    Construct F_{1+2} by merging the triples of two source functors.
    This simulates the 'gluing' of two local causal knowledge patches.
    """
    import tempfile, json
    from pathlib import Path

    merged_triples = f1.triples + f2.triples

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        for t in merged_triples:
            tmp.write(json.dumps(t) + "\n")
        tmp_path = Path(tmp.name)

    try:
        joint = CausalFunctor(tmp_path, domain_name=joint_name)
    finally:
        tmp_path.unlink(missing_ok=True)
    return joint


def run_sheaf_test(
    source_functors: Dict[str, CausalFunctor],
    ground_truth_functor: CausalFunctor,
    target_queries: List[str],
    kan_func=None,
    out_dir: Path = None,
) -> pd.DataFrame:
    """
    Test the sheaf gluing condition:
        F1(joint) > max(F1(individual sources))

    Args:
        source_functors:       {"medical": F_med, "legal": F_legal}
        ground_truth_functor:  F_econ
        target_queries:        20 held-out query strings
        kan_func:              defaults to left_kan_extension
    """
    if kan_func is None:
        from kan.coend import left_kan_extension as kan_func

    embedder = SharedEmbedder.get()
    names    = list(source_functors.keys())

    # Individual Kan extensions
    individual_preds: Dict[str, Dict[str, nx.DiGraph]] = {}
    for name, functor in source_functors.items():
        print(f"\n[Sheaf] Computing {name} Kan extension…")
        individual_preds[name] = kan_func(functor, target_queries)

    # Joint Kan extension
    print("\n[Sheaf] Computing JOINT Kan extension…")
    functors_list = list(source_functors.values())
    joint_functor = functors_list[0]
    for f in functors_list[1:]:
        joint_functor = make_joint_functor(joint_functor, f)
    joint_preds = kan_func(joint_functor, target_queries)

    # Evaluate all
    rows = []
    sheaf_holds_count = 0

    for q in target_queries:
        gt = ground_truth_functor(q)

        f1_individual = {}
        for name in names:
            m = evaluate(individual_preds[name].get(q, nx.DiGraph()), gt, embedder=embedder)
            f1_individual[name] = m["edge_f1"]

        m_joint = evaluate(joint_preds.get(q, nx.DiGraph()), gt, embedder=embedder)
        f1_joint = m_joint["edge_f1"]

        max_individual = max(f1_individual.values()) if f1_individual else 0.0
        sheaf_holds    = f1_joint > max_individual

        if sheaf_holds:
            sheaf_holds_count += 1

        row = {
            "query":        q,
            "f1_joint":     f1_joint,
            "max_individual": max_individual,
            "sheaf_holds":  sheaf_holds,
            "improvement":  f1_joint - max_individual,
        }
        for name in names:
            row[f"f1_{name}"] = f1_individual[name]
        rows.append(row)

        symbol = "✓" if sheaf_holds else "✗"
        print(f"  {symbol}  joint={f1_joint:.3f}  "
              + "  ".join(f"{n}={v:.3f}" for n, v in f1_individual.items())
              + f"  Δ={f1_joint - max_individual:+.3f}")

    df = pd.DataFrame(rows)
    frac = sheaf_holds_count / len(target_queries)
    print(f"\n[Sheaf] Condition holds for {sheaf_holds_count}/{len(target_queries)} queries ({frac:.0%})")
    print(f"  Mean improvement: {df['improvement'].mean():+.4f}")
    print(f"  INTERPRETATION: {'SHEAF CONDITION HOLDS' if frac > 0.5 else 'SHEAF CONDITION FAILS'}")

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "sheaf_test_results.csv", index=False)
        summary = {
            "sheaf_holds_fraction": frac,
            "mean_improvement": float(df["improvement"].mean()),
            "n_queries": len(target_queries),
        }
        (out_dir / "sheaf_summary.json").write_text(json.dumps(summary, indent=2))

    return df
