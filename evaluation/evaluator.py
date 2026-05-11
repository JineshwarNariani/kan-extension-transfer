"""
evaluation/evaluator.py

Runs the full evaluation comparing Left Kan, Right Kan, and Naive RAG
against the ground-truth economic functor on the 20 held-out queries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd

from evaluation.metrics import evaluate, summarise_results
from evaluation.semantic_eval import run_semantic_evaluation  # noqa: F401 — re-exported
from functor.causal_functor import CausalFunctor
from functor.embedder import SharedEmbedder


def run_evaluation(
    target_queries: List[str],
    ground_truth_functor: CausalFunctor,
    method_preds: Dict[str, Dict[str, nx.DiGraph]],
    out_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Full evaluation across all methods and queries.

    Args:
        target_queries:       list of held-out economic query strings
        ground_truth_functor: F_econ evaluated on test split
        method_preds:         {"left_kan": {q→G}, "right_kan": {q→G}, "naive_rag": {q→G}}
        out_dir:              if set, saves results CSV and summary JSON there

    Returns:
        pd.DataFrame with one row per (method, query)
    """
    embedder = SharedEmbedder.get()
    rows = []

    for method, preds in method_preds.items():
        for q in target_queries:
            gt   = ground_truth_functor(q)
            pred = preds.get(q, nx.DiGraph())

            metrics = evaluate(pred, gt, embedder=embedder)
            rows.append({
                "query":  q,
                "method": method,
                **metrics,
            })

    df = pd.DataFrame(rows)

    # Print summary table
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    summary = df.groupby("method")[
        ["edge_f1", "relation_f1", "node_jaccard"]
    ].agg(["mean", "std"])
    print(summary.to_string())
    print("="*70 + "\n")

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "evaluation_results.csv", index=False)
        print(f"[Eval] Saved per-query results → {out_dir / 'evaluation_results.csv'}")

        # Per-method summary
        summary_records = {}
        for method in df["method"].unique():
            method_rows = df[df["method"] == method].to_dict("records")
            summary_records[method] = summarise_results(method_rows)

        summary_path = out_dir / "evaluation_summary.json"
        summary_path.write_text(json.dumps(summary_records, indent=2), encoding="utf-8")
        print(f"[Eval] Saved summary → {summary_path}")

    return df


def run_sensitivity_analysis(
    target_queries: List[str],
    ground_truth_functor: CausalFunctor,
    source_functor: CausalFunctor,
    out_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Sweep over ALL pre-registered hyperparameter ranges:
      - k × sim_threshold for both left_kan and right_kan
      - consensus_frac for right_kan (at primary k and sim_threshold)
    """
    from kan.coend import left_kan_extension
    from kan.end   import right_kan_extension
    from config    import (KAN_K_RANGE, KAN_SIM_RANGE, KAN_CONSENSUS_RANGE,
                           KAN_K_PRIMARY, KAN_SIM_THRESHOLD)

    records = []
    embedder = SharedEmbedder.get()

    # Sweep 1: k × sim_threshold for left_kan and right_kan
    for k in KAN_K_RANGE:
        for sim_thr in KAN_SIM_RANGE:
            left_preds  = left_kan_extension(source_functor, target_queries, k=k, sim_threshold=sim_thr)
            right_preds = right_kan_extension(source_functor, target_queries, k=k, sim_threshold=sim_thr)

            for method, preds in [("left_kan", left_preds), ("right_kan", right_preds)]:
                f1s = [
                    evaluate(preds.get(q, nx.DiGraph()), ground_truth_functor(q), embedder=embedder)["edge_f1"]
                    for q in target_queries
                ]
                records.append({
                    "method":        method,
                    "k":             k,
                    "sim_threshold": sim_thr,
                    "consensus_frac": 0.6,
                    "mean_edge_f1":  float(np.mean(f1s)),
                    "std_edge_f1":   float(np.std(f1s)),
                })
                print(f"  k={k} sim={sim_thr:.2f} {method}: F1={np.mean(f1s):.3f}")

    # Sweep 2: consensus_frac for right_kan (at primary k and sim_threshold)
    for cfrac in KAN_CONSENSUS_RANGE:
        right_preds = right_kan_extension(
            source_functor, target_queries,
            k=KAN_K_PRIMARY, sim_threshold=KAN_SIM_THRESHOLD,
            consensus_frac=cfrac,
        )
        f1s = [
            evaluate(right_preds.get(q, nx.DiGraph()), ground_truth_functor(q), embedder=embedder)["edge_f1"]
            for q in target_queries
        ]
        records.append({
            "method":        "right_kan",
            "k":             KAN_K_PRIMARY,
            "sim_threshold": KAN_SIM_THRESHOLD,
            "consensus_frac": cfrac,
            "mean_edge_f1":  float(np.mean(f1s)),
            "std_edge_f1":   float(np.std(f1s)),
        })
        print(f"  consensus={cfrac:.2f} right_kan: F1={np.mean(f1s):.3f}")

    df = pd.DataFrame(records)
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "sensitivity_analysis.csv", index=False)
        # Also save consensus-sweep subset separately for easy reading
        consensus_df = df[df["consensus_frac"] != 0.6]
        if not consensus_df.empty:
            consensus_df.to_csv(out_dir / "consensus_sensitivity.csv", index=False)
    return df
