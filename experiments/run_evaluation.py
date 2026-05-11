"""
experiments/run_evaluation.py

Step 3: Run full evaluation + all three novel experiments.
Run after run_kan.py has completed.

Usage:
    python experiments/run_evaluation.py
    python experiments/run_evaluation.py --source medical
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import EXTRACTION_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR
from functor.causal_functor import CausalFunctor
from functor.query_vocab    import load_test_queries
from evaluation.evaluator   import run_evaluation, run_sensitivity_analysis
from evaluation.naturality_test import run_naturality_test
from evaluation.sheaf_test  import run_sheaf_test
from evaluation.domain_proximity import domain_proximity, pairwise_domain_proximities
from visualization.causal_graph_viz import draw_comparison
from visualization.ablation_plots   import (
    plot_kan_crossover, plot_sheaf_heatmap, plot_sensitivity
)
from kan.coend import left_kan_extension
from kan.end   import right_kan_extension


def _load_functor(corpus: str) -> CausalFunctor:
    path = EXTRACTION_DIR / corpus / "relational_triples.jsonl"
    return CausalFunctor(path, domain_name=corpus)


def _load_predictions(src_name: str) -> dict:
    path = RESULTS_DIR / "kan_predictions" / f"predictions_{src_name}.pkl"
    with path.open("rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["medical", "legal", "all"], default="all")
    parser.add_argument("--skip-sensitivity", action="store_true")
    args = parser.parse_args()

    test_queries  = load_test_queries(RESULTS_DIR / "query_split")
    econ_functor  = _load_functor("economic")

    source_names = ["medical", "legal"] if args.source == "all" else [args.source]
    source_functors = {n: _load_functor(n) for n in source_names}

    # ── 1. Main Evaluation ─────────────────────────────────────────────────
    print("\n" + "="*60 + "\n1. MAIN EVALUATION\n" + "="*60)
    for src_name in source_names:
        bundle = _load_predictions(src_name)
        eval_out = TABLES_DIR / f"eval_{src_name}"
        df = run_evaluation(
            target_queries=test_queries,
            ground_truth_functor=econ_functor,
            method_preds={
                "left_kan":  bundle["left_kan"],
                "right_kan": bundle["right_kan"],
                "naive_rag": bundle["naive_rag"],
            },
            out_dir=eval_out,
        )

        # Visualise first 3 queries
        for i, q in enumerate(test_queries[:3]):
            gt = econ_functor(q)
            import networkx as nx
            draw_comparison(
                query=q, gt_graph=gt,
                predicted={
                    "left_kan":  bundle["left_kan"].get(q)  or nx.DiGraph(),
                    "right_kan": bundle["right_kan"].get(q) or nx.DiGraph(),
                    "naive_rag": bundle["naive_rag"].get(q) or nx.DiGraph(),
                },
                out_path=FIGURES_DIR / f"comparison_{src_name}_q{i}.png",
            )

    # ── 2. Sensitivity Analysis ────────────────────────────────────────────
    if not args.skip_sensitivity:
        print("\n" + "="*60 + "\n2. SENSITIVITY ANALYSIS\n" + "="*60)
        for src_name, src_functor in source_functors.items():
            sens_df = run_sensitivity_analysis(
                test_queries, econ_functor, src_functor,
                out_dir=TABLES_DIR / "sensitivity"
            )
            plot_sensitivity(sens_df, FIGURES_DIR / f"sensitivity_{src_name}.png")

    # ── 3. Naturality Test ─────────────────────────────────────────────────
    print("\n" + "="*60 + "\n3. NATURALITY TEST\n" + "="*60)
    for src_name, src_functor in source_functors.items():
        for method_name, kan_func in [
            ("left_kan",  left_kan_extension),
            ("right_kan", right_kan_extension),
        ]:
            run_naturality_test(
                source_functor=src_functor,
                ground_truth_functor=econ_functor,
                kan_func=kan_func,
                method_name=f"{method_name}_{src_name}",
                out_dir=TABLES_DIR / "naturality",
            )

    # ── 4. Sheaf Gluing Test ───────────────────────────────────────────────
    if len(source_functors) > 1:
        print("\n" + "="*60 + "\n4. SHEAF GLUING TEST\n" + "="*60)
        sheaf_df = run_sheaf_test(
            source_functors=source_functors,
            ground_truth_functor=econ_functor,
            target_queries=test_queries,
            out_dir=TABLES_DIR / "sheaf",
        )
        plot_sheaf_heatmap(
            sheaf_df,
            source_names=source_names,
            out_path=FIGURES_DIR / "sheaf_heatmap.png",
        )

    # ── 5. Kan Crossover Plot ──────────────────────────────────────────────
    print("\n" + "="*60 + "\n5. KAN CROSSOVER CURVE\n" + "="*60)
    proximities = pairwise_domain_proximities(source_functors, econ_functor)
    print(f"Domain proximities: {proximities}")

    # Load per-domain F1 for crossover plot
    left_f1s  = {}
    right_f1s = {}
    for src_name in source_names:
        eval_csv = TABLES_DIR / f"eval_{src_name}" / "evaluation_results.csv"
        if eval_csv.exists():
            import pandas as pd
            df_eval = pd.read_csv(eval_csv)
            left_f1s[src_name]  = df_eval[df_eval["method"] == "left_kan"]["edge_f1"].mean()
            right_f1s[src_name] = df_eval[df_eval["method"] == "right_kan"]["edge_f1"].mean()

    if left_f1s and right_f1s:
        plot_kan_crossover(
            domain_proximities=proximities,
            left_f1s=left_f1s,
            right_f1s=right_f1s,
            out_path=FIGURES_DIR / "kan_crossover.png",
        )

    print("\n[EVALUATION COMPLETE]")
    print(f"Results: {TABLES_DIR}")
    print(f"Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
