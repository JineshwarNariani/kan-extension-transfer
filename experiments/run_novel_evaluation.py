"""
experiments/run_novel_evaluation.py

Novel Multi-Perspective Evaluation — runs all 7 new evaluation frameworks
on the already-computed predictions (no new SLURM job or LLM extraction needed).

Evaluations implemented:

  Semantic (metric fix):
    0. Semantic Soft-F1     — BERTScore-style triple matching (Hungarian + greedy)
                              Addresses the root cause of exact-match F1=0

  Inspired by parallels from other fields:
    1. Structural Motifs    — Gentner's Structure-Mapping Theory (cognitive science)
                              Relational pattern fingerprinting; ignores entity labels
    2. Biological Homology  — Comparative anatomy (biology)
                              Graph skeleton similarity; same "bone structure"
    3. Universality Classes — Renormalization Group Theory (physics)
                              Which of 6 universal causal patterns dominate?
    4. Back-Translation     — Nida's Dynamic Equivalence (linguistics/translation)
                              LLM round-trip fidelity test (requires LLM)
    5. Soft Kan / RN Layer  — Causal Density Function (Mahadevan textbook)
                              Re-run Kan with ρ(c,d)=sim/freq^α weighting
    6. Interpretive Coherence — Gadamer's Fusion of Horizons (hermeneutics)
                              LLM-as-judge: does the transfer make economic sense?

Usage:
    # All evaluations (LLM-based need API key set)
    python experiments/run_novel_evaluation.py

    # Skip LLM-dependent evaluations (back-translation and coherence)
    python experiments/run_novel_evaluation.py --skip-llm

    # Single source domain
    python experiments/run_novel_evaluation.py --source legal

    # Adjust soft-F1 similarity threshold
    python experiments/run_novel_evaluation.py --soft-threshold 0.35

Prerequisites:
    results/kan_predictions/predictions_medical.pkl   (from run_kan.py)
    results/kan_predictions/predictions_legal.pkl
    results/extraction/{medical,legal,economic}/relational_triples.jsonl
    results/query_split/  (test queries from run_kan.py)
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Force unbuffered output so SLURM log captures progress in real time
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

from config import EXTRACTION_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR
from functor.causal_functor import CausalFunctor
from functor.query_vocab    import load_test_queries
from functor.embedder       import SharedEmbedder

from evaluation.semantic_eval          import run_semantic_evaluation
from evaluation.structural_motif_eval  import run_motif_evaluation
from evaluation.homology_eval          import run_homology_evaluation
from evaluation.universality_eval      import run_universality_evaluation

from kan.soft_coend import soft_left_kan_extension
from kan.soft_end   import soft_right_kan_extension


NOVEL_TABLES = TABLES_DIR / "novel_evaluation"


# ── Data loading helpers ───────────────────────────────────────────────────────

def _load_functor(corpus: str) -> CausalFunctor:
    path = EXTRACTION_DIR / corpus / "relational_triples.jsonl"
    return CausalFunctor(path, domain_name=corpus)


def _load_predictions(src_name: str) -> dict:
    path = RESULTS_DIR / "kan_predictions" / f"predictions_{src_name}.pkl"
    with path.open("rb") as f:
        return pickle.load(f)


def _build_gt_cache_bulk(
    functor: "CausalFunctor",
    queries: List[str],
    embedder: "SharedEmbedder",
) -> dict:
    """
    Build {query → DiGraph} cache using ONE bulk encode call instead of N.

    CausalFunctor.__call__ re-encodes the query every invocation (~10s overhead).
    This function encodes all queries in a single batch, then computes similarity
    to the functor's pre-computed topic embeddings to build each graph.
    """
    query_embeds = embedder.encode(queries)   # (N_queries, 384) — ONE call
    sim_matrix   = embedder.cosine_similarity_matrix(
        query_embeds, functor.topic_embeds
    )  # (N_queries, N_topics)

    cache = {}
    for i, q in enumerate(queries):
        sims    = sim_matrix[i]
        top_idx = np.argsort(sims)[::-1][: functor.fuzzy_k]
        G = nx.DiGraph()
        for idx in top_idx:
            w = float(sims[idx])
            if w < functor.sim_threshold:
                continue
            topic = functor.topics[idx]
            for triple in functor.topic_to_triples[topic]:
                from functor.causal_functor import _normalise_entity
                subj = _normalise_entity(triple.get("subj", ""))
                obj  = _normalise_entity(triple.get("obj",  ""))
                rel  = triple.get("rel", "affects")
                if not subj or not obj:
                    continue
                if G.has_edge(subj, obj):
                    G[subj][obj]["weight"] = max(G[subj][obj]["weight"], w)
                else:
                    G.add_edge(subj, obj, relation=rel, weight=w, source_topic=topic)
        cache[q] = G
    return cache


# ── Summary printer ────────────────────────────────────────────────────────────

def _print_master_summary(all_results: dict) -> None:
    """Print a consolidated table of all evaluation dimensions, across all methods."""

    METRIC_MAP = {
        "semantic":         ("soft_f1",            "SoftF1"),
        "motif":            ("motif_similarity",    "Motif"),
        "homology":         ("homology_score",      "Homol"),
        "universality":     ("universality_score",  "Univ"),
        "back_translation": ("fidelity",            "BackT"),
        "coherence":        ("coherence_score",     "Coher"),
        "soft_kan":         ("soft_f1",             "ρSoftF1"),
    }

    print("\n" + "="*100)
    print("MASTER NOVEL EVALUATION SUMMARY")
    print("="*100)
    print(f"{'Method':<34} {'SoftF1':>8} {'Motif':>8} {'Homol':>8} "
          f"{'Univ':>8} {'BackT':>8} {'Coher':>8} {'ρSoftF1':>8}")
    print("-"*100)

    for src_name, evals in all_results.items():
        # Collect all unique methods across all DataFrames
        all_methods: list = []
        for df in evals.values():
            if df is not None and not df.empty and "method" in df.columns:
                for m in df["method"].unique():
                    if m not in all_methods:
                        all_methods.append(m)

        # Print standard methods first, then soft Kan
        ordered = [m for m in ["left_kan", "right_kan", "naive_rag"] if m in all_methods]
        ordered += [m for m in all_methods if m not in ordered]

        for method in ordered:
            row_vals: dict = {}
            for eval_name, (col, label) in METRIC_MAP.items():
                df = evals.get(eval_name)
                if df is None or df.empty:
                    continue
                if "method" not in df.columns:
                    continue
                sub = df[df["method"] == method]
                if sub.empty:
                    continue
                if col in sub.columns:
                    row_vals[label] = float(sub[col].mean())

            label_str = f"{method}/{src_name}"
            def _fmt(key): return f"{row_vals[key]:>8.3f}" if key in row_vals else f"{'—':>8}"
            print(f"  {label_str:<32} "
                  f"{_fmt('SoftF1')} {_fmt('Motif')} {_fmt('Homol')} "
                  f"{_fmt('Univ')} {_fmt('BackT')} {_fmt('Coher')} {_fmt('ρSoftF1')}")

        print()   # blank line between source domains

    print("="*100)
    print("SoftF1 = semantic soft-F1 (BERTScore-style)")
    print("ρSoftF1 = soft-F1 after RN-derivative reweighting (soft Kan extension)")
    print("Motif = structural motif fingerprint cosine sim (Gentner SME)")
    print("Homol  = graph skeleton cosine sim (biological homology)")
    print("Univ   = universality class preservation (1 - JSD, physics RG)")
    print("BackT  = back-translation fidelity (Nida dynamic equivalence, needs LLM)")
    print("Coher  = interpretive coherence 0–1 (Gadamer fusion, needs LLM)")
    print("="*100 + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Novel multi-perspective evaluation")
    parser.add_argument("--source", choices=["medical", "legal", "all"], default="all")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM-dependent evaluations (back-translation + coherence)")
    parser.add_argument("--soft-threshold", type=float, default=0.40,
                        help="Cosine similarity threshold for semantic soft-F1 (default 0.40)")
    parser.add_argument("--soft-kan-alpha", type=float, default=0.5,
                        help="RN-derivative exponent α for soft Kan (0=uniform, 1=full RN)")
    parser.add_argument("--max-llm-queries", type=int, default=10,
                        help="Max queries for LLM evaluations (back-translation, coherence)")
    args = parser.parse_args()

    source_names = ["medical", "legal"] if args.source == "all" else [args.source]

    NOVEL_TABLES.mkdir(parents=True, exist_ok=True)

    # ── Load shared resources ──────────────────────────────────────────────────
    print("\n[Setup] Loading embedder and functors…")
    embedder     = SharedEmbedder.get()
    test_queries = load_test_queries(RESULTS_DIR / "query_split")
    econ_functor = _load_functor("economic")

    print(f"[Setup] {len(test_queries)} test queries loaded.")

    all_results: dict = {}

    for src_name in source_names:
        print(f"\n{'='*70}")
        print(f"SOURCE DOMAIN: {src_name.upper()}")
        print(f"{'='*70}")

        src_functor  = _load_functor(src_name)
        bundle       = _load_predictions(src_name)
        method_preds = {
            "left_kan":  bundle["left_kan"],
            "right_kan": bundle["right_kan"],
            "naive_rag": bundle["naive_rag"],
        }

        src_out = NOVEL_TABLES / src_name
        src_out.mkdir(parents=True, exist_ok=True)

        evals: dict = {}

        # Build shared GT graph cache with ONE bulk encode call instead of N.
        # _build_gt_cache_bulk encodes all queries in one batch, then builds graphs.
        print("\n[Setup] Pre-computing GT graphs (bulk query-encode)…")
        gt_cache = _build_gt_cache_bulk(econ_functor, test_queries, embedder)
        print(f"[Setup] {len(gt_cache)} GT graphs cached.")

        # ── 0. Semantic Soft-F1 ────────────────────────────────────────────────
        print(f"\n── 0. SEMANTIC SOFT-F1 (threshold={args.soft_threshold}) ──")
        df_sem = run_semantic_evaluation(
            test_queries, econ_functor, method_preds,
            embedder=embedder,
            sim_threshold=args.soft_threshold,
            out_dir=src_out / "semantic",
            _gt_cache=gt_cache,
        )
        evals["semantic"] = df_sem

        # ── 1. Structural Motif Evaluation (Gentner) ───────────────────────────
        print("\n── 1. STRUCTURAL MOTIF EVALUATION (Gentner SME) ──")
        df_motif = run_motif_evaluation(
            test_queries, econ_functor, method_preds,
            out_dir=src_out / "motif",
            _gt_cache=gt_cache,
        )
        evals["motif"] = df_motif

        # ── 2. Homology Evaluation (Biology) ──────────────────────────────────
        print("\n── 2. HOMOLOGY EVALUATION (Biological Skeleton) ──")
        df_homology = run_homology_evaluation(
            test_queries, econ_functor, method_preds,
            out_dir=src_out / "homology",
            _gt_cache=gt_cache,
        )
        evals["homology"] = df_homology

        # ── 3. Universality Class Evaluation (Physics) ─────────────────────────
        print("\n── 3. UNIVERSALITY CLASS EVALUATION (RG Theory) ──")
        df_univ = run_universality_evaluation(
            test_queries, econ_functor, method_preds,
            embedder=embedder,
            out_dir=src_out / "universality",
            _gt_cache=gt_cache,
        )
        evals["universality"] = df_univ

        # ── 4. Soft Kan (RN-Derivative) ────────────────────────────────────────
        print(f"\n── 4. SOFT KAN EXTENSION (RN-derivative α={args.soft_kan_alpha}) ──")
        soft_left_preds  = soft_left_kan_extension(
            src_functor, test_queries, alpha=args.soft_kan_alpha
        )
        soft_right_preds = soft_right_kan_extension(
            src_functor, test_queries, alpha=args.soft_kan_alpha
        )
        soft_method_preds = {
            "soft_left_kan":  soft_left_preds,
            "soft_right_kan": soft_right_preds,
        }
        # Evaluate soft Kan with semantic soft-F1 (reuse GT cache)
        df_soft = run_semantic_evaluation(
            test_queries, econ_functor, soft_method_preds,
            embedder=embedder,
            sim_threshold=args.soft_threshold,
            out_dir=src_out / "soft_kan",
            _gt_cache=gt_cache,
        )
        evals["soft_kan"] = df_soft
        # Also run motif/homology on soft Kan predictions (reuse GT cache)
        run_motif_evaluation(
            test_queries, econ_functor, soft_method_preds,
            out_dir=src_out / "soft_kan",
            _gt_cache=gt_cache,
        )

        # ── 5. Back-Translation (Dynamic Equivalence) ─────────────────────────
        df_bt = None
        if not args.skip_llm:
            print("\n── 5. BACK-TRANSLATION FIDELITY (Nida Dynamic Equivalence) ──")
            from evaluation.back_translation_eval import run_back_translation_evaluation
            df_bt = run_back_translation_evaluation(
                test_queries,
                source_functor=src_functor,
                source_domain=src_name,
                method_preds=method_preds,
                embedder=embedder,
                max_triples_per_query=5,
                max_queries=args.max_llm_queries,
                out_dir=src_out / "back_translation",
            )
        else:
            print("\n── 5. BACK-TRANSLATION FIDELITY — SKIPPED (--skip-llm) ──")
        evals["back_translation"] = df_bt

        # ── 6. Interpretive Coherence (Gadamer) ───────────────────────────────
        df_coh = None
        if not args.skip_llm:
            print("\n── 6. INTERPRETIVE COHERENCE (Gadamer Fusion-of-Horizons) ──")
            from evaluation.interpretive_coherence_eval import run_coherence_evaluation
            df_coh = run_coherence_evaluation(
                test_queries,
                source_domain=src_name,
                method_preds=method_preds,
                max_triples_per_query=5,
                max_queries=args.max_llm_queries,
                out_dir=src_out / "coherence",
            )
        else:
            print("\n── 6. INTERPRETIVE COHERENCE — SKIPPED (--skip-llm) ──")
        evals["coherence"] = df_coh

        all_results[src_name] = evals

        # ── Per-source combined summary ─────────────────────────────────────────
        _save_source_summary(src_name, evals, src_out)

    # ── Master summary ─────────────────────────────────────────────────────────
    _print_master_summary(all_results)

    # Save master summary as JSON
    _save_master_json(all_results, NOVEL_TABLES / "master_summary.json")

    print("[NOVEL EVALUATION COMPLETE]")
    print(f"Results: {NOVEL_TABLES}")


def _save_source_summary(src_name: str, evals: dict, out_dir: Path) -> None:
    """Save a per-source CSV combining all evaluation dimensions."""
    dfs = []
    for eval_name, df in evals.items():
        if df is None or df.empty:
            continue
        df = df.copy()
        df["eval_type"] = eval_name
        dfs.append(df)
    if dfs:
        combined = pd.concat(dfs, ignore_index=True, sort=False)
        combined.to_csv(out_dir / "combined_evaluations.csv", index=False)
        print(f"\n[Summary] Combined results for {src_name} → {out_dir / 'combined_evaluations.csv'}")


def _save_master_json(all_results: dict, out_path: Path) -> None:
    """Save mean scores per method per eval type as JSON for quick inspection."""
    summary = {}
    for src_name, evals in all_results.items():
        summary[src_name] = {}
        for eval_name, df in evals.items():
            if df is None or df.empty:
                summary[src_name][eval_name] = "skipped"
                continue
            per_method = {}
            metric_cols = {
                "semantic":         "soft_f1",
                "motif":            "motif_similarity",
                "homology":         "homology_score",
                "universality":     "universality_score",
                "back_translation": "fidelity",
                "coherence":        "coherence_score",
                "soft_kan":         "soft_f1",
            }
            col = metric_cols.get(eval_name)
            if col and col in df.columns:
                for method in df["method"].unique() if "method" in df.columns else []:
                    sub = df[df["method"] == method]
                    per_method[method] = {
                        "mean": round(float(sub[col].mean()), 4),
                        "std":  round(float(sub[col].std()),  4),
                    }
            summary[src_name][eval_name] = per_method

    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[Summary] Master JSON → {out_path}")


if __name__ == "__main__":
    main()
