"""
experiments/run_kan.py

Step 2: Compute Left Kan, Right Kan, and Naive RAG predictions.
Run after run_extraction.py has completed.

Usage:
    python experiments/run_kan.py
    python experiments/run_kan.py --source medical    # single source domain
    python experiments/run_kan.py --source legal
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import EXTRACTION_DIR, RESULTS_DIR, RANDOM_SEED
from functor.causal_functor import CausalFunctor
from functor.query_vocab import build_query_split, save_query_split, load_test_queries
from kan.coend    import left_kan_extension
from kan.end      import right_kan_extension
from kan.baseline import naive_rag_baseline


def _load_functor(corpus: str) -> CausalFunctor:
    triples_path = EXTRACTION_DIR / corpus / "relational_triples.jsonl"
    if not triples_path.exists():
        raise FileNotFoundError(
            f"No triples for '{corpus}'. Run run_extraction.py first."
        )
    return CausalFunctor(triples_path, domain_name=corpus)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["medical", "legal", "all"], default="all")
    parser.add_argument("--k",             type=int,   default=10)
    parser.add_argument("--sim-threshold", type=float, default=0.25)
    parser.add_argument("--consensus",     type=float, default=0.60)
    args = parser.parse_args()

    out_dir = RESULTS_DIR / "kan_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Build query split from economic corpus ──────────────────────────────
    econ_triples_path = EXTRACTION_DIR / "economic" / "relational_triples.jsonl"
    split_dir = RESULTS_DIR / "query_split"
    test_queries_path = split_dir / "test_queries.json"

    if not test_queries_path.exists():
        train_topics, test_topics = build_query_split(
            econ_triples_path, seed=RANDOM_SEED, n_test_target=20
        )
        save_query_split(train_topics, test_topics, split_dir)
    else:
        print(f"[QuerySplit] Loaded from {test_queries_path}")

    test_queries = load_test_queries(split_dir)
    print(f"\n[KAN] {len(test_queries)} test queries loaded.")

    # ── Load functors ───────────────────────────────────────────────────────
    source_names = ["medical", "legal"] if args.source == "all" else [args.source]
    source_functors = {name: _load_functor(name) for name in source_names}
    econ_functor    = _load_functor("economic")

    # ── Run all methods per source domain ───────────────────────────────────
    for src_name, src_functor in source_functors.items():
        print(f"\n{'─'*50}")
        print(f"SOURCE: {src_name.upper()}")
        print(f"{'─'*50}")

        t0 = time.time()

        left_preds  = left_kan_extension(
            src_functor, test_queries,
            k=args.k, sim_threshold=args.sim_threshold
        )
        right_preds = right_kan_extension(
            src_functor, test_queries,
            k=args.k, sim_threshold=args.sim_threshold,
            consensus_frac=args.consensus
        )
        rag_preds   = naive_rag_baseline(src_functor, test_queries, k=args.k * 2)

        elapsed = time.time() - t0
        print(f"[KAN] Computed all predictions in {elapsed:.1f}s")

        # Save predictions
        preds_bundle = {
            "source":        src_name,
            "test_queries":  test_queries,
            "left_kan":      left_preds,
            "right_kan":     right_preds,
            "naive_rag":     rag_preds,
            "params": {
                "k":             args.k,
                "sim_threshold": args.sim_threshold,
                "consensus":     args.consensus,
            },
        }
        pkl_path = out_dir / f"predictions_{src_name}.pkl"
        with pkl_path.open("wb") as f:
            pickle.dump(preds_bundle, f)
        print(f"[KAN] Saved → {pkl_path}")

    print("\n[KAN] Done. Run run_evaluation.py next.")


if __name__ == "__main__":
    main()
