"""
test_smoke.py — Full pipeline smoke test. No API key needed.

Creates synthetic triples that mimic what Democritus produces, then runs
the complete kan → evaluation → visualization chain.  Should complete in
under 60 seconds and confirm there are no import or logic errors before
you submit the SLURM job.

Run from the project root:
    python test_smoke.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

PASS = []
FAIL = []


def check(name: str):
    class _ctx:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, tb):
            if exc_type:
                FAIL.append((name, traceback.format_exc()))
                print(f"  FAIL  {name}")
                return True   # suppress so remaining checks run
            PASS.append(name)
            print(f"  pass  {name}")
    return _ctx()


# ── 1. Core imports ────────────────────────────────────────────────────────────
print("\n[1] Imports")

with check("config"):
    from config import (
        MED_DIR, ECON_DIR, LEGAL_DIR, EXTRACTION_DIR, RESULTS_DIR,
        FIGURES_DIR, TABLES_DIR, KAN_K_PRIMARY, KAN_SIM_THRESHOLD,
        KAN_CONSENSUS_FRAC, KAN_K_RANGE, KAN_SIM_RANGE, KAN_CONSENSUS_RANGE,
        SBERT_MODEL, PRIMARY_MODEL, RANDOM_SEED,
    )

with check("functor.embedder"):
    from functor.embedder import SharedEmbedder

with check("functor.causal_functor"):
    from functor.causal_functor import CausalFunctor, _normalise_entity

with check("functor.query_vocab"):
    from functor.query_vocab import build_query_split, save_query_split, load_test_queries

with check("kan.coend"):
    from kan.coend import left_kan_extension

with check("kan.end"):
    from kan.end import right_kan_extension

with check("kan.baseline"):
    from kan.baseline import naive_rag_baseline

with check("evaluation.metrics"):
    from evaluation.metrics import evaluate, summarise_results

with check("evaluation.evaluator"):
    from evaluation.evaluator import run_evaluation, run_sensitivity_analysis

with check("evaluation.naturality_test"):
    from evaluation.naturality_test import run_naturality_test, REFINEMENT_PAIRS

with check("evaluation.sheaf_test"):
    from evaluation.sheaf_test import run_sheaf_test, make_joint_functor

with check("evaluation.domain_proximity"):
    from evaluation.domain_proximity import domain_proximity, pairwise_domain_proximities

with check("visualization.ablation_plots"):
    from visualization.ablation_plots import (
        plot_kan_crossover, plot_sheaf_heatmap, plot_sensitivity
    )

with check("visualization.causal_graph_viz"):
    from visualization.causal_graph_viz import draw_comparison


# ── 2. Embedder ────────────────────────────────────────────────────────────────
print("\n[2] Embedder (loads all-MiniLM-L6-v2)")

with check("SharedEmbedder.get()"):
    emb = SharedEmbedder.get()

with check("encode returns (N,384)"):
    import numpy as np
    out = emb.encode(["hello world", "monetary policy", "clinical trial"])
    assert out.shape == (3, 384), f"got {out.shape}"

with check("cosine_similarity_matrix shape"):
    a = emb.encode(["query one"])
    b = emb.encode(["doc one", "doc two", "doc three"])
    sim = emb.cosine_similarity_matrix(a, b)
    assert sim.shape == (1, 3), f"got {sim.shape}"

with check("embeddings are L2-normalised"):
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"norms={norms}"


# ── 3. Synthetic triples ───────────────────────────────────────────────────────
print("\n[3] Synthetic corpus")

MED_TRIPLES = [
    {"subj": "statin therapy",       "rel": "reduces",    "obj": "LDL cholesterol",     "topic": "cardiovascular treatment"},
    {"subj": "statin therapy",       "rel": "reduces",    "obj": "myocardial infarction","topic": "cardiovascular treatment"},
    {"subj": "hypertension",         "rel": "leads to",   "obj": "stroke",               "topic": "cardiovascular risk"},
    {"subj": "insulin resistance",   "rel": "causes",     "obj": "type 2 diabetes",      "topic": "metabolic disease"},
    {"subj": "metformin",            "rel": "reduces",    "obj": "blood glucose",        "topic": "diabetes treatment"},
    {"subj": "chemotherapy",         "rel": "inhibits",   "obj": "tumor growth",         "topic": "oncology"},
    {"subj": "antibiotic resistance","rel": "increases",  "obj": "infection mortality",  "topic": "infectious disease"},
    {"subj": "vaccine",              "rel": "prevents",   "obj": "infection",            "topic": "infectious disease"},
    {"subj": "blood pressure",       "rel": "affects",    "obj": "kidney function",      "topic": "renal health"},
    {"subj": "obesity",              "rel": "increases",  "obj": "cardiovascular risk",  "topic": "metabolic disease"},
]

ECON_TRIPLES = [
    {"subj": "interest rate hike",   "rel": "reduces",    "obj": "inflation",            "topic": "monetary policy"},
    {"subj": "interest rate hike",   "rel": "increases",  "obj": "unemployment",         "topic": "monetary policy"},
    {"subj": "quantitative easing",  "rel": "increases",  "obj": "asset prices",         "topic": "monetary policy"},
    {"subj": "fiscal stimulus",      "rel": "increases",  "obj": "GDP growth",           "topic": "fiscal policy"},
    {"subj": "trade deficit",        "rel": "weakens",    "obj": "currency value",       "topic": "trade policy"},
    {"subj": "credit tightening",    "rel": "reduces",    "obj": "business investment",  "topic": "credit markets"},
    {"subj": "wage growth",          "rel": "increases",  "obj": "consumer spending",    "topic": "labor market"},
    {"subj": "supply shock",         "rel": "causes",     "obj": "price inflation",      "topic": "supply chain"},
    {"subj": "federal funds rate",   "rel": "affects",    "obj": "mortgage rates",       "topic": "interest rates"},
    {"subj": "corporate tax cut",    "rel": "increases",  "obj": "business investment",  "topic": "fiscal policy"},
    {"subj": "unemployment rate",    "rel": "affects",    "obj": "consumer confidence",  "topic": "labor market"},
    {"subj": "money supply",         "rel": "causes",     "obj": "inflation",            "topic": "monetary policy"},
]

LEGAL_TRIPLES = [
    {"subj": "antitrust enforcement","rel": "reduces",    "obj": "market concentration", "topic": "competition law"},
    {"subj": "regulatory penalty",   "rel": "deters",     "obj": "consumer fraud",       "topic": "enforcement"},
    {"subj": "merger review",        "rel": "prevents",   "obj": "monopoly formation",   "topic": "competition law"},
    {"subj": "disclosure requirement","rel": "increases", "obj": "market transparency",  "topic": "securities law"},
    {"subj": "credit regulation",    "rel": "limits",     "obj": "predatory lending",    "topic": "financial regulation"},
]

def _write_triples(triples: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for t in triples:
            f.write(json.dumps(t) + "\n")


with tempfile.TemporaryDirectory() as tmpdir:
    tmp = Path(tmpdir)
    med_path   = tmp / "medical"   / "relational_triples.jsonl"
    econ_path  = tmp / "economic"  / "relational_triples.jsonl"
    legal_path = tmp / "legal"     / "relational_triples.jsonl"
    _write_triples(MED_TRIPLES,   med_path)
    _write_triples(ECON_TRIPLES,  econ_path)
    _write_triples(LEGAL_TRIPLES, legal_path)

    # ── 4. CausalFunctor ──────────────────────────────────────────────────────
    print("\n[4] CausalFunctor")
    with check("load medical functor"):
        F_med   = CausalFunctor(med_path,   domain_name="medical")

    with check("load economic functor"):
        F_econ  = CausalFunctor(econ_path,  domain_name="economic")

    with check("load legal functor"):
        F_legal = CausalFunctor(legal_path, domain_name="legal")

    with check("F_econ('monetary policy') returns DiGraph"):
        import networkx as nx
        G = F_econ("monetary policy")
        assert isinstance(G, nx.DiGraph)
        assert len(G.edges()) > 0, "no edges for monetary policy query"

    with check("domain_centroid shape (384,)"):
        c = F_med.domain_centroid()
        assert c.shape == (384,)

    with check("topic_summary keys"):
        s = F_med.topic_summary()
        assert "n_triples" in s and "n_topics" in s

    # ── 5. Query split ─────────────────────────────────────────────────────────
    print("\n[5] Query split")
    split_dir = tmp / "query_split"

    with check("build_query_split"):
        train, test = build_query_split(econ_path, seed=42, n_test_target=4)
        assert len(test) >= 1

    with check("save and reload query split"):
        save_query_split(train, test, split_dir)
        loaded = load_test_queries(split_dir)
        assert loaded == test

    # ── 6. Kan extensions ─────────────────────────────────────────────────────
    print("\n[6] Kan extensions")
    test_qs = test  # use the 4 held-out economic topics as queries

    with check("left_kan_extension returns dict[str, DiGraph]"):
        left_preds = left_kan_extension(F_med, test_qs, k=3, sim_threshold=0.10)
        assert isinstance(left_preds, dict)
        assert all(isinstance(v, nx.DiGraph) for v in left_preds.values())

    with check("right_kan_extension returns dict[str, DiGraph]"):
        right_preds = right_kan_extension(F_med, test_qs, k=3, sim_threshold=0.10, consensus_frac=0.4)
        assert isinstance(right_preds, dict)

    with check("naive_rag_baseline returns dict[str, DiGraph]"):
        rag_preds = naive_rag_baseline(F_med, test_qs, k=6)
        assert isinstance(rag_preds, dict)

    with check("left_kan edges have weight and sources"):
        for G in left_preds.values():
            for u, v, d in G.edges(data=True):
                assert "weight" in d, f"edge ({u},{v}) missing weight"
                assert "sources" in d, f"edge ({u},{v}) missing sources"
                break

    with check("right_kan edges have consensus field"):
        for G in right_preds.values():
            for u, v, d in G.edges(data=True):
                assert "consensus" in d, f"edge ({u},{v}) missing consensus"
                break

    with check("right_kan(consensus=1.0) is subset of left_kan"):
        strict_right = right_kan_extension(F_med, test_qs[:1], k=5, sim_threshold=0.05, consensus_frac=1.0)
        loose_left   = left_kan_extension(F_med,  test_qs[:1], k=5, sim_threshold=0.05)
        q = test_qs[0]
        right_edges = set(strict_right[q].edges())
        left_edges  = set(loose_left[q].edges())
        assert right_edges.issubset(left_edges), \
            f"Ran(consensus=1.0) has edges not in Lan: {right_edges - left_edges}"

    # ── 7. Metrics ────────────────────────────────────────────────────────────
    print("\n[7] Evaluation metrics")
    q0 = test_qs[0]
    gt = F_econ(q0)

    with check("evaluate returns all required keys"):
        m = evaluate(left_preds[q0], gt, embedder=emb, semantic_threshold=0.80)
        for key in ["edge_f1", "relation_f1", "node_jaccard", "pred_size", "true_size"]:
            assert key in m, f"missing key: {key}"

    with check("evaluate empty prediction gives zero F1"):
        m_empty = evaluate(nx.DiGraph(), gt, embedder=emb)
        assert m_empty["edge_f1"] == 0.0
        assert m_empty["edge_recall"] == 0.0

    with check("evaluate identical prediction gives F1=1.0"):
        m_perfect = evaluate(gt, gt, embedder=emb)
        assert m_perfect["edge_f1"] == 1.0, f"got {m_perfect['edge_f1']}"

    with check("summarise_results"):
        rows = [m, m_empty, m_perfect]
        summary = summarise_results(rows)
        assert "mean_edge_f1" in summary

    # ── 8. run_evaluation (mini) ──────────────────────────────────────────────
    print("\n[8] run_evaluation (mini, no I/O)")
    with check("run_evaluation DataFrame shape"):
        import pandas as pd
        df = run_evaluation(
            target_queries=test_qs,
            ground_truth_functor=F_econ,
            method_preds={"left_kan": left_preds, "right_kan": right_preds, "naive_rag": rag_preds},
            out_dir=None,
        )
        expected_rows = len(test_qs) * 3   # 3 methods
        assert len(df) == expected_rows, f"expected {expected_rows} rows, got {len(df)}"
        assert "edge_f1" in df.columns

    # ── 9. Naturality test ────────────────────────────────────────────────────
    print("\n[9] Naturality test")
    with check("run_naturality_test runs without crash"):
        nat_df = run_naturality_test(
            source_functor=F_med,
            ground_truth_functor=F_econ,
            kan_func=left_kan_extension,
            method_name="left_kan_medical",
            out_dir=None,
        )
        assert "natural" in nat_df.columns
        assert len(nat_df) == len(REFINEMENT_PAIRS)

    # ── 10. Sheaf test ────────────────────────────────────────────────────────
    print("\n[10] Sheaf test")
    with check("make_joint_functor merges triples"):
        F_joint = make_joint_functor(F_med, F_legal)
        assert len(F_joint.triples) == len(F_med.triples) + len(F_legal.triples)

    with check("run_sheaf_test runs without crash"):
        sheaf_df = run_sheaf_test(
            source_functors={"medical": F_med, "legal": F_legal},
            ground_truth_functor=F_econ,
            target_queries=test_qs,
            out_dir=None,
        )
        assert "sheaf_holds" in sheaf_df.columns
        assert "f1_joint" in sheaf_df.columns

    # ── 11. Domain proximity ──────────────────────────────────────────────────
    print("\n[11] Domain proximity")
    with check("domain_proximity in [-1, 1]"):
        prox = domain_proximity(F_med, F_econ)
        assert -1.0 <= prox <= 1.0, f"got {prox}"
        print(f"        medical↔economic proximity: {prox:.3f}")

    with check("pairwise_domain_proximities"):
        d = pairwise_domain_proximities({"medical": F_med, "legal": F_legal}, F_econ)
        assert set(d.keys()) == {"medical", "legal"}
        print(f"        medical={d['medical']:.3f}  legal={d['legal']:.3f}")

    # ── 12. Sensitivity analysis (tiny sweep) ─────────────────────────────────
    print("\n[12] Sensitivity analysis (2 k values × 2 sim values + consensus sweep)")
    sens_df = None
    with check("run_sensitivity_analysis runs and returns DataFrame"):
        import unittest.mock as mock
        # run_sensitivity_analysis does `from config import ...` inside the function,
        # so we must patch config directly (not the evaluator module's namespace).
        with mock.patch("config.KAN_K_RANGE", [3, 5]), \
             mock.patch("config.KAN_SIM_RANGE", [0.10, 0.25]), \
             mock.patch("config.KAN_CONSENSUS_RANGE", [0.4, 0.8]), \
             mock.patch("config.KAN_K_PRIMARY", 3), \
             mock.patch("config.KAN_SIM_THRESHOLD", 0.10):
            sens_df = run_sensitivity_analysis(
                target_queries=test_qs,
                ground_truth_functor=F_econ,
                source_functor=F_med,
                out_dir=None,
            )
        assert "mean_edge_f1" in sens_df.columns
        assert "consensus_frac" in sens_df.columns

    # ── 13. Visualizations ────────────────────────────────────────────────────
    print("\n[13] Visualizations")
    fig_dir = tmp / "figures"
    fig_dir.mkdir()

    with check("draw_comparison saves PNG"):
        draw_comparison(
            query=test_qs[0],
            gt_graph=F_econ(test_qs[0]),
            predicted={
                "left_kan":  left_preds.get(test_qs[0]) or nx.DiGraph(),
                "right_kan": right_preds.get(test_qs[0]) or nx.DiGraph(),
                "naive_rag": rag_preds.get(test_qs[0]) or nx.DiGraph(),
            },
            out_path=fig_dir / "smoke_comparison.png",
        )
        assert (fig_dir / "smoke_comparison.png").exists()

    with check("plot_sensitivity saves PNG"):
        if sens_df is None:
            raise RuntimeError("sens_df not available (sensitivity check failed)")
        plot_sensitivity(sens_df, fig_dir / "smoke_sensitivity.png")
        assert (fig_dir / "smoke_sensitivity.png").exists()

    with check("plot_sheaf_heatmap saves PNG"):
        plot_sheaf_heatmap(
            sheaf_df,
            source_names=["medical", "legal"],
            out_path=fig_dir / "smoke_sheaf.png",
        )
        assert (fig_dir / "smoke_sheaf.png").exists()

    with check("plot_kan_crossover saves PNG"):
        plot_kan_crossover(
            domain_proximities={"medical": 0.31, "legal": 0.47},
            left_f1s={"medical": 0.22, "legal": 0.28},
            right_f1s={"medical": 0.18, "legal": 0.31},
            out_path=fig_dir / "smoke_crossover.png",
        )
        assert (fig_dir / "smoke_crossover.png").exists()

    # ── 14. Corpus check ──────────────────────────────────────────────────────
    print("\n[14] Live corpus readiness")
    for domain, path in [("medical", MED_DIR), ("economic", ECON_DIR), ("legal", LEGAL_DIR)]:
        docs = list(path.glob("*.txt"))
        chars = sum(len(p.read_text(encoding="utf-8", errors="replace")) for p in docs)
        pages = chars // 1500
        with check(f"{domain}: {len(docs)} docs / {pages} pages"):
            assert len(docs) > 0, f"no docs in {path}"


# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"SMOKE TEST COMPLETE: {len(PASS)} passed, {len(FAIL)} failed")
print("="*60)

if FAIL:
    print("\nFAILURES:")
    for name, tb in FAIL:
        print(f"\n  [{name}]")
        for line in tb.strip().split("\n"):
            print(f"    {line}")
    sys.exit(1)
else:
    print("\nAll checks passed. Safe to submit the SLURM job.")
    print("(Run corpus fetchers first if corpus pages are still low.)")
