# Kan-Extension Transfer for Cross-Domain Causal Reasoning

**Final project — CMPSCI 692CT: Category Theory for AGI (UMass Amherst, Spring 2026).**
**Author:** Jineshwar Nariani.

This repository contains the implementation, results, and report for the project
*"Cross-Domain Causal Transfer via Kan Extensions: An Empirical Test of the Coend / End Asymmetry and a Multi-Perspective Evaluation Framework"*, which empirically tests Mahadevan's *Kan-Do-Calculus* (the categorical formulation of cross-domain causal transfer from the *Categories for AGI* textbook) on three real corpora and twenty held-out target queries.

The full project report lives at the repository root: **[`REPORT.md`](../REPORT.md)** (markdown) and **[`report.tex`](../report.tex)** / **[`report.pdf`](../report.pdf)** (LaTeX + PDF). The figures it references all live in `results/figures/`.

---

## What is in this repository

```
kan_transfer/
├── README.md                       # this file
├── config.py                       # paths and shared constants
├── unity_job.sh                    # SLURM script for the UNITY cluster
│
├── data/acquire/                   # corpus fetchers
│   ├── fed_fetcher.py              # Federal Reserve FOMC + Beige Book + speeches
│   ├── econ_expanded_fetcher.py    # additional economic content
│   ├── legal_fetcher.py            # SEC litigation releases + enforcement actions
│   ├── pubmed_fetcher.py           # PubMed abstract pull (deprecated)
│   └── pubmed_fulltext_fetcher.py  # PubMed full-text via PMC
│
├── extraction/                     # Democritus-driven causal-triple extraction
│   ├── democritus_runner.py
│   └── anthropic_client.py         # optional Anthropic Claude backend
│
├── functor/                        # category-theoretic primitives
│   ├── causal_functor.py           # F : C_queries → C_causalGraphs
│   ├── embedder.py                 # shared SentenceTransformer
│   └── query_vocab.py              # train / test query split
│
├── kan/                            # four prediction methods
│   ├── coend.py                    # left Kan extension (colimit / union)
│   ├── end.py                      # right Kan extension (limit / intersection)
│   ├── soft_coend.py               # RN-derivative-weighted soft left Kan
│   ├── soft_end.py                 # RN-derivative-weighted soft right Kan
│   └── baseline.py                 # naive RAG baseline
│
├── evaluation/                     # seven evaluators
│   ├── metrics.py                  # exact-match edge-F1
│   ├── semantic_eval.py            # BERTScore-style soft-F1
│   ├── structural_motif_eval.py    # Gentner structure-mapping
│   ├── homology_eval.py            # comparative-anatomy skeleton features
│   ├── universality_eval.py        # Wilson renormalization-group classes
│   ├── back_translation_eval.py    # Nida dynamic equivalence
│   ├── interpretive_coherence_eval.py  # Gadamer fusion-of-horizons
│   ├── domain_proximity.py
│   ├── naturality_test.py
│   ├── sheaf_test.py
│   └── evaluator.py                # batch runner
│
├── experiments/                    # top-level entry points
│   ├── run_extraction.py
│   ├── run_kan.py
│   ├── run_evaluation.py
│   ├── run_novel_evaluation.py     # the main 7-dimension run
│   └── run_all.py
│
├── visualization/
│   ├── causal_graph_viz.py         # per-query graph comparison plots
│   └── ablation_plots.py
│
├── cliff_integration/
│   └── kan_transfer_agentic.py     # CLIFF agentic-router stub (WIP)
│
└── results/
    ├── extraction/{economic,legal,medical}/   # extracted triples
    ├── kan_predictions/                       # predictions_*.pkl (gitignored)
    ├── query_split/                           # train / test queries
    ├── figures/                               # 10 plots used in the report
    └── tables/
        ├── eval_{legal,medical}/              # exact-match edge-F1 (zero)
        ├── sensitivity/                       # k / τ / consensus sweeps
        ├── sheaf/                             # sheaf-gluing test
        ├── naturality/                        # naturality square checks
        └── novel_evaluation/                  # the 7-dimension master table
            ├── master_summary.json            # ← canonical headline numbers
            ├── legal/  {semantic, motif, homology, universality, back_translation,
            │            coherence, soft_kan}/
            └── medical/{ same subdirectories }
```

## Headline result

The `master_summary.json` for the legal → economic source (proximity 0.480):

| Method      | SoftF1 | Motif | Homol. | Univ. | BackT | Coher. |
|-------------|-------:|------:|-------:|------:|------:|-------:|
| left_kan    |  0.029 | 0.135 |  0.592 | 0.766 | 0.339 |  **0.500** |
| right_kan   |  0.003 | 0.100 |  0.095 | 0.743 | 0.072 |  0.100 |
| naive_rag   |  0.031 | 0.435 |  0.997 | 0.870 | 0.703 |  0.984 |
| soft_left_kan | 0.027 | —     | —      | —     | —     |  —     |

**Take-aways.**
1. **Left Kan beats Right Kan on every dimension** — empirical confirmation of the coend / end asymmetry predicted by *Kan-Do-Calculus*.
2. **Naive RAG appears to "win" most absolute scores but does not actually transfer:** its homology score of 0.997 is by construction (RAG returns source-domain triples verbatim, and a graph is similar to itself). The metric that rewards honest cross-domain transfer — coherence under Gadamer's fusion-of-horizons — favours Left Kan over Right Kan by a 5× margin.
3. **Proximity predicts transfer:** medical → economic (proximity 0.035) yields zero transfer on every metric; legal → economic (proximity 0.480) yields populated transfer.
4. **Soft Kan (RN-derivative reweighting) densifies the predicted graph 5×** without improving semantic F1 — empirical evidence that the bottleneck is *vocabulary grounding*, not *coend selectivity*. This motivates the still-unimplemented RN entity-translation layer in the textbook's three-layer Sheaf–Kan–RN framework.

## Reproducing the full pipeline

```bash
# 1. Acquire corpora
python data/acquire/fed_fetcher.py
python data/acquire/legal_fetcher.py
python data/acquire/pubmed_fulltext_fetcher.py

# 2. Extract causal triples via Democritus
python experiments/run_extraction.py

# 3. Compute Kan extensions and the naive-RAG baseline
python experiments/run_kan.py

# 4. Run the seven-dimension evaluation suite
python experiments/run_novel_evaluation.py             # full run (needs LLM key)
python experiments/run_novel_evaluation.py --skip-llm  # without back-trans/coherence

# Or use the SLURM job script on UNITY
sbatch unity_job.sh
```

The full run completes in ~30 min on UNITY (job 56498824, 250 GB allocation, single node). Without LLM-based evaluators it completes in ~12 min on a laptop with a GPU.

## Dependencies

See `requirements.txt`. Core stack: `sentence-transformers`, `networkx`, `pandas`, `numpy`, `scipy`, `openai` (for LLM-based evaluators).

`OPENAI_API_KEY` and `DEMOC_LLM_BASE_URL` (the UMass TheKeymaker gateway URL) are read from `.env`.

## Related project repositories (Prof. Mahadevan)

This project builds on Mahadevan's research toolchain:

- **CLIFF** — first AGI chatbot built on FunctorFlow: <https://github.com/sridharmahadevan/CLIFF_CatAgi>
- **Democritus** — causal-triple extraction: <https://github.com/sridharmahadevan/Democritus_OpenAI>
- **BASKET** — agentic workflows from textual documents: <https://github.com/sridharmahadevan/BASKET>
- **FunctorFlow.jl** — Julia port with symbolic Diagrammatic Backpropagation: <https://juliaknowledge.github.io/FunctorFlow.jl/dev>

## License

MIT. Course-project code; not production-hardened.

## Acknowledgements

Prof. Sridhar Mahadevan for the *Categories for AGI* textbook, the toolchain, and the project conversations that shaped this work. AI coding assistance was used extensively — see the AI-disclosure section in the report.
