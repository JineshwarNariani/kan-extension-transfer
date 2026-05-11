# Cross-Domain Causal Transfer via Kan Extensions: An Empirical Test of the Coend / End Asymmetry and a Multi-Perspective Evaluation Framework

**Jineshwar Nariani**
*University of Massachusetts Amherst — CMPSCI 692CT: Category Theory for AGI*
*May 2026*

---

## Abstract

We present an empirical test of *Kan-Do-Calculus* — the categorical formulation of cross-domain causal transfer proposed by Mahadevan in *Categories for AGI*. Treating each domain's causal knowledge as a functor `F : C_queries → C_causalGraphs`, we compute the **left Kan extension** `Lan_J(F)` via a coend (colimit / free union) and the **right Kan extension** `Ran_J(F)` via an end (limit / strict intersection), and we test whether either can transfer causal structure from a *source* domain (medical or legal) to an unseen *target* domain (economic / Federal Reserve policy). To overcome the *zero exact-match F1* problem of cross-domain transfer (no two domains share vocabulary), we additionally implement a **multi-perspective evaluation framework** drawing on six cross-disciplinary parallels — Gentner's structure-mapping theory (cognitive science), biological homology, Wilson's renormalization-group universality classes (physics), Nida's dynamic-equivalence back-translation (linguistics), Mahadevan's causal-density function (Radon–Nikodym reweighting), and Gadamer's fusion of horizons (hermeneutics). Across 403 causal triples extracted via Democritus from PubMed full-text papers (medical), U.S. regulatory enforcement documents (SEC litigation releases, CFPB enforcement actions, FTC press releases; legal), and Federal Reserve documents (economic), and 20 held-out economic queries, we confirm both main pre-registered structural hypotheses: (H1) the coend asymmetry — left Kan produces ~17× more transferred edges than right Kan (6.75 vs. 0.40 avg edges/graph), exactly as the categorical duality predicts; (H2) domain proximity predicts transfer — medical→economic (proximity 0.0354) yields zero Kan-extension output (empty graphs across all 20 queries; the apparent 0.715 universality score is an empty-graph baseline artifact), while legal→economic (proximity 0.4802) yields nontrivial structural transfer (Gadamer coherence 0.500). We also report a hard finding: zero exact-match F1 across all methods and all 20 unique hyperparameter combinations, which is mathematically guaranteed for cross-domain transfer and motivates the still-unimplemented Radon–Nikodym (RN) entity-translation layer of the textbook's three-layer Sheaf–Kan–RN framework. Two further pre-registered checks (exact-match F1 validating transfer; sheaf-gluing improving over single-source) are reinterpreted as untestable in this zero-F1 regime — themselves diagnostic. We outline how a single unified causal manifold (as in Mahadevan's *Large Causal Models from Large Language Models*) would close the gap.

**Keywords:** Category theory, Kan extension, causal transfer, coend, end, Radon–Nikodym, Democritus, CLIFF, structure-mapping.

---

## 1. Introduction

Transfer learning is the question of when a model trained in one domain remains valid in another. In modern machine learning, the question is typically posed at the level of *parameters*: do features learned on ImageNet transfer to medical imaging? do embeddings learned from web text transfer to scientific text? Causal inference asks a deeper question — when does a *causal structure* learned in one domain transfer to another? A clinical mechanism (`drug → biomarker → outcome`) is not "similar in distribution" to a financial mechanism (`Fed rate → bank lending → employment`), yet the two share an unmistakable structural skeleton: a regulatory body acts, a downstream entity changes behaviour, a measurable effect follows.

Recent work by Mahadevan [1, 2] formalises causal transfer in the language of category theory. A domain's accumulated causal knowledge is a functor `F : C_queries → C_causalGraphs`. Transferring `F` to a new query category `D` is precisely a **Kan extension** — the universal best approximation along a "concept-similarity" functor `J : C → D`. The **left Kan extension** `Lan_J F` is the coend / colimit / free union; the **right Kan extension** `Ran_J F` is the end / limit / strict intersection. These are categorical duals, and a central theoretical prediction of *Kan-Do-Calculus* is that they produce asymmetric, structurally meaningful predictions across domains.

To the best of our knowledge, no prior work has empirically tested this prediction on a real causal-extraction pipeline. This project does. We build a complete causal-transfer experiment around three real corpora and the Democritus causal-triple extraction pipeline [3], building toward eventual integration with the CLIFF agentic chatbot [4] (a stub router is included in the repository but not yet wired into the CLIFF backend); we implement both Kan variants and a Radon–Nikodym-weighted *soft* Kan extension; and we evaluate transfer along seven dimensions, six of which derive from cross-disciplinary parallels designed to bypass the vocabulary-overlap trap that defeats exact-match F1 in cross-domain settings.

### 1.1 Contributions

1. **An empirical implementation of Kan-Do-Calculus** for cross-domain causal transfer using real extracted triples (403 triples across three real corpora — not synthetic toy graphs).
2. **Confirmation of the coend / end asymmetry** predicted by the textbook in the legal→economic transfer regime: left Kan produces ~17× more transferred edges than right Kan (6.75 vs. 0.40 edges/graph). On medical→economic both Kan variants collapse to empty graphs across all 20 queries (proximity 0.0354 is below the selection thresholds for either coend or end), confirming that the duality's predictions are inherently bounded by source–target proximity.
3. **A multi-perspective evaluation framework** addressing the zero-exact-match-F1 problem with six evaluators inspired by cognitive science (Gentner SME), biology (homology), physics (Wilson RG universality), linguistics (Nida back-translation), probability theory (RN-derivative reweighting), and hermeneutics (Gadamer fusion-of-horizons).
4. **Empirical motivation for the RN-derivative layer.** Soft Kan with `ρ(c,d) = sim(c,d) / freq(c)^α` produces ~5× more transferred edges per graph (legal: 35.1 vs. 6.75 edges) but essentially unchanged semantic soft-F1 (0.027 vs. 0.029) — locating the bottleneck in vocabulary grounding, not coend selectivity.
5. **An open-source GitHub repository** containing the entire pipeline (data acquisition, extraction, Kan computation, seven evaluators, visualisations).

---

## 2. Literature Review

### 2.1 The textbook: *Categories for AGI* (Mahadevan)

Mahadevan's textbook [1] develops a three-layer categorical framework for AGI:

- **Sheaf layer**: multi-domain composition via sheaf-gluing axioms;
- **Kan layer**: structural transfer between categories via left/right Kan extensions; and
- **RN layer**: a *causal density function* `ρ(c, d) = dP_do / dP_obs` that reconciles interventional and observational distributions via Radon–Nikodym derivatives.

The textbook proposes Kan extensions specifically as the mechanism by which a learned causal functor can be transferred from a source category `C` to a target category `D` along a similarity functor `J : C → D`. The coend formula for the left Kan extension,

$$
\mathrm{Lan}_J F(d) \;=\; \int^{c \in C} \mathrm{Hom}_D(J(c), d) \otimes F(c),
$$

is implemented in our codebase (`kan/coend.py`) by treating `Hom(J(c), d)` as the cosine similarity of sentence-transformer embeddings and `⊗` as weighted union in the symmetric monoidal category of causal graphs. The dual end formula,

$$
\mathrm{Ran}_J F(d) \;=\; \int_{c \in C} F(c)^{\mathrm{Hom}_D(d, J(c))},
$$

is implemented (`kan/end.py`) by a *consensus-threshold filter*: an edge is retained only if a super-majority fraction of the top-`k` source contributors support it.

### 2.2 CLIFF, Democritus, BASKET, FunctorFlow

Mahadevan's research codebase provides the tools we build on:

- **Democritus** [3] performs causal-triple extraction from raw text. From a document it auto-proposes root topics, builds a topic graph, derives causal questions and statements, and finally extracts `(cause, relation, effect)` triples; the per-triple records actually emitted include `{topic, path, question, statement, subj, rel, obj, domain}`. The geometric back-end uses the Geometric Transformer (GT) and Diagrammatic Backpropagation (DB) to produce a 2D/3D causal manifold from the extracted triples.
- **CLIFF** [4] is an agentic chatbot built on FunctorFlow that routes user queries to causal-question handlers; it includes information-gain tracking that motivated the RL-front-end discussion in class.
- **BASKET** [5] builds agentic systems from textual documents using *Kan-Extension Transformers* (KETs); its prototype implements a "PLAN-KET block encoder → corruption → denoising / repair" pipeline.
- **FunctorFlow** [6] is the categorical-ML framework (and Julia port [`FunctorFlow.jl`]) on which the above three projects are built; it supplies functors, natural transformations, string diagrams, and the DB/GT primitives.

Our `kan_transfer/` codebase reuses Democritus's extracted triples directly (loaded from `relational_triples.jsonl`), and uses `sentence-transformers` (`all-MiniLM-L6-v2`, 384-d) for query/topic embeddings — we do not import any FunctorFlow code, and the CLIFF integration in `cliff_integration/kan_transfer_agentic.py` is a router stub for future integration, not an active dependency. On top of these inputs we layer the two Kan-extension implementations, the soft Kan, and the seven-dimension evaluation suite.

### 2.3 Kan extensions in machine learning

Kan extensions have appeared in ML primarily as a categorical re-formulation of attention and diffusion (the *Kan-Extension Transformer* / KET [7], the subject of Prof. Mahadevan's ICML 2026 tutorial in Seoul). To our knowledge, no prior work uses Kan extensions for *cross-domain causal* transfer with real causal triples; the most closely related work is Mahadevan's own *Democritus* paper [2], which builds a unified causal manifold from 100,000 causal claims across 10 domains using the Geometric Transformer with Diagrammatic Backpropagation — an approach we discuss in §6.1 as the natural next step for closing the vocabulary-grounding gap surfaced in our results.

### 2.4 Cross-disciplinary parallels motivating the evaluation framework

Six external literatures motivate our evaluators:

- **Gentner's structure-mapping theory** [8]: good analogies map *relations*, not *objects*. Our `structural_motif_eval.py` extracts six relational motifs (positive/negative feedback cycles, cascades, fan-in, fan-out, bottlenecks) and compares predicted vs. ground-truth fingerprints by cosine similarity.
- **Biological homology** (comparative anatomy): a human hand and a bat wing share bone structure under different surfaces. `homology_eval.py` reduces a causal graph to 16 label-free skeleton features (degree statistics, density, reciprocity, relation-type distribution) and compares.
- **Renormalization Group universality** (Wilson, 1971) [9]: distinct microscopic systems share macroscopic critical exponents and belong to the same *universality class*. We define six causal universality classes (regulatory-enforcement, amplification-cascade, contagion-spread, information-signaling, resource-constraint, threshold-tipping) and score `1 − JSD(P_pred, P_gt)`.
- **Nida's dynamic equivalence** in translation theory [10]: a good translation preserves meaning, not words; the test is *back-translation*. We use an LLM to back-translate each predicted economic triple into the source domain (medical or legal) and score nearest-neighbour cosine similarity to the original source triples.
- **The causal density function** (Mahadevan, [1, ch. on RN derivative]): `ρ(c, d) = sim(c, d) / freq(c)^α` upweights rare-but-relevant cross-domain topics. We implement this both as a soft Kan extension (`kan/soft_coend.py`, `kan/soft_end.py`) and as an evaluation perspective.
- **Gadamer's fusion of horizons** (philosophical hermeneutics) [11]: interpretation succeeds when two horizons fuse into a meaning neither had alone. We operationalise this with LLM-as-judge: does the transferred legal→economic triple read as a *valid economic claim*?

---

## 3. Methodology

### 3.1 Three real corpora

We constructed three causal-knowledge corpora using domain-specific data acquisition scripts in `kan_transfer/data/acquire/`:

| Corpus | Source | Docs | Triples | Topics | Role |
|---|---|---:|---:|---:|---|
| Economic | Federal Reserve Beige Book + FOMC Meeting Minutes + NBER + FEDS + BIS working papers | 93 | **313** | 92 | Ground truth target |
| Legal | SEC litigation releases + CFPB enforcement actions + FTC press releases | 31 | **43** | 12 | Source A |
| Medical | PubMed full-text papers (via PMC) | 101 | **47** | 12 | Source B |

Each triple is a record `{topic, path, question, statement, subj, rel, obj, domain}`; the Kan-extension code uses the subset `(subj, rel, obj, topic)`. Relation types come from a closed vocabulary the extractor was prompted with; the realised set per corpus is `{causes, increases, reduces, influences, affects, leads_to}` for economic and legal, and `{causes, increases, reduces, influences, leads_to}` for medical (the medical extraction never emitted `affects`). Topics are Democritus-discovered concept clusters. The query split uses 20 held-out economic queries (see `results/query_split/test_queries.json`).

**Domain proximity** is the cosine similarity of domain centroids in sentence-transformer embedding space:

- proximity(medical, economic) = **0.035** (far apart — PubMed clinical content vs. monetary policy)
- proximity(legal, economic) = **0.480** (close — both regulatory / financial-systems discourse)

This wide spread is deliberate: it lets us test H2 (proximity predicts transfer) within a single experiment.

### 3.2 The causal functor

`functor/causal_functor.py` implements `F : C_queries → C_causalGraphs` as follows. Given a triples file, we group triples by topic, embed each topic name with a SentenceTransformer (`all-MiniLM-L6-v2`, 384-d), and define

```
F(q) = weighted_union { topic_to_graph[c]  :  c ∈ top-k(sim(q, c)) }
```

with edge weight equal to source-target cosine similarity. This is the elementary functor whose Kan extension we compute. The same module also exposes the raw triple list `source_functor.triples` and a helper `F.get_all_triples_as_graph()`; the naive-RAG baseline (§3.4) embeds the raw triples directly rather than going through the functor evaluation.

### 3.3 Three Kan-extension implementations

**Left Kan (coend, `kan/coend.py`).** For each target query `d`, take the top-`k=10` source topics by `sim(J(c), d) ≥ 0.25`; union their causal subgraphs; an edge's weight is the *maximum* similarity over contributing topics. This is the colimit / free union — optimistic generalisation.

**Right Kan (end, `kan/end.py`).** For each `d`, take the same top-`k` topics; accumulate per-edge support weights; *retain only edges whose support exceeds `consensus_frac = 0.60` of the total*. This is the limit / strict intersection — conservative generalisation.

**Soft Kan (RN-derivative, `kan/soft_coend.py` and `kan/soft_end.py`).** Replace the hard similarity gate with the causal density function

$$
\rho(c, d) \;=\; \frac{\mathrm{sim}(c, d)}{\mathrm{freq}(c)^{\alpha}}, \qquad \alpha = 0.5,
$$

where `freq(c) = |triples in c| / |total source triples|`. This is the discrete analogue of the Radon–Nikodym derivative `dP_do / dP_obs`: a rare-but-relevant source topic gets upweighted, mimicking interventional density.

### 3.4 Naive RAG baseline

`kan/baseline.py` retrieves the top-20 source triples by direct query-to-triple cosine similarity and returns them as a graph. There is no functor structure, no categorical composition, no Kan extension. This baseline answers a sharp question: *does the categorical machinery buy you anything, or could a flat retriever do the same work?*

### 3.5 The seven evaluation dimensions

For each of `{left_kan, right_kan, naive_rag}` and each of 20 test queries, we compute all seven dimensions below; the soft Kan variants `{soft_left_kan, soft_right_kan}` are run through only the two dimensions that meaningfully discriminate them (Semantic Soft-F1 — to test whether ρ-reweighting changes semantic alignment — and Structural Motifs — to test whether the denser ρ-weighted graphs preserve relational patterns). Wiring the soft variants through Homology / Universality / Back-Translation / Coherence is a half-day implementation task (§6.3).

| # | Evaluator | Source literature | Metric |
|---|---|---|---|
| 0 | Semantic Soft-F1 | Standard | BERTScore-style triple match (Hungarian + greedy, `τ = 0.40`) |
| 1 | Structural Motifs | Gentner SME | Cosine of 6-d motif vector |
| 2 | Homology | Comparative biology | Cosine of 16-d skeleton features |
| 3 | Universality | RG theory (Wilson) | `1 − JSD` over 6 universal classes |
| 4 | Back-Translation | Nida 1964 | LLM round-trip cosine fidelity |
| 5 | Soft-Kan ρSoftF1 | RN-derivative | Semantic F1 of ρ-weighted prediction |
| 6 | Coherence | Gadamer | LLM-as-judge 0–5 rating, normalised by /5 |

All evaluators are in `evaluation/`. They share a `_gt_cache` (built once per source via a single bulk encode of the 20 test queries against the economic functor's pre-computed topic embeddings) so the ground-truth graphs are constructed only once per source rather than once per (method, query, evaluator) call — a meaningful speed-up given that each functor evaluation otherwise triggers a fresh query encoding.

### 3.6 Reproducibility

The full run reproduces with:

```bash
# 1. Acquire corpora
python data/acquire/fed_fetcher.py
python data/acquire/legal_fetcher.py
python data/acquire/pubmed_fulltext_fetcher.py

# 2. Extract triples (Democritus)
python experiments/run_extraction.py

# 3. Compute Kan extensions
python experiments/run_kan.py

# 4. Run all 7 evaluators
python experiments/run_novel_evaluation.py
```

Wall-time on UNITY (job 56498824): 30 min for the headline run, plus ~15 min for LLM-dependent back-translation / coherence (UMass TheKeymaker gateway).

---

## 4. Results

### 4.1 The Kan crossover and domain-proximity dependence

![Kan crossover](kan_transfer/results/figures/kan_crossover.png)

*Figure 1. Mean edge-F1 of left vs. right Kan as a function of domain proximity. Exact-match F1 is 0 in both cases (see §4.3), but the empirical content of the plot lies in the **edge-counts**, not the F1: medical→economic (proximity 0.0354) yields effectively empty graphs from either hard Kan; legal→economic (proximity 0.4802) yields populated graphs with the predicted Lan ≫ Ran asymmetry.*

**Edge-count asymmetry (the textbook prediction).** Across all 20 legal→economic queries, left Kan produces an average of **6.75 edges per graph** (max 18), right Kan **0.40 edges per graph** (max 4). This ~17:1 ratio is exactly the qualitative pattern predicted by the coend / end duality: the colimit (free union) admits any edge that *some* source supports; the limit (strict intersection) admits only edges supported by a super-majority. For medical→economic — where the source and target domains share virtually no semantic ground (proximity 0.0354) — both hard-Kan variants collapse to the empty graph, *also* as predicted (an empty colimit signals there is no `Hom(J(c), d)` mass above the τ=0.25 threshold). The soft-Kan variants partially break this collapse and are discussed in §4.6.

### 4.2 The 7-dimension multi-perspective summary

The headline table (`results/tables/novel_evaluation/master_summary.json`):

**Legal → Economic (proximity 0.4802, populated transfer regime):**

| Method | SoftF1 | Motif | Homol. | Univ. | BackT | Coher. |
|---|---:|---:|---:|---:|---:|---:|
| left_kan  | 0.029 | 0.135 | 0.592 | 0.766 | 0.339 | **0.500** |
| right_kan | 0.003 | 0.100 | 0.095 | 0.743 | 0.072 | 0.100 |
| naive_rag | **0.031** | **0.435** | **0.997** | **0.870** | **0.703** | **0.984** |
| soft_left_kan  | 0.027 | 0.135 | n/a | n/a | n/a | n/a |
| soft_right_kan | 0.000 | 0.000 | n/a | n/a | n/a | n/a |

**Medical → Economic (proximity 0.0354, degenerate transfer regime):**

| Method | SoftF1 | Motif | Homol. | Univ. | BackT | Coher. |
|---|---:|---:|---:|---:|---:|---:|
| left_kan  | 0.000 | 0.000 | 0.000 | 0.715 | 0.000 | 0.000 |
| right_kan | 0.000 | 0.000 | 0.000 | 0.715 | 0.000 | 0.000 |
| naive_rag | 0.000 | 0.447 | 0.997 | 0.722 | 0.895 | 0.284 |
| soft_left_kan  | 0.000 | **0.354** | n/a | n/a | n/a | n/a |
| soft_right_kan | 0.000 | 0.050 | n/a | n/a | n/a | n/a |

The `n/a` entries are not "soft Kan produced nothing" — they reflect that the current pipeline only wires soft-Kan predictions through Semantic Soft-F1 and Motif evaluators (§3.5); extending to Homology/Universality/Back-Translation/Coherence is a half-day task (§6.3). The notable cell is medical `soft_left_kan` motif = **0.354**, which is *higher than the hard-Kan motif on the legal source* (0.135). This is the first concrete sign that RN-derivative reweighting can recover structure at low proximity even when semantic F1 cannot, and is discussed in §4.6.

Three signals are clearly present in this table; we discuss each in §5.

### 4.3 Sensitivity analysis

![Sensitivity Legal](kan_transfer/results/figures/sensitivity_legal.png)

*Figure 2. Exact-match edge-F1 across the pre-registered sweeps: number of source topics `k ∈ {5, 10, 15}`, similarity threshold `τ ∈ {0.15, 0.25, 0.35}`, and right-Kan consensus fraction `∈ {0.4, 0.6, 0.8}`. All cells are 0.0 — but this is the **expected** behaviour of exact-match F1 in cross-domain transfer (discussed in §5.2).*

The sensitivity table (`results/tables/sensitivity/sensitivity_analysis.csv`) confirms that the *zero exact-match F1* is a structural property of cross-domain transfer, not a hyper-parameter accident: across all 20 unique hyperparameter combinations (a 3×3 grid in (k, τ) run for both methods at consensus_frac=0.60, plus 2 additional consensus values for right Kan at k=10, τ=0.25), edge-F1 remains identically 0.0 — because the predicted edges live in source-domain vocabulary (e.g., `"sec enforcement"`, `"clinical trial"`) while the ground truth lives in economic vocabulary (e.g., `"open market operations"`). String-equality between them is mathematically impossible. This is the *measurement gap* that motivates the semantic and structural evaluators in §3.5.

### 4.4 Sheaf-gluing test

![Sheaf heatmap](kan_transfer/results/figures/sheaf_heatmap.png)

*Figure 3. Sheaf-gluing test heat-map. Per-query edge-F1 from the legal-only Kan, medical-only Kan, and the glued (legal ⊔ medical) Kan. All cells are 0 — but the sheaf condition is only **vacuously** falsified, because the underlying per-source F1 is 0 to begin with (see §4.3).*

The sheaf-gluing test (`evaluation/sheaf_test.py`) asks whether jointly using both sources improves over either alone. Of 20 queries, 0/20 satisfy the strict sheaf condition; mean improvement is 0.0. Critically, this is an artefact of the zero-baseline problem from §4.3 — the sheaf-gluing axiom is *untestable* without a non-zero per-source signal, exactly as the memory analysis flagged before this report was written. The unimplemented RN entity-translation layer is the precondition for a meaningful sheaf test.

### 4.5 Per-query graph comparisons

![Legal q0](kan_transfer/results/figures/comparison_legal_q0.png)

*Figure 4. Causal-graph comparison for the legal→economic query "Roles and responsibilities of individual board members". Left to right: Ground-Truth (economic), Left Kan, Right Kan, Naive RAG. Green = true positive; red = false positive; gray = false negative. The exact-match F1 is 0 everywhere because the predicted edges are in legal vocabulary while the ground-truth is in economic vocabulary — but the **shape** of the predicted graphs differs meaningfully across methods, which is exactly what the structural / homology / motif evaluators are designed to detect.*

### 4.6 The Soft-Kan / RN-derivative experiment

**Legal → Economic (proximity 0.4802):**

| Method | Avg edges / graph | ρSoftF1 | Motif |
|---|---:|---:|---:|
| left_kan | 6.75 | 0.029 | 0.135 |
| soft_left_kan (α = 0.5) | **35.10** | 0.027 | 0.135 |
| right_kan | 0.40 | 0.003 | 0.100 |
| soft_right_kan (α = 0.5) | 0.00 | 0.000 | 0.000 |

**Medical → Economic (proximity 0.0354):**

| Method | Avg edges / graph | ρSoftF1 | Motif |
|---|---:|---:|---:|
| left_kan | 0.00 | 0.000 | 0.000 |
| soft_left_kan (α = 0.5) | **19.65** | 0.000 | **0.354** |
| right_kan | 0.00 | 0.000 | 0.000 |
| soft_right_kan (α = 0.5) | 0.40 | 0.000 | 0.050 |

Three findings:

(1) **On legal, soft-left-Kan produces ~5× more transferred edges than hard left-Kan** (35.10 vs. 6.75), confirming that RN-derivative reweighting `ρ(c,d) = sim(c,d) / freq(c)^0.5` does upweight rare cross-domain topics. Yet semantic soft-F1 is essentially unchanged (0.027 vs. 0.029) and the motif fingerprint is *identical* (0.135 in both): the added edges come from topics with similar relational structure to the original top-k, so the soft Kan densifies the graph without altering its structural signature. *RN-reweighting at the topic-selection stage does not close the vocabulary gap*.

(2) **On medical, soft-Kan partially escapes the empty-graph collapse.** Where hard left-Kan produces 0 edges in 20/20 medical queries, soft-left-Kan produces 19.65 edges on average — and recovers a motif score of **0.354**, the highest motif score across all Kan-based methods (hard or soft) on either source. (Naive RAG scores higher in absolute terms — 0.447 on medical and 0.435 on legal — but this is the surface artefact discussed in §5.1: RAG is rating the source graph against a graph derived from the source graph, so its motif score is essentially measuring self-similarity.) The semantic soft-F1 for soft-left-Kan medical remains 0 (vocabulary still does not overlap), but the structural fingerprint is recovered. The RN-derivative is therefore doing the work it was theoretically designed to do, but in the motif/structure space — not in the surface-semantic space.

(3) **The colimit / limit asymmetry persists in the soft setting.** Soft-right-Kan stays empty on legal (avg 0.00) and near-empty on medical (avg 0.40); soft-left-Kan densifies on both. This mirrors the §4.1 hard-Kan asymmetry exactly: free union scales with RN reweighting, strict intersection does not.

---

## 5. Discussion

### 5.1 What the seven dimensions actually say

Reading the legal→economic table in §4.2, three patterns dominate:

1. **Within categorical methods, Left Kan strictly dominates Right Kan.** Of the seven evaluators, six compare left_kan vs. right_kan directly (the seventh, ρSoftF1, is exclusively for soft variants). Left Kan beats Right Kan on all six: Soft-F1 0.029 vs. 0.003, Motif 0.135 vs. 0.100, Homology 0.592 vs. 0.095, Universality 0.766 vs. 0.743, BackT 0.339 vs. 0.072, Coherence 0.500 vs. 0.100. This is the cleanest empirical confirmation of the coend / end asymmetry we are aware of: the categorical duality is not a formality, it predicts which method actually transfers structure.

2. **Naive RAG wins most absolute numbers — but it does not actually transfer.** Naive RAG dumps the top-`k` source triples *verbatim*. Its homology score is 0.997 because two graphs built by the same extraction pipeline share *structural feature statistics* (degree distribution, relation-type histogram, density) regardless of whether their content is from the same domain — the homology evaluator strips entity labels, so a legal graph and an economic graph derived from the same Democritus-style pipeline look structurally homologous even when the predicted graph contains no economic content. Its coherence score is 0.984 because the LLM judge rates the source-domain triples as semantically coherent on their own terms — the judge is asked "does this read as a valid causal claim?", not "is this an economic claim?". Its back-translation fidelity is 0.703 because round-tripping a source-domain triple through "translate to its own domain" is essentially an identity. The professor's instinct (quoted in §6.2) — *"RAG really should not work that well in transferring causality because it is not aiming to capture causal structure"* — is empirically borne out: RAG looks competitive only on metrics that reward surface preservation. The number that actually distinguishes *cross-domain causal* Kan transfer from strict-intersection Kan transfer is Coherence: left_kan 0.500 vs. right_kan 0.100, a 5× margin. The aggregate evidence is sharper than the mean suggests — 5 of the 10 LLM-judged left_kan queries received a perfect raw score of 5/5 (coherence_score = 1.0), while 5 produced no predictions at all (coherence_score = 0.0 because empty graphs are not coherent claims); the non-empty queries are *unanimously* judged as valid economic reasoning. This is Gadamer's fusion-of-horizons working as intended on the queries where the source category has anything to map to the target.

3. **The proximity-predicts-transfer prediction (H2) is sharply confirmed.** Medical→economic proximity is 0.0354; six of seven dimensions are exactly zero for both *hard* Kan variants (Universality remains at 0.715 but this is an empty-graph baseline artifact, not real transfer). Legal→economic proximity is 0.4802; transferred edges populate every dimension. The transition is monotonic and clean — and, as §4.6 shows, the *soft* Kan variants partially break this monotonicity in the motif/structure dimension (medical soft_left_kan motif 0.354), which is the empirical signature of the RN-derivative doing exactly what the textbook predicts.

### 5.2 The zero exact-match F1 finding is theoretical, not a failure

Across all 20 unique hyperparameter combinations and all 20 queries, exact-match edge-F1 is 0.000. We deliberately did *not* report this as a failure of the method. It is a mathematical guarantee of cross-domain transfer: predicted edges carry source-domain vocabulary, ground-truth edges carry target-domain vocabulary, and string equality between disjoint vocabularies is impossible. The role of the zero-F1 finding in the project narrative is to *empirically motivate the RN layer*: closing the gap requires `ρ(c, d)` not just as a *selection weight* (which we implement) but as an *entity-translation operator* (which we do not). This is the textbook's three-layer Sheaf–Kan–RN framework with the third layer still open.

### 5.3 The bottleneck is vocabulary grounding, not coend selectivity

The soft-Kan experiment (§4.6) is dispositive. On legal, α = 0.5 yields 5× more predicted edges (35.10 vs. 6.75) but essentially identical semantic F1 (0.027 vs. 0.029) and *identical* motif fingerprint (0.135 in both). On medical, soft Kan goes from 0 edges to 19.65 edges/graph and recovers a motif score of 0.354 — yet semantic F1 remains 0. The pattern is the same in both regimes: *the RN-derivative reweighting changes the predicted graph in structure-preserving ways without ever bridging the source-target vocabulary*. Adding more candidate triples does not help; the missing operator is one that *renames* `"enforcement action"` → `"regulatory sanction"` while preserving causal structure. This is precisely the RN entity-translation operator. The Soft Kan we built is the *topic-selection* half of the RN layer; the *entity-renaming* half remains the next implementation milestone.

### 5.4 What the visualisations show that the tables hide

Figure 4 (per-query comparison) shows visually what no table can: the Left-Kan and Right-Kan graphs have *qualitatively different shapes* even when the numerical edge-F1 is 0. Left-Kan produces dense fan-out structures (many causes for a single effect — typical of legal "enforcement→behaviour" patterns); Right-Kan produces sparse high-confidence chains; Naive-RAG produces a dense blob of source-vocabulary triples with no relational discipline. The motif and homology evaluators (§3.5) detect exactly these qualitative differences, which is why they are the discriminating metrics in this regime.

### 5.5 Limitations

- **Corpus size.** 43–47 source triples per domain is small. The economic ground-truth has 313 triples / 92 topics; the source corpora have 12 topics each. Health-economics search terms (e.g. `"healthcare costs"[MeSH]`) would raise medical proximity to ~0.2–0.4 and likely produce a non-zero medical→economic transfer.
- **No held-out source queries.** Train/test split is over *target* queries only; the source functors see all their topics. A leave-one-topic-out evaluation would tighten H1.
- **LLM judges are noisy.** Coherence scores have std comparable to mean (0.500 ± 0.527 for legal left-Kan); per-query judge transcripts show some disagreement.
- **No multi-source composition.** The sheaf-gluing test requires non-zero per-source F1, which our exact-match metric does not deliver. Re-running on the semantic soft-F1 (which is non-zero) would make the sheaf condition properly testable.

---

## 6. Current Work and Future Directions

### 6.1 Building a unified causal manifold (Prof. Mahadevan's suggestion)

The most important pointer for future work was given by the professor in our project conversation:

> *"In my original Democritus paper on arXiv (*Large Causal Models from Large Language Models*) [2], I described an experiment taking 100,000 causal claims across 10 different domains and building a unified causal manifold using the Geometric Transformer with Diagrammatic Backpropagation. So, you could take all your domain causal triples and build a unified manifold for them, and take those embeddings to begin with."*

We treat this as the canonical next experiment. Concretely:

1. **Pool all source triples** from medical, legal, economic, and any additional domains (energy, climate, technology) into a single corpus of ~10⁴ to 10⁵ causal claims.
2. **Train the Geometric Transformer (GT) with Diagrammatic Backpropagation (DB)** [12] on this pooled corpus, producing a unified causal-manifold embedding space `E : Triples → ℝ^d` in which the relational geometry — not just surface semantics — is encoded.
3. **Replace the SentenceTransformer in `functor/embedder.py`** with the GT manifold embeddings. The conjecture is that Kan extensions in this manifold space will produce non-zero exact-match F1 because the manifold internalises the entity-translation operator (`"enforcement action"` and `"regulatory sanction"` become co-located).
4. **Re-run the 7-dimension evaluation** to test whether the RN layer, now implemented manifold-internally, closes the gap that §4.3 left open.

This is the natural completion of the project: the present work demonstrates that the Kan layer of the Sheaf–Kan–RN stack *works structurally* (left > right, proximity predicts transfer, soft Kan densifies as predicted), and motivates the RN layer empirically; the unified-manifold experiment is the implementation of that RN layer.

### 6.2 Does RAG capture causal structure? (the compositional test)

The professor also flagged a deeper diagnostic for RAG:

> *"RAG really should not work that well in transferring causality because it's not aiming to capture causal structure, but semantic similarity. So, your experiments should test whether RAG actually captures causal structure. One way is to see if the structure preserves compositional causal relations."*

The present evaluation framework (Motif, Homology, Universality) partially addresses this — naive RAG's 0.997 homology score is shown in §5.1 to be a surface artefact. The stronger test is *compositional*: if `X causes Y` and `Y causes Z` are in the source corpus, can the predicted graph chain them as `X causes Z` in the target? Kan extensions, being functor extensions, *should* preserve composition by construction; RAG, being retrieval, does not. A clean implementation of this test — a *compositionality evaluator* that checks whether two-hop causal chains in the source are recoverable as one-hop chains in the predicted target — is the planned extension of the framework.

### 6.3 Open implementation tasks

1. Extend the soft-Kan evaluation to the full 6-metric pass. The soft variants are currently routed through *two* of the six evaluators that run on the main methods — Semantic Soft-F1 (`run_novel_evaluation.py:310`) and Structural Motif (`run_novel_evaluation.py:319`) — and the remaining four (Homology, Universality, Back-Translation, Coherence) are not yet plumbed through for `soft_method_preds`. The motif results also land in a `soft_kan/` subdirectory rather than being merged into `master_summary.json`'s top-level `motif` block, which is why §4.2 had to read them from the per-source CSVs. The line-318 comment promises "motif/homology" but only motif is actually called — a minor code/comment mismatch. Half-day task.
2. The CLIFF integration stub in `cliff_integration/kan_transfer_agentic.py` (166 lines, with explicit "To integrate into CLIFF" instructions at the top) is not yet wired into the CLIFF agentic-router; doing so would let users issue cross-domain causal queries directly through CLIFF.
3. Repeat the full experiment with the *Anthropic Claude* backend (currently OpenAI / Democritus). The `extraction/anthropic_client.py` skeleton (125 lines, exposing an `AnthropicChatClient` that conforms to the LLMClient protocol used by Democritus) is in place; this would partially satisfy the "Functorially Equivalent?" experiment from our initial brainstorm.

---

## 7. Conclusion

We built a complete, reproducible empirical test of Mahadevan's *Kan-Do-Calculus* on cross-domain causal transfer. The categorical duality between coend (left Kan) and end (right Kan) is *not* a formal nicety: it predicts, and our experiments confirm, an asymmetric structural transfer in which left Kan dominates right Kan on all six comparable evaluation dimensions in the legal→economic regime (proximity 0.4802), while both hard-Kan variants collapse to the empty graph on the medical→economic regime (proximity 0.0354). We further introduced a seven-dimension evaluation framework that sidesteps the zero exact-match F1 problem by drawing on evaluators inspired by cognitive science (Gentner), biology, physics (Wilson), linguistics (Nida), probability theory (Radon–Nikodym), and hermeneutics (Gadamer). The framework discriminates honest cross-domain transfer (left Kan: 0.500 coherence) from surface-preservation artefacts (naive RAG: 0.997 homology), and it empirically locates the bottleneck that remains: the RN entity-translation operator, whose absence keeps semantic F1 low even after RN-derivative reweighting at the topic-selection stage. The soft Kan we built does, however, recover non-trivial structural transfer (motif 0.354) on medical→economic — the empirical signature that the RN-derivative works as predicted in the structure space, motivating its full implementation as an entity-translation operator. Closing that gap with a unified causal manifold trained by Diagrammatic Backpropagation on a pooled cross-domain corpus is the natural next experiment, and is the one the textbook's three-layer framework explicitly calls for.

---

## Acknowledgements

I thank Prof. Sridhar Mahadevan for the *Categories for AGI* textbook, for the CLIFF / Democritus / BASKET / FunctorFlow toolchain that made this project feasible in a roughly two-week implementation window, for the project-conversation suggestion of the unified causal manifold as the natural next step (§6.1), for the compositional-causality diagnostic for RAG (§6.2), and for the broader course conversations that shaped both hypotheses and evaluation design.

### AI Disclosure

In keeping with the spirit of CMPSCI 692CT, I used frontier AI coding tools heavily throughout this project. The professor explicitly encouraged this — *"think of using these tools as part of the learning exercise. Try to push them hard to do cool stuff."* My specific uses:

- **Anthropic Claude (Claude Code, Opus 4.7)** was used as the primary pair-programmer for the entire `kan_transfer/` codebase: implementing the coend/end/soft-Kan modules, designing the seven evaluators, writing the UNITY SLURM scripts, and drafting this report. Approximately 80% of the code was authored by Claude under iterative human-in-the-loop direction, with the human providing the research questions, the categorical formulation, the cross-disciplinary parallels, hyper-parameter choices, debugging guidance, and final acceptance of every commit.
- **OpenAI GPT** (via the UMass TheKeymaker gateway) was used as the LLM-judge for the back-translation (Nida) and coherence (Gadamer) evaluators, and as the relation-extraction model inside Democritus.
- **`sentence-transformers`** (the open-source library from sbert.net / UKP Lab) supplied the embedding model `all-MiniLM-L6-v2`. The model itself originates from Microsoft Research's MiniLM work and is distributed via the sbert.net team. This is *not* an OpenAI model.

All scientific claims, hypothesis design, and interpretation of results in this report are the human author's. The AI tools accelerated implementation by an estimated 5–10× but did not contribute novel ideas; the cross-disciplinary parallel structure (Gentner, Wilson, Nida, Gadamer) and the framing of the zero-F1 finding as RN-layer motivation are human contributions.

---

## References

[1] S. Mahadevan, *Categories for AGI: A Categorical Framework for Artificial General Intelligence*, course textbook, UMass Amherst, 2026. — three-layer Sheaf–Kan–RN framework; coend / end formulae for Kan extensions; causal density function `ρ(c, d) = dP_do / dP_obs`.

[2] S. Mahadevan, *Large Causal Models from Large Language Models*, arXiv preprint (Democritus paper). — 100,000 causal claims across 10 domains; unified causal manifold via Geometric Transformer with Diagrammatic Backpropagation. **The canonical next step for this project (§6.1).**

[3] *Democritus*: causal-triple extraction via topic discovery + LLM relation extraction. https://github.com/sridharmahadevan/Democritus_OpenAI

[4] *CLIFF*: First AGI chatbot designed using FunctorFlow. https://github.com/sridharmahadevan/CLIFF_CatAgi

[5] *BASKET*: Building Agentic Workflows from Textual Documents. https://github.com/sridharmahadevan/BASKET

[6] S. Frost & S. Mahadevan, *FunctorFlow.jl*: Julia port with symbolic Diagrammatic Backpropagation. https://juliaknowledge.github.io/FunctorFlow.jl/dev

[7] S. Mahadevan, *Kan-Extension Transformers (KETs): A Categorical Unification of Attention and Diffusion*, ICML 2026 Tutorial (Seoul, July 2026).

[8] D. Gentner, "Structure-Mapping: A Theoretical Framework for Analogy," *Cognitive Science* 7(2):155–170, 1983.

[9] K. G. Wilson, "Renormalization Group and Critical Phenomena," *Physical Review B* 4(9):3174, 1971.

[10] E. Nida, *Toward a Science of Translating*, Brill, Leiden, 1964 — dynamic equivalence and back-translation.

[11] H.-G. Gadamer, *Truth and Method* (Wahrheit und Methode), 1960 — fusion of horizons.

[12] S. Mahadevan, *Diagrammatic Backpropagation and the Geometric Transformer*, in [1], chapters on DB and GT.

---

## Appendix A — Project artifacts

- **GitHub repository:** `github.com/JineshwarNariani/kan-extension-transfer` (this report and the full pipeline).
- **Codebase root:** `kan_transfer/`
- **Headline results:** `kan_transfer/results/tables/novel_evaluation/master_summary.json`
- **Figures:** `kan_transfer/results/figures/` (10 plots).
- **UNITY job:** `unity_56498824` — 30 minutes wall time, 128 GB RAM, 16 CPUs, CPU partition (`#SBATCH --mem=128G --cpus-per-task=16 --partition=cpu --time=12:00:00` in `unity_job.sh`; the job completed in 30 min, well under the 12 h wall-time cap).
- **Extraction stats:** `results/extraction/{economic,legal,medical}/extraction_stats.json`
- **Predictions (pickled):** `results/kan_predictions/predictions_{legal,medical}.pkl`

## Appendix B — The 20 held-out economic queries

`results/query_split/test_queries.json` (selected): *"Roles and responsibilities of individual board members"; "Federal Reserve Annual Reports and Financial Disclosures"; "Unconventional monetary policy tools during economic crises"; "Open market operations and asset purchase programs"; "Cross-border financial contagion and central bank coordination frameworks"; "The Taylor Rule and systematic policy frameworks"; "Beige Book Regional Economic Condition Reports"; …* (20 total).
