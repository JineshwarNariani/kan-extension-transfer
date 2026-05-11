"""
cliff_integration/kan_transfer_agentic.py

Extends CLIFF with a new 'kan_transfer' route that answers cross-domain
causal questions by applying pre-computed Kan extensions.

To integrate into CLIFF:
  1. Add to CLIFF_CatAgi/functorflow_v3/query_router_agentic.py:
         from kan_transfer.cliff_integration.kan_transfer_agentic import (
             KanTransferAgenticRunner, KanTransferAgenticConfig
         )
  2. Register route in cliff.py route_ff2_query():
         if any(p in q_lower for p in ["transfer", "cross-domain", "generalize from"]):
             return "kan_transfer"
  3. Add _run_kan_transfer() dispatch in FF2QueryRouter.run()
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "Democritus_OpenAI"))

from functor.causal_functor import CausalFunctor
from functor.embedder       import SharedEmbedder
from kan.coend              import left_kan_extension
from kan.end                import right_kan_extension
from config                 import EXTRACTION_DIR


@dataclass
class KanTransferAgenticConfig:
    query:      str
    outdir:     Path
    source_domain: str = "medical"   # "medical" | "legal" | "auto"
    method:        str = "left_kan"  # "left_kan" | "right_kan" | "both"
    k:             int = 10
    sim_threshold: float = 0.25


class KanTransferAgenticRunner:
    """
    CLIFF route: 'kan_transfer'

    Answers a query by finding the Kan extension of the source-domain
    causal functor over the target query.

    Example queries this route handles:
      "What can medical trial evidence tell us about monetary policy causality?"
      "Transfer causal knowledge from clinical trials to economic forecasting"
      "Generalize from drug treatment studies to fiscal policy effects"
    """

    TRANSFER_PATTERNS = [
        "transfer", "cross-domain", "what can", "tell us about",
        "generalize from", "medical knowledge to", "apply to",
        "kan extension", "causal transfer",
    ]

    def __init__(self, config: KanTransferAgenticConfig):
        self.config = config
        self._functors: dict[str, CausalFunctor] = {}

    def _load_functor(self, domain: str) -> CausalFunctor:
        if domain not in self._functors:
            triples_path = EXTRACTION_DIR / domain / "relational_triples.jsonl"
            if not triples_path.exists():
                raise RuntimeError(
                    f"No extracted triples for domain '{domain}'. "
                    "Run experiments/run_extraction.py first."
                )
            self._functors[domain] = CausalFunctor(triples_path, domain_name=domain)
        return self._functors[domain]

    def _detect_source_domain(self, query: str) -> str:
        q = query.lower()
        if any(w in q for w in ["medical", "clinical", "trial", "drug", "patient"]):
            return "medical"
        if any(w in q for w in ["legal", "regulatory", "enforcement", "ftc", "sec"]):
            return "legal"
        return self.config.source_domain

    def run(self) -> str:
        query = self.config.query

        src_domain = (
            self.config.source_domain
            if self.config.source_domain != "auto"
            else self._detect_source_domain(query)
        )

        try:
            src_functor = self._load_functor(src_domain)
        except RuntimeError as e:
            return f"[KanTransfer] Cannot run: {e}"

        results_text = [
            f"**Kan Extension Transfer Query**\n",
            f"Source domain: **{src_domain}**",
            f"Query: _{query}_\n",
        ]

        if self.config.method in ("left_kan", "both"):
            left_preds = left_kan_extension(
                src_functor, [query],
                k=self.config.k, sim_threshold=self.config.sim_threshold
            )
            G_left = left_preds.get(query)
            if G_left and G_left.edges():
                results_text.append("\n**Left Kan Extension (optimistic/colimit):**")
                results_text.append(
                    f"Predicted {len(G_left.edges())} causal relationships:"
                )
                for u, v, d in sorted(G_left.edges(data=True), key=lambda e: -e[2].get("weight", 0))[:10]:
                    results_text.append(f"  • {u} **{d.get('relation','→')}** {v}  (sim={d.get('weight',0):.2f})")
            else:
                results_text.append("Left Kan: no predictions above threshold.")

        if self.config.method in ("right_kan", "both"):
            right_preds = right_kan_extension(
                src_functor, [query],
                k=self.config.k, sim_threshold=self.config.sim_threshold
            )
            G_right = right_preds.get(query)
            if G_right and G_right.edges():
                results_text.append("\n**Right Kan Extension (conservative/limit):**")
                results_text.append(
                    f"Predicted {len(G_right.edges())} high-consensus causal relationships:"
                )
                for u, v, d in sorted(G_right.edges(data=True), key=lambda e: -e[2].get("weight", 0))[:10]:
                    results_text.append(f"  • {u} **{d.get('relation','→')}** {v}  (consensus={d.get('consensus',0):.2f})")
            else:
                results_text.append("Right Kan: no predictions met consensus threshold.")

        return "\n".join(results_text)


# ── Route registration snippet (paste into CLIFF's query_router_agentic.py) ──

KAN_TRANSFER_ROUTE_SNIPPET = '''
# In route_ff2_query():
KAN_PATTERNS = [
    "transfer", "cross-domain", "what can X tell us", "generalize from",
    "medical knowledge to", "kan extension", "causal transfer",
]
if any(p in query_lower for p in KAN_PATTERNS):
    return "kan_transfer"

# In FF2QueryRouter.run():
elif route == "kan_transfer":
    from kan_transfer.cliff_integration.kan_transfer_agentic import (
        KanTransferAgenticRunner, KanTransferAgenticConfig
    )
    runner = KanTransferAgenticRunner(KanTransferAgenticConfig(
        query=self.config.query,
        outdir=self.config.outdir,
    ))
    result = runner.run()
    # Write result to output file as CLIFF would
    (self.config.outdir / "kan_transfer_result.txt").write_text(result)
'''
