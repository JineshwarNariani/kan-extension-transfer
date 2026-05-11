"""
experiments/run_extraction.py

Step 1 of the pipeline: run Democritus on all three corpora.
Call this FIRST — it takes 4–8 hours on UNITY and produces the
relational_triples.jsonl files that everything else depends on.

Usage:
    python experiments/run_extraction.py --corpus medical
    python experiments/run_extraction.py --corpus economic
    python experiments/run_extraction.py --corpus legal
    python experiments/run_extraction.py --corpus all

Set before running (UMass GenAI platform — thekeymaker.umass.edu):
    export OPENAI_API_KEY=d440...            ← your UMass platform key
    export DEMOC_LLM_BASE_URL=https://thekeymaker.umass.edu
    export DEMOC_LLM_MODEL=gpt-4.1-mini     ← cheap for bulk extraction
    export KAN_CLAUDE_MODEL=claude-sonnet-4-6  ← used for robustness check only

The UMass platform speaks OpenAI-compatible format for ALL models (both Claude and GPT).
Use --backend openai (default) for everything. The --backend anthropic flag is only
for when a native Anthropic key (sk-ant-...) is used directly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "Democritus_OpenAI"))

from config import (
    MED_DIR, ECON_DIR, LEGAL_DIR,
    EXTRACTION_DIR,
    ANTHROPIC_API_KEY, OPENAI_API_KEY,
    DEMOCRITUS_NUM_ROOT_TOPICS, DEMOCRITUS_DEPTH_LIMIT, DEMOCRITUS_MAX_TOPICS,
)
from extraction.democritus_runner import (
    discover_topics_from_corpus,
    run_democritus_pipeline,
    load_triples,
    validate_triples,
)


def make_llm_client(backend: str, model_override: str = ""):
    """
    Construct LLM client.

    For UMass GenAI platform (thekeymaker.umass.edu):
        backend = "openai"  — works for BOTH Claude and GPT models
        OPENAI_API_KEY      = your UMass platform key (d440...)
        DEMOC_LLM_BASE_URL  = https://thekeymaker.umass.edu
        DEMOC_LLM_MODEL     = gpt-4.1-mini  (cheap) or claude-sonnet-4-6 (comparison)

    For native Anthropic key (sk-ant-...) directly:
        backend = "anthropic"
        ANTHROPIC_API_KEY   = sk-ant-...
    """
    if backend == "anthropic":
        # Native Anthropic SDK — only works with sk-ant-... keys against api.anthropic.com
        from extraction.anthropic_client import AnthropicChatClient
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        return AnthropicChatClient(
            model=model_override or os.getenv("KAN_PRIMARY_MODEL", "claude-sonnet-4-6"),
            max_tokens=256,
        )
    elif backend == "openai":
        # OpenAI-compatible client — works for UMass platform (both Claude and GPT models)
        from llms.openai_client import OpenAIChatClient  # type: ignore
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY not set.\n"
                "Run in your terminal:\n"
                "  export OPENAI_API_KEY=d440...\n"
                "  export DEMOC_LLM_BASE_URL=https://thekeymaker.umass.edu\n"
                "  export DEMOC_LLM_MODEL=gpt-4.1-mini"
            )
        model = model_override or os.getenv("DEMOC_LLM_MODEL", "gpt-4.1-mini")
        base_url = os.getenv("DEMOC_LLM_BASE_URL", "https://api.openai.com")
        print(f"[LLM] Using model: {model} via {base_url}")
        return OpenAIChatClient(model=model, max_tokens=256, max_batch_size=8)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'anthropic' or 'openai'.")


CORPUS_MAP = {
    "medical":  (MED_DIR,  "medical"),
    "economic": (ECON_DIR, "economic"),
    "legal":    (LEGAL_DIR, "legal"),
}


def run_corpus_extraction(
    corpus_name: str,
    backend: str,
    force: bool = False,
) -> Path:
    corpus_dir, _ = CORPUS_MAP[corpus_name]
    out_dir = EXTRACTION_DIR / corpus_name
    topics_file    = out_dir / "configs" / "root_topics.txt"
    triples_path   = out_dir / "relational_triples.jsonl"

    print(f"\n{'='*60}")
    print(f"EXTRACTING: {corpus_name.upper()} corpus")
    print(f"  Source dir:  {corpus_dir}")
    print(f"  Output dir:  {out_dir}")
    print(f"  Backend:     {backend}")
    print(f"{'='*60}\n")

    if triples_path.exists() and not force:
        print(f"[Skip] {triples_path} already exists. Use --force to re-run.")
        triples = load_triples(triples_path)
        validate_triples(triples, corpus_name)
        return triples_path

    doc_files = list(corpus_dir.glob("*.txt"))
    if not doc_files:
        raise RuntimeError(
            f"No .txt files in {corpus_dir}. "
            "Run data fetchers first:\n"
            "  python data/acquire/pubmed_fetcher.py\n"
            "  python data/acquire/fed_fetcher.py"
        )

    print(f"[Corpus] {len(doc_files)} documents found.")
    t0 = time.time()

    llm_client = make_llm_client(backend)

    # Step 1: Discover topics from corpus
    if not topics_file.exists() or force:
        discover_topics_from_corpus(
            corpus_dir=corpus_dir,
            out_topics_file=topics_file,
            llm_client=llm_client,
            num_root_topics=DEMOCRITUS_NUM_ROOT_TOPICS,
        )
    else:
        print(f"[Skip] Topics file already exists: {topics_file}")

    # Step 2: Run full Democritus pipeline
    triples_path = run_democritus_pipeline(
        topics_file=topics_file,
        out_dir=out_dir,
        llm_client=llm_client,
        depth_limit=DEMOCRITUS_DEPTH_LIMIT,
        max_topics=DEMOCRITUS_MAX_TOPICS,
    )

    # Step 3: Validate
    triples = load_triples(triples_path)
    stats   = validate_triples(triples, corpus_name)

    # Save stats
    stats_path = out_dir / "extraction_stats.json"
    stats_path.write_text(json.dumps({**stats, "elapsed_seconds": time.time() - t0}, indent=2))

    elapsed = time.time() - t0
    print(f"\n[Done] {corpus_name} extraction complete in {elapsed/60:.1f} min")
    print(f"  Triples: {triples_path}")

    # Report cost if using Anthropic
    if hasattr(llm_client, "usage_summary"):
        cost_info = llm_client.usage_summary()
        print(f"  API cost: ${cost_info.get('total_cost_usd', 0):.4f}")
        print(f"  Tokens:   {cost_info.get('input_tokens', 0)} in / {cost_info.get('output_tokens', 0)} out")

    return triples_path


def main():
    parser = argparse.ArgumentParser(description="Run Democritus extraction on corpora.")
    parser.add_argument(
        "--corpus", choices=["medical", "economic", "legal", "all"],
        default="all", help="Which corpus to process"
    )
    parser.add_argument(
        "--backend", choices=["anthropic", "openai"],
        default="openai", help="LLM backend (openai = UMass platform; anthropic = native sk-ant-... key)"
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if output already exists")
    args = parser.parse_args()

    corpora = list(CORPUS_MAP.keys()) if args.corpus == "all" else [args.corpus]

    for corpus in corpora:
        try:
            run_corpus_extraction(corpus, backend=args.backend, force=args.force)
        except Exception as exc:
            print(f"\n[ERROR] {corpus} extraction failed: {exc}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
