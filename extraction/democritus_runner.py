"""
democritus_runner.py

Wraps the Democritus pipeline to run on a corpus directory.
Patches the LLM factory with whichever client is passed (Anthropic or OpenAI).
Produces a single canonical triples.jsonl per corpus with checkpointing.

The Democritus pipeline expects:
  1. A root_topics.txt file (we generate this from documents)
  2. An output directory
  3. An LLM client implementing ask() / ask_batch()
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from extraction.anthropic_client import AnthropicChatClient


DEMOCRITUS_ROOT = Path(__file__).resolve().parents[2] / "Democritus_OpenAI"


def _inject_python_path():
    """Add Democritus to sys.path so its modules are importable."""
    dr = str(DEMOCRITUS_ROOT)
    if dr not in sys.path:
        sys.path.insert(0, dr)


def _add_retry(llm_client, max_retries: int = 5, base_delay: float = 3.0) -> None:
    """
    Wrap llm_client._single_chat with exponential-backoff retry in-place.

    Retries on transient errors (rate limits, 5xx, timeouts).
    Raises immediately on auth errors (401/403) — those won't self-heal.
    Delays: 3s, 6s, 12s, 24s, 48s (+ up to 2s jitter each).
    """
    original = llm_client._single_chat

    def _retrying(prompt: str) -> str:
        for attempt in range(max_retries + 1):
            try:
                return original(prompt)
            except RuntimeError as exc:
                msg = str(exc)
                if "401" in msg or "403" in msg:
                    raise  # auth failures never self-heal
                if attempt == max_retries:
                    raise
                delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                print(f"[Retry] attempt {attempt + 1}/{max_retries} failed: "
                      f"{msg[:120]}  — retrying in {delay:.1f}s…")
                time.sleep(delay)

    llm_client._single_chat = _retrying


def _patch_factory(llm_client) -> None:
    """
    Monkey-patch Democritus's make_llm_client() to return our client.
    This must be called BEFORE importing any Democritus pipeline modules.
    Also wraps _single_chat with retry logic.
    """
    _inject_python_path()
    _add_retry(llm_client)
    import llms.factory as factory  # type: ignore
    factory.make_llm_client = lambda **kw: llm_client


def discover_topics_from_corpus(
    corpus_dir: Path,
    out_topics_file: Path,
    llm_client,
    num_root_topics: int = 12,
) -> List[str]:
    """
    Run Democritus document_topic_discovery on all .txt files in corpus_dir.
    Concatenates all documents into a single text before topic discovery
    so that topics span the full corpus.
    """
    _patch_factory(llm_client)

    from scripts.document_topic_discovery import discover_topics_from_text  # type: ignore

    docs = list(corpus_dir.glob("*.txt"))
    if not docs:
        raise RuntimeError(f"No .txt files found in {corpus_dir}")

    # Shuffle so the 120k-char sample spans the whole corpus, not just
    # whichever files sort first alphabetically.
    import random as _random
    _random.seed(42)
    _random.shuffle(docs)

    print(f"[Topics] Sampling from {len(docs)} documents in {corpus_dir.name}…")

    # Build a uniformly-sampled text: take up to 1200 chars from each doc in
    # shuffle order until we hit 120,000 chars total.
    BUDGET   = 120_000
    PER_DOC  = max(200, BUDGET // max(len(docs), 1))
    combined = []
    total    = 0
    for doc in docs:
        snippet = doc.read_text(encoding="utf-8", errors="replace")[:PER_DOC]
        combined.append(f"=== {doc.stem} ===\n{snippet}")
        total += len(snippet)
        if total >= BUDGET:
            break
    full_text = "\n\n".join(combined)
    print(f"[Topics] Using {len(combined)}/{len(docs)} docs, {len(full_text):,} chars.")

    topics = discover_topics_from_text(
        full_text,
        num_root_topics=num_root_topics,
        topics_per_chunk=6,
        batch_size=8,
        max_tokens=128,
    )

    out_topics_file.parent.mkdir(parents=True, exist_ok=True)
    out_topics_file.write_text("\n".join(topics) + "\n", encoding="utf-8")
    print(f"[Topics] {len(topics)} root topics → {out_topics_file}")
    return topics


def run_democritus_pipeline(
    topics_file: Path,
    out_dir: Path,
    llm_client,
    depth_limit: int = 2,
    max_topics: int = 150,
) -> Path:
    """
    Run the full Democritus pipeline (topic graph → questions → statements → triples).
    Returns path to the relational_triples.jsonl output.

    All Democritus output files are written to out_dir.
    """
    _patch_factory(llm_client)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save current directory and switch to out_dir so Democritus writes files there
    orig_cwd = os.getcwd()
    os.chdir(out_dir)

    # Also ensure configs/root_topics.txt is accessible
    configs_dir = out_dir / "configs"
    configs_dir.mkdir(exist_ok=True)
    import shutil
    dst = configs_dir / "root_topics.txt"
    if topics_file.resolve() != dst.resolve():
        shutil.copy(topics_file, dst)

    try:
        print("\n[Democritus] === Step 1: Building topic graph ===")
        from scripts.topic_graph_builder import main as build_topics  # type: ignore
        build_topics(
            topics_file=str(configs_dir / "root_topics.txt"),
            depth_limit=depth_limit,
            max_total_topics=max_topics,
        )

        print("\n[Democritus] === Step 2: Generating causal questions ===")
        from scripts.causal_question_builder import main as build_questions  # type: ignore
        build_questions()

        print("\n[Democritus] === Step 3: Generating causal statements ===")
        from scripts.causal_statement_builder import main as build_statements  # type: ignore
        build_statements()

        print("\n[Democritus] === Step 4: Extracting relational triples ===")
        from scripts.relational_triple_extractor import main as extract_triples  # type: ignore
        extract_triples()

    finally:
        os.chdir(orig_cwd)

    triples_path = out_dir / "relational_triples.jsonl"
    if not triples_path.exists():
        # Some Democritus versions write to different names
        alt = list(out_dir.glob("*triples*.jsonl"))
        if alt:
            triples_path = alt[0]
            print(f"[Democritus] Found triples at {triples_path}")
        else:
            raise RuntimeError(f"relational_triples.jsonl not found in {out_dir}")

    count = sum(1 for _ in triples_path.open())
    print(f"\n[Democritus] Pipeline complete. {count} triples → {triples_path}")
    return triples_path


def load_triples(triples_path: Path) -> List[dict]:
    """Load relational_triples.jsonl into a list of dicts."""
    triples = []
    with triples_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                triples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"[Triples] Loaded {len(triples)} triples from {triples_path.name}")
    return triples


def validate_triples(triples: List[dict], corpus_name: str) -> dict:
    """Basic sanity checks on extracted triples."""
    total  = len(triples)
    unique_subj = {t.get("subj", "") for t in triples}
    unique_obj  = {t.get("obj",  "") for t in triples}
    unique_rels = {t.get("rel",  "") for t in triples}
    unique_topics = {t.get("topic", "") for t in triples}

    print(f"\n[Validate] {corpus_name}:")
    print(f"  Total triples:   {total}")
    print(f"  Unique subjects: {len(unique_subj)}")
    print(f"  Unique objects:  {len(unique_obj)}")
    print(f"  Relation types:  {unique_rels}")
    print(f"  Unique topics:   {len(unique_topics)}")

    if total < 20:
        print(f"  WARNING: Very few triples ({total}). Consider increasing num_root_topics.")

    return {
        "corpus":        corpus_name,
        "total_triples": total,
        "n_subjects":    len(unique_subj),
        "n_objects":     len(unique_obj),
        "n_topics":      len(unique_topics),
        "relation_types": list(unique_rels),
    }
