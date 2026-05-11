"""
query_vocab.py

Builds the train/test split of economic topics.
Test topics become the 20 held-out evaluation queries.
The split is deterministic (fixed seed) and pre-committed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


def build_query_split(
    econ_triples_path: Path,
    test_fraction: float = 0.40,
    seed: int = 42,
    n_test_target: int = 20,
) -> Tuple[List[str], List[str]]:
    """
    Split discovered economic topics into train / test.

    Train topics are used to build the ground-truth functor F_econ
    (the "known" part of the economic causal graph).

    Test topics are the held-out evaluation queries — these are the queries
    d ∈ C_econ for which we compare Lan_J(F_med)(d) vs F_econ(d).

    Returns:
        (train_topics, test_topics)
    """
    records = []
    with econ_triples_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    all_topics = list({r.get("topic") or r.get("domain") or "unknown" for r in records})
    all_topics = [t for t in all_topics if t and t != "unknown"]

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(all_topics))
    all_topics_shuffled = [all_topics[i] for i in idx]

    n_test = min(n_test_target, max(1, int(len(all_topics_shuffled) * test_fraction)))
    test_topics  = all_topics_shuffled[:n_test]
    train_topics = all_topics_shuffled[n_test:]

    print(f"[QueryVocab] {len(all_topics)} total topics → "
          f"{len(train_topics)} train / {len(test_topics)} test")

    return train_topics, test_topics


def save_query_split(
    train: List[str],
    test: List[str],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_topics.json").write_text(
        json.dumps(train, indent=2), encoding="utf-8"
    )
    (out_dir / "test_queries.json").write_text(
        json.dumps(test, indent=2), encoding="utf-8"
    )
    print(f"[QueryVocab] Saved train/test split to {out_dir}")


def load_test_queries(split_dir: Path) -> List[str]:
    path = split_dir / "test_queries.json"
    return json.loads(path.read_text(encoding="utf-8"))
