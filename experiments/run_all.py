"""
experiments/run_all.py

Master pipeline: fetch → extract → kan → evaluate.
Runs all steps end-to-end.  Can be launched on UNITY as a single job.

Usage:
    python experiments/run_all.py --backend anthropic
    python experiments/run_all.py --backend openai --skip-legal
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list, check: bool = True) -> int:
    print(f"\n[Pipeline] Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=False)
    if check and result.returncode != 0:
        print(f"[Pipeline] FAILED with code {result.returncode}")
        sys.exit(result.returncode)
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["anthropic", "openai"], default="openai")
    parser.add_argument("--skip-fetch",  action="store_true")
    parser.add_argument("--skip-legal",  action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-sensitivity", action="store_true")
    args = parser.parse_args()

    py = sys.executable

    # Step 0: Fetch corpora (expanded versions — full text, not truncated)
    if not args.skip_fetch:
        print("\n" + "="*60 + "\nSTEP 0: FETCH CORPORA\n" + "="*60)
        run([py, str(ROOT / "data" / "acquire" / "pubmed_fulltext_fetcher.py")])
        run([py, str(ROOT / "data" / "acquire" / "econ_expanded_fetcher.py")])
        if not args.skip_legal:
            run([py, str(ROOT / "data" / "acquire" / "legal_fetcher.py")])

    # Step 1: Extract
    if not args.skip_extract:
        print("\n" + "="*60 + "\nSTEP 1: EXTRACTION\n" + "="*60)
        run([py, str(ROOT / "experiments" / "run_extraction.py"),
             "--corpus", "medical",  "--backend", args.backend])
        run([py, str(ROOT / "experiments" / "run_extraction.py"),
             "--corpus", "economic", "--backend", args.backend])
        if not args.skip_legal:
            run([py, str(ROOT / "experiments" / "run_extraction.py"),
                 "--corpus", "legal", "--backend", args.backend])

    # Step 2: Kan predictions
    print("\n" + "="*60 + "\nSTEP 2: KAN EXTENSIONS\n" + "="*60)
    source = "medical" if args.skip_legal else "all"
    run([py, str(ROOT / "experiments" / "run_kan.py"), "--source", source])

    # Step 3: Evaluation
    print("\n" + "="*60 + "\nSTEP 3: EVALUATION\n" + "="*60)
    eval_cmd = [py, str(ROOT / "experiments" / "run_evaluation.py"), "--source", source]
    if args.skip_sensitivity:
        eval_cmd.append("--skip-sensitivity")
    run(eval_cmd)

    print("\n" + "="*60)
    print("ALL STEPS COMPLETE.")
    print(f"Results → {ROOT / 'results'}")
    print("="*60)


if __name__ == "__main__":
    main()
