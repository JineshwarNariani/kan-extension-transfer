"""
visualization/ablation_plots.py

Generates the key experimental figures:
  1. Kan Crossover Curve: Left vs Right Kan F1 as function of domain proximity
  2. Sheaf Gluing Heatmap: per-query F1 by source domain
  3. Sensitivity plots: F1 vs k, F1 vs sim_threshold
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd


def plot_kan_crossover(
    domain_proximities: Dict[str, float],   # {"medical": 0.31, "legal": 0.47}
    left_f1s: Dict[str, float],             # {"medical": 0.22, "legal": 0.28}
    right_f1s: Dict[str, float],
    out_path: Path,
) -> None:
    """
    Plot Left Kan vs Right Kan F1 against domain proximity.
    The crossover (if it exists) is marked with a dashed vertical line.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    domains = sorted(domain_proximities, key=lambda d: domain_proximities[d])
    prox    = [domain_proximities[d] for d in domains]
    left    = [left_f1s.get(d, 0.0)  for d in domains]
    right   = [right_f1s.get(d, 0.0) for d in domains]

    ax.plot(prox, left,  "o-", color="#2980b9", linewidth=2.5, markersize=8,
            label="Left Kan (Lan) — colimit")
    ax.plot(prox, right, "s-", color="#e67e22", linewidth=2.5, markersize=8,
            label="Right Kan (Ran) — limit")

    for i, d in enumerate(domains):
        ax.annotate(d, (prox[i], max(left[i], right[i]) + 0.01),
                    ha="center", fontsize=9)

    # Mark crossover if it exists
    for i in range(len(prox) - 1):
        if (left[i] >= right[i]) != (left[i+1] >= right[i+1]):
            cross_x = (prox[i] + prox[i+1]) / 2
            ax.axvline(cross_x, color="#7f8c8d", linestyle="--", alpha=0.7,
                       label=f"Crossover ≈ {cross_x:.2f}")
            break

    ax.set_xlabel("Domain Proximity (cosine similarity of domain centroids)", fontsize=12)
    ax.set_ylabel("Mean Edge F1", fontsize=12)
    ax.set_title("Kan Crossover: Left vs Right Kan as a Function of Domain Proximity",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[Viz] Crossover plot → {out_path}")
    plt.close(fig)


def plot_sheaf_heatmap(
    sheaf_df: pd.DataFrame,
    source_names: List[str],
    out_path: Path,
) -> None:
    """
    Heatmap of per-query F1 by source domain.
    Columns: medical, legal, joint
    Rows: query index
    """
    columns  = [f"f1_{n}" for n in source_names] + ["f1_joint"]
    col_labels = source_names + ["JOINT (glued)"]

    queries = sheaf_df["query"].tolist()
    matrix  = sheaf_df[columns].values   # (N_queries, N_sources+1)

    fig, ax = plt.subplots(figsize=(max(6, len(columns) * 2.5), max(6, len(queries) * 0.45)))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=0.6, aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(queries)))
    ax.set_yticklabels([q[:45] + "…" if len(q) > 45 else q for q in queries],
                       fontsize=7)

    # Annotate cells with F1 values
    for i in range(len(queries)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            color = "white" if val < 0.3 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Edge F1")
    ax.set_title("Sheaf Gluing Test: F1 by Source Domain per Query",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[Viz] Sheaf heatmap → {out_path}")
    plt.close(fig)


def plot_sensitivity(
    sensitivity_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Three-panel sensitivity plot covering all pre-registered sweeps:
      Left:   F1 vs k              (fix sim_threshold=0.25)
      Centre: F1 vs sim_threshold  (fix k=10)
      Right:  F1 vs consensus_frac (right_kan only, fix k=10, sim=0.25)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"left_kan": "#2980b9", "right_kan": "#e67e22"}

    # Filter to the k × sim sweep rows (consensus_frac == 0.6 is the primary)
    sweep_df = sensitivity_df[sensitivity_df["consensus_frac"] == 0.6]

    for method, grp in sweep_df.groupby("method"):
        color = colors.get(method, "#666")

        # Panel 0: F1 vs k (fix sim_threshold at primary value 0.25)
        sub_k = grp[grp["sim_threshold"] == 0.25]
        if not sub_k.empty:
            axes[0].errorbar(sub_k["k"], sub_k["mean_edge_f1"],
                             yerr=sub_k["std_edge_f1"],
                             label=method, color=color, marker="o", linewidth=2)

        # Panel 1: F1 vs sim_threshold (fix k at primary value 10)
        sub_t = grp[grp["k"] == 10]
        if not sub_t.empty:
            axes[1].errorbar(sub_t["sim_threshold"], sub_t["mean_edge_f1"],
                             yerr=sub_t["std_edge_f1"],
                             label=method, color=color, marker="s", linewidth=2)

    # Panel 2: F1 vs consensus_frac (right_kan only, k=10, sim=0.25)
    cons_df = sensitivity_df[
        (sensitivity_df["method"] == "right_kan") &
        (sensitivity_df["k"] == 10) &
        (sensitivity_df["sim_threshold"] == 0.25)
    ].sort_values("consensus_frac")
    if not cons_df.empty:
        axes[2].errorbar(cons_df["consensus_frac"], cons_df["mean_edge_f1"],
                         yerr=cons_df["std_edge_f1"],
                         label="right_kan", color="#e67e22", marker="^", linewidth=2)

    axes[0].set_xlabel("k (number of source topics)", fontsize=12)
    axes[0].set_ylabel("Mean Edge F1", fontsize=12)
    axes[0].set_title("F1 vs k", fontsize=12, fontweight="bold")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Similarity threshold", fontsize=12)
    axes[1].set_ylabel("Mean Edge F1", fontsize=12)
    axes[1].set_title("F1 vs Similarity Threshold", fontsize=12, fontweight="bold")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel("Consensus fraction (Right Kan)", fontsize=12)
    axes[2].set_ylabel("Mean Edge F1", fontsize=12)
    axes[2].set_title("F1 vs Consensus Fraction", fontsize=12, fontweight="bold")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.suptitle("Sensitivity Analysis (pre-registered ranges)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[Viz] Sensitivity plot → {out_path}")
    plt.close(fig)
