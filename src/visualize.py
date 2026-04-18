"""
visualize.py
------------
All plotting in one place. Every figure we produce is explainable:

  1. dendrogram.png       - the hierarchical clustering tree. This IS the
                             taxonomy, visualized. Each leaf is a paper.
  2. year_distribution.png - bar chart showing how the 100 papers are split
                             across 2021-2026.
  3. silhouette_vs_k.png   - shows silhouette at different cluster counts,
                             justifying our chosen k.
  4. ga_convergence.png    - GA best/mean fitness per generation. Shows the
                             GA is actually improving, not random.
  5. cluster_sizes.png     - bar chart of papers per cluster.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    labels: list[str],
    cluster_labels: np.ndarray,
    out_path: Path,
    color_threshold: float | None = None,
    title: str = "Hierarchical Clustering Dendrogram of APT Research Papers",
) -> None:
    """The main event - the dendrogram that represents our automated taxonomy."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(14, len(labels) * 0.22), 9))

    # Use cluster labels as the leaf coloring guide
    ddata = dendrogram(
        linkage_matrix,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=7,
        color_threshold=color_threshold,
        above_threshold_color="#7f7f7f",
        ax=ax,
    )

    ax.set_title(title, fontsize=13, pad=14)
    ax.set_xlabel("Paper (leaf)", fontsize=10)
    ax.set_ylabel("Cosine distance", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def plot_year_distribution(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = df["year"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(counts.index.astype(int).astype(str), counts.values,
                  color="#2C7FB8", edgecolor="#023858")
    ax.set_title("Corpus Distribution by Publication Year", fontsize=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Papers")
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2, str(int(v)),
                ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def plot_silhouette_sweep(scores: dict[int, float], chosen_k: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ks = sorted(scores.keys())
    vals = [scores[k] for k in ks]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, vals, marker="o", color="#2ca02c")
    ax.axvline(chosen_k, color="red", linestyle="--", alpha=0.7, label=f"chosen k = {chosen_k}")
    ax.set_title("Silhouette Score vs Number of Clusters (k)", fontsize=12)
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette score (cosine)")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def plot_ga_convergence(history: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gens = [h["generation"] for h in history]
    best = [h["best_fitness"] for h in history]
    mean = [h["mean_fitness"] for h in history]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(gens, best, marker="o", label="best fitness", color="#d62728")
    ax.plot(gens, mean, marker="s", label="mean fitness", color="#1f77b4", alpha=0.7)
    ax.set_title("Genetic Algorithm Convergence", fontsize=12)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (silhouette - parsimony penalty)")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def plot_cluster_sizes(labels: np.ndarray, cluster_names: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    uniq, counts = np.unique(labels, return_counts=True)
    order = np.argsort(counts)[::-1]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    names = [cluster_names[i] for i in uniq[order]]
    vals = counts[order]
    bars = ax.barh(range(len(uniq)), vals, color="#6a51a3")
    ax.set_yticks(range(len(uniq)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Papers")
    ax.set_title("Papers per Cluster")
    ax.grid(axis="x", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(v + 0.2, b.get_y() + b.get_height() / 2, str(int(v)),
                va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.name}")
