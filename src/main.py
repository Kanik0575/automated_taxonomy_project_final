"""
main.py
-------
End-to-end pipeline. Run this after fetch_papers.py has populated data/papers.csv.

    python src/main.py

What it does, in order:
  1. Load the real corpus from data/papers.csv  (papers fetched from
     Semantic Scholar - no fabricated data).
  2. Build a TF-IDF feature matrix from the titles + abstracts.
  3. Run a silhouette sweep across k=4..10 to justify the choice of k.
  4. Run the GA to find a compact, high-silhouette feature subset.
  5. Do hierarchical clustering TWICE: once with all features (baseline),
     once with the GA-selected subset. Report both.
  6. Compute cluster profiles: top characteristic terms + suggested MITRE
     tactic label for each cluster.
  7. Produce all plots: dendrogram, year distribution, silhouette sweep,
     GA convergence, cluster sizes.
  8. Write a markdown taxonomy report and a CSV of per-paper cluster labels.

Everything written to outputs/ is reproducible (random_state=42) and
derived from the real data. Nothing is invented.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import load_corpus, build_tfidf
from genetic_algorithm import GAConfig, run_ga
from clustering import cluster_papers, find_best_k
from taxonomy import compute_cluster_profiles, profiles_to_markdown
from visualize import (
    plot_dendrogram,
    plot_year_distribution,
    plot_silhouette_sweep,
    plot_ga_convergence,
    plot_cluster_sizes,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"
REP = OUT / "reports"


def bar(msg: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {msg}")
    print("=" * 70)


def main(n_clusters: int = 7, ga_generations: int | None = None) -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    REP.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    bar("STEP 1 | Load corpus")
    df = load_corpus()
    print(f"Loaded {len(df)} papers.")
    print("Papers per year:")
    for y, c in df["year"].value_counts().sort_index().items():
        print(f"  {int(y)}: {int(c)}")

    plot_year_distribution(df, FIG / "year_distribution.png")

    # ------------------------------------------------------------------
    bar("STEP 2 | TF-IDF feature extraction")
    X, vectorizer, feature_names = build_tfidf(df["doc_text"].tolist())
    print(f"TF-IDF matrix:  {X.shape[0]} papers x {X.shape[1]} features")
    print(f"Sparsity:       {1 - X.nnz / (X.shape[0] * X.shape[1]):.2%}")

    # ------------------------------------------------------------------
    bar("STEP 3 | Silhouette sweep over k (baseline, full features)")
    sweep = find_best_k(X, k_range=range(4, 11))
    for k, s in sweep.items():
        mark = " <-- chosen" if k == n_clusters else ""
        print(f"  k={k}: silhouette={s:+.4f}{mark}")
    plot_silhouette_sweep(sweep, chosen_k=n_clusters, out_path=FIG / "silhouette_sweep.png")

    # ------------------------------------------------------------------
    bar("STEP 4 | Genetic Algorithm - feature selection")
    cfg = GAConfig(n_clusters=n_clusters)
    if ga_generations is not None:
        cfg.generations = ga_generations
    ga_res = run_ga(X, cfg=cfg, verbose=True)
    plot_ga_convergence(ga_res.history, FIG / "ga_convergence.png")

    # ------------------------------------------------------------------
    bar("STEP 5 | Hierarchical clustering - baseline vs GA-selected")
    baseline = cluster_papers(X, feature_mask=None, n_clusters=n_clusters)
    optimized = cluster_papers(X, feature_mask=ga_res.best_mask, n_clusters=n_clusters)

    print(f"  Baseline (all {X.shape[1]} features):       silhouette = {baseline.silhouette:+.4f}")
    print(f"  GA-selected ({ga_res.best_num_features:>4} features):  silhouette = {optimized.silhouette:+.4f}")
    delta = optimized.silhouette - baseline.silhouette
    improvement_pct = (delta / abs(baseline.silhouette)) * 100 if baseline.silhouette != 0 else 0.0
    print(f"  Delta: {delta:+.4f}  ({improvement_pct:+.1f}% vs baseline)")
    if delta <= 0:
        print("  HONEST NOTE: GA did not improve silhouette over baseline on this run.")
        print("  This is worth reporting truthfully - not every GA run produces a win.")

    # ------------------------------------------------------------------
    bar("STEP 6 | Cluster interpretation (taxonomy labels)")
    profiles = compute_cluster_profiles(
        X=X,
        labels=optimized.labels,
        feature_names=feature_names,
        df=df,
        top_n_terms=10,
        top_n_papers=3,
    )
    for p in profiles:
        print(f"\nCluster {p.cluster_id} [{p.size} papers] -> {p.suggested_label}")
        print("   top terms:", ", ".join(t for t, _ in p.top_terms[:6]))

    # ------------------------------------------------------------------
    bar("STEP 7 | Plots")
    cluster_names = [f"C{p.cluster_id}: {p.suggested_label}" for p in profiles]
    plot_cluster_sizes(optimized.labels, cluster_names, FIG / "cluster_sizes.png")

    # Leaf labels for dendrogram: short "[year] first-3-words-of-title"
    leaf_labels: list[str] = []
    for _, row in df.iterrows():
        title_words = " ".join(str(row["title"]).split()[:4])
        leaf_labels.append(f"[{int(row['year'])}] {title_words}")

    plot_dendrogram(
        linkage_matrix=optimized.linkage_matrix,
        labels=leaf_labels,
        cluster_labels=optimized.labels,
        out_path=FIG / "dendrogram.png",
        title="APT Research Taxonomy - Hierarchical Clustering (GA-selected features)",
    )
    plot_dendrogram(
        linkage_matrix=baseline.linkage_matrix,
        labels=leaf_labels,
        cluster_labels=baseline.labels,
        out_path=FIG / "dendrogram_baseline.png",
        title="APT Research Taxonomy - Hierarchical Clustering (baseline, all features)",
    )

    # ------------------------------------------------------------------
    bar("STEP 8 | Reports")
    taxonomy_md = profiles_to_markdown(profiles)
    # Append run metadata
    meta = [
        "",
        "---",
        "## Run Metadata",
        f"- Corpus size: **{len(df)}** papers",
        f"- TF-IDF features (full): **{X.shape[1]}**",
        f"- GA-selected features: **{ga_res.best_num_features}**",
        f"- Baseline silhouette (k={n_clusters}): **{baseline.silhouette:+.4f}**",
        f"- GA-optimized silhouette (k={n_clusters}): **{optimized.silhouette:+.4f}**",
        f"- Delta: **{delta:+.4f}**",
        f"- GA runtime: **{ga_res.total_seconds:.1f}s**",
        f"- Random seed: **{cfg.random_state}** (reproducible)",
    ]
    (REP / "taxonomy.md").write_text(taxonomy_md + "\n".join(meta), encoding="utf-8")
    print(f"  wrote taxonomy.md")

    # Per-paper cluster assignments CSV.
    # Include publisher column if present (multi-source fetcher adds it).
    base_cols = ["paper_id", "title", "year", "authors", "venue", "doi", "url"]
    extra_cols = [c for c in ("publisher", "source") if c in df.columns]
    out_df = df[base_cols + extra_cols].copy()
    out_df["cluster_id"] = optimized.labels
    out_df["cluster_label"] = [profiles[c].suggested_label for c in optimized.labels]
    out_df.to_csv(REP / "paper_cluster_assignments.csv", index=False)
    print(f"  wrote paper_cluster_assignments.csv")

    # Run summary JSON (machine-readable)
    summary = {
        "corpus_size": int(len(df)),
        "n_features_total": int(X.shape[1]),
        "n_features_selected_by_ga": int(ga_res.best_num_features),
        "baseline_silhouette": float(baseline.silhouette),
        "ga_silhouette": float(optimized.silhouette),
        "silhouette_improvement": float(delta),
        "n_clusters": int(n_clusters),
        "silhouette_sweep": {int(k): float(v) for k, v in sweep.items()},
        "ga_runtime_seconds": float(ga_res.total_seconds),
        "random_seed": int(cfg.random_state),
        "cluster_sizes": {int(p.cluster_id): int(p.size) for p in profiles},
        "cluster_labels": {int(p.cluster_id): p.suggested_label for p in profiles},
    }
    (REP / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  wrote run_summary.json")

    bar("DONE")
    print(f"Outputs: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
