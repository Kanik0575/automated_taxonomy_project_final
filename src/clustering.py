"""
clustering.py
-------------
Hierarchical (agglomerative) clustering on the GA-selected feature subset.

We do clustering TWICE for honesty:
  (a) On the full TF-IDF feature set (baseline).
  (b) On the GA-selected subset (our proposed method).
This lets us report whether GA feature selection actually improved silhouette
vs the baseline - or whether it didn't, which is equally valid to report.

Output:
  - Cluster label for each of the 100 papers
  - Linkage matrix for dendrogram plotting
  - Silhouette score
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


@dataclass
class ClusteringResult:
    labels: np.ndarray         # int array, length n_papers
    linkage_matrix: np.ndarray # scipy linkage format, for dendrogram
    silhouette: float
    n_clusters: int
    feature_indices: np.ndarray  # which columns of the full matrix were used
    distance_matrix_condensed: np.ndarray  # for dendrogram consistency


def _cosine_condensed(X_dense: np.ndarray) -> np.ndarray:
    """Compute condensed cosine distance matrix in the format scipy.linkage wants."""
    from scipy.spatial.distance import pdist
    # Use cosine - text similarity is scale-invariant so cosine is natural.
    return pdist(X_dense, metric="cosine")


def cluster_papers(
    X: csr_matrix,
    feature_mask: np.ndarray | None = None,
    n_clusters: int = 7,
    linkage_method: str = "average",
) -> ClusteringResult:
    """Cluster papers using hierarchical clustering.

    Args:
        X: full TF-IDF matrix (sparse, papers x features).
        feature_mask: optional boolean/0-1 array of length X.shape[1].
                      If None, use all features.
        n_clusters: number of flat clusters to cut the tree at.
        linkage_method: 'average', 'ward', 'complete', 'single'.
                        'average' pairs well with cosine distance for text.

    Returns: ClusteringResult
    """
    if feature_mask is None:
        feature_mask = np.ones(X.shape[1], dtype=np.uint8)

    feature_indices = np.where(feature_mask)[0]
    X_sub = X[:, feature_indices].toarray()

    # Labels via sklearn (for convenience)
    sk_model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="cosine",
        linkage=linkage_method,
    )
    labels = sk_model.fit_predict(X_sub)
    sil = silhouette_score(X_sub, labels, metric="cosine") if len(set(labels)) > 1 else -1.0

    # Linkage matrix via scipy (needed for the dendrogram)
    condensed = _cosine_condensed(X_sub)
    Z = scipy_linkage(condensed, method=linkage_method)

    return ClusteringResult(
        labels=labels,
        linkage_matrix=Z,
        silhouette=float(sil),
        n_clusters=n_clusters,
        feature_indices=feature_indices,
        distance_matrix_condensed=condensed,
    )


def find_best_k(
    X: csr_matrix,
    feature_mask: np.ndarray | None = None,
    k_range: range = range(4, 11),
    linkage_method: str = "average",
) -> dict[int, float]:
    """Silhouette sweep over k, to justify our choice of n_clusters."""
    scores: dict[int, float] = {}
    for k in k_range:
        res = cluster_papers(X, feature_mask, n_clusters=k, linkage_method=linkage_method)
        scores[k] = res.silhouette
    return scores


if __name__ == "__main__":
    from preprocess import load_corpus, build_tfidf
    df = load_corpus()
    X, _, _ = build_tfidf(df["doc_text"].tolist())
    print(f"Silhouette sweep across k (full features):")
    for k, s in find_best_k(X).items():
        print(f"  k={k}: silhouette={s:+.4f}")
