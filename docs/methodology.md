# Methodology

## 1. Research question

How can a collection of academic papers on Advanced Persistent Threats (APTs) be automatically organised into a coherent thematic taxonomy using unsupervised machine learning, and does genetic-algorithm-based feature selection improve cluster quality over a baseline that uses all TF-IDF features?

## 2. Corpus construction

### 2.1 Source
The corpus is drawn from the Semantic Scholar Graph API (`api.semanticscholar.org/graph/v1`). Semantic Scholar aggregates records from IEEE, ACM, Springer, arXiv, Elsevier, and other publishers, providing a cross-database meta-index. Each paper's underlying publication venue is preserved in the `venue` column of `data/papers.csv`.

### 2.2 Search queries
Five overlapping queries are issued against the API, each year-restricted, to maximise recall:

1. `"Advanced Persistent Threat taxonomy"`
2. `"APT attack classification"`
3. `"Advanced Persistent Threat MITRE ATT&CK"`
4. `"APT detection framework machine learning"`
5. `"APT lifecycle classification"`

Results across queries are deduplicated by Semantic Scholar paper ID.

### 2.3 Eligibility filter
A paper is retained if and only if:
- It has a non-empty abstract of at least 50 characters.
- Its abstract contains at least one topic term (`advanced persistent threat`, `apt`, `threat actor`, `cyber attack`, `targeted attack`, `nation-state`, `threat intelligence`).
- Its abstract contains at least one method term (`taxonomy`, `classification`, `framework`, `mitre`, `att&ck`, `machine learning`, `deep learning`, `clustering`, `detection`, `neural network`, `ontology`, `knowledge graph`).

Rejected papers are logged with reasons to `data/rejected_log.csv`.

### 2.4 Year distribution
To prevent the over-representation of recent years (which carry larger publication volumes), a fixed per-year quota is applied:

| Year | Quota |
|------|-------|
| 2021 | 17    |
| 2022 | 17    |
| 2023 | 17    |
| 2024 | 17    |
| 2025 | 16    |
| 2026 | 16    |
| **Total** | **100** |

Within each year, eligible papers are ranked by abstract length (as a proxy for informational completeness) and the top N are retained.

## 3. Feature extraction

Each paper's title and abstract are concatenated, lowercased, and stripped of non-alphanumeric characters (except `&` and `-`, preserved for terms like *MITRE ATT&CK* and *low-and-slow*). TF-IDF features are extracted with the following configuration:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `ngram_range` | (1, 2) | Captures bigrams like *lateral movement*, *dns tunneling* |
| `min_df` | 2 | Drop hapax legomena (terms in ≤ 1 paper) |
| `max_df` | 0.85 | Drop terms in > 85% of papers (non-discriminative) |
| `sublinear_tf` | True | Use 1 + log(tf) to dampen term repetition |
| `norm` | L2 | Standard for cosine-based similarity |
| `stop_words` | English + domain-specific | Removes *apt*, *attack*, *paper*, *study*, etc. |

Domain-specific stopwords are added because every paper in the corpus concerns APTs; the word *apt* itself carries no discriminative signal.

## 4. Genetic algorithm for feature selection

### 4.1 Representation
Each chromosome is a binary vector of length V (vocabulary size). A 1 at position i indicates that feature i is retained; 0 indicates removal.

### 4.2 Constraints
Chromosomes are constrained to select between 30 and 200 features. Chromosomes outside this range are rejected (fitness = −1) or repaired by randomly toggling bits back to the feasible region.

### 4.3 Fitness function

```
fitness(chromosome) = silhouette(clustering(X_masked)) − λ · |chromosome|
```

where:
- `X_masked` is the TF-IDF matrix restricted to selected features
- `clustering()` is Agglomerative Hierarchical Clustering with cosine distance and average linkage, cut at k = 7 clusters
- `silhouette()` is the mean silhouette coefficient across all samples, computed with cosine distance
- `λ = 0.0005` is a parsimony weight that lightly penalises larger feature sets

### 4.4 Operators

| Operator | Setting |
|----------|---------|
| Population size | 30 |
| Generations | 25 |
| Selection | Tournament, size 3 |
| Crossover | Uniform, probability 0.9 |
| Mutation | Bit-flip, expected 2 flips per chromosome |
| Elitism | Top 2 chromosomes survive unchanged |
| Random seed | 42 |

### 4.5 Termination
The GA runs for a fixed 25 generations and returns the best chromosome encountered across all generations (not merely the last generation's best).

## 5. Clustering

Agglomerative Hierarchical Clustering is applied twice:

1. **Baseline:** on the full TF-IDF matrix (all V features).
2. **GA-optimised:** on the feature subset selected by the GA.

Both runs use:
- Distance metric: cosine
- Linkage method: average
- Number of clusters (flat cut): k = 7

The choice of k = 7 is justified by a silhouette sweep over k ∈ {4, 5, 6, 7, 8, 9, 10}. The sweep plot is saved to `outputs/figures/silhouette_sweep.png`.

The linkage matrix from each run is used to render a dendrogram (`outputs/figures/dendrogram.png`). The dendrogram is the primary visual representation of the learned taxonomy.

## 6. Cluster interpretation

For each cluster C, the most *characteristic* terms are identified by computing, for every feature j:

```
distinctiveness(j, C) = mean_TFIDF(j | paper ∈ C) − mean_TFIDF(j | paper ∉ C)
```

The top-10 terms by distinctiveness become the cluster's keyword signature. Each cluster is then heuristically mapped to the closest MITRE ATT&CK tactic by keyword matching its signature against a pre-defined tactic→keyword dictionary (see `src/taxonomy.py`). Clusters whose signature does not match any tactic are labelled *Unclassified (inspect top terms)*.

This heuristic mapping is suggestive, not authoritative. The researcher is expected to manually review and, where necessary, refine the labels.

## 7. Evaluation

The central quantitative metric is the silhouette score. The claim is not that silhouette represents ground-truth taxonomic quality (there is no ground truth for unsupervised clustering of research papers), but that silhouette measures how tight and well-separated the clusters are relative to one another. An improvement in silhouette from baseline to GA-optimised indicates that feature selection has reduced the noise component of the similarity metric.

All metrics are written to `outputs/reports/run_summary.json` at runtime:

- `baseline_silhouette` – silhouette with all features
- `ga_silhouette` – silhouette with GA-selected features
- `silhouette_improvement` – absolute difference
- `n_features_selected_by_ga` – parsimony
- `ga_runtime_seconds` – computational cost

## 8. Reproducibility

All random operations use `random_state = 42`. Given the same `data/papers.csv`, running `python src/main.py` produces byte-identical figures and numerics. The only source of between-run variation is the Semantic Scholar query, whose results may change as new papers are indexed or existing records are updated.

## 9. Limitations

A full discussion of limitations is given in `docs/limitations.md`. Summary:

- n = 100 documents is small; feature-selection overfitting to silhouette is a real concern.
- Only abstracts are analysed, not full paper bodies; significant methodological detail may be lost.
- Silhouette on sparse high-dimensional TF-IDF data is known to be noisy.
- The heuristic MITRE tactic labelling is a keyword overlap rule, not a learned classifier.
- Clusters reflect *research vocabulary* in the literature, not attacker behaviour; the dendrogram does not replace MITRE ATT&CK.
