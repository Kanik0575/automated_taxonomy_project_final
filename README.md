# Automated APT Taxonomy via Genetic Algorithm + Hierarchical Clustering

A systematic clustering study of academic papers on Advanced Persistent Threats (APTs) published 2021–2026. Real papers are fetched from the Semantic Scholar API. A Genetic Algorithm selects an informative TF-IDF feature subset, and Agglomerative Hierarchical Clustering groups the papers into a taxonomy visualized as a dendrogram.

> **What this project IS:** An unsupervised, data-driven grouping of APT research papers into thematic clusters using machine learning. The dendrogram shows how the research *literature* organizes itself.
>
> **What this project IS NOT:** A new taxonomy of APT attack techniques. The definitive attacker-behaviour taxonomy is MITRE ATT&CK. This project clusters *papers about APTs*, and then heuristically maps each cluster to the closest MITRE tactic for interpretability.

---

## Data sources

Papers are pulled from three independent **free** aggregators and merged:

| Source | Why | Key needed? |
|--------|-----|-------------|
| **Semantic Scholar** (bulk endpoint) | Large coverage, has abstracts, covers IEEE/ACM/Springer papers | No |
| **arXiv** | Adds cs.CR preprints | No |
| **OpenAlex** | Broad journal/conference coverage, inverted-index abstracts decoded | No |

**Why not IEEE / ACM / Springer / Scopus APIs directly?**

- **IEEE Xplore API**: needs a developer key (approval 1-3 days)
- **ACM Digital Library**: no public search API exists
- **Springer Nature API**: needs a free key (immediate, but another signup step)
- **Scopus API**: requires a paid Elsevier subscription

The aggregators above *do* index papers from IEEE, ACM, Springer, and Elsevier venues. Each paper in the final corpus is tagged with its real publisher using a DOI prefix lookup:

| DOI prefix | Publisher |
|------------|-----------|
| `10.1109/` | IEEE |
| `10.1145/` | ACM |
| `10.1007/` | Springer |
| `10.1016/` | Elsevier (Scopus-indexed) |
| `10.3390/` | MDPI |
| `10.1002/` | Wiley |
| `10.48550/` | arXiv |

The `publisher` column in `data/papers.csv` shows this distribution after the fetch completes.

---

## Pipeline

```
  Semantic Scholar API              (fetch_papers.py)
         │
         ▼
  100 real papers (2021-2026, distributed 17/17/17/17/16/16)
         │
         ▼
  TF-IDF features                   (preprocess.py)
  (titles + abstracts, 1-2 ngrams, sublinear_tf)
         │
         ▼
  Genetic Algorithm                 (genetic_algorithm.py)
  (binary chromosome = feature mask
   fitness = silhouette − parsimony penalty)
         │
         ▼
  Reduced feature set (30-200 features)
         │
         ▼
  Agglomerative Hierarchical Clustering   (clustering.py)
  (cosine distance, average linkage, k=7)
         │
         ├──► Dendrogram  (visualize.py)
         ├──► Cluster labels via top TF-IDF terms  (taxonomy.py)
         └──► Per-cluster MITRE ATT&CK suggestion  (taxonomy.py)
```

---

## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Fetch real papers from Semantic Scholar

```bash
python src/fetch_papers.py
```

This queries the Semantic Scholar Graph API, pulls candidate papers for each year 2021–2026, filters them (must have an abstract; abstract must contain both an APT-topic term and a method term), and writes the accepted papers to `data/papers.csv`. A log of rejected papers (with reasons) goes to `data/rejected_log.csv` so nothing is hidden.

Runtime: 3–6 minutes depending on Semantic Scholar's rate limits.

**If the fetch falls short of the 100-paper quota**, the script tells you how many papers it got per year. You can then either relax the filter in `fetch_papers.py` (add more `METHOD_TERMS`, more `SEARCH_QUERIES`) or accept the smaller corpus and document the shortfall in your report.

### 3. Run the ML pipeline

```bash
python src/main.py
```

This runs preprocessing, the silhouette-vs-k sweep, the GA, the final clustering (baseline + GA-optimized), and generates all reports and plots.

Runtime: 30–90 seconds for GA + clustering.

### Outputs

After running `main.py`:

```
outputs/
├── figures/
│   ├── dendrogram.png              ← THE taxonomy (GA-selected features)
│   ├── dendrogram_baseline.png     ← comparison (all features)
│   ├── ga_convergence.png          ← GA improves each generation
│   ├── silhouette_sweep.png        ← justifies choice of k
│   ├── cluster_sizes.png           ← papers per cluster
│   └── year_distribution.png       ← corpus balance check
└── reports/
    ├── taxonomy.md                 ← human-readable taxonomy report
    ├── paper_cluster_assignments.csv  ← each paper's cluster label
    └── run_summary.json            ← machine-readable metrics
```

---

## Methodology notes (also see `docs/methodology.md`)

### Why Semantic Scholar
Free public API, aggregates papers from IEEE, ACM, Springer, arXiv, Elsevier, and others. We are not indexing one specific database — we are accessing a meta-index. The honest way to describe it is "Semantic Scholar acts as a cross-database meta-index; each paper's underlying venue is recorded in the `venue` column of `data/papers.csv`."

### Why hierarchical clustering (not k-means, DBSCAN, HDBSCAN)
Hierarchical clustering produces a dendrogram — a complete tree of merges at every granularity. That tree *is* the visual artefact we care about, because it shows both high-level thematic groups and fine-grained sub-groups in one picture. Flat methods like k-means lose this structure.

### Why a GA for feature selection
TF-IDF on 100 short abstracts gives ~500–3000 features. Most are noise. Feature-subset selection is combinatorial (2^V subsets) and the objective (silhouette after clustering) is not differentiable, which rules out gradient methods. GAs are well-suited to this kind of combinatorial, non-differentiable search.

### Honest caveat: n=100 is small
With only 100 documents, any feature-selection method can overfit to the silhouette score — we may be finding features that happen to cluster *these* 100 papers well, not features that generalize. This is acknowledged in `docs/limitations.md`, not hidden.

---

## Testing the pipeline without internet

If you cannot reach Semantic Scholar (rate-limited, offline, etc.), you can validate that the pipeline runs by generating a synthetic test corpus:

```bash
python tools/generate_test_corpus.py
python src/main.py
```

**Do not submit the results from synthetic data.** The synthetic corpus produces cleaner clusters than real data because I designed the themes to be distinct. Real abstracts are messier. Synthetic runs are for debugging only.

---

## Project layout

```
apt-taxonomy-v2/
├── README.md                        (this file)
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── fetch_papers.py              Semantic Scholar API fetcher
│   ├── preprocess.py                TF-IDF feature extraction
│   ├── genetic_algorithm.py         GA for feature selection (from scratch, numpy)
│   ├── clustering.py                Hierarchical clustering + silhouette sweep
│   ├── taxonomy.py                  Cluster interpretation + MITRE tactic mapping
│   ├── visualize.py                 All plots (dendrogram, GA convergence, etc.)
│   └── main.py                      End-to-end orchestrator
│
├── tools/
│   └── generate_test_corpus.py      Synthetic data for offline pipeline testing
│
├── data/                            (populated by fetch_papers.py)
│   ├── papers.csv                   (with `publisher` column!)
│   ├── papers.json
│   └── rejected_log.csv
│
├── outputs/                         (populated by main.py)
│   ├── figures/
│   └── reports/
│
└── docs/
    ├── methodology.md               Academic write-up
    ├── how_to_present.md            Viva prep: what to tell the professor
    └── limitations.md               Honest limitations section
```

---

## Reproducibility

Every run uses `random_state = 42` (set in `genetic_algorithm.py`). Re-running `main.py` on the same `papers.csv` produces identical results. The only source of variation between runs is the Semantic Scholar query — new papers published since your last fetch may appear in subsequent runs.

---

## Academic integrity

This project intentionally avoids any fabricated data. If the fetcher retrieves fewer than 100 papers, the shortfall is reported, not silently padded. All cluster labels are derived from actual TF-IDF term frequencies in the abstracts, not invented. All metrics (silhouette scores, runtimes, feature counts) are computed at runtime and written to `outputs/reports/run_summary.json`.
