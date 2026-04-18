"""
genetic_algorithm.py
--------------------
Feature selection via a Genetic Algorithm (GA).

THE CORE IDEA
  TF-IDF gives us ~1000-3000 features from 100 abstracts. Most are noise.
  Hierarchical clustering on the full feature set tends to produce blurry
  clusters because every feature (including noisy ones) contributes to the
  distance metric.
  
  The GA tries to find a small subset of features (say 50-150) that, when
  used as the input to hierarchical clustering, produces the tightest and
  most separated clusters - as measured by the silhouette score.

WHAT THE GA IS DOING, STEP BY STEP
  1. Create a random population of "chromosomes". Each chromosome is a binary
     vector of length V (vocab size). A 1 means "keep this feature", 0 means
     "drop it".
  2. Evaluate each chromosome's fitness:
        a. Subset the TF-IDF matrix to just the selected features.
        b. Run hierarchical clustering on this reduced matrix.
        c. Compute the silhouette score of the resulting clusters.
        d. Apply a light penalty for chromosomes that select too many features
           (to encourage parsimony).
  3. Select parents via tournament selection.
  4. Create children via uniform crossover + bit-flip mutation.
  5. Apply elitism: the best 2 chromosomes survive unchanged.
  6. Repeat for N generations. Return the best chromosome found.

WHY A GA (AND WHEN NOT TO USE ONE)
  A GA is useful here because feature-subset selection is a combinatorial
  search problem: 2^V possible subsets, and the objective (silhouette after
  clustering) is not differentiable, so gradient methods don't apply.
  
  Honest caveat: with only 100 documents, ANY feature-selection method can
  overfit the silhouette score - we may be finding features that happen to
  cluster these 100 papers well, not features that generalize. This is
  acknowledged in the project report, not hidden.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


# ---------------------------------------------------------------------------
# GA hyperparameters. These are reasonable defaults; tune if needed.
# ---------------------------------------------------------------------------
@dataclass
class GAConfig:
    population_size: int = 30
    generations: int = 25
    tournament_size: int = 3
    elitism: int = 2
    crossover_prob: float = 0.9
    # Per-bit mutation probability scales with chromosome length so that, on
    # average, about 2 bits flip per mutation event.
    mutation_bits_target: float = 2.0
    # Constraints on how many features a chromosome may select.
    min_features: int = 30
    max_features: int = 200
    # Clustering used inside the fitness evaluation.
    n_clusters: int = 7
    # Penalty per selected feature - small enough to not dominate silhouette,
    # large enough to prefer smaller feature sets when silhouette ties.
    parsimony_weight: float = 0.0005
    random_state: int = 42


@dataclass
class GAResult:
    best_mask: np.ndarray                 # binary array, length V
    best_fitness: float
    best_silhouette: float
    best_num_features: int
    history: list[dict] = field(default_factory=list)
    total_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------
def _fitness(mask: np.ndarray, X: csr_matrix, cfg: GAConfig) -> tuple[float, float, int]:
    """Return (fitness, silhouette, num_features_selected).

    Fitness = silhouette_score - parsimony_weight * num_features
    Invalid chromosomes (too few / too many features) get -1.
    """
    n_selected = int(mask.sum())
    if n_selected < cfg.min_features or n_selected > cfg.max_features:
        return -1.0, -1.0, n_selected

    X_sub = X[:, mask.astype(bool)]
    # Agglomerative clustering needs a dense matrix for 'cosine' affinity
    X_dense = X_sub.toarray()

    # Edge case: if all rows are zero after subsetting (extremely unlikely but
    # possible if mask picks only terms nobody has), fail gracefully.
    if np.linalg.norm(X_dense, axis=1).min() == 0:
        return -1.0, -1.0, n_selected

    try:
        model = AgglomerativeClustering(
            n_clusters=cfg.n_clusters,
            metric="cosine",
            linkage="average",
        )
        labels = model.fit_predict(X_dense)
        if len(set(labels)) < 2:
            return -1.0, -1.0, n_selected
        sil = silhouette_score(X_dense, labels, metric="cosine")
    except Exception:
        return -1.0, -1.0, n_selected

    fit = sil - cfg.parsimony_weight * n_selected
    return float(fit), float(sil), n_selected


# ---------------------------------------------------------------------------
# GA operators
# ---------------------------------------------------------------------------
def _init_population(vocab_size: int, cfg: GAConfig, rng: np.random.Generator) -> np.ndarray:
    """Each chromosome starts with a random subset sized within the valid range."""
    pop = np.zeros((cfg.population_size, vocab_size), dtype=np.uint8)
    for i in range(cfg.population_size):
        k = rng.integers(cfg.min_features, cfg.max_features + 1)
        chosen = rng.choice(vocab_size, size=k, replace=False)
        pop[i, chosen] = 1
    return pop


def _tournament(pop: np.ndarray, fitnesses: np.ndarray, cfg: GAConfig, rng: np.random.Generator) -> np.ndarray:
    idx = rng.integers(0, len(pop), size=cfg.tournament_size)
    best = idx[np.argmax(fitnesses[idx])]
    return pop[best].copy()


def _uniform_crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    mask = rng.random(p1.shape) < 0.5
    c1 = np.where(mask, p1, p2).astype(np.uint8)
    c2 = np.where(mask, p2, p1).astype(np.uint8)
    return c1, c2


def _mutate(chrom: np.ndarray, cfg: GAConfig, rng: np.random.Generator) -> np.ndarray:
    V = chrom.shape[0]
    p_bit = cfg.mutation_bits_target / V
    flips = rng.random(V) < p_bit
    chrom = chrom.copy()
    chrom[flips] ^= 1
    # Repair: if the mutation pushes us outside the feature-count bounds,
    # randomly toggle bits until we are back within the valid range.
    k = int(chrom.sum())
    if k < cfg.min_features:
        zero_idx = np.where(chrom == 0)[0]
        need = cfg.min_features - k
        add = rng.choice(zero_idx, size=need, replace=False)
        chrom[add] = 1
    elif k > cfg.max_features:
        one_idx = np.where(chrom == 1)[0]
        remove = rng.choice(one_idx, size=k - cfg.max_features, replace=False)
        chrom[remove] = 0
    return chrom


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def run_ga(X: csr_matrix, cfg: GAConfig | None = None, verbose: bool = True) -> GAResult:
    cfg = cfg or GAConfig()
    rng = np.random.default_rng(cfg.random_state)
    V = X.shape[1]

    pop = _init_population(V, cfg, rng)
    fitnesses = np.array([_fitness(c, X, cfg)[0] for c in pop])

    t0 = time.time()
    history: list[dict] = []
    best_ever_mask = pop[int(np.argmax(fitnesses))].copy()
    best_ever_fit = float(fitnesses.max())

    for gen in range(cfg.generations):
        # Elitism - top `elitism` chromosomes survive unchanged.
        elite_idx = np.argsort(fitnesses)[-cfg.elitism:]
        new_pop = [pop[i].copy() for i in elite_idx]

        while len(new_pop) < cfg.population_size:
            p1 = _tournament(pop, fitnesses, cfg, rng)
            p2 = _tournament(pop, fitnesses, cfg, rng)
            if rng.random() < cfg.crossover_prob:
                c1, c2 = _uniform_crossover(p1, p2, rng)
            else:
                c1, c2 = p1, p2
            c1 = _mutate(c1, cfg, rng)
            c2 = _mutate(c2, cfg, rng)
            new_pop.append(c1)
            if len(new_pop) < cfg.population_size:
                new_pop.append(c2)

        pop = np.array(new_pop, dtype=np.uint8)
        fitnesses = np.array([_fitness(c, X, cfg)[0] for c in pop])

        cur_best = int(np.argmax(fitnesses))
        if fitnesses[cur_best] > best_ever_fit:
            best_ever_fit = float(fitnesses[cur_best])
            best_ever_mask = pop[cur_best].copy()

        history.append({
            "generation": gen + 1,
            "best_fitness": float(fitnesses.max()),
            "mean_fitness": float(fitnesses[fitnesses > -1].mean()) if (fitnesses > -1).any() else -1.0,
            "best_num_features": int(pop[cur_best].sum()),
        })
        if verbose:
            print(f"  gen {gen+1:02d} | best fit = {fitnesses.max():+.4f} | "
                  f"mean fit = {history[-1]['mean_fitness']:+.4f} | "
                  f"features = {history[-1]['best_num_features']}")

    # Report best ever, not just last generation's best
    best_fit, best_sil, best_k = _fitness(best_ever_mask, X, cfg)
    result = GAResult(
        best_mask=best_ever_mask,
        best_fitness=best_fit,
        best_silhouette=best_sil,
        best_num_features=best_k,
        history=history,
        total_seconds=time.time() - t0,
    )
    if verbose:
        print(f"\nGA finished in {result.total_seconds:.1f}s")
        print(f"  best fitness   : {result.best_fitness:+.4f}")
        print(f"  best silhouette: {result.best_silhouette:+.4f}")
        print(f"  features kept  : {result.best_num_features} / {V}")
    return result


if __name__ == "__main__":
    # Smoke test: requires preprocess.py to have run first.
    from preprocess import load_corpus, build_tfidf
    df = load_corpus()
    X, _, feats = build_tfidf(df["doc_text"].tolist())
    print(f"Corpus: {len(df)} papers, {X.shape[1]} features")
    res = run_ga(X)
    kept_feats = [feats[i] for i in np.where(res.best_mask)[0]][:30]
    print("\nSome kept features:", kept_feats)
