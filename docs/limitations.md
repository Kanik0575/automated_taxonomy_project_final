# Limitations

An honest discussion of where this project is weak. Include a condensed version of this in the final report. Acknowledging limitations signals research maturity; hiding them invites harder questions in a viva.

---

## 1. Small sample size (n = 100)

The corpus is 100 papers. With only 100 documents:

- **Silhouette is noisy.** Silhouette assumes sufficient density within clusters; with 14–17 papers per cluster, individual outliers can swing the score noticeably.
- **Feature-selection overfitting is a real risk.** The GA finds features that happen to cluster these 100 papers well. Held-out validation on a disjoint set of papers would be needed to show generalisation, and we do not do this.
- **Cluster stability across runs is not tested.** A proper study would bootstrap-resample the corpus and check whether the same clusters emerge — we do not currently do this.

**Mitigation in the code:** a parsimony penalty in the GA fitness function discourages selecting features just to improve silhouette marginally. This reduces but does not eliminate overfitting risk.

## 2. Abstract-only analysis

We process only titles and abstracts. Abstracts are summary artefacts — they highlight contributions but may omit:

- Dataset descriptions
- Detailed methodology
- Evaluation protocols
- Limitations and failure modes

A cluster built on abstract vocabulary may miss papers that are methodologically similar but described with different marketing language, and may conflate papers that use identical vocabulary but address fundamentally different subproblems.

**Why we didn't use full text:** Semantic Scholar provides full text only for open-access papers, and the open-access fraction varies by year and publisher. A mixed abstract/full-text corpus would introduce an uncontrolled source of variation.

## 3. Semantic Scholar coverage gaps

Semantic Scholar is excellent but not exhaustive. Known gaps:

- Some IEEE conference proceedings are indexed with delay.
- Some Springer chapters are missing entirely.
- Preprints (arXiv) and published versions are sometimes treated as separate records.

This means our corpus is a sample of what Semantic Scholar currently indexes, not a census of all published APT research.

## 4. The MITRE ATT&CK labelling is heuristic

Cluster labels are assigned by counting, for each cluster's top-10 distinctive terms, how many overlap with a pre-defined tactic keyword dictionary. This is:

- Deterministic and explainable (good)
- But not learned from data (weak)
- And limited to keywords we thought of in advance (fragile)

A cluster whose characteristic terms are novel — say, a new attack paradigm not yet in our keyword dictionary — will be labelled "Unclassified." The dictionary should be reviewed and expanded periodically. An LLM-based labeller would be a more robust next step and is a natural future extension.

## 5. Silhouette is not ground truth

Silhouette measures intra-cluster tightness vs inter-cluster separation under a given distance metric. It does NOT measure:

- Whether the clusters are *useful* to a human reader.
- Whether the clusters correspond to research-community consensus on APT subtopics.
- Whether the clusters are stable across perturbations.

A clustering can score high silhouette while producing arbitrary groupings. We mitigate this by also presenting the top characteristic terms per cluster — a human can inspect whether the terms form a coherent theme. But this mitigation is informal.

## 6. Research-literature clustering is not attack-behaviour taxonomy

This is the most important conceptual limitation and the one most likely to be probed in a viva. Our clusters reflect *how researchers write about APTs*, not *how APTs behave*. A cluster labelled "Detection Methods" contains papers that talk about detection methods — it is not an attack stage. The canonical taxonomy of attacker behaviour is MITRE ATT&CK, maintained manually by domain experts.

A reader who takes our dendrogram as "the taxonomy of APT attacks" is reading it wrong. The work is better described as a bibliometric thematic map of the APT research literature, automatically constructed.

## 7. Year stratification is coarse

We enforce equal representation across 2021–2026, but we do not model *within-year* popularity drift. A research theme that exploded in late 2023 will be under-represented in our 2023 slice if our relevance filter retrieves older-but-similar papers first. A time-aware sampling strategy (e.g., quarterly quotas) would produce a finer picture at the cost of more engineering.

## 8. Language and venue bias

- The corpus is English-only. APT research in non-English venues is excluded.
- Open-access venues are over-represented because Semantic Scholar tends to have better metadata for them.
- Conference proceedings outside major CS venues are under-represented.

## 9. No human validation

A proper taxonomic study would ask domain experts to independently label a subset of papers, then measure whether our clusters agree with expert judgments (e.g., via Fleiss' kappa). We do not do this. The clusters are presented as an automated first-pass grouping for exploratory analysis, not as a validated taxonomy.

---

## What these limitations mean in practice

- **For your viva:** state these limitations openly. Your professor will respect it.
- **For the report:** include a condensed limitations section (around 3–4 of the items above) — don't hide them.
- **For future work:** items #2 (full-text analysis), #4 (LLM labelling), and #9 (human validation) are the three most tractable extensions if this becomes a larger project.
