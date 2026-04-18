"""
taxonomy.py
-----------
Turns raw cluster assignments into a labelled taxonomy.

A cluster number by itself means nothing to a reader. For each cluster we
extract the TF-IDF terms that are most *characteristic* of papers in that
cluster (i.e. terms whose mean TF-IDF inside the cluster is much higher
than in the rest of the corpus). These characteristic terms become the
cluster label.

We also try to map each cluster to a MITRE ATT&CK tactic by keyword matching
the top terms against known tactic-specific vocabulary. This map is clearly
labelled as "suggested" - the user should eyeball it before trusting it.

Output per cluster:
  - size
  - top N characteristic terms
  - suggested MITRE tactic (or "unclassified")
  - representative paper titles
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


# Rough keyword -> MITRE tactic mapping. This is used purely to SUGGEST a label
# for each cluster; it is not ground truth. The user should review.
TACTIC_KEYWORDS: dict[str, list[str]] = {
    "Reconnaissance":       ["reconnaissance", "osint", "scanning", "footprint"],
    "Resource Development": ["infrastructure", "domain generation", "dga", "staging"],
    "Initial Access":       ["phishing", "spearphishing", "initial access", "supply chain", "exploit"],
    "Execution":            ["execution", "powershell", "script", "living off the land", "lolbin"],
    "Persistence":          ["persistence", "web shell", "registry", "startup", "backdoor"],
    "Privilege Escalation": ["privilege escalation", "token", "kernel exploit"],
    "Defense Evasion":      ["evasion", "obfuscation", "rootkit", "anti-forensic", "timestomp"],
    "Credential Access":    ["credential", "password", "lsass", "kerberoasting", "hash"],
    "Discovery":            ["discovery", "enumeration", "active directory", "ldap"],
    "Lateral Movement":     ["lateral movement", "pass the hash", "pass the ticket", "rdp", "smb"],
    "Collection":           ["collection", "keylog", "screen capture", "staged data"],
    "Command and Control":  ["command and control", "c2", "beacon", "dns tunnel",
                             "jaspi", "ja3", "covert channel", "exfiltration channel"],
    "Exfiltration":         ["exfiltration", "data exfiltration", "covert exfil",
                             "cloud storage", "low and slow"],
    "Impact":               ["wiper", "ransomware", "destructive", "ics", "scada"],
    "Detection Methods":    ["anomaly detection", "classifier", "neural network",
                             "graph neural", "lstm", "transformer", "bert", "federated"],
    "Attribution":          ["attribution", "threat actor", "apt group", "tactic technique"],
    "Survey/Review":        ["survey", "systematic review", "literature review", "overview"],
}


@dataclass
class ClusterProfile:
    cluster_id: int
    size: int
    top_terms: list[tuple[str, float]]
    suggested_label: str
    representative_papers: list[str]


def compute_cluster_profiles(
    X: csr_matrix,
    labels: np.ndarray,
    feature_names: np.ndarray,
    df: pd.DataFrame,
    top_n_terms: int = 10,
    top_n_papers: int = 3,
) -> list[ClusterProfile]:
    """For each cluster, find the most characteristic terms and representative papers."""
    X_dense = X.toarray()
    profiles: list[ClusterProfile] = []

    for cid in sorted(set(labels)):
        in_cluster = labels == cid
        out_cluster = ~in_cluster
        size = int(in_cluster.sum())

        # Characteristic = high mean tfidf inside - mean tfidf outside
        mean_in = X_dense[in_cluster].mean(axis=0)
        if out_cluster.any():
            mean_out = X_dense[out_cluster].mean(axis=0)
        else:
            mean_out = np.zeros_like(mean_in)
        distinctiveness = mean_in - mean_out

        top_idx = np.argsort(distinctiveness)[::-1][:top_n_terms]
        top_terms = [(str(feature_names[i]), float(distinctiveness[i])) for i in top_idx if distinctiveness[i] > 0]

        label = _suggest_label([t for t, _ in top_terms])

        # Representative papers: those closest to the cluster centroid
        centroid = mean_in
        cluster_X = X_dense[in_cluster]
        if cluster_X.shape[0] > 0:
            # cosine similarity to centroid
            centroid_norm = np.linalg.norm(centroid) + 1e-12
            row_norms = np.linalg.norm(cluster_X, axis=1) + 1e-12
            sims = (cluster_X @ centroid) / (row_norms * centroid_norm)
            rep_idx_local = np.argsort(sims)[::-1][:top_n_papers]
            rep_idx_global = np.where(in_cluster)[0][rep_idx_local]
            rep_titles = [str(df.iloc[i]["title"]) for i in rep_idx_global]
        else:
            rep_titles = []

        profiles.append(ClusterProfile(
            cluster_id=int(cid),
            size=size,
            top_terms=top_terms,
            suggested_label=label,
            representative_papers=rep_titles,
        ))

    return profiles


def _suggest_label(top_terms: list[str]) -> str:
    """Pick the MITRE-like tactic whose keywords best match the top terms."""
    if not top_terms:
        return "Unclassified"
    joined = " ".join(top_terms).lower()
    scores: dict[str, int] = {}
    for tactic, keywords in TACTIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in joined)
        if score > 0:
            scores[tactic] = score
    if not scores:
        return "Unclassified (inspect top terms)"
    # If there's a tie, concat the top two
    best = sorted(scores.items(), key=lambda x: -x[1])
    if len(best) > 1 and best[0][1] == best[1][1]:
        return f"{best[0][0]} / {best[1][0]}"
    return best[0][0]


def profiles_to_markdown(profiles: list[ClusterProfile]) -> str:
    """Render the taxonomy as a readable markdown document."""
    lines = ["# Automated Taxonomy of APT Research Literature", ""]
    lines.append(f"Derived from hierarchical clustering of {sum(p.size for p in profiles)} papers.")
    lines.append("")
    lines.append("Each cluster below represents a thematic group of papers that use")
    lines.append("similar vocabulary in their abstracts. The **suggested label** is")
    lines.append("a heuristic match to MITRE ATT&CK tactics - verify by inspection.")
    lines.append("")
    lines.append("---")
    lines.append("")

    for p in profiles:
        lines.append(f"## Cluster {p.cluster_id}: {p.suggested_label}")
        lines.append(f"- **Size:** {p.size} papers")
        lines.append(f"- **Top characteristic terms:** " +
                     ", ".join(f"`{t}` ({s:.3f})" for t, s in p.top_terms[:8]))
        if p.representative_papers:
            lines.append("- **Representative papers (closest to centroid):**")
            for t in p.representative_papers:
                lines.append(f"  - {t}")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    from preprocess import load_corpus, build_tfidf
    from clustering import cluster_papers

    df = load_corpus()
    X, _, feats = build_tfidf(df["doc_text"].tolist())
    res = cluster_papers(X, n_clusters=7)
    profiles = compute_cluster_profiles(X, res.labels, feats, df)
    print(profiles_to_markdown(profiles))
