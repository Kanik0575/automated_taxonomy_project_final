"""
generate_test_corpus.py
-----------------------
SYNTHETIC DATA GENERATOR - DO NOT USE FOR SUBMISSION.

Produces 100 procedurally-generated paper records and writes them to
data/papers.csv. These are NOT real papers. They exist only so you can:

  - Smoke-test the pipeline without reaching Semantic Scholar.
  - Demonstrate that the ML pipeline runs end-to-end when the API is
    temporarily unreachable.

Any figure or report derived from this synthetic corpus will produce
cleaner-than-real clusters, because the synthetic abstracts are built
around clean thematic vocabulary pools. Real-world silhouette scores
on Semantic Scholar data will be lower.

If you are seeing this file being run as part of the "real" pipeline,
stop. Run `python src/fetch_papers.py` instead to fetch real papers
from Semantic Scholar.
"""
import csv
import random
from pathlib import Path

random.seed(7)

# Themed vocabulary pools - representative of real APT research topics
THEMES = {
    "c2_detection": {
        "title_bits": ["detecting command and control", "C2 channel analysis",
                       "DNS tunneling detection", "beacon traffic classification",
                       "encrypted C2 identification", "TLS fingerprinting for APT C2"],
        "vocab": ["command and control", "c2", "beacon", "dns tunnel", "tls", "ja3",
                  "encrypted traffic", "covert channel", "domain fronting",
                  "network flow", "traffic analysis", "deep packet inspection"],
    },
    "ml_classification": {
        "title_bits": ["machine learning classifier for APT", "deep learning APT detection",
                       "neural network attack classification", "LSTM-based threat detection",
                       "transformer models for APT TTP extraction"],
        "vocab": ["machine learning", "deep learning", "neural network", "lstm", "transformer",
                  "bert", "classifier", "supervised learning", "feature engineering",
                  "random forest", "support vector", "convolutional"],
    },
    "lateral_movement": {
        "title_bits": ["detecting lateral movement", "pass the hash detection",
                       "internal network traversal analysis", "credential-based lateral movement",
                       "kerberos ticket abuse detection"],
        "vocab": ["lateral movement", "pass the hash", "pass the ticket", "kerberos",
                  "credential", "internal network", "authentication", "smb", "rdp",
                  "active directory", "hash", "token"],
    },
    "taxonomy_framework": {
        "title_bits": ["automated taxonomy of APT", "ATT&CK-based classification framework",
                       "ontology for APT attacks", "APT kill chain framework",
                       "unified classification schema"],
        "vocab": ["taxonomy", "ontology", "framework", "mitre att&ck", "kill chain",
                  "classification schema", "tactic technique procedure", "ttp",
                  "knowledge graph", "adversary behavior"],
    },
    "attribution": {
        "title_bits": ["APT group attribution", "threat actor attribution via TTPs",
                       "campaign attribution using graph analysis", "behavioral fingerprinting for attribution",
                       "nation-state attribution techniques"],
        "vocab": ["attribution", "threat actor", "apt group", "campaign",
                  "behavioral fingerprint", "nation state", "apt29", "apt41",
                  "lazarus", "diamond model", "graph embedding"],
    },
    "exfiltration": {
        "title_bits": ["low and slow exfiltration detection", "data exfiltration analysis",
                       "covert exfiltration channel detection", "time-series exfiltration modeling"],
        "vocab": ["exfiltration", "data leak", "dlp", "cloud storage upload",
                  "time series", "low and slow", "steganography", "covert channel",
                  "volume anomaly", "dns exfiltration"],
    },
    "initial_access": {
        "title_bits": ["spearphishing detection", "supply chain compromise analysis",
                       "phishing email classification", "zero-day exploit detection",
                       "initial access vector analysis"],
        "vocab": ["spearphishing", "phishing", "supply chain", "initial access",
                  "exploit", "zero day", "malicious attachment", "watering hole",
                  "email header", "sandbox detonation"],
    },
}

YEARS = [2021, 2022, 2023, 2024, 2025, 2026]
QUOTAS = {2021: 17, 2022: 17, 2023: 17, 2024: 17, 2025: 16, 2026: 16}

def make_abstract(theme_name: str) -> tuple[str, str]:
    theme = THEMES[theme_name]
    title = random.choice(theme["title_bits"]).capitalize()
    # Pad title with a year-or-venue variation
    title += ": " + random.choice(["A Systematic Study", "A Novel Approach", "An Empirical Evaluation",
                                     "Methods and Results", "Framework and Analysis"])
    n_vocab_terms = random.randint(6, 10)
    vocab = random.sample(theme["vocab"], min(n_vocab_terms, len(theme["vocab"])))
    cross_theme = random.sample(list(THEMES.keys()), 2)
    extra = random.sample(THEMES[cross_theme[0]]["vocab"], 2) + random.sample(THEMES[cross_theme[1]]["vocab"], 2)
    all_terms = vocab + extra
    random.shuffle(all_terms)
    abstract = (
        f"This paper addresses the challenge of {random.choice(['detecting', 'classifying', 'characterizing', 'attributing'])} "
        f"advanced persistent threats by leveraging {all_terms[0]} and {all_terms[1]}. "
        f"We propose a technique that uses {all_terms[2]}, {all_terms[3]} and {all_terms[4]} "
        f"to build a robust classifier for APT attack campaigns. "
        f"Our approach integrates {all_terms[5]} with {all_terms[6] if len(all_terms) > 6 else all_terms[0]} "
        f"and evaluates it across multiple datasets. Experimental results demonstrate that the method "
        f"improves detection accuracy over baseline approaches. The framework has implications for "
        f"threat intelligence analysis and can be integrated into enterprise security pipelines."
    )
    return title, abstract


def main():
    # Write to the project's data/ directory regardless of where this is run from
    import os
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    out = project_root / "data" / "papers.csv"
    out.parent.mkdir(exist_ok=True, parents=True)
    print("=" * 70)
    print("WARNING: Generating SYNTHETIC test data. DO NOT submit results")
    print("         derived from this corpus. For real data, run:")
    print("             python src/fetch_papers.py")
    print("=" * 70)
    rows = []
    paper_id = 0
    theme_cycle = list(THEMES.keys())
    for year, quota in QUOTAS.items():
        for i in range(quota):
            theme = theme_cycle[(paper_id) % len(theme_cycle)]
            title, abstract = make_abstract(theme)
            paper_id += 1
            rows.append({
                "paper_id": f"TEST{paper_id:04d}",
                "title": title,
                "year": year,
                "authors": "Smith J.; Kumar A.",
                "venue": "Test Venue",
                "publication_types": "JournalArticle",
                "doi": f"10.9999/test.{paper_id}",
                "arxiv_id": "",
                "url": f"https://example.org/test/{paper_id}",
                "open_access_pdf": "",
                "abstract": abstract,
                "source_query": "test",
            })
    fieldnames = list(rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} synthetic test papers to {out}")


if __name__ == "__main__":
    main()
