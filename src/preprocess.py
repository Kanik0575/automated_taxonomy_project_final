"""
preprocess.py
-------------
Turns the 100 paper abstracts into a TF-IDF feature matrix.

Design choices and why they matter:
  - min_df=2 drops any term that appears in only one paper. Terms that unique
    carry no cross-paper signal and just inflate the feature space.
  - max_df=0.85 drops terms that appear in more than 85% of papers. Words like
    "apt" or "attack" are in almost every paper and are useless for clustering.
  - ngram_range=(1,2) captures both unigrams and bigrams. "lateral movement"
    is far more informative than "lateral" + "movement" separately.
  - sublinear_tf=True uses 1+log(tf) instead of raw tf. Standard practice for
    text classification; dampens the effect of term repetition.
  - English stopwords are removed. Domain stopwords (apt, attack, paper, etc.)
    are added below - a paper on APTs will always contain these words, so they
    carry no discriminative signal.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CORPUS_CSV = DATA_DIR / "papers.csv"

# Domain-specific stopwords: these words are too generic in this specific corpus.
DOMAIN_STOPWORDS = {
    "apt", "attack", "attacks", "attacker", "attackers", "paper", "papers",
    "study", "studies", "research", "approach", "approaches", "method", "methods",
    "propose", "proposed", "proposes", "proposing", "result", "results",
    "experiment", "experiments", "experimental", "work", "works", "abstract",
    "introduction", "conclusion", "conclusions", "future", "show", "shows",
    "shown", "using", "used", "use", "uses", "based", "also", "however",
    "therefore", "thus", "moreover", "furthermore", "additionally", "advanced",
    "persistent", "threat", "threats",  # literally in the query
}


def clean_text(text: str) -> str:
    """Basic text cleaning before TF-IDF. Keep it minimal - TF-IDF handles a lot."""
    text = text.lower()
    # Replace non-alphanumeric (keep & - which matters for MITRE ATT&CK)
    text = re.sub(r"[^a-z0-9&\-\s]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_corpus(path: Path = CORPUS_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"\n\n  papers.csv not found at {path}\n"
            f"  Run `python src/fetch_papers.py` first to build the corpus.\n"
        )
    df = pd.read_csv(path)
    # Defensive: require non-empty abstracts. Should already be true after fetch.
    df = df[df["abstract"].notna() & (df["abstract"].str.len() > 50)].copy()
    df["doc_text"] = (df["title"].fillna("") + ". " + df["abstract"].fillna("")).apply(clean_text)
    return df.reset_index(drop=True)


def build_tfidf(docs: list[str]):
    """Build TF-IDF matrix. Returns (matrix, vectorizer, feature_names)."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    stopwords = list(ENGLISH_STOP_WORDS.union(DOMAIN_STOPWORDS))
    vectorizer = TfidfVectorizer(
        stop_words=stopwords,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        sublinear_tf=True,
        norm="l2",
    )
    X = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names_out()
    return X, vectorizer, feature_names


if __name__ == "__main__":
    df = load_corpus()
    print(f"Loaded {len(df)} papers from {CORPUS_CSV}")
    X, vec, feats = build_tfidf(df["doc_text"].tolist())
    print(f"TF-IDF matrix shape: {X.shape}  (papers x features)")
    print(f"Feature vocabulary size: {len(feats)}")
    print(f"Sparsity: {1 - X.nnz / (X.shape[0] * X.shape[1]):.2%}")
    print("\nFirst 20 features:", list(feats[:20]))
    print("Last 20 features:", list(feats[-20:]))
