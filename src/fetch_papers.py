"""
fetch_papers.py
---------------
Fetches real academic papers on Advanced Persistent Threats (APTs) from
THREE independent free aggregators and merges them into a single corpus
of 100 papers distributed across 2021-2026.

WHY THREE SOURCES
  - Semantic Scholar: large, has abstracts, covers most venues.
  - arXiv:            adds preprints in cs.CR that S2 sometimes misses.
  - OpenAlex:         broad coverage of published journal + conference work;
                      abstracts come as an inverted index we decode here.

WHY NOT IEEE / ACM / SCOPUS DIRECTLY
  - IEEE Xplore API requires a key (approval can take days).
  - ACM has no public search API at all.
  - Scopus requires a paid Elsevier subscription.
  - All three publishers' papers ARE still reachable via the aggregators
    above. We tag each paper with its real publisher using DOI prefixes
    (10.1109=IEEE, 10.1145=ACM, 10.1007=Springer, 10.1016=Elsevier/Scopus).

OUTPUT
  data/papers.csv       - accepted corpus (target: 100 papers)
  data/papers.json      - same, JSON
  data/rejected_log.csv - every paper we saw but did not accept, with reason

Run:
    python src/fetch_papers.py
"""

from __future__ import annotations

import csv
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
YEAR_RANGE = (2021, 2026)
YEAR_QUOTAS: dict[int, int] = {
    2021: 17, 2022: 17, 2023: 17, 2024: 17, 2025: 16, 2026: 16,
}

USER_AGENT = "APTTaxonomyResearchProject/1.0 (university research; mailto:student@example.edu)"

# Queries - broader so we get hits across all three APIs
SEARCH_QUERIES = [
    "advanced persistent threat taxonomy",
    "advanced persistent threat classification",
    "advanced persistent threat detection framework",
    "APT MITRE ATT&CK",
    "APT machine learning detection",
    "APT threat intelligence",
    "APT attack lifecycle",
    "targeted cyber attack classification",
    "nation-state threat actor detection",
    "APT campaign analysis attribution",
]

# Max papers per query per source
MAX_PER_QUERY_S2 = 400
MAX_PER_QUERY_OPENALEX = 200
MAX_PER_QUERY_ARXIV = 200

# Filters
TOPIC_TERMS = [
    "advanced persistent threat", "apt", "threat actor", "targeted attack",
    "nation-state", "nation state", "threat intelligence", "cyber espionage",
    "cyber-attack", "cyber attack",
]
METHOD_TERMS = [
    "taxonomy", "classification", "framework", "mitre", "att&ck",
    "machine learning", "deep learning", "clustering", "detection",
    "neural network", "ontology", "knowledge graph", "automated",
    "systematic", "analysis", "attribution", "identification", "model",
    "algorithm", "feature", "characterization",
]

# Paths
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_CSV = ROOT / "data" / "papers.csv"
OUTPUT_JSON = ROOT / "data" / "papers.json"
REJECTED_LOG = ROOT / "data" / "rejected_log.csv"

# DOI prefix -> publisher. Used to tag each paper with its real publisher.
DOI_PUBLISHERS: list[tuple[str, str]] = [
    ("10.1109",   "IEEE"),
    ("10.1145",   "ACM"),
    ("10.1007",   "Springer"),
    ("10.1016",   "Elsevier (Scopus-indexed)"),
    ("10.1002",   "Wiley"),
    ("10.1561",   "NOW Publishers"),
    ("10.1093",   "Oxford University Press"),
    ("10.3390",   "MDPI"),
    ("10.48550",  "arXiv"),
    ("10.1017",   "Cambridge University Press"),
    ("10.1080",   "Taylor & Francis"),
    ("10.1177",   "SAGE"),
]

# Standard field schema we normalise every source to
FIELDS = [
    "paper_id", "title", "year", "authors", "venue", "publisher",
    "publication_types", "doi", "arxiv_id", "url", "open_access_pdf",
    "abstract", "source", "source_query",
]


# ---------------------------------------------------------------------------
# Generic HTTP with exponential backoff
# ---------------------------------------------------------------------------
def http_get(url: str, params: dict | None = None, max_retries: int = 6, timeout: int = 60) -> requests.Response | None:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code == 429:
                wait = 5 * (2 ** attempt)
                print(f"    [429 rate-limited, attempt {attempt+1}/{max_retries}, waiting {wait}s]")
                time.sleep(wait)
                continue
            if 500 <= resp.status_code < 600:
                wait = 3 * (2 ** attempt)
                print(f"    [{resp.status_code} server error, waiting {wait}s]")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.exceptions.Timeout:
            wait = 3 * (2 ** attempt)
            print(f"    [timeout, waiting {wait}s]")
            time.sleep(wait)
        except requests.exceptions.RequestException as exc:
            print(f"    [request error: {exc}]")
            if attempt == max_retries - 1:
                return None
            time.sleep(3 * (2 ** attempt))
    return None


def tag_publisher(doi: str, venue: str, source: str) -> str:
    """Return a publisher label given DOI/venue/source."""
    if doi:
        for prefix, pub in DOI_PUBLISHERS:
            if doi.startswith(prefix):
                return pub
    vl = (venue or "").lower()
    if "ieee" in vl or "transactions on" in vl:
        return "IEEE"
    if "acm" in vl or "sigsac" in vl or "ccs" in vl:
        return "ACM"
    if "springer" in vl or "lecture notes" in vl or "lncs" in vl:
        return "Springer"
    if "elsevier" in vl or "computers & security" in vl:
        return "Elsevier (Scopus-indexed)"
    if source == "arxiv":
        return "arXiv"
    return "Other"


# ===========================================================================
# SOURCE 1: Semantic Scholar (bulk endpoint)
# ===========================================================================
S2_BULK = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
S2_FIELDS = "title,abstract,authors,year,externalIds,url,venue,publicationTypes,openAccessPdf"


def fetch_semantic_scholar(query: str, max_total: int = MAX_PER_QUERY_S2) -> list[dict]:
    collected: list[dict] = []
    token: str | None = None
    page = 0
    year_str = f"{YEAR_RANGE[0]}-{YEAR_RANGE[1]}"
    while len(collected) < max_total:
        params = {"query": query, "year": year_str, "fields": S2_FIELDS}
        if token:
            params["token"] = token
        resp = http_get(S2_BULK, params)
        if not resp:
            break
        try:
            data = resp.json()
        except json.JSONDecodeError:
            break
        batch = data.get("data", []) or []
        collected.extend(batch)
        page += 1
        print(f"    S2 page {page}: {len(batch)} papers (total {len(collected)})")
        token = data.get("token")
        if not token or not batch:
            break
        time.sleep(1.3)
    return [_normalize_s2(r, query) for r in collected[:max_total]]


def _normalize_s2(raw: dict, source_query: str) -> dict:
    authors = raw.get("authors") or []
    author_names = "; ".join(a.get("name", "") for a in authors if a.get("name"))
    ext = raw.get("externalIds") or {}
    pub_types = raw.get("publicationTypes") or []
    oa = raw.get("openAccessPdf") or {}
    doi = ext.get("DOI", "") or ""
    venue = (raw.get("venue") or "").strip()
    return {
        "paper_id": f"s2:{raw.get('paperId','')}",
        "title": (raw.get("title") or "").strip(),
        "year": raw.get("year"),
        "authors": author_names,
        "venue": venue,
        "publisher": tag_publisher(doi, venue, "semantic_scholar"),
        "publication_types": "; ".join(pub_types) if pub_types else "",
        "doi": doi,
        "arxiv_id": ext.get("ArXiv", "") or "",
        "url": raw.get("url", "") or "",
        "open_access_pdf": (oa.get("url", "") if oa else "") or "",
        "abstract": (raw.get("abstract") or "").strip(),
        "source": "semantic_scholar",
        "source_query": source_query,
    }


# ===========================================================================
# SOURCE 2: arXiv (Atom XML)
# ===========================================================================
ARXIV_API = "http://export.arxiv.org/api/query"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def fetch_arxiv(query: str, max_total: int = MAX_PER_QUERY_ARXIV) -> list[dict]:
    # arXiv doesn't take year as a query param directly; we filter after.
    # Restrict to cs.CR (Cryptography and Security) for relevance.
    search = f'cat:cs.CR AND (abs:"{query}" OR ti:"{query}")'
    collected: list[dict] = []
    start = 0
    page_size = 100
    while len(collected) < max_total:
        params = {
            "search_query": search,
            "start": start,
            "max_results": page_size,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        resp = http_get(ARXIV_API, params, timeout=60)
        if not resp:
            break
        try:
            root = ET.fromstring(resp.text)
        except ET.ParseError:
            break
        entries = root.findall("atom:entry", ARXIV_NS)
        if not entries:
            break
        batch = [_normalize_arxiv(e, query) for e in entries]
        batch = [p for p in batch if p and p.get("year") and YEAR_RANGE[0] <= p["year"] <= YEAR_RANGE[1]]
        collected.extend(batch)
        print(f"    arXiv start={start}: {len(batch)} in-range papers (total {len(collected)})")
        start += page_size
        if len(entries) < page_size:
            break
        time.sleep(3.0)  # arXiv asks for >3s between calls
    return collected[:max_total]


def _normalize_arxiv(entry: ET.Element, source_query: str) -> dict | None:
    try:
        arxiv_id_full = entry.findtext("atom:id", default="", namespaces=ARXIV_NS)
        arxiv_id = arxiv_id_full.rsplit("/", 1)[-1] if arxiv_id_full else ""
        title = (entry.findtext("atom:title", default="", namespaces=ARXIV_NS) or "").strip().replace("\n", " ")
        abstract = (entry.findtext("atom:summary", default="", namespaces=ARXIV_NS) or "").strip().replace("\n", " ")
        published = entry.findtext("atom:published", default="", namespaces=ARXIV_NS) or ""
        year = int(published[:4]) if published[:4].isdigit() else None
        authors = "; ".join(
            a.findtext("atom:name", default="", namespaces=ARXIV_NS) or ""
            for a in entry.findall("atom:author", ARXIV_NS)
        )
        pdf_url = ""
        for link in entry.findall("atom:link", ARXIV_NS):
            if link.get("type") == "application/pdf":
                pdf_url = link.get("href", "")
        doi = entry.findtext("arxiv:doi", default="", namespaces=ARXIV_NS) or ""
        return {
            "paper_id": f"arxiv:{arxiv_id}",
            "title": title,
            "year": year,
            "authors": authors,
            "venue": "arXiv preprint",
            "publisher": tag_publisher(doi, "arXiv", "arxiv"),
            "publication_types": "Preprint",
            "doi": doi,
            "arxiv_id": arxiv_id,
            "url": arxiv_id_full,
            "open_access_pdf": pdf_url,
            "abstract": re.sub(r"\s+", " ", abstract),
            "source": "arxiv",
            "source_query": source_query,
        }
    except Exception:
        return None


# ===========================================================================
# SOURCE 3: OpenAlex
# ===========================================================================
OPENALEX_WORKS = "https://api.openalex.org/works"


def fetch_openalex(query: str, max_total: int = MAX_PER_QUERY_OPENALEX) -> list[dict]:
    collected: list[dict] = []
    cursor = "*"
    per_page = 100
    page = 0
    while len(collected) < max_total and cursor:
        params = {
            "search": query,
            "filter": f"publication_year:{YEAR_RANGE[0]}-{YEAR_RANGE[1]},type:article",
            "per-page": per_page,
            "cursor": cursor,
        }
        resp = http_get(OPENALEX_WORKS, params)
        if not resp:
            break
        try:
            data = resp.json()
        except json.JSONDecodeError:
            break
        results = data.get("results", []) or []
        batch = [_normalize_openalex(r, query) for r in results]
        batch = [p for p in batch if p]
        collected.extend(batch)
        page += 1
        meta = data.get("meta", {}) or {}
        cursor = meta.get("next_cursor")
        print(f"    OpenAlex page {page}: {len(batch)} papers (total {len(collected)})")
        if not results:
            break
        time.sleep(1.2)
    return collected[:max_total]


def _decode_abstract_inverted_index(idx: dict | None) -> str:
    """OpenAlex returns abstracts as {word: [positions]}. Reconstruct text."""
    if not idx:
        return ""
    positions: list[tuple[int, str]] = []
    for word, pos_list in idx.items():
        for p in pos_list:
            positions.append((p, word))
    positions.sort()
    return " ".join(w for _, w in positions)


def _normalize_openalex(raw: dict, source_query: str) -> dict | None:
    try:
        oa_id = raw.get("id", "").rsplit("/", 1)[-1] if raw.get("id") else ""
        title = (raw.get("title") or raw.get("display_name") or "").strip()
        year = raw.get("publication_year")
        authorships = raw.get("authorships") or []
        authors = "; ".join(
            (a.get("author", {}) or {}).get("display_name", "") for a in authorships
        )
        doi = (raw.get("doi") or "").replace("https://doi.org/", "")
        abstract = _decode_abstract_inverted_index(raw.get("abstract_inverted_index"))
        host = raw.get("primary_location") or {}
        source = host.get("source") or {}
        venue = (source.get("display_name") or "").strip()
        oa_pdf = (host.get("pdf_url") or "") if host else ""
        pub_types = raw.get("type") or ""
        return {
            "paper_id": f"oa:{oa_id}",
            "title": title,
            "year": year,
            "authors": authors,
            "venue": venue,
            "publisher": tag_publisher(doi, venue, "openalex"),
            "publication_types": pub_types,
            "doi": doi,
            "arxiv_id": "",
            "url": raw.get("id", ""),
            "open_access_pdf": oa_pdf,
            "abstract": abstract.strip(),
            "source": "openalex",
            "source_query": source_query,
        }
    except Exception:
        return None


# ===========================================================================
# Filters, dedup, allocation
# ===========================================================================
def passes_strict(p: dict) -> tuple[bool, str]:
    abstract = (p.get("abstract") or "").strip()
    if not abstract or len(abstract) < 50:
        return False, "no_or_short_abstract"
    combined = ((p.get("title") or "") + " " + abstract).lower()
    if not any(t in combined for t in TOPIC_TERMS):
        return False, "no_topic_term"
    if not any(m in combined for m in METHOD_TERMS):
        return False, "no_method_term"
    return True, "accepted"


def passes_relaxed(p: dict) -> bool:
    abstract = (p.get("abstract") or "").strip()
    if not abstract or len(abstract) < 30:
        return False
    combined = ((p.get("title") or "") + " " + abstract).lower()
    return any(t in combined for t in TOPIC_TERMS)


def dedupe_key(p: dict) -> str:
    """Dedupe by DOI if present, else normalised title."""
    doi = (p.get("doi") or "").strip().lower()
    if doi:
        return f"doi:{doi}"
    arx = (p.get("arxiv_id") or "").strip().lower()
    if arx:
        return f"arxiv:{arx}"
    title = (p.get("title") or "").lower()
    title = re.sub(r"[^a-z0-9]+", " ", title).strip()
    return f"title:{title[:80]}"


def build_corpus() -> tuple[list[dict], list[dict]]:
    all_papers: dict[str, dict] = {}

    # --- Stage 1: Fetch from all sources ---
    for i, q in enumerate(SEARCH_QUERIES, 1):
        print(f"\n=== Query {i}/{len(SEARCH_QUERIES)}: {q!r}")

        print("  [Semantic Scholar]")
        s2_batch = fetch_semantic_scholar(q)
        _merge(all_papers, s2_batch, "S2")

        print("  [arXiv]")
        arx_batch = fetch_arxiv(q)
        _merge(all_papers, arx_batch, "arXiv")

        print("  [OpenAlex]")
        oa_batch = fetch_openalex(q)
        _merge(all_papers, oa_batch, "OpenAlex")

        print(f"  => unique pool after query {i}: {len(all_papers)} papers")

    print(f"\n{'='*60}")
    print(f"Total unique candidates pooled across all sources: {len(all_papers)}")
    print(f"{'='*60}")

    # --- Stage 2: Strict filter + year bucket ---
    eligible: dict[int, list[dict]] = {y: [] for y in YEAR_QUOTAS}
    rejected: list[dict] = []
    for p in all_papers.values():
        year = p.get("year")
        if year not in YEAR_QUOTAS:
            rejected.append({**p, "reject_reason": "year_out_of_range"})
            continue
        ok, reason = passes_strict(p)
        if ok:
            eligible[year].append(p)
        else:
            rejected.append({**p, "reject_reason": reason})

    print("\nAfter strict filter - eligible per year:")
    for y in YEAR_QUOTAS:
        print(f"  {y}: {len(eligible[y])}")

    # --- Stage 3: Allocate per year ---
    accepted: list[dict] = []
    shortfalls: list[tuple[int, int]] = []
    for year, quota in YEAR_QUOTAS.items():
        pool = sorted(
            eligible[year],
            key=lambda x: (-len(x.get("abstract") or ""), x.get("title") or ""),
        )
        taken = pool[:quota]
        accepted.extend(taken)
        for p in pool[quota:]:
            rejected.append({**p, "reject_reason": "over_year_quota"})
        if len(taken) < quota:
            shortfalls.append((year, quota - len(taken)))

    # --- Stage 4: Relaxed rescue pass ---
    if shortfalls:
        print(f"\nShortfalls: {shortfalls}  running RELAXED rescue pass...")
        accepted_keys = {dedupe_key(p) for p in accepted}
        for year, needed in shortfalls:
            candidates = [
                p for p in rejected
                if p.get("year") == year
                and dedupe_key(p) not in accepted_keys
                and p.get("reject_reason") in ("no_method_term", "no_topic_term")
                and passes_relaxed(p)
            ]
            candidates.sort(key=lambda x: (-len(x.get("abstract") or ""), x.get("title") or ""))
            chosen = candidates[:needed]
            for p in chosen:
                accepted.append(p)
                accepted_keys.add(dedupe_key(p))
                p["reject_reason"] = "RESCUED_via_relaxed_filter"
            print(f"  {year}: rescued {len(chosen)}/{needed}")

    return accepted, rejected


def _merge(pool: dict[str, dict], batch: list[dict], label: str) -> None:
    added = 0
    for p in batch:
        if not p:
            continue
        k = dedupe_key(p)
        if k in pool:
            # keep whichever has the longer abstract
            existing = pool[k]
            if len(p.get("abstract") or "") > len(existing.get("abstract") or ""):
                pool[k] = p
            continue
        pool[k] = p
        added += 1
    print(f"    merged {added} new unique papers from {label} "
          f"({len(batch) - added} duplicates with pool)")


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def write_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    print("Multi-source APT paper fetcher")
    print(f"  Sources: Semantic Scholar + arXiv + OpenAlex")
    print(f"  Year range: {YEAR_RANGE[0]}-{YEAR_RANGE[1]}")
    print(f"  Per-year quotas: {YEAR_QUOTAS}")
    print(f"  Queries: {len(SEARCH_QUERIES)}")

    accepted, rejected = build_corpus()

    write_csv(accepted, OUTPUT_CSV, FIELDS)
    write_csv(rejected, REJECTED_LOG, FIELDS + ["reject_reason"])
    with OUTPUT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(accepted, fh, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Accepted: {len(accepted)}  |  Rejected: {len(rejected)}")

    by_year: dict[int, int] = {}
    by_pub: dict[str, int] = {}
    by_src: dict[str, int] = {}
    for p in accepted:
        by_year[p["year"]] = by_year.get(p["year"], 0) + 1
        by_pub[p["publisher"]] = by_pub.get(p["publisher"], 0) + 1
        by_src[p["source"]] = by_src.get(p["source"], 0) + 1

    print("\nYear distribution (target in parens):")
    all_ok = True
    for y in sorted(YEAR_QUOTAS):
        got = by_year.get(y, 0)
        target = YEAR_QUOTAS[y]
        mark = "OK" if got == target else "!!"
        if got != target:
            all_ok = False
        print(f"  {mark} {y}: {got}/{target}")

    print("\nPublisher distribution (via DOI prefix):")
    for pub, n in sorted(by_pub.items(), key=lambda x: -x[1]):
        print(f"  {pub}: {n}")

    print("\nSource distribution (aggregator):")
    for src, n in sorted(by_src.items(), key=lambda x: -x[1]):
        print(f"  {src}: {n}")

    print(f"\nCorpus: {OUTPUT_CSV}")
    print(f"Rejected log: {REJECTED_LOG}")

    if not all_ok:
        print("\nStill short of 100. Options:")
        print("  1. Re-run in 5 minutes (rate-limit windows clear).")
        print("  2. Increase MAX_PER_QUERY_S2 / OPENALEX / ARXIV at top of file.")
        print("  3. Add more SEARCH_QUERIES.")
        return 2

    print("\nAll year quotas met. Ready for: python src/main.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
