"""
Phase 1: Download arXiv papers at intersection of AI/philosophy/neuroscience.
Uses arXiv API (rate limit: 1 request per 3 seconds).

Usage:
    python corpus/scripts/download_arxiv.py
"""

import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time
import os
import json
from pathlib import Path

QUERIES = [
    'all:"consciousness" AND cat:cs.AI',
    'all:"philosophy of mind" AND cat:cs.AI',
    'all:"neural correlates consciousness"',
    'all:"hard problem consciousness"',
    'all:"integrated information theory"',
    'all:"global workspace theory"',
    'all:"phenomenal consciousness" AND cat:q-bio.NC',
    'all:"qualia" AND (cat:cs.AI OR cat:cs.CL)',
    'all:"zombie argument" AND cat:cs.AI',
    'all:"formal verification" AND "philosophical"',
]

# Use project-relative paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "corpus" / "raw" / "cross_disciplinary" / "arxiv"
METADATA_DIR = PROJECT_ROOT / "corpus" / "metadata"

BASE_URL = "http://export.arxiv.org/api/query"


def search_arxiv(query: str, max_results: int = 100):
    """Search arXiv API for papers matching query."""
    params = urllib.parse.urlencode({
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance"
    })
    url = f"{BASE_URL}?{params}"

    req = urllib.request.Request(url, headers={
        "User-Agent": "PhilosopherEngine/1.0 (Academic Research Corpus Builder)"
    })
    response = urllib.request.urlopen(req, timeout=30)
    root = ET.fromstring(response.read())

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", ns)

    results = []
    for entry in entries:
        paper_id = entry.find("atom:id", ns).text.split("/abs/")[-1]
        title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
        summary = entry.find("atom:summary", ns).text.strip().replace("\n", " ")

        # Get categories
        categories = []
        for cat in entry.findall("atom:category", ns):
            term = cat.get("term")
            if term:
                categories.append(term)

        # Get authors
        authors = []
        for author in entry.findall("atom:author", ns):
            name = author.find("atom:name", ns)
            if name is not None:
                authors.append(name.text)

        results.append({
            "id": paper_id,
            "title": title,
            "summary": summary,
            "authors": authors,
            "categories": categories,
        })

    return results


def download_pdf(paper_id: str) -> bool:
    """Download PDF for a given arXiv paper ID."""
    clean_id = paper_id.replace("/", "_")
    filepath = OUTPUT_DIR / f"{clean_id}.pdf"
    if filepath.exists():
        return True

    url = f"https://arxiv.org/pdf/{paper_id}.pdf"
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "PhilosopherEngine/1.0 (Academic Research Corpus Builder)"
        })
        urllib.request.urlretrieve(url, filepath)
        return True
    except Exception as e:
        print(f"  [FAIL] {paper_id} ({e})")
        return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    all_papers = {}

    print("Searching arXiv for relevant papers...")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Rate limit: 1 request per 3 seconds\n")

    for query in QUERIES:
        print(f"Searching: {query}")
        try:
            results = search_arxiv(query)
            new_count = 0
            for paper in results:
                if paper["id"] not in all_papers:
                    all_papers[paper["id"]] = paper
                    new_count += 1
            print(f"  Found {len(results)} results ({new_count} new)")
        except Exception as e:
            print(f"  [ERROR] {e}")
        time.sleep(3)

    print(f"\nFound {len(all_papers)} unique papers. Downloading PDFs...")

    success = 0
    for i, pid in enumerate(all_papers, 1):
        print(f"  [{i}/{len(all_papers)}] {pid}: {all_papers[pid]['title'][:60]}...")
        if download_pdf(pid):
            success += 1
        time.sleep(1)

    # Save metadata
    meta_path = METADATA_DIR / "arxiv_download_metadata.json"
    meta = {
        "source": "arXiv",
        "queries": QUERIES,
        "total_unique_papers": len(all_papers),
        "pdfs_downloaded": success,
        "papers": list(all_papers.values()),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\nDone: {success}/{len(all_papers)} PDFs downloaded")
    print(f"Metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
