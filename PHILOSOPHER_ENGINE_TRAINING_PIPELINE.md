# Philosopher Engine: CPT/SFT Training Pipeline
## Claude Code Instructional Guide

This guide is designed to be executed session-by-session in Claude Code. Each phase is a self-contained session with clear inputs, outputs, and validation steps. Copy the relevant phase into Claude Code and execute.

---

## Pipeline Overview

```
Phase 1: Corpus Assembly        → Raw philosophical texts
Phase 2: Text Extraction        → Clean plaintext from PDFs/EPUBs
Phase 3: Cleaning & Filtering   → Normalized, deduplicated corpus
Phase 4: CPT Data Formatting    → Tokenized JSONL ready for training
Phase 5: CPT Training           → Domain-adapted base model
Phase 6: SFT Data Generation    → LLM council + Z3 validated examples
Phase 7: SFT Training           → Instruction-tuned philosophical model
Phase 8: Evaluation             → Benchmarks + qualitative assessment
```

**Total estimated time**: 3-6 weeks part-time
**Total estimated cost**: $8,000-$28,000 (GPU + API)
**Required infrastructure**: Cloud GPU (A100 80GB x4 minimum for 70B CPT)

---

## Phase 1: Corpus Assembly

### Session Goal
Identify and download all source texts for the CPT corpus. Target: 500M-2B tokens.

### Claude Code Instructions

```
You are assembling a philosophical text corpus for continued pre-training. 
The target is 500M-2B tokens with this mixing ratio:

- 40% Philosophy of Mind (target: 200M-800M tokens)
- 20% Neuroscience (target: 100M-400M tokens)  
- 15% Broader Philosophy (target: 75M-300M tokens)
- 15% Cognitive Science (target: 75M-300M tokens)
- 10% Cross-disciplinary bridges (target: 50M-200M tokens)

CRITICAL: Only original published texts. Never LLM-generated content.

Create a directory structure at ~/corpus/ and a manifest tracking all sources.
```

### Source Inventory

```yaml
# corpus_config.yaml

corpus_target_tokens: 1_000_000_000  # 1B tokens target

sources:

  # ---- PRIMARY: Open Access Philosophy ----
  
  stanford_encyclopedia:
    url: "https://plato.stanford.edu/"
    type: "web_scrape"
    estimated_tokens: 80_000_000
    category: "philosophy_of_mind"  # + broader_philosophy
    notes: >
      ~1,800 articles. Focus on: consciousness, qualia, 
      functionalism, physicalism, dualism, mental causation,
      personal identity, free will, intentionality, perception,
      philosophy of neuroscience, cognitive science entries.
    priority: 1
    
  philpapers_open_access:
    url: "https://philpapers.org/"
    type: "pdf_download"
    estimated_tokens: 200_000_000
    category: "philosophy_of_mind"
    notes: >
      Filter for: open access papers in philosophy of mind,
      philosophy of cognitive science, metaphysics of mind.
      PhilPapers API available for metadata.
    priority: 1

  arxiv_philosophy:
    url: "https://arxiv.org/"
    type: "pdf_download"
    categories: ["cs.AI", "cs.CL", "q-bio.NC"]
    estimated_tokens: 50_000_000
    category: "cross_disciplinary"
    notes: >
      Papers at intersection of AI, philosophy, neuroscience.
      Search terms: consciousness, qualia, phenomenal experience,
      hard problem, neural correlates of consciousness.
    priority: 2

  # ---- NEUROSCIENCE ----
  
  pubmed_central_open:
    url: "https://www.ncbi.nlm.nih.gov/pmc/"
    type: "pdf_download"
    estimated_tokens: 150_000_000
    category: "neuroscience"
    notes: >
      Open access neuroscience papers. Focus on:
      neural correlates of consciousness (NCC),
      global workspace theory empirical work,
      IIT empirical predictions, predictive processing,
      attention and consciousness, blindsight, split-brain.
    priority: 1

  # ---- COGNITIVE SCIENCE ----
  
  cognitive_science_society:
    url: "https://cognitivesciencesociety.org/"
    type: "pdf_download"
    estimated_tokens: 50_000_000
    category: "cognitive_science"
    notes: "Proceedings of CogSci conferences, open access papers."
    priority: 2

  # ---- BOOKS (Public Domain / Open Access) ----
  
  classic_philosophy_texts:
    type: "gutenberg_download"
    estimated_tokens: 100_000_000
    category: "broader_philosophy"
    texts:
      - "Descartes - Meditations on First Philosophy"
      - "Hume - A Treatise of Human Nature"
      - "Kant - Critique of Pure Reason"
      - "James - Principles of Psychology"
      - "Russell - The Analysis of Mind"
      - "Ryle - The Concept of Mind"
    priority: 2

  # ---- CONTEMPORARY (Fair Use / Author-Permitted) ----
  
  author_repositories:
    type: "direct_download"
    estimated_tokens: 100_000_000
    category: "philosophy_of_mind"
    notes: >
      Many philosophers host papers on personal websites.
      Check: David Chalmers' papers page, Daniel Dennett's 
      papers page, Ned Block, Tyler Burge, Frank Jackson,
      Thomas Nagel (selected), Giulio Tononi (IIT papers).
      ALWAYS verify license/permissions.
    priority: 1
```

### Directory Structure

```bash
# Execute in Claude Code:

mkdir -p ~/corpus/{raw,extracted,cleaned,formatted}
mkdir -p ~/corpus/raw/{philosophy_of_mind,neuroscience,broader_philosophy,cognitive_science,cross_disciplinary}
mkdir -p ~/corpus/metadata

cat > ~/corpus/manifest.json << 'EOF'
{
  "created": "2026-02-15",
  "target_tokens": 1000000000,
  "mixing_ratio": {
    "philosophy_of_mind": 0.40,
    "neuroscience": 0.20,
    "broader_philosophy": 0.15,
    "cognitive_science": 0.15,
    "cross_disciplinary": 0.10
  },
  "sources": [],
  "total_documents": 0,
  "total_tokens_estimated": 0,
  "status": "assembling"
}
EOF
```

### Download Scripts

```python
# ~/corpus/scripts/download_sep.py
"""
Download Stanford Encyclopedia of Philosophy articles.
Respects robots.txt and rate limits.
"""

import requests
import time
import json
import os
from pathlib import Path

# Target categories for philosophy of mind
TARGET_ENTRIES = [
    "consciousness", "qualia", "zombies", "chinese-room",
    "functionalism", "physicalism", "dualism", "epiphenomenalism",
    "mental-causation", "multiple-realizability", "supervenience",
    "consciousness-neuroscience", "consciousness-higher",
    "consciousness-representational", "consciousness-temporal",
    "intentionality", "propositional-attitudes", "mental-content",
    "mind-body-problem", "personal-identity", "self-knowledge",
    "perception-problem", "other-minds", "panpsychism",
    "neutral-monism", "emergent-properties", "identity-theory",
    "anomalous-monism", "behaviorism", "eliminative-materialism",
    "folk-psychology", "computational-mind", "connectionism",
    "embodied-cognition", "extended-mind", "free-will",
    "consciousness-animal", "pain", "pleasure", "emotion",
    "imagination", "memory", "attention", "sleep",
    # Broader philosophy entries
    "modality-epistemology", "possible-worlds", "necessity-possibility",
    "a-priori", "analytic-synthetic", "thought-experiments",
    "abduction", "scientific-explanation"
]

OUTPUT_DIR = Path(os.path.expanduser("~/corpus/raw/philosophy_of_mind/sep"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def download_entry(entry_name: str):
    url = f"https://plato.stanford.edu/entries/{entry_name}/"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            filepath = OUTPUT_DIR / f"{entry_name}.html"
            filepath.write_text(resp.text)
            print(f"  Downloaded: {entry_name}")
            return True
    except Exception as e:
        print(f"  Failed: {entry_name} ({e})")
    return False

if __name__ == "__main__":
    print(f"Downloading {len(TARGET_ENTRIES)} SEP entries...")
    success = 0
    for entry in TARGET_ENTRIES:
        if download_entry(entry):
            success += 1
        time.sleep(2)  # Rate limit: 1 request per 2 seconds
    print(f"Done: {success}/{len(TARGET_ENTRIES)} entries downloaded")
```

```python
# ~/corpus/scripts/download_arxiv.py
"""
Download arXiv papers at intersection of AI/philosophy/neuroscience.
Uses arXiv API (rate limit: 1 request per 3 seconds).
"""

import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time
import os
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

OUTPUT_DIR = Path(os.path.expanduser("~/corpus/raw/cross_disciplinary/arxiv"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "http://export.arxiv.org/api/query"

def search_arxiv(query: str, max_results: int = 100):
    params = urllib.parse.urlencode({
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance"
    })
    url = f"{BASE_URL}?{params}"
    
    response = urllib.request.urlopen(url)
    root = ET.fromstring(response.read())
    
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", ns)
    
    results = []
    for entry in entries:
        paper_id = entry.find("atom:id", ns).text.split("/abs/")[-1]
        title = entry.find("atom:title", ns).text.strip()
        summary = entry.find("atom:summary", ns).text.strip()
        results.append({"id": paper_id, "title": title, "summary": summary})
    
    return results

def download_pdf(paper_id: str):
    clean_id = paper_id.replace("/", "_")
    filepath = OUTPUT_DIR / f"{clean_id}.pdf"
    if filepath.exists():
        return True
    
    url = f"https://arxiv.org/pdf/{paper_id}.pdf"
    try:
        urllib.request.urlretrieve(url, filepath)
        return True
    except Exception as e:
        print(f"  Failed: {paper_id} ({e})")
        return False

if __name__ == "__main__":
    all_papers = {}
    for query in QUERIES:
        print(f"Searching: {query}")
        results = search_arxiv(query)
        for paper in results:
            all_papers[paper["id"]] = paper
        time.sleep(3)
    
    print(f"\nFound {len(all_papers)} unique papers. Downloading PDFs...")
    success = 0
    for pid in all_papers:
        if download_pdf(pid):
            success += 1
        time.sleep(1)
    
    print(f"Done: {success}/{len(all_papers)} PDFs downloaded")
    
    # Save metadata
    import json
    meta_path = OUTPUT_DIR / "metadata.json"
    meta_path.write_text(json.dumps(list(all_papers.values()), indent=2))
```

### Validation Checkpoint

```bash
# Run after Phase 1 to verify corpus assembly:

echo "=== Corpus Assembly Status ==="
echo "Philosophy of Mind:"
find ~/corpus/raw/philosophy_of_mind -type f | wc -l
echo "Neuroscience:"
find ~/corpus/raw/neuroscience -type f | wc -l
echo "Broader Philosophy:"
find ~/corpus/raw/broader_philosophy -type f | wc -l
echo "Cognitive Science:"
find ~/corpus/raw/cognitive_science -type f | wc -l
echo "Cross-disciplinary:"
find ~/corpus/raw/cross_disciplinary -type f | wc -l
echo ""
echo "Total files:"
find ~/corpus/raw -type f | wc -l
echo "Total size:"
du -sh ~/corpus/raw
```

**Minimum threshold to proceed**: 5,000+ documents, 2GB+ raw text.

---

## Phase 2: Text Extraction

### Session Goal
Convert all PDFs, HTMLs, EPUBs into clean plaintext preserving argumentative structure.

### Claude Code Instructions

```
You are extracting text from raw philosophical documents.

CRITICAL REQUIREMENTS:
1. Preserve paragraph structure (philosophical arguments span paragraphs)
2. Preserve footnotes inline (philosophers put substantive content in footnotes)
3. Handle French/German diacritics (Meillassoux, Heidegger, etc.)
4. Remove page headers/footers, page numbers, DOIs
5. Preserve section headings as markers
6. One output file per input document
7. Track extraction quality metrics

Input: ~/corpus/raw/
Output: ~/corpus/extracted/
```

### Extraction Scripts

```python
# ~/corpus/scripts/extract_pdf.py
"""
PDF text extraction using PyMuPDF (fitz).
Handles academic PDFs with footnotes, columns, and diacritics.
"""

import fitz  # PyMuPDF
import re
import os
import json
from pathlib import Path

INPUT_DIR = Path(os.path.expanduser("~/corpus/raw"))
OUTPUT_DIR = Path(os.path.expanduser("~/corpus/extracted"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = {"total": 0, "success": 0, "failed": 0, "empty": 0}


def extract_pdf(pdf_path: Path) -> str:
    """Extract text from PDF preserving structure."""
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        METRICS["failed"] += 1
        return ""
    
    full_text = []
    
    for page_num, page in enumerate(doc):
        blocks = page.get_text("blocks")  # Returns positioned text blocks
        
        # Sort blocks by vertical position (top to bottom)
        blocks.sort(key=lambda b: (b[1], b[0]))
        
        page_text = []
        for block in blocks:
            text = block[4].strip()
            
            # Skip page headers/footers (typically very top/bottom)
            if block[1] < 50 or block[1] > page.rect.height - 50:
                # Check if it's a footnote (small font, bottom of page)
                if block[1] > page.rect.height - 200:
                    # Likely footnote — keep but mark
                    if text and not re.match(r'^\d+$', text):
                        page_text.append(f"[FOOTNOTE: {text}]")
                continue
            
            # Skip if just a page number
            if re.match(r'^\d{1,4}$', text):
                continue
            
            # Skip DOIs and URLs on their own line
            if re.match(r'^(doi:|https?://|DOI:)', text):
                continue
            
            if text:
                page_text.append(text)
        
        if page_text:
            full_text.append("\n\n".join(page_text))
    
    doc.close()
    return "\n\n".join(full_text)


def extract_html(html_path: Path) -> str:
    """Extract text from HTML (SEP articles, web pages)."""
    from html.parser import HTMLParser
    
    class TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self.text_parts = []
            self.current_tag = ""
            self.skip_tags = {"script", "style", "nav", "header", "footer"}
            self.in_skip = 0
            
        def handle_starttag(self, tag, attrs):
            self.current_tag = tag
            if tag in self.skip_tags:
                self.in_skip += 1
            if tag in ("h1", "h2", "h3", "h4"):
                self.text_parts.append(f"\n\n## ")
            elif tag == "p":
                self.text_parts.append("\n\n")
            elif tag == "br":
                self.text_parts.append("\n")
                
        def handle_endtag(self, tag):
            if tag in self.skip_tags:
                self.in_skip -= 1
                
        def handle_data(self, data):
            if self.in_skip <= 0:
                self.text_parts.append(data)
    
    html_content = html_path.read_text(encoding="utf-8", errors="replace")
    extractor = TextExtractor()
    extractor.feed(html_content)
    return "".join(extractor.text_parts).strip()


def process_all():
    """Process all files in corpus/raw recursively."""
    
    for category_dir in INPUT_DIR.iterdir():
        if not category_dir.is_dir():
            continue
        
        category = category_dir.name
        out_category = OUTPUT_DIR / category
        out_category.mkdir(parents=True, exist_ok=True)
        
        for filepath in category_dir.rglob("*"):
            if not filepath.is_file():
                continue
            
            METRICS["total"] += 1
            
            # Determine extraction method
            suffix = filepath.suffix.lower()
            if suffix == ".pdf":
                text = extract_pdf(filepath)
            elif suffix in (".html", ".htm"):
                text = extract_html(filepath)
            elif suffix == ".txt":
                text = filepath.read_text(encoding="utf-8", errors="replace")
            else:
                continue
            
            if not text or len(text) < 100:
                METRICS["empty"] += 1
                continue
            
            METRICS["success"] += 1
            
            # Save extracted text
            out_name = filepath.stem + ".txt"
            out_path = out_category / out_name
            # Handle nested directories
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text, encoding="utf-8")
    
    # Save metrics
    metrics_path = OUTPUT_DIR / "extraction_metrics.json"
    metrics_path.write_text(json.dumps(METRICS, indent=2))
    print(f"\nExtraction complete:")
    print(f"  Total files: {METRICS['total']}")
    print(f"  Successful:  {METRICS['success']}")
    print(f"  Empty/short: {METRICS['empty']}")
    print(f"  Failed:      {METRICS['failed']}")


if __name__ == "__main__":
    process_all()
```

### Validation Checkpoint

```bash
# Check extraction quality:

echo "=== Extraction Stats ==="
cat ~/corpus/extracted/extraction_metrics.json

echo ""
echo "=== Sample lengths (first 10 files) ==="
for f in $(find ~/corpus/extracted -name "*.txt" | head -10); do
    echo "$(wc -c < $f) bytes: $(basename $f)"
done

echo ""
echo "=== Diacritics test ==="
grep -rl "Meillassoux\|phénoménologie\|Bewußtsein\|Gemüt" ~/corpus/extracted/ | wc -l
echo "files with non-ASCII philosophical terms preserved"
```

---

## Phase 3: Cleaning & Filtering

### Session Goal
Normalize text, remove duplicates, filter low-quality documents, verify mixing ratios.

### Claude Code Instructions

```
You are cleaning the extracted philosophical corpus.

OPERATIONS:
1. Unicode normalization (NFC)
2. Fix hyphenation artifacts from PDF extraction (re-join split words)
3. Reconnect paragraphs split across pages
4. Remove duplicate documents (MinHash deduplication)
5. Filter out documents with low argument density
6. Verify mixing ratio matches target
7. Produce final cleaned corpus with quality report

Input: ~/corpus/extracted/
Output: ~/corpus/cleaned/
```

### Cleaning Pipeline

```python
# ~/corpus/scripts/clean_corpus.py
"""
Corpus cleaning pipeline for philosophical texts.
"""

import re
import os
import json
import unicodedata
import hashlib
from pathlib import Path
from collections import Counter
from typing import List, Tuple

INPUT_DIR = Path(os.path.expanduser("~/corpus/extracted"))
OUTPUT_DIR = Path(os.path.expanduser("~/corpus/cleaned"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_unicode(text: str) -> str:
    """NFC normalization — canonical decomposition + composition."""
    return unicodedata.normalize("NFC", text)


def fix_hyphenation(text: str) -> str:
    """Rejoin words split by line-end hyphenation in PDFs.
    
    'con-\nsciousness' → 'consciousness'
    But preserve intentional hyphens: 'mind-body' stays 'mind-body'
    """
    # Pattern: word fragment + hyphen + newline + lowercase continuation
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    # Also handle hyphen + space + newline
    text = re.sub(r'(\w)- \n(\w)', r'\1\2', text)
    return text


def reconnect_paragraphs(text: str) -> str:
    """Reconnect paragraphs split across PDF pages.
    
    A paragraph continues if a line ends without sentence-final 
    punctuation and the next line starts with a lowercase letter.
    """
    lines = text.split('\n')
    result = []
    buffer = ""
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if buffer:
                result.append(buffer)
                buffer = ""
            result.append("")
            continue
        
        if buffer:
            # Check if this continues the previous line
            if (stripped[0].islower() or stripped[0] in '("\'') and \
               buffer and buffer[-1] not in '.!?:;':
                buffer += " " + stripped
            else:
                result.append(buffer)
                buffer = stripped
        else:
            buffer = stripped
    
    if buffer:
        result.append(buffer)
    
    return "\n".join(result)


def remove_headers_footers(text: str) -> str:
    """Remove repeated headers/footers (journal names, page numbers)."""
    lines = text.split('\n')
    
    # Count line frequencies — headers/footers repeat across pages
    line_counts = Counter(line.strip() for line in lines if line.strip())
    
    # If a short line appears more than 3 times, it's likely a header/footer
    repeated = {line for line, count in line_counts.items() 
                if count > 3 and len(line) < 100}
    
    filtered = [line for line in lines if line.strip() not in repeated]
    return "\n".join(filtered)


def clean_text(text: str) -> str:
    """Full cleaning pipeline for a single document."""
    text = normalize_unicode(text)
    text = fix_hyphenation(text)
    text = remove_headers_footers(text)
    text = reconnect_paragraphs(text)
    
    # Remove excessive whitespace
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    # Remove common PDF artifacts
    text = re.sub(r'\x0c', '\n', text)  # Form feed
    text = re.sub(r'[\x00-\x08\x0b\x0e-\x1f]', '', text)  # Control chars
    
    return text.strip()


# ============================================================
# DEDUPLICATION (MinHash)
# ============================================================

def compute_shingles(text: str, k: int = 5) -> set:
    """Compute k-character shingles for MinHash."""
    words = text.lower().split()
    if len(words) < k:
        return set()
    return {tuple(words[i:i+k]) for i in range(len(words) - k + 1)}


def minhash_signature(shingles: set, num_hashes: int = 128) -> List[int]:
    """Compute MinHash signature."""
    if not shingles:
        return [float('inf')] * num_hashes
    
    signature = []
    for i in range(num_hashes):
        min_hash = float('inf')
        for shingle in shingles:
            h = hash((i, shingle)) & 0xFFFFFFFF
            if h < min_hash:
                min_hash = h
        signature.append(min_hash)
    return signature


def jaccard_from_minhash(sig_a: List[int], sig_b: List[int]) -> float:
    """Estimate Jaccard similarity from MinHash signatures."""
    assert len(sig_a) == len(sig_b)
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def deduplicate(documents: List[Tuple[Path, str]], 
                threshold: float = 0.8) -> List[Tuple[Path, str]]:
    """Remove near-duplicate documents using MinHash."""
    
    print(f"  Computing MinHash signatures for {len(documents)} documents...")
    signatures = []
    for path, text in documents:
        shingles = compute_shingles(text)
        sig = minhash_signature(shingles)
        signatures.append(sig)
    
    # Find duplicates
    duplicates = set()
    for i in range(len(documents)):
        if i in duplicates:
            continue
        for j in range(i + 1, len(documents)):
            if j in duplicates:
                continue
            similarity = jaccard_from_minhash(signatures[i], signatures[j])
            if similarity > threshold:
                # Keep the longer document
                len_i = len(documents[i][1])
                len_j = len(documents[j][1])
                if len_i >= len_j:
                    duplicates.add(j)
                else:
                    duplicates.add(i)
    
    print(f"  Found {len(duplicates)} duplicates (threshold={threshold})")
    return [doc for i, doc in enumerate(documents) if i not in duplicates]


# ============================================================
# QUALITY FILTERING
# ============================================================

# Philosophical argument indicators — documents with high density
# of these terms are more likely to contain substantive arguments
ARGUMENT_INDICATORS = [
    "therefore", "thus", "hence", "consequently", "it follows",
    "implies", "entails", "because", "since", "given that",
    "suppose", "assume", "consider", "if we accept", "granted that",
    "however", "nevertheless", "on the other hand", "objection",
    "contra", "despite", "although", "whereas",
    "I argue", "I contend", "I maintain", "I claim",
    "the argument", "the objection", "the problem", "the thesis",
    "premise", "conclusion", "inference", "valid", "sound",
    "necessary", "sufficient", "possible", "impossible",
    "conceivable", "metaphysically", "a priori", "a posteriori",
    "supervenes", "reduces to", "identical to", "constituted by"
]


def argument_density(text: str) -> float:
    """Score document by density of philosophical argument indicators.
    
    Returns indicators per 1000 words. Typical values:
    - Philosophy papers: 15-40
    - Neuroscience papers: 5-15
    - Non-philosophical text: 0-5
    """
    words = text.lower().split()
    if len(words) < 100:
        return 0.0
    
    count = sum(text.lower().count(indicator) for indicator in ARGUMENT_INDICATORS)
    return (count / len(words)) * 1000


def filter_quality(documents: List[Tuple[Path, str]],
                   min_length: int = 500,
                   min_argument_density: float = 3.0) -> List[Tuple[Path, str]]:
    """Filter out low-quality documents."""
    
    filtered = []
    stats = {"too_short": 0, "low_density": 0, "passed": 0}
    
    for path, text in documents:
        if len(text) < min_length:
            stats["too_short"] += 1
            continue
        
        density = argument_density(text)
        
        # Use lower threshold for neuroscience (less argumentative language)
        category = path.parts[-2] if len(path.parts) > 1 else ""
        threshold = 1.5 if "neuroscience" in str(path) else min_argument_density
        
        if density < threshold:
            stats["low_density"] += 1
            continue
        
        stats["passed"] += 1
        filtered.append((path, text))
    
    print(f"  Quality filter: {stats}")
    return filtered


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_cleaning_pipeline():
    """Execute the full cleaning pipeline."""
    
    print("=" * 60)
    print("CORPUS CLEANING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load all extracted documents
    print("\n[1/5] Loading extracted documents...")
    documents = []
    for filepath in INPUT_DIR.rglob("*.txt"):
        text = filepath.read_text(encoding="utf-8", errors="replace")
        rel_path = filepath.relative_to(INPUT_DIR)
        documents.append((rel_path, text))
    print(f"  Loaded {len(documents)} documents")
    
    # Step 2: Clean each document
    print("\n[2/5] Cleaning texts...")
    cleaned = [(path, clean_text(text)) for path, text in documents]
    
    # Step 3: Deduplicate
    print("\n[3/5] Deduplicating...")
    deduped = deduplicate(cleaned)
    
    # Step 4: Quality filter
    print("\n[4/5] Quality filtering...")
    filtered = filter_quality(deduped)
    
    # Step 5: Save and compute statistics
    print("\n[5/5] Saving cleaned corpus...")
    
    category_stats = {}
    total_tokens_est = 0
    
    for rel_path, text in filtered:
        out_path = OUTPUT_DIR / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        
        # Track per-category stats
        category = rel_path.parts[0] if rel_path.parts else "unknown"
        if category not in category_stats:
            category_stats[category] = {"docs": 0, "tokens_est": 0}
        
        # Rough token estimate: words * 1.3
        token_est = int(len(text.split()) * 1.3)
        category_stats[category]["docs"] += 1
        category_stats[category]["tokens_est"] += token_est
        total_tokens_est += token_est
    
    # Save report
    report = {
        "total_documents": len(filtered),
        "total_tokens_estimated": total_tokens_est,
        "category_breakdown": category_stats,
        "mixing_ratios_actual": {
            cat: round(stats["tokens_est"] / total_tokens_est, 3)
            for cat, stats in category_stats.items()
        },
        "pipeline_stats": {
            "input_documents": len(documents),
            "after_dedup": len(deduped),
            "after_quality_filter": len(filtered),
            "removed_total": len(documents) - len(filtered)
        }
    }
    
    report_path = OUTPUT_DIR / "cleaning_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    
    print(f"\n{'=' * 60}")
    print(f"CLEANING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Documents: {len(documents)} → {len(filtered)}")
    print(f"Estimated tokens: {total_tokens_est:,}")
    print(f"\nCategory breakdown:")
    for cat, stats in sorted(category_stats.items()):
        ratio = stats["tokens_est"] / total_tokens_est
        print(f"  {cat}: {stats['docs']} docs, "
              f"~{stats['tokens_est']:,} tokens ({ratio:.1%})")
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    run_cleaning_pipeline()
```

### Validation Checkpoint

```bash
cat ~/corpus/cleaned/cleaning_report.json | python3 -m json.tool

# Verify mixing ratios are within tolerance:
# philosophy_of_mind: 35-45% (target 40%)
# neuroscience: 15-25% (target 20%)
# broader_philosophy: 10-20% (target 15%)
# cognitive_science: 10-20% (target 15%)
# cross_disciplinary: 5-15% (target 10%)
```

**Minimum threshold to proceed**: 500M+ estimated tokens, mixing ratios within 5% of targets.

---

## Phase 4: CPT Data Formatting

### Session Goal
Tokenize the cleaned corpus and format for training.

### Claude Code Instructions

```
You are formatting the cleaned corpus for continued pre-training.

FORMAT: JSONL where each line is one document.
TOKENIZER: Use the base model's native tokenizer.
SPLIT: 95% train, 5% validation.
KEY RULE: Each document is a single continuous sequence. 
No chunking — preserve long-range argumentative structure.
If a document exceeds max context length, split at 
paragraph boundaries (never mid-sentence).
```

### Formatting Script

```python
# ~/corpus/scripts/format_cpt.py
"""
Format cleaned corpus into tokenized JSONL for CPT.

Supports: Llama 3.1 (default), Mixtral
"""

import json
import os
import random
from pathlib import Path
from typing import List

# If using Llama tokenizer:
# pip install transformers sentencepiece protobuf
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-70B")

INPUT_DIR = Path(os.path.expanduser("~/corpus/cleaned"))
OUTPUT_DIR = Path(os.path.expanduser("~/corpus/formatted"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
MAX_SEQ_LENGTH = 8192      # Tokens per training sequence
TRAIN_SPLIT = 0.95
SEED = 42


def estimate_tokens(text: str) -> int:
    """Rough token estimate without loading tokenizer.
    For precise count, use the actual tokenizer.
    """
    return int(len(text.split()) * 1.3)


def split_at_paragraph(text: str, max_tokens: int) -> List[str]:
    """Split long documents at paragraph boundaries.
    
    NEVER split mid-sentence — philosophical arguments lose 
    coherence when cut at arbitrary points.
    """
    paragraphs = text.split("\n\n")
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for para in paragraphs:
        para_tokens = estimate_tokens(para)
        
        if current_tokens + para_tokens > max_tokens and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_tokens = para_tokens
        else:
            current_chunk.append(para)
            current_tokens += para_tokens
    
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    
    return chunks


def format_corpus():
    """Format the entire corpus into training JSONL."""
    
    print("Formatting corpus for CPT...")
    
    # Collect all documents
    all_docs = []
    for filepath in INPUT_DIR.rglob("*.txt"):
        if filepath.name.startswith("cleaning"):
            continue
        text = filepath.read_text(encoding="utf-8")
        category = filepath.relative_to(INPUT_DIR).parts[0]
        all_docs.append({
            "text": text,
            "category": category,
            "source": str(filepath.relative_to(INPUT_DIR))
        })
    
    print(f"  Loaded {len(all_docs)} documents")
    
    # Split long documents
    all_sequences = []
    for doc in all_docs:
        tokens = estimate_tokens(doc["text"])
        
        if tokens <= MAX_SEQ_LENGTH:
            all_sequences.append(doc)
        else:
            chunks = split_at_paragraph(doc["text"], MAX_SEQ_LENGTH)
            for i, chunk in enumerate(chunks):
                all_sequences.append({
                    "text": chunk,
                    "category": doc["category"],
                    "source": f"{doc['source']}__chunk_{i}"
                })
    
    print(f"  After splitting: {len(all_sequences)} sequences")
    
    # Shuffle
    random.seed(SEED)
    random.shuffle(all_sequences)
    
    # Split train/val
    split_idx = int(len(all_sequences) * TRAIN_SPLIT)
    train_seqs = all_sequences[:split_idx]
    val_seqs = all_sequences[split_idx:]
    
    # Write JSONL
    train_path = OUTPUT_DIR / "train.jsonl"
    val_path = OUTPUT_DIR / "val.jsonl"
    
    total_train_tokens = 0
    with open(train_path, 'w') as f:
        for seq in train_seqs:
            f.write(json.dumps({"text": seq["text"]}) + "\n")
            total_train_tokens += estimate_tokens(seq["text"])
    
    total_val_tokens = 0
    with open(val_path, 'w') as f:
        for seq in val_seqs:
            f.write(json.dumps({"text": seq["text"]}) + "\n")
            total_val_tokens += estimate_tokens(seq["text"])
    
    # Write dataset card
    card = {
        "name": "philosopher-engine-cpt",
        "description": "Continued pre-training corpus for philosophical reasoning",
        "train_sequences": len(train_seqs),
        "val_sequences": len(val_seqs),
        "train_tokens_estimated": total_train_tokens,
        "val_tokens_estimated": total_val_tokens,
        "max_seq_length": MAX_SEQ_LENGTH,
        "seed": SEED,
        "category_distribution": {}
    }
    
    # Compute category distribution
    for seq in train_seqs:
        cat = seq["category"]
        if cat not in card["category_distribution"]:
            card["category_distribution"][cat] = 0
        card["category_distribution"][cat] += 1
    
    card_path = OUTPUT_DIR / "dataset_card.json"
    card_path.write_text(json.dumps(card, indent=2))
    
    print(f"\n  Train: {len(train_seqs)} sequences, ~{total_train_tokens:,} tokens")
    print(f"  Val:   {len(val_seqs)} sequences, ~{total_val_tokens:,} tokens")
    print(f"  Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    format_corpus()
```

### Validation Checkpoint

```bash
# Verify formatted data:
echo "=== Dataset Card ==="
cat ~/corpus/formatted/dataset_card.json | python3 -m json.tool

echo ""
echo "=== File sizes ==="
ls -lh ~/corpus/formatted/

echo ""
echo "=== Sample sequence (first 500 chars) ==="
head -1 ~/corpus/formatted/train.jsonl | python3 -c "import sys,json; print(json.loads(sys.stdin.read())['text'][:500])"

echo ""
echo "=== Line counts ==="
wc -l ~/corpus/formatted/train.jsonl ~/corpus/formatted/val.jsonl
```

---

## Phase 5: CPT Training

### Session Goal
Run continued pre-training on the base model. This is the most compute-intensive phase.

### Option A: Full CPT (Budget: $10K-$20K)

```bash
# Environment: 4x A100 80GB (e.g., Lambda Labs, RunPod, AWS p4d.24xlarge)

# Install dependencies
pip install torch transformers accelerate deepspeed datasets \
    wandb bitsandbytes flash-attn

# Login
huggingface-cli login
wandb login
```

```python
# ~/training/run_cpt.py
"""
Continued Pre-Training on Llama 3.1 70B.

Uses DeepSpeed ZeRO-3 for distributed training across 4x A100 80GB.
"""

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import os

# ---- Configuration ----
MODEL_NAME = "meta-llama/Meta-Llama-3.1-70B"
OUTPUT_DIR = os.path.expanduser("~/models/philosopher-cpt-70b")
DATA_DIR = os.path.expanduser("~/corpus/formatted")

TRAINING_CONFIG = {
    # Learning rate: LOW to avoid catastrophic forgetting
    "learning_rate": 2e-5,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    
    # Batch size: effective = per_device * gradient_accum * num_gpus
    # 2 * 4 * 4 = 32 sequences per step
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    
    # Duration: 1-3 epochs over the corpus
    "num_train_epochs": 2,
    "max_steps": -1,  # Let epochs determine
    
    # Precision: bf16 for A100s
    "bf16": True,
    "tf32": True,
    
    # Checkpointing
    "save_strategy": "steps",
    "save_steps": 500,
    "save_total_limit": 3,
    
    # Evaluation
    "eval_strategy": "steps",
    "eval_steps": 250,
    
    # Logging
    "logging_steps": 10,
    "report_to": "wandb",
    "run_name": "philosopher-cpt-70b",
    
    # Memory optimization
    "gradient_checkpointing": True,
    "deepspeed": "ds_config.json",
    
    # Output
    "output_dir": OUTPUT_DIR,
}


def main():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    dataset = load_dataset("json", data_files={
        "train": f"{DATA_DIR}/train.jsonl",
        "validation": f"{DATA_DIR}/val.jsonl"
    })
    
    # Tokenize
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=8192,
            padding=False,
            return_special_tokens_mask=True
        )
    
    tokenized = dataset.map(tokenize, batched=True, 
                            remove_columns=["text"],
                            num_proc=8)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    
    # Data collator for causal LM (shifts labels by 1)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training
    args = TrainingArguments(**TRAINING_CONFIG)
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
    )
    
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"\nCPT complete. Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
```

```json
// ~/training/ds_config.json
// DeepSpeed ZeRO-3 config for 4x A100 80GB
{
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu", "pin_memory": true},
        "offload_param": {"device": "none"},
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": 2,
    "wall_clock_breakdown": false
}
```

```bash
# Launch training:
deepspeed --num_gpus=4 ~/training/run_cpt.py
```

### Option B: QLoRA CPT (Budget: $2K-$5K)

```python
# ~/training/run_cpt_qlora.py
"""
QLoRA CPT — fits on a single A100 80GB or 2x A100 40GB.
Trains ~100M adapter parameters instead of all 70B.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset
import os

MODEL_NAME = "meta-llama/Meta-Llama-3.1-70B"
OUTPUT_DIR = os.path.expanduser("~/models/philosopher-cpt-qlora")
DATA_DIR = os.path.expanduser("~/corpus/formatted")

# QLoRA quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# LoRA config — target attention + MLP layers
lora_config = LoraConfig(
    r=64,                    # Rank — higher = more capacity
    lora_alpha=128,          # Scaling factor
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({trainable/total:.2%})")
    
    dataset = load_dataset("json", data_files={
        "train": f"{DATA_DIR}/train.jsonl",
        "validation": f"{DATA_DIR}/val.jsonl"
    })
    
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,      # Higher LR for LoRA adapters
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        gradient_checkpointing=True,
        report_to="wandb",
        run_name="philosopher-cpt-qlora",
    )
    
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=8192,
        packing=True,  # Pack short sequences together for efficiency
    )
    
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"\nQLoRA CPT complete. Adapter saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
```

### Validation Checkpoint

```python
# ~/training/eval_cpt.py
"""
Evaluate CPT model: philosophical perplexity + general capability retention.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import math

def compute_perplexity(model, tokenizer, texts, max_length=2048):
    """Compute perplexity on a list of texts."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", 
                             truncation=True, max_length=max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
            total_tokens += inputs["input_ids"].shape[1]
    
    return math.exp(total_loss / total_tokens)


# Test texts
PHILOSOPHY_TEXTS = [
    "The hard problem of consciousness concerns the question of why and "
    "how physical processes in the brain give rise to subjective experience. "
    "Even if we could explain all the functional and behavioral aspects of "
    "consciousness, there would remain the further question of why these "
    "processes are accompanied by phenomenal experience at all.",
    
    "The zombie argument proceeds from the premise that it is conceivable "
    "that there exists a being physically identical to a conscious being "
    "but lacking any phenomenal experience. If conceivability entails "
    "metaphysical possibility, then such zombie worlds are possible, "
    "which contradicts the physicalist thesis that phenomenal properties "
    "supervene with metaphysical necessity on physical properties.",
]

GENERAL_TEXTS = [
    "The process of photosynthesis converts carbon dioxide and water into "
    "glucose and oxygen using light energy from the sun. This occurs in "
    "the chloroplasts of plant cells.",
    
    "To implement a binary search algorithm, first sort the array. Then "
    "compare the target value to the middle element. If the target is less "
    "than the middle element, search the left half; otherwise search the right.",
]

# Compare base model vs CPT model
# base_ppl = compute_perplexity(base_model, tokenizer, PHILOSOPHY_TEXTS)
# cpt_ppl = compute_perplexity(cpt_model, tokenizer, PHILOSOPHY_TEXTS)
# general_base = compute_perplexity(base_model, tokenizer, GENERAL_TEXTS)
# general_cpt = compute_perplexity(cpt_model, tokenizer, GENERAL_TEXTS)

# PASS CRITERIA:
# 1. Philosophy perplexity should DROP (lower = better)
# 2. General perplexity should NOT increase by more than 10%
```

**Pass criteria for Phase 5:**
- Philosophy perplexity: decreased vs base model
- General perplexity: within 10% of base model
- Training loss: converged (no divergence in final 20% of steps)

---

## Phase 6: SFT Data Generation

### Session Goal
Generate 5K-10K supervised fine-tuning examples using the LLM council + Z3 validation pipeline from the Philosopher Engine architecture.

### Claude Code Instructions

```
You are generating SFT training examples for philosophical reasoning.

METHOD: LLM Council (Claude + GPT-4 + Gemini)
- Each generates draft examples from source passages
- Cross-validate: each critiques the others' outputs
- Z3 validates formal structure of arguments
- Flag disagreements for human review

EXAMPLE TYPES:
Type A — Exposition: Reconstruct argument's logical structure
Type B — Critical Engagement: Strongest objection + response
Type C — Cross-Disciplinary: Connect to consciousness research
Type D — Passage Comprehension: Deep understanding questions

TARGET: 200-400 examples per philosopher, 20-30 philosophers
```

### SFT Generation Pipeline

```python
# ~/training/sft/generate_examples.py
"""
SFT example generation using LLM council + Z3 validation.

This script orchestrates the generation of high-quality training 
examples by having multiple LLMs generate, cross-validate, and 
refine philosophical reasoning examples.
"""

import json
import os
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field

OUTPUT_DIR = Path(os.path.expanduser("~/training/sft/examples"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class SFTExample:
    id: str
    type: str                  # A, B, C, or D
    system_prompt: str
    user_prompt: str
    assistant_response: str
    philosopher: str
    source_passage: str
    difficulty_tier: int       # 1, 2, or 3
    z3_validated: bool = False
    council_agreement: float = 0.0  # 0-1
    human_reviewed: bool = False
    review_status: str = "pending"  # pending, approved, edited, rejected


SYSTEM_PROMPT = """You are a philosophical reasoning assistant specializing \
in philosophy of mind and consciousness studies. You analyze arguments with \
formal rigor, identify argumentation schemes, detect logical fallacies, \
and connect philosophical reasoning to empirical findings in neuroscience \
and cognitive science. When assessing arguments, you distinguish deductive \
from defeasible inference, track which premises are contested, and identify \
exactly where rival theories disagree. You express appropriate uncertainty \
about contested claims and never present philosophical positions as settled \
when genuine debate exists."""


# ---- EXAMPLE TEMPLATES ----

TYPE_A_TEMPLATE = """Below is a passage from {philosopher}'s work. \
Reconstruct the argument's logical structure step by step, identifying:
1. The main thesis
2. Each premise (mark as [STRICT] if deductive or [DEFEASIBLE] if presumptive)
3. The inference pattern (modus ponens, reductio, IBE, analogy, etc.)
4. Any implicit premises
5. The argumentation scheme(s) used (from Walton's taxonomy if applicable)

PASSAGE:
{passage}"""

TYPE_B_TEMPLATE = """Consider the following argument from {philosopher}:

{passage}

Present the strongest objection to this argument, then provide the most \
rigorous defense. Structure your response as:
1. OBJECTION: The strongest counterargument (identify which premise or \
inference step it targets, and whether it undermines, rebuts, or undercuts)
2. DEFENSE: How {philosopher} (or a defender) would respond
3. ASSESSMENT: Which side has the stronger case, and what would settle \
the debate"""

TYPE_C_TEMPLATE = """Connect the following philosophical argument to \
current empirical research in neuroscience or cognitive science:

{passage}

Identify:
1. What empirical predictions (if any) this philosophical position makes
2. Which neuroscientific findings are relevant (NCC studies, IIT predictions, \
GWT evidence, etc.)
3. Whether the empirical evidence supports, undermines, or is neutral toward \
the philosophical claim
4. What further experiments could help adjudicate the philosophical question"""

TYPE_D_TEMPLATE = """Carefully read the following passage from {philosopher}:

{passage}

Answer these questions:
1. What is {philosopher}'s central claim in this passage?
2. What technical terms does {philosopher} use, and how should they be \
understood in context?
3. How does this argument relate to the broader debate about consciousness?
4. What would a {rival_philosopher} say in response to this passage?"""


def generate_type_a(philosopher: str, passage: str, 
                    llm_client=None) -> str:
    """Generate a Type A (Exposition) example."""
    prompt = TYPE_A_TEMPLATE.format(
        philosopher=philosopher, passage=passage
    )
    
    # In production: call LLM API
    # response = llm_client.generate(
    #     system=SYSTEM_PROMPT,
    #     user=prompt,
    #     temperature=0.3
    # )
    # return response
    
    return ""  # Placeholder


def generate_type_b(philosopher: str, passage: str,
                    llm_client=None) -> str:
    """Generate a Type B (Critical Engagement) example."""
    prompt = TYPE_B_TEMPLATE.format(
        philosopher=philosopher, passage=passage
    )
    return ""  # Placeholder


def generate_type_c(philosopher: str, passage: str,
                    llm_client=None) -> str:
    """Generate a Type C (Cross-Disciplinary) example."""
    prompt = TYPE_C_TEMPLATE.format(passage=passage)
    return ""  # Placeholder


def generate_type_d(philosopher: str, passage: str,
                    rival: str, llm_client=None) -> str:
    """Generate a Type D (Comprehension) example."""
    prompt = TYPE_D_TEMPLATE.format(
        philosopher=philosopher, passage=passage,
        rival_philosopher=rival
    )
    return ""  # Placeholder


# ---- LLM COUNCIL ----

def council_generate(philosopher: str, passage: str, 
                     example_type: str) -> Dict:
    """Generate an example using the LLM council.
    
    Process:
    1. All three LLMs generate independently
    2. Each critiques the other two
    3. Compute agreement score
    4. Select best or merge
    """
    
    generator = {
        "A": generate_type_a,
        "B": generate_type_b,
        "C": generate_type_c,
        "D": generate_type_d,
    }[example_type]
    
    # Generate from each LLM
    # claude_response = generator(philosopher, passage, claude_client)
    # gpt4_response = generator(philosopher, passage, gpt4_client)
    # gemini_response = generator(philosopher, passage, gemini_client)
    
    # Cross-validate
    # claude_critique_of_gpt = critique(claude_client, gpt4_response)
    # gpt_critique_of_claude = critique(gpt4_client, claude_response)
    # etc.
    
    # Compute agreement (simplified)
    # agreement = compute_agreement(claude_response, gpt4_response, gemini_response)
    
    return {
        "responses": {},  # All three responses
        "critiques": {},  # Cross-critiques
        "agreement": 0.0,
        "selected": "",   # Best response
    }


# ---- Z3 VALIDATION ----

def z3_validate_example(example: SFTExample) -> bool:
    """Validate the formal structure of an SFT example using Z3.
    
    Checks:
    1. If the response claims an argument is valid, verify with Z3
    2. If it identifies specific schemes, verify premises match template
    3. If it claims inconsistency, verify with Z3 consistency check
    4. If it identifies counterexample, verify countermodel is genuine
    """
    
    # In production: parse the assistant response, extract formal claims,
    # translate to Z3, and verify. Use the GVR loop from the architecture.
    
    return True  # Placeholder


# ---- MAIN GENERATION LOOP ----

# Target philosophers and their primary rivals
PHILOSOPHER_PAIRS = [
    ("Chalmers", "Dennett"),
    ("Dennett", "Chalmers"),
    ("Nagel", "Churchland"),
    ("Jackson", "Lewis"),
    ("Searle", "Dennett"),
    ("Block", "Dennett"),
    ("Tononi", "Searle"),
    ("Koch", "Churchland"),
    ("Dehaene", "Block"),
    ("Levine", "Loar"),
    ("Kim", "Davidson"),
    ("Putnam", "Smart"),
    ("Fodor", "Churchland"),
    ("Carruthers", "Block"),
    ("Rosenthal", "Block"),
    ("Baars", "Tononi"),
    ("Tye", "Block"),
    ("Dretske", "Fodor"),
    ("Clark", "Adams"),
    ("Thompson", "Churchland"),
]


def generate_all_examples():
    """Generate the full SFT dataset."""
    
    all_examples = []
    
    for philosopher, rival in PHILOSOPHER_PAIRS:
        print(f"\nGenerating examples for {philosopher}...")
        
        # Load source passages for this philosopher
        # passages = load_passages(philosopher)
        passages = []  # Placeholder
        
        examples_for_philosopher = 0
        
        for passage in passages:
            # Generate one of each type per passage
            for etype in ["A", "B", "C", "D"]:
                council_result = council_generate(
                    philosopher, passage, etype
                )
                
                example = SFTExample(
                    id=f"{philosopher}_{etype}_{examples_for_philosopher}",
                    type=etype,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt="",  # Set from template
                    assistant_response=council_result.get("selected", ""),
                    philosopher=philosopher,
                    source_passage=passage,
                    difficulty_tier=2,  # Assess per example
                    council_agreement=council_result.get("agreement", 0),
                )
                
                # Z3 validation
                example.z3_validated = z3_validate_example(example)
                
                all_examples.append(example)
                examples_for_philosopher += 1
        
        print(f"  Generated {examples_for_philosopher} examples")
    
    # Save
    output_path = OUTPUT_DIR / "sft_examples_raw.jsonl"
    with open(output_path, 'w') as f:
        for ex in all_examples:
            record = {
                "messages": [
                    {"role": "system", "content": ex.system_prompt},
                    {"role": "user", "content": ex.user_prompt},
                    {"role": "assistant", "content": ex.assistant_response}
                ],
                "metadata": {
                    "id": ex.id,
                    "type": ex.type,
                    "philosopher": ex.philosopher,
                    "tier": ex.difficulty_tier,
                    "z3_validated": ex.z3_validated,
                    "council_agreement": ex.council_agreement,
                    "human_reviewed": ex.human_reviewed,
                }
            }
            f.write(json.dumps(record) + "\n")
    
    print(f"\nTotal examples: {len(all_examples)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    generate_all_examples()
```

### Human Review Interface

```python
# ~/training/sft/review_interface.py
"""
Terminal-based human review interface for SFT examples.

Presents each example and allows: approve, edit, reject, skip.
Tracks review progress and inter-annotator agreement.
"""

import json
import os
import sys
from pathlib import Path

INPUT = Path(os.path.expanduser("~/training/sft/examples/sft_examples_raw.jsonl"))
OUTPUT = Path(os.path.expanduser("~/training/sft/examples/sft_examples_reviewed.jsonl"))

def review_session():
    examples = []
    with open(INPUT) as f:
        for line in f:
            examples.append(json.loads(line))
    
    # Filter for unreviewed
    pending = [e for e in examples if not e["metadata"].get("human_reviewed")]
    
    print(f"Review session: {len(pending)} examples pending")
    print("Commands: [a]pprove  [e]dit  [r]eject  [s]kip  [q]uit")
    print("=" * 60)
    
    reviewed_count = 0
    
    for i, example in enumerate(pending):
        meta = example["metadata"]
        msgs = example["messages"]
        
        print(f"\n--- Example {i+1}/{len(pending)} ---")
        print(f"ID: {meta['id']}  Type: {meta['type']}  "
              f"Philosopher: {meta['philosopher']}")
        print(f"Z3 Validated: {meta['z3_validated']}  "
              f"Council Agreement: {meta['council_agreement']:.2f}")
        print(f"\nUSER: {msgs[1]['content'][:300]}...")
        print(f"\nASSISTANT: {msgs[2]['content'][:500]}...")
        
        while True:
            action = input("\n> ").strip().lower()
            
            if action == 'a':
                meta["human_reviewed"] = True
                meta["review_status"] = "approved"
                reviewed_count += 1
                break
            elif action == 'e':
                print("Enter corrected response (Ctrl+D to finish):")
                new_response = sys.stdin.read()
                msgs[2]["content"] = new_response
                meta["human_reviewed"] = True
                meta["review_status"] = "edited"
                reviewed_count += 1
                break
            elif action == 'r':
                meta["human_reviewed"] = True
                meta["review_status"] = "rejected"
                reviewed_count += 1
                break
            elif action == 's':
                break
            elif action == 'q':
                print(f"\nReviewed {reviewed_count} examples this session.")
                # Save progress
                with open(OUTPUT, 'w') as f:
                    for ex in examples:
                        f.write(json.dumps(ex) + "\n")
                return
            else:
                print("Invalid command. Use: a/e/r/s/q")
    
    # Save all
    with open(OUTPUT, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    
    print(f"\nSession complete. Reviewed {reviewed_count} examples.")


if __name__ == "__main__":
    review_session()
```

---

## Phase 7: SFT Training

### Session Goal
Fine-tune the CPT model on the curated SFT examples.

```python
# ~/training/run_sft.py
"""
Supervised Fine-Tuning with LoRA on the CPT-adapted model.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
import os

# Use the CPT model as base (not the original Llama)
BASE_MODEL = os.path.expanduser("~/models/philosopher-cpt-70b")  
# Or for QLoRA CPT: merge adapters first, then use merged model
OUTPUT_DIR = os.path.expanduser("~/models/philosopher-sft")
SFT_DATA = os.path.expanduser("~/training/sft/examples/sft_examples_reviewed.jsonl")

lora_config = LoraConfig(
    r=32,            # Lower rank than CPT — SFT needs less capacity
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)


def format_chat(example):
    """Format SFT example into chat template."""
    messages = example["messages"]
    # Filter to only approved/edited examples
    if example.get("metadata", {}).get("review_status") == "rejected":
        return {"text": ""}
    
    # Apply chat template
    # text = tokenizer.apply_chat_template(messages, tokenize=False)
    # return {"text": text}
    
    # Simple format if no chat template:
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|{role}|>\n{content}")
    parts.append("<|end|>")
    return {"text": "\n".join(parts)}


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    model = get_peft_model(model, lora_config)
    
    dataset = load_dataset("json", data_files=SFT_DATA, split="train")
    dataset = dataset.map(format_chat)
    dataset = dataset.filter(lambda x: len(x["text"]) > 50)
    
    # Split 95/5
    split = dataset.train_test_split(test_size=0.05, seed=42)
    
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        gradient_checkpointing=True,
        report_to="wandb",
        run_name="philosopher-sft",
    )
    
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=4096,
        packing=False,  # Don't pack SFT examples
    )
    
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"\nSFT complete. Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
```

---

## Phase 8: Evaluation

### Session Goal
Benchmark the fine-tuned model on philosophical reasoning tasks.

```python
# ~/training/eval/run_eval.py
"""
Evaluation suite for the Philosopher Engine model.

Tests:
1. Argument validity detection (binary classification)
2. Scheme identification (multi-class)  
3. Inconsistency detection (binary + localization)
4. Cross-theory comparison (open-ended, LLM-judged)
5. General capability retention (MMLU subset)
"""

import json
from pathlib import Path

EVAL_DIR = Path("~/training/eval").expanduser()
EVAL_DIR.mkdir(exist_ok=True)


# ---- Test 1: Argument Validity ----

VALIDITY_TESTS = [
    {
        "argument": (
            "P1: Zombies are conceivable.\n"
            "P2: Whatever is conceivable is metaphysically possible.\n"
            "P3: If zombie worlds are possible, physicalism is false.\n"
            "C: Physicalism is false."
        ),
        "valid": True,
        "label": "zombie_argument_valid_structure"
    },
    {
        "argument": (
            "P1: The brain produces consciousness.\n"
            "P2: Computers are not brains.\n"
            "C: Computers cannot be conscious."
        ),
        "valid": False,
        "label": "substrate_fallacy_invalid"
    },
    {
        "argument": (
            "P1: If functionalism is true, then any system with the right "
            "functional organization is conscious.\n"
            "P2: The Chinese Room has the right functional organization.\n"
            "P3: The Chinese Room is not conscious.\n"
            "C: Functionalism is false."
        ),
        "valid": True,
        "label": "chinese_room_valid_modus_tollens"
    },
    {
        "argument": (
            "P1: Mary knows all physical facts about color.\n"
            "P2: Mary learns something new when she sees red.\n"
            "C: There are non-physical facts.\n"
            "Note: Implicit premise needed — 'If knowing all physical facts "
            "leaves something unknown, there are non-physical facts.'"
        ),
        "valid": True,  # Valid IF implicit premise is made explicit
        "label": "knowledge_argument_enthymeme"
    },
    {
        "argument": (
            "P1: Consciousness exists.\n"
            "P2: Physics cannot explain consciousness.\n"
            "C: God must have created consciousness."
        ),
        "valid": False,
        "label": "god_of_gaps_invalid"
    },
]


# ---- Test 2: Scheme Identification ----

SCHEME_TESTS = [
    {
        "text": "Just as a bat's sonar experience is alien to us, our "
                "visual experience may be alien to creatures with different "
                "sensory modalities. If we cannot know what it is like to be "
                "a bat, then we cannot fully understand bat consciousness.",
        "expected_scheme": "argument_from_analogy",
    },
    {
        "text": "The best explanation for why physical duplicates would "
                "have identical experiences is that consciousness supervenes "
                "on physical properties. No rival theory explains this "
                "correlation as well.",
        "expected_scheme": "inference_to_best_explanation",
    },
    {
        "text": "If we accept epiphenomenalism, then consciousness has no "
                "causal effects on behavior. But this implies we could never "
                "know we are conscious, which is absurd. Therefore, "
                "epiphenomenalism is false.",
        "expected_scheme": "argument_from_consequences",
    },
]


# ---- Test 3: Inconsistency Detection ----

INCONSISTENCY_TESTS = [
    {
        "claims": [
            "Physicalism holds that all facts are physical facts.",
            "Qualia are non-physical properties of experience.",
            "Qualia exist."
        ],
        "inconsistent": True,
        "conflicting_claims": [0, 1, 2],  # All three jointly inconsistent
    },
    {
        "claims": [
            "Functionalism identifies mental states with functional roles.",
            "Multiple realizability shows the same mental state can be "
            "realized by different physical substrates.",
            "The brain is one substrate that realizes mental states."
        ],
        "inconsistent": False,
    },
    {
        "claims": [
            "IIT holds that consciousness is identical to integrated information.",
            "A thermostat has some integrated information (Φ > 0).",
            "Thermostats are not conscious.",
            "Anything with Φ > 0 is conscious."
        ],
        "inconsistent": True,
        "conflicting_claims": [2, 3],  # Claims 2 and 3 directly conflict
    },
]


def evaluate_model(model, tokenizer):
    """Run full evaluation suite."""
    
    results = {"validity": [], "schemes": [], "inconsistency": []}
    
    # Test 1: Validity
    print("Testing argument validity detection...")
    for test in VALIDITY_TESTS:
        prompt = (f"Is the following argument logically valid? "
                  f"Answer 'VALID' or 'INVALID' and explain briefly.\n\n"
                  f"{test['argument']}")
        
        # response = generate(model, tokenizer, prompt)
        # predicted_valid = "VALID" in response.upper().split('\n')[0]
        # correct = predicted_valid == test["valid"]
        # results["validity"].append({"label": test["label"], "correct": correct})
    
    # Test 2: Scheme identification
    print("Testing scheme identification...")
    for test in SCHEME_TESTS:
        prompt = (f"Identify the argumentation scheme used in this passage. "
                  f"Choose from: argument_from_analogy, "
                  f"inference_to_best_explanation, argument_from_consequences, "
                  f"argument_from_sign, conceivability_to_possibility.\n\n"
                  f"{test['text']}")
        
        # response = generate(model, tokenizer, prompt)
        # correct = test["expected_scheme"] in response.lower()
        # results["schemes"].append({"correct": correct})
    
    # Test 3: Inconsistency detection
    print("Testing inconsistency detection...")
    for test in INCONSISTENCY_TESTS:
        claims_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(test["claims"]))
        prompt = (f"Are the following claims jointly consistent? "
                  f"If inconsistent, identify which claims conflict.\n\n"
                  f"{claims_text}")
        
        # response = generate(model, tokenizer, prompt)
        # predicted_inconsistent = "inconsistent" in response.lower()
        # correct = predicted_inconsistent == test["inconsistent"]
        # results["inconsistency"].append({"correct": correct})
    
    # Summary
    for category, tests in results.items():
        if tests:
            accuracy = sum(t["correct"] for t in tests) / len(tests)
            print(f"  {category}: {accuracy:.1%}")
    
    return results


if __name__ == "__main__":
    # model, tokenizer = load_model(...)
    # results = evaluate_model(model, tokenizer)
    pass
```

### Pass Criteria

```
MINIMUM THRESHOLDS TO DEPLOY:

Argument Validity Detection:   ≥ 85% accuracy
Scheme Identification:         ≥ 70% accuracy  
Inconsistency Detection:       ≥ 75% accuracy
General Capability (MMLU):     ≤ 5% drop from base model

If any threshold is missed → iterate:
1. Analyze failure cases
2. Generate targeted SFT examples for failure patterns
3. Re-run Z3 validation on new examples
4. Retrain (expect 3-5 iteration cycles)
```

---

## Quick Reference: Phase Dependencies

```
Phase 1 (Assembly)     →  nothing (start here)
Phase 2 (Extraction)   →  Phase 1
Phase 3 (Cleaning)     →  Phase 2
Phase 4 (Formatting)   →  Phase 3
Phase 5 (CPT)          →  Phase 4 + GPU access
Phase 6 (SFT Data)     →  Philosopher Engine architecture (separate doc)
Phase 7 (SFT)          →  Phase 5 + Phase 6
Phase 8 (Evaluation)   →  Phase 7

Phase 6 can run IN PARALLEL with Phases 1-5.
Start SFT data generation while corpus is being assembled.
```

---

## Cost Summary

| Phase | Compute | API Costs | Human Time |
|-------|---------|-----------|------------|
| 1-4 (Data) | Minimal (CPU) | $0 | 30-40 hrs |
| 5 (CPT) | $5K-20K GPU | $0 | 5-10 hrs (monitoring) |
| 6 (SFT Data) | Minimal | $1K-3K LLM API | 100-200 hrs (review) |
| 7 (SFT) | $500-2K GPU | $0 | 5-10 hrs |
| 8 (Eval) | Minimal | $200-500 LLM API | 20-30 hrs |
| **Total** | **$5.5K-22K** | **$1.2K-3.5K** | **160-290 hrs** |
