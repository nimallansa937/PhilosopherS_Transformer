#!/usr/bin/env python3
"""
Book Preparation Pipeline: Extract text from local PDFs, EPUBs, and other
formats into corpus/extracted/ for Philosopher Engine training.

Handles:
  1. PDFs via PyMuPDF (fitz) - preserves structure, footnotes, diacritics
  2. EPUBs via ebooklib - extracts chapter text from XHTML spine
  3. TXT/MD pass-through
  4. DJVU skipped (no reliable pure-Python reader)

Sources processed:
  - corpus/raw/**  (existing raw corpus)
  - "Descartes 2/" directory (user's local Spinoza/Descartes collection)
  - descart/ directory deduplication check

Usage:
    python corpus/scripts/prepare_books.py
    python corpus/scripts/prepare_books.py --source "path/to/Descartes 2"
    python corpus/scripts/prepare_books.py --skip-raw  (only process --source)
"""

import io
import json
import os
import re
import sys
import hashlib
import argparse
from pathlib import Path
from html.parser import HTMLParser
from typing import List, Dict, Optional, Tuple

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace"
        )

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    print("WARNING: PyMuPDF not installed. PDF extraction disabled.")
    print("Install: pip install PyMuPDF")

try:
    import ebooklib
    from ebooklib import epub
    HAS_EPUB = True
except ImportError:
    HAS_EPUB = False
    print("WARNING: ebooklib not installed. EPUB extraction disabled.")
    print("Install: pip install ebooklib")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "corpus" / "raw"
EXTRACTED_DIR = PROJECT_ROOT / "corpus" / "extracted"
DESCARTES2_DIR = PROJECT_ROOT / "Descartes 2"
DESCART_DIR = PROJECT_ROOT / "descart"

# Metrics
METRICS = {
    "total_files": 0,
    "pdfs_extracted": 0,
    "epubs_extracted": 0,
    "txt_copied": 0,
    "skipped_small": 0,
    "skipped_duplicate": 0,
    "skipped_crdownload": 0,
    "skipped_unsupported": 0,
    "failed": 0,
    "total_chars": 0,
}


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_extracted_text(text: str) -> str:
    """Clean extracted text: normalize whitespace, remove artifacts."""
    # Normalize Unicode whitespace
    text = text.replace('\xa0', ' ')
    text = text.replace('\u200b', '')  # zero-width space

    # Remove excessive blank lines (more than 2 in a row)
    text = re.sub(r'\n{4,}', '\n\n\n', text)

    # Remove page number artifacts on their own line
    text = re.sub(r'\n\s*\d{1,4}\s*\n', '\n', text)

    # Remove standalone DOIs
    text = re.sub(r'\n\s*(doi|DOI|https?://doi\.org)[^\n]*\n', '\n', text)

    # Normalize dashes
    text = text.replace('\u2013', '-').replace('\u2014', ' - ')

    # Fix common OCR/extraction artifacts
    text = re.sub(r'(?<=[a-z])-\n(?=[a-z])', '', text)  # Dehyphenate
    text = re.sub(r'  +', ' ', text)  # Collapse double spaces

    return text.strip()


# ============================================================
# PDF EXTRACTION (via PyMuPDF)
# ============================================================

def extract_pdf(pdf_path: Path) -> str:
    """Extract text from PDF preserving paragraph structure and footnotes."""
    if not HAS_FITZ:
        return ""

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        print(f"    [FAIL] Could not open PDF: {e}")
        METRICS["failed"] += 1
        return ""

    full_text = []

    for page_num, page in enumerate(doc):
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: (b[1], b[0]))

        page_text = []
        page_height = page.rect.height

        for block in blocks:
            text = block[4].strip()
            y_pos = block[1]

            # Skip empty blocks
            if not text:
                continue

            # Skip page numbers (standalone digits)
            if re.match(r'^\d{1,4}$', text):
                continue

            # Skip standalone DOIs/URLs
            if re.match(r'^(doi:|https?://|DOI:|www\.)', text, re.I):
                continue

            # Header region (top 50px) - skip unless substantial
            if y_pos < 50 and len(text) < 100:
                continue

            # Footer region - check for footnotes vs page artifacts
            if y_pos > page_height - 60:
                if len(text) > 30 and not re.match(r'^\d{1,4}$', text):
                    # Likely a footnote - keep with marker
                    page_text.append(f"[FOOTNOTE: {text}]")
                continue

            page_text.append(text)

        if page_text:
            full_text.append("\n\n".join(page_text))

    doc.close()
    return clean_extracted_text("\n\n".join(full_text))


# ============================================================
# EPUB EXTRACTION (via ebooklib)
# ============================================================

class EPUBTextExtractor(HTMLParser):
    """Extract text from EPUB XHTML content."""

    def __init__(self):
        super().__init__()
        self.text_parts: List[str] = []
        self.skip_tags = {"script", "style", "nav", "head"}
        self.in_skip = 0
        self.current_tag = ""

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        if tag in self.skip_tags:
            self.in_skip += 1
        elif tag in ("h1", "h2", "h3", "h4"):
            self.text_parts.append("\n\n## ")
        elif tag == "p":
            self.text_parts.append("\n\n")
        elif tag == "br":
            self.text_parts.append("\n")
        elif tag == "li":
            self.text_parts.append("\n- ")

    def handle_endtag(self, tag):
        if tag in self.skip_tags:
            self.in_skip -= 1
        if tag in ("h1", "h2", "h3", "h4"):
            self.text_parts.append("\n")

    def handle_data(self, data):
        if self.in_skip <= 0:
            self.text_parts.append(data)


def extract_epub(epub_path: Path) -> str:
    """Extract text from EPUB file, chapter by chapter."""
    if not HAS_EPUB:
        return ""

    try:
        book = epub.read_epub(str(epub_path), options={"ignore_ncx": True})
    except Exception as e:
        print(f"    [FAIL] Could not open EPUB: {e}")
        METRICS["failed"] += 1
        return ""

    full_text = []

    # Process spine items (chapters in reading order)
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            content = item.get_content()
            try:
                html_text = content.decode("utf-8", errors="replace")
            except Exception:
                html_text = str(content)

            extractor = EPUBTextExtractor()
            try:
                extractor.feed(html_text)
            except Exception:
                continue

            chapter_text = "".join(extractor.text_parts).strip()

            # Skip very short items (likely TOC, copyright pages)
            if len(chapter_text) > 100:
                full_text.append(chapter_text)

    return clean_extracted_text("\n\n".join(full_text))


# ============================================================
# FILENAME UTILITIES
# ============================================================

def safe_filename(name: str, max_len: int = 120) -> str:
    """Create a safe, short filename from a potentially long name."""
    # Remove bracketed series info at the start
    name = re.sub(r'^\[.*?\]\s*', '', name)

    # Remove common suffixes
    name = re.sub(r'\s*-\s*libgen\.\w+$', '', name, flags=re.I)
    name = re.sub(r'\s*\(\d{4}.*?\)$', '', name)

    # Clean characters
    name = re.sub(r'[^\w\s\-.]', '', name)
    name = re.sub(r'\s+', '_', name.strip())

    # Truncate
    if len(name) > max_len:
        # Keep meaningful prefix + hash suffix
        name_hash = hashlib.md5(name.encode()).hexdigest()[:8]
        name = name[:max_len - 9] + '_' + name_hash

    return name or "unnamed"


def categorize_file(filepath: Path) -> str:
    """Determine corpus category for a file based on its name/content."""
    name_lower = filepath.stem.lower()

    # Descartes primary texts
    descartes_primary_keywords = [
        'meditations', 'discourse on method', 'principles of philosophy',
        'passions of the soul', 'rules for the direction',
        'descartes_ error', 'descartes_error',
    ]
    for kw in descartes_primary_keywords:
        if kw.replace(' ', '_') in name_lower or kw.replace(' ', '-') in name_lower:
            return 'descartes_primary'

    # Rationalist tradition (Spinoza, Leibniz, Malebranche)
    rationalist_keywords = [
        'spinoza', 'leibniz', 'malebranche', 'rationalist',
        'ethics_spinoza', 'monadology', 'occasionalism',
    ]
    for kw in rationalist_keywords:
        if kw in name_lower:
            return 'rationalist_tradition'

    # Descartes scholarship
    scholarship_keywords = [
        'descartes', 'cartesian', 'cogito', 'meditation',
    ]
    for kw in scholarship_keywords:
        if kw in name_lower:
            return 'descartes_scholarship'

    # Philosophy of mind
    mind_keywords = [
        'consciousness', 'mind', 'qualia', 'zombie', 'phenomenal',
        'physicalism', 'dualism', 'mental', 'brain',
    ]
    for kw in mind_keywords:
        if kw in name_lower:
            return 'philosophy_of_mind'

    # Default: broader philosophy
    return 'broader_philosophy'


# ============================================================
# DEDUPLICATION
# ============================================================

def build_dedup_set(descart_dir: Path) -> set:
    """Build a set of normalized filenames from descart/ for dedup."""
    seen = set()
    if not descart_dir.exists():
        return seen

    for f in descart_dir.rglob("*"):
        if f.is_file():
            # Normalize: lowercase, remove extension, remove common suffixes
            stem = f.stem.lower()
            stem = re.sub(r'\s*-\s*libgen\.\w+$', '', stem)
            stem = re.sub(r'[_\-\s]+', ' ', stem).strip()
            # Use first 40 chars as dedup key
            seen.add(stem[:40])

    return seen


def is_duplicate(filepath: Path, dedup_set: set) -> bool:
    """Check if file is likely a duplicate of something in descart/."""
    stem = filepath.stem.lower()
    stem = re.sub(r'\s*-\s*libgen\.\w+$', '', stem)
    stem = re.sub(r'[_\-\s]+', ' ', stem).strip()
    key = stem[:40]
    return key in dedup_set


# ============================================================
# MAIN EXTRACTION PIPELINE
# ============================================================

def process_file(filepath: Path,
                 output_dir: Path,
                 category: str,
                 dedup_set: set) -> Optional[str]:
    """Process a single file: extract text and save to output_dir/category/."""
    METRICS["total_files"] += 1
    suffix = filepath.suffix.lower()

    # Skip incomplete downloads
    if suffix == '.crdownload':
        METRICS["skipped_crdownload"] += 1
        return None

    # Skip unsupported formats
    if suffix == '.djvu':
        METRICS["skipped_unsupported"] += 1
        print(f"    [SKIP] DJVU not supported: {filepath.name[:60]}")
        return None

    # Skip unknown formats
    if suffix not in ('.pdf', '.epub', '.txt', '.text', '.md', '.html', '.htm'):
        METRICS["skipped_unsupported"] += 1
        return None

    # Dedup check
    if is_duplicate(filepath, dedup_set):
        METRICS["skipped_duplicate"] += 1
        print(f"    [DUP]  Already in descart/: {filepath.name[:60]}")
        return None

    # Extract text
    if suffix == '.pdf':
        text = extract_pdf(filepath)
        if text:
            METRICS["pdfs_extracted"] += 1
    elif suffix == '.epub':
        text = extract_epub(filepath)
        if text:
            METRICS["epubs_extracted"] += 1
    elif suffix in ('.txt', '.text', '.md'):
        try:
            text = filepath.read_text(encoding="utf-8", errors="replace")
            text = clean_extracted_text(text)
            METRICS["txt_copied"] += 1
        except Exception as e:
            print(f"    [FAIL] Could not read: {e}")
            METRICS["failed"] += 1
            return None
    elif suffix in ('.html', '.htm'):
        try:
            from corpus.scripts.extract_text import extract_html
            text = extract_html(filepath)
            METRICS["txt_copied"] += 1
        except Exception:
            # Inline HTML extraction
            html_content = filepath.read_text(encoding="utf-8", errors="replace")
            extractor = EPUBTextExtractor()  # Works for general HTML too
            extractor.feed(html_content)
            text = "".join(extractor.text_parts).strip()
            METRICS["txt_copied"] += 1
    else:
        return None

    # Check minimum size
    if not text or len(text) < 100:
        METRICS["skipped_small"] += 1
        return None

    # Build output path
    out_dir = output_dir / category
    out_dir.mkdir(parents=True, exist_ok=True)

    out_name = safe_filename(filepath.stem) + ".txt"
    out_path = out_dir / out_name

    # Handle Windows long path
    if len(str(out_path)) > 250:
        name_hash = hashlib.md5(filepath.name.encode()).hexdigest()[:12]
        out_path = out_dir / f"doc_{name_hash}.txt"

    # Avoid overwriting existing files
    if out_path.exists():
        name_hash = hashlib.md5(filepath.name.encode()).hexdigest()[:8]
        out_path = out_dir / f"{out_path.stem}_{name_hash}.txt"

    try:
        out_path.write_text(text, encoding="utf-8")
    except (OSError, FileNotFoundError) as e:
        name_hash = hashlib.md5(filepath.name.encode()).hexdigest()[:12]
        out_path = out_dir / f"doc_{name_hash}.txt"
        try:
            out_path.write_text(text, encoding="utf-8")
        except Exception as e2:
            print(f"    [FAIL] Could not write: {e2}")
            METRICS["failed"] += 1
            return None

    METRICS["total_chars"] += len(text)
    print(f"    [OK]   {filepath.name[:55]:55s} -> {category}/{out_path.name} "
          f"({len(text):,} chars)")
    return str(out_path)


def process_directory(source_dir: Path,
                      output_dir: Path,
                      dedup_set: set,
                      label: str = "",
                      auto_categorize: bool = True):
    """Process all files in a directory tree."""
    if not source_dir.exists():
        print(f"  [WARN] Directory not found: {source_dir}")
        return

    print(f"\n{'=' * 60}")
    print(f"Processing: {label or source_dir}")
    print(f"{'=' * 60}")

    files = sorted([f for f in source_dir.rglob("*") if f.is_file()])
    print(f"  Found {len(files)} files")

    for filepath in files:
        if auto_categorize:
            # Try to detect category from parent dir first
            rel_parts = filepath.relative_to(source_dir).parts
            if len(rel_parts) > 1:
                parent_dir = rel_parts[0].lower()
                known_cats = {
                    'descartes_primary', 'rationalist_tradition',
                    'philosophy_of_mind', 'contemporary_responses',
                    'formal_logic', 'descartes_scholarship',
                    'broader_philosophy', 'cognitive_science',
                    'cross_disciplinary', 'neuroscience',
                    'gutenberg', 'archive_org',
                }
                if parent_dir in known_cats:
                    category = parent_dir
                else:
                    category = categorize_file(filepath)
            else:
                category = categorize_file(filepath)
        else:
            category = categorize_file(filepath)

        process_file(filepath, output_dir, category, dedup_set)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare books for Philosopher Engine training")
    parser.add_argument(
        "--source", type=str, default=None,
        help="Additional source directory (e.g. 'Descartes 2/')")
    parser.add_argument(
        "--skip-raw", action="store_true",
        help="Skip processing corpus/raw/ (only process --source)")
    parser.add_argument(
        "--output", type=str, default=str(EXTRACTED_DIR),
        help="Output directory for extracted text")
    parser.add_argument(
        "--no-dedup", action="store_true",
        help="Disable deduplication against descart/")

    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BOOK PREPARATION PIPELINE")
    print("=" * 60)
    print(f"  Output:    {output_dir}")
    print(f"  PyMuPDF:   {'YES' if HAS_FITZ else 'NO'}")
    print(f"  ebooklib:  {'YES' if HAS_EPUB else 'NO'}")

    # Build dedup set
    if not args.no_dedup:
        dedup_set = build_dedup_set(DESCART_DIR)
        print(f"  Dedup set: {len(dedup_set)} files from descart/")
    else:
        dedup_set = set()

    # 1. Process corpus/raw/
    if not args.skip_raw:
        process_directory(
            RAW_DIR, output_dir, dedup_set,
            label="corpus/raw/ (existing raw corpus)")

    # 2. Process Descartes 2/ (or --source)
    source_dir = Path(args.source) if args.source else DESCARTES2_DIR
    if source_dir.exists():
        process_directory(
            source_dir, output_dir, dedup_set,
            label=f"{source_dir.name} (local book collection)",
            auto_categorize=True)
    else:
        print(f"\n  [INFO] Source directory not found: {source_dir}")

    # Save metrics
    metrics_path = output_dir / "preparation_metrics.json"
    metrics_path.write_text(json.dumps(METRICS, indent=2))

    # Summary
    print(f"\n{'=' * 60}")
    print("PREPARATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total files scanned:  {METRICS['total_files']}")
    print(f"  PDFs extracted:       {METRICS['pdfs_extracted']}")
    print(f"  EPUBs extracted:      {METRICS['epubs_extracted']}")
    print(f"  TXT/HTML copied:      {METRICS['txt_copied']}")
    print(f"  Skipped (small):      {METRICS['skipped_small']}")
    print(f"  Skipped (duplicate):  {METRICS['skipped_duplicate']}")
    print(f"  Skipped (incomplete): {METRICS['skipped_crdownload']}")
    print(f"  Skipped (format):     {METRICS['skipped_unsupported']}")
    print(f"  Failed:               {METRICS['failed']}")
    print(f"  Total chars output:   {METRICS['total_chars']:,}")
    print(f"\n  Metrics: {metrics_path}")

    # Show output category breakdown
    print(f"\n  Output categories:")
    for cat_dir in sorted(output_dir.iterdir()):
        if cat_dir.is_dir():
            count = sum(1 for f in cat_dir.rglob("*.txt"))
            print(f"    {cat_dir.name}: {count} files")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
