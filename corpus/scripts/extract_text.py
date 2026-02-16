"""
Phase 2: Text Extraction from PDFs, HTMLs, and plain text files.
Handles academic PDFs with footnotes, columns, and diacritics.

CRITICAL REQUIREMENTS:
1. Preserve paragraph structure (philosophical arguments span paragraphs)
2. Preserve footnotes inline (philosophers put substantive content in footnotes)
3. Handle French/German diacritics (Meillassoux, Heidegger, etc.)
4. Remove page headers/footers, page numbers, DOIs
5. Preserve section headings as markers
6. One output file per input document
7. Track extraction quality metrics

Usage:
    python corpus/scripts/extract_text.py
"""

import re
import os
import json
from pathlib import Path
from html.parser import HTMLParser

# Optional: PyMuPDF for PDF extraction
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    print("WARNING: PyMuPDF (fitz) not installed. PDF extraction disabled.")
    print("Install with: pip install PyMuPDF")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_DIR = PROJECT_ROOT / "corpus" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "corpus" / "extracted"

METRICS = {"total": 0, "success": 0, "failed": 0, "empty": 0, "skipped_no_fitz": 0}


def extract_pdf(pdf_path: Path) -> str:
    """Extract text from PDF preserving structure."""
    if not HAS_FITZ:
        METRICS["skipped_no_fitz"] += 1
        return ""

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        METRICS["failed"] += 1
        print(f"  [FAIL] Could not open PDF: {pdf_path.name} ({e})")
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
                    # Likely footnote - keep but mark
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


class TextExtractor(HTMLParser):
    """HTML parser that extracts article text, preserving structure."""

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
            self.text_parts.append("\n\n## ")
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


def extract_html(html_path: Path) -> str:
    """Extract text from HTML (SEP articles, web pages)."""
    html_content = html_path.read_text(encoding="utf-8", errors="replace")
    extractor = TextExtractor()
    extractor.feed(html_content)
    return "".join(extractor.text_parts).strip()


def process_all():
    """Process all files in corpus/raw recursively."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PHASE 2: TEXT EXTRACTION")
    print("=" * 60)
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    for category_dir in sorted(INPUT_DIR.iterdir()):
        if not category_dir.is_dir():
            continue

        category = category_dir.name
        out_category = OUTPUT_DIR / category
        out_category.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing category: {category}")

        for filepath in sorted(category_dir.rglob("*")):
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

            # Save extracted text preserving subdirectory structure
            rel_path = filepath.relative_to(category_dir)
            out_name = rel_path.with_suffix(".txt")
            out_path = out_category / out_name

            # Truncate filename if path would exceed Windows 260 char limit
            if len(str(out_path)) > 250:
                stem = out_path.stem[:80]  # Truncate to 80 chars
                # Add hash suffix to avoid collisions
                import hashlib as _hl
                name_hash = _hl.md5(filepath.name.encode()).hexdigest()[:8]
                out_path = out_path.parent / f"{stem}_{name_hash}.txt"

            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                out_path.write_text(text, encoding="utf-8")
            except (OSError, FileNotFoundError) as e:
                # Last resort: use very short name
                name_hash = __import__('hashlib').md5(
                    filepath.name.encode()).hexdigest()[:12]
                out_path = out_category / f"doc_{name_hash}.txt"
                out_path.write_text(text, encoding="utf-8")

            print(f"  [OK] {filepath.name[:60]}... -> {out_path.name} "
                  f"({len(text):,} chars)")

    # Save metrics
    metrics_path = OUTPUT_DIR / "extraction_metrics.json"
    metrics_path.write_text(json.dumps(METRICS, indent=2))

    print(f"\n{'=' * 60}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total files:        {METRICS['total']}")
    print(f"  Successful:         {METRICS['success']}")
    print(f"  Empty/short (<100): {METRICS['empty']}")
    print(f"  Failed:             {METRICS['failed']}")
    if METRICS['skipped_no_fitz']:
        print(f"  Skipped (no fitz):  {METRICS['skipped_no_fitz']}")
    print(f"\nMetrics saved to: {metrics_path}")


if __name__ == "__main__":
    process_all()
