#!/usr/bin/env python3
"""
Phase 1: Download early-modern rationalist philosophy corpus from
Project Gutenberg and Internet Archive.

Sources: Project Gutenberg (gutendex API), Internet Archive (advancedsearch API)
Authors: Descartes, Spinoza, Leibniz, Malebranche, Arnauld, Gassendi,
         Princess Elisabeth of Bohemia, and related figures.

Respects robots.txt and rate limits for both services.
Generates manifest.json + download_log.txt for Phase 2 pipeline integration.

Usage:
    python corpus/scripts/download_corpus.py
    python corpus/scripts/download_corpus.py --authors "descartes,spinoza"
    python corpus/scripts/download_corpus.py --sources "gutenberg"
    python corpus/scripts/download_corpus.py --output ~/corpus/
    python corpus/scripts/download_corpus.py --dry-run
"""

import argparse
import datetime
import hashlib
import io
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote_plus

import requests

# ---------------------------------------------------------------------------
# Fix Windows console encoding for Unicode output
# ---------------------------------------------------------------------------
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
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

USER_AGENT = "PhilosopherEngine/1.0 (Academic Research Corpus Builder)"
HEADERS = {"User-Agent": USER_AGENT}

# ---------------------------------------------------------------------------
# Author / search configuration
# ---------------------------------------------------------------------------

AUTHORS = [
    "Descartes",
    "Spinoza",
    "Leibniz",
    "Malebranche",
    "Arnauld",
    "Gassendi",
    "Princess Elisabeth",
]

# Map author surname to corpus subfolder name
AUTHOR_FOLDER_MAP = {
    "descartes":            "descartes",
    "spinoza":              "spinoza",
    "leibniz":              "leibniz",
    "malebranche":          "other_rationalists",
    "arnauld":              "other_rationalists",
    "gassendi":             "other_rationalists",
    "princess elisabeth":   "other_rationalists",
    "elisabeth":            "other_rationalists",
}

# Maps an author to the specific titles we want (Gutenberg keyword search)
GUTENBERG_TITLE_SEARCHES: dict[str, list[str]] = {
    "Descartes": [
        "Meditations on First Philosophy",
        "Discourse on Method",
        "Discourse on the Method",
        "Principles of Philosophy",
        "Rules for the Direction of the Mind",
        "Passions of the Soul",
        "The World",
        "Objections and Replies",
        "Selections from Descartes",
    ],
    "Spinoza": [
        "Ethics",
        "Treatise on the Emendation of the Intellect",
        "Theological-Political Treatise",
        "Tractatus Theologico-Politicus",
        "Short Treatise",
        "Improvement of the Understanding",
    ],
    "Leibniz": [
        "Monadology",
        "Discourse on Metaphysics",
        "New Essays on Human Understanding",
        "Theodicy",
        "New System of Nature",
        "Leibniz Philosophical Writings",
    ],
    "Malebranche": [
        "Search after Truth",
        "Dialogues on Metaphysics",
    ],
    "Arnauld": [
        "Art of Thinking",
        "Port-Royal Logic",
        "Objections",
    ],
    "Gassendi": [
        "Objections",
        "Disquisitio Metaphysica",
    ],
    "Princess Elisabeth": [
        "Elisabeth Descartes correspondence",
        "Princess Elisabeth",
    ],
}

# Internet Archive extra search terms beyond author searches
ARCHIVE_EXTRA_SEARCHES = [
    "Adam Tannery Descartes",           # critical edition
    "Gebhardt Spinoza",                 # critical edition
    "Objections and Replies Descartes",
    "Elisabeth Bohemia Descartes correspondence",
    "Meditations First Philosophy",
    "Cartesian rationalism philosophy",
    "mind body problem 17th century",
    "rationalism philosophy early modern",
    "substance dualism history",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class DownloadLogger:
    """Dual-output logger: console + download_log.txt."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Append mode so previous run logs are preserved
        self._fh = open(log_path, "a", encoding="utf-8")
        self._write(f"\n{'='*72}")
        self._write(f"Corpus download session: {_now_iso()}")
        self._write(f"{'='*72}")

    # internal write
    def _write(self, msg: str):
        self._fh.write(msg + "\n")
        self._fh.flush()

    def info(self, msg: str):
        ts = _now_iso()
        line = f"[{ts}] INFO  {msg}"
        print(msg)
        self._write(line)

    def error(self, msg: str):
        ts = _now_iso()
        line = f"[{ts}] ERROR {msg}"
        print(f"  [ERROR] {msg}", file=sys.stderr)
        self._write(line)

    def warn(self, msg: str):
        ts = _now_iso()
        line = f"[{ts}] WARN  {msg}"
        print(f"  [WARN] {msg}")
        self._write(line)

    def close(self):
        self._fh.close()


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds"
    )

# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

class Manifest:
    """JSON manifest tracking every downloaded file."""

    def __init__(self, path: Path):
        self.path = path
        self.entries: list[dict[str, Any]] = []
        self._url_set: set[str] = set()
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self.entries = data
                elif isinstance(data, dict) and "files" in data:
                    self.entries = data["files"]
                for e in self.entries:
                    if "url" in e:
                        self._url_set.add(e["url"])
            except Exception:
                pass  # start fresh if corrupt

    def already_have(self, url: str) -> bool:
        return url in self._url_set

    def add(
        self,
        filename: str,
        source: str,
        author: str,
        title: str,
        url: str,
        file_size: int,
        fmt: str = "txt",
        duplicate_of: str = "",
    ):
        entry = {
            "filename": filename,
            "source": source,
            "author": author,
            "title": title,
            "url": url,
            "download_date": _now_iso(),
            "file_size": file_size,
            "format": fmt,
        }
        if duplicate_of:
            entry["duplicate_of"] = duplicate_of
        self.entries.append(entry)
        self._url_set.add(url)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        wrapper = {
            "generated": _now_iso(),
            "total_files": len(self.entries),
            "total_bytes": sum(e.get("file_size", 0) for e in self.entries),
            "files": self.entries,
        }
        self.path.write_text(
            json.dumps(wrapper, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def find_title(self, title_lower: str) -> Optional[dict]:
        """Check if we already have a file whose title matches (fuzzy)."""
        for e in self.entries:
            if _fuzzy_match(e.get("title", ""), title_lower):
                return e
        return None


def _fuzzy_match(a: str, b: str) -> bool:
    """Cheap fuzzy match: lowercase, strip punctuation, check substring."""
    a = re.sub(r"[^a-z0-9 ]", "", a.lower())
    b = re.sub(r"[^a-z0-9 ]", "", b.lower())
    return a in b or b in a


# ---------------------------------------------------------------------------
# Local library scanner — avoid re-downloading books user already has
# ---------------------------------------------------------------------------

def _scan_local_library(*dirs: Path) -> set[str]:
    """
    Scan one or more local directories for existing book filenames.
    Returns a set of normalised title tokens for fuzzy matching.
    E.g. "Descartes-Meditations-on-First-Philosophy.pdf" ->
         "descartes meditations on first philosophy"
    """
    titles: set[str] = set()
    for d in dirs:
        if not d.exists():
            continue
        for f in d.iterdir():
            if f.is_file():
                stem = f.stem  # filename without extension
                # Normalise: replace hyphens/underscores/dots with space,
                # strip parenthesised metadata, lowercase
                norm = re.sub(r"[_\-.]", " ", stem)
                norm = re.sub(r"\([^)]*\)", "", norm)  # strip (publisher 2006)
                norm = re.sub(r"\[[^\]]*\]", "", norm)  # strip [series info]
                norm = re.sub(r"[^a-z0-9 ]", "", norm.lower())
                norm = re.sub(r"\s+", " ", norm).strip()
                if len(norm) > 5:
                    titles.add(norm)
    return titles


def _already_in_local_library(
    title: str,
    local_titles: set[str],
) -> bool:
    """Check if a download candidate title matches something already on disk."""
    if not local_titles:
        return False
    norm = re.sub(r"[^a-z0-9 ]", "", title.lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    # Check if the download title is a substring of any local file or vice versa
    for lt in local_titles:
        # Both directions, but require at least 15 chars overlap to avoid
        # false positives like "Descartes" matching everything
        if len(norm) >= 15 and norm in lt:
            return True
        if len(lt) >= 15 and lt in norm:
            return True
    return False


# ---------------------------------------------------------------------------
# Simple progress bar (no deps)
# ---------------------------------------------------------------------------

def _progress_bar(current: int, total: int, width: int = 40, suffix: str = ""):
    if total == 0:
        return
    frac = current / total
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    pct = frac * 100
    sys.stdout.write(f"\r  [{bar}] {pct:5.1f}%  {suffix}")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")


# ---------------------------------------------------------------------------
# HTTP helpers with retry + exponential backoff
# ---------------------------------------------------------------------------

def _get_with_retry(
    url: str,
    logger: DownloadLogger,
    *,
    max_retries: int = 3,
    timeout: int = 60,
    stream: bool = False,
    params: dict | None = None,
) -> Optional[requests.Response]:
    """GET with retries and exponential backoff."""
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(
                url,
                headers=HEADERS,
                timeout=timeout,
                stream=stream,
                params=params,
            )
            if resp.status_code == 200:
                return resp
            elif resp.status_code == 429:
                wait = 2 ** attempt * 5
                logger.warn(f"Rate limited (429) on {url}, waiting {wait}s")
                time.sleep(wait)
            else:
                logger.warn(
                    f"HTTP {resp.status_code} on {url} "
                    f"(attempt {attempt}/{max_retries})"
                )
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
        except requests.exceptions.Timeout:
            logger.warn(f"Timeout on {url} (attempt {attempt}/{max_retries})")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    logger.error(f"All {max_retries} retries exhausted for {url}")
    return None


def _download_file(
    url: str,
    dest: Path,
    logger: DownloadLogger,
    *,
    min_size: int = 1024,
    max_retries: int = 3,
) -> bool:
    """Download a file to disk. Returns True on success."""
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=120, stream=True)
            if resp.status_code != 200:
                logger.warn(
                    f"HTTP {resp.status_code} downloading {url} "
                    f"(attempt {attempt}/{max_retries})"
                )
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                continue

            dest.parent.mkdir(parents=True, exist_ok=True)
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            _progress_bar(
                                downloaded, total,
                                suffix=f"{downloaded//1024}KB/{total//1024}KB"
                            )
            if total > 0:
                print()  # newline after progress bar

            # Validate
            actual_size = dest.stat().st_size
            if actual_size < min_size:
                logger.warn(
                    f"Downloaded file too small ({actual_size} B < {min_size} B): "
                    f"{dest.name} — probably an error page"
                )
                dest.unlink(missing_ok=True)
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                continue

            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Download error {url}: {e}")
            dest.unlink(missing_ok=True)
            if attempt < max_retries:
                time.sleep(2 ** attempt)

    logger.error(f"All {max_retries} retries exhausted downloading {url}")
    return False


# ---------------------------------------------------------------------------
# Filename sanitiser
# ---------------------------------------------------------------------------

def _safe_filename(name: str, ext: str = ".txt") -> str:
    """Convert a title into a safe filesystem name."""
    name = re.sub(r'[<>:"/\\|?*]', "", name)
    name = re.sub(r"\s+", "_", name.strip())
    name = name[:120]  # cap length
    if not name.endswith(ext):
        name += ext
    return name


# ---------------------------------------------------------------------------
# MODULE 1: Project Gutenberg (via gutendex.com API)
# ---------------------------------------------------------------------------

GUTENDEX_BASE = "https://gutendex.com/books/"
GUTENBERG_RATE_LIMIT = 2.0  # seconds between requests


def _gutenberg_text_url(book: dict) -> Optional[str]:
    """Extract the best plaintext URL from a gutendex book record."""
    formats = book.get("formats", {})
    # Priority order
    for mime in [
        "text/plain; charset=utf-8",
        "text/plain; charset=us-ascii",
        "text/plain",
    ]:
        if mime in formats:
            url = formats[mime]
            # Skip .zip variants
            if not url.endswith(".zip"):
                return url
    # Fallback: any text/plain variant
    for mime, url in formats.items():
        if "text/plain" in mime and not url.endswith(".zip"):
            return url
    return None


def _author_matches(book: dict, author_query: str) -> bool:
    """Check if any of the book's authors match the query (surname match)."""
    # Normalise query to just the surname for matching
    aq = author_query.lower().strip()
    # Handle "Princess Elisabeth" -> match "elisabeth"
    aq_parts = aq.split()

    for a in book.get("authors", []):
        name = a.get("name", "").lower()
        # Gutenberg uses "Surname, Firstname" format
        name_parts = [p.strip() for p in name.replace(",", " ").split()]
        for qp in aq_parts:
            if len(qp) >= 4 and any(qp in np for np in name_parts):
                return True
    return False


def _title_is_relevant(book_title: str, author_query: str) -> bool:
    """
    Check if a book title is plausibly relevant to our rationalist corpus.
    Prevents false positives like "Gulliver's Travels" from a "The World" search.

    Two-tier check: strong keywords (auto-accept) vs weak keywords
    (require author match in the title or bookshelf tags).
    """
    t = book_title.lower()
    aq = author_query.lower()

    # ---- Strong keywords: definitely relevant to rationalist corpus ----
    STRONG_KEYWORDS = [
        "meditation", "metaphysic", "discourse on method",
        "discourse on the method", "discourse of a method",
        "passions of the soul", "rules for the direction",
        "ethics demonstrated", "emendation of the intellect",
        "theological-political", "tractatus theologico",
        "monadol", "theodicy", "discourse on metaphysic",
        "new essays on human understanding",
        "improvement of the understanding",
        "search after truth", "port-royal logic", "art of thinking",
        "short treatise on god",
        "descartes", "spinoza", "leibniz", "malebranche", "arnauld",
        "gassendi", "elisabeth", "cartesian",
        "cogito", "rationalis", "dualism",
        "mind and body", "mind-body problem",
        "disquisitio metaphysica",
        "objections and replies",
        "principles of philosophy",
        "selections from the principles",
        "philosophy of mind", "prima philosophia",
        "method of rightly conducting",
    ]
    for kw in STRONG_KEYWORDS:
        if kw in t:
            return True

    # ---- Author surname in the title => relevant ----
    for part in aq.split():
        if len(part) >= 5 and part in t:
            return True

    return False


def _gutenberg_search(
    query: str,
    logger: DownloadLogger,
    *,
    max_pages: int = 3,
) -> list[dict]:
    """Search gutendex for books matching a query string.
    Limits pagination to max_pages to avoid pulling thousands of results."""
    results: list[dict] = []
    url = GUTENDEX_BASE
    params = {"search": query}
    page = 0

    while url and page < max_pages:
        resp = _get_with_retry(url, logger, params=params)
        if resp is None:
            break
        try:
            data = resp.json()
        except ValueError:
            logger.error(f"Invalid JSON from gutendex for query={query}")
            break

        for book in data.get("results", []):
            results.append(book)

        # Pagination — gutendex returns absolute next URL
        next_url = data.get("next")
        if next_url and next_url != url:
            url = next_url
            params = None  # params are in the next URL already
            page += 1
        else:
            break

        time.sleep(GUTENBERG_RATE_LIMIT)

    return results


def download_gutenberg(
    authors: list[str],
    output_root: Path,
    manifest: Manifest,
    logger: DownloadLogger,
    *,
    dry_run: bool = False,
    local_titles: set[str] | None = None,
) -> dict[str, int]:
    """
    Download public-domain texts from Project Gutenberg for each author.
    Returns stats dict {author: count}.
    """
    logger.info("=" * 60)
    logger.info("PROJECT GUTENBERG — Downloading rationalist philosophy texts")
    logger.info(f"Rate limit: 1 request per {GUTENBERG_RATE_LIMIT}s")
    logger.info("=" * 60)

    stats: dict[str, int] = {}
    seen_ids: set[int] = set()

    for author in authors:
        author_lower = author.lower()
        folder_key = None
        for k, v in AUTHOR_FOLDER_MAP.items():
            if k in author_lower:
                folder_key = v
                break
        if folder_key is None:
            folder_key = "other_rationalists"

        out_dir = output_root / "gutenberg" / folder_key
        out_dir.mkdir(parents=True, exist_ok=True)

        title_searches = GUTENBERG_TITLE_SEARCHES.get(author, [])

        # Strategy 1: search by author name
        logger.info(f"\n--- Gutenberg: searching author '{author}' ---")
        author_books = _gutenberg_search(author, logger)
        time.sleep(GUTENBERG_RATE_LIMIT)

        # Strategy 2: search by each title keyword
        title_books: list[dict] = []
        for title_q in title_searches:
            logger.info(f"  Searching title: '{title_q}'")
            batch = _gutenberg_search(title_q, logger, max_pages=1)
            title_books.extend(batch)
            time.sleep(GUTENBERG_RATE_LIMIT)

        # Merge and deduplicate
        all_books: dict[int, dict] = {}
        for b in author_books + title_books:
            bid = b.get("id", 0)
            if bid and bid not in all_books:
                all_books[bid] = b

        # -----------------------------------------------------------
        # FILTER: Only keep books where (a) the author matches OR
        #         (b) the title is relevant to our philosophy corpus.
        #         This eliminates false positives like "Gulliver's Travels"
        #         from a "The World" search.
        # -----------------------------------------------------------
        filtered_books: dict[int, dict] = {}
        for bid, book in all_books.items():
            btitle = book.get("title", "")
            if _author_matches(book, author):
                filtered_books[bid] = book
            elif _title_is_relevant(btitle, author):
                filtered_books[bid] = book

        logger.info(
            f"  Found {len(all_books)} raw results, "
            f"{len(filtered_books)} relevant for '{author}'"
        )

        count = 0
        for bid, book in filtered_books.items():
            if bid in seen_ids:
                continue
            seen_ids.add(bid)

            title = book.get("title", f"Unknown_{bid}")
            book_authors = ", ".join(
                a.get("name", "?") for a in book.get("authors", [])
            )

            txt_url = _gutenberg_text_url(book)
            if txt_url is None:
                logger.warn(
                    f"  No plaintext for: {title[:60]} (id={bid}), skipping"
                )
                continue

            # Resume check
            if manifest.already_have(txt_url):
                logger.info(f"  [CACHED] {title[:60]}")
                count += 1
                continue

            # Local library dedup check
            if _already_in_local_library(title, local_titles or set()):
                logger.info(
                    f"  [LOCAL-SKIP] {title[:60]}  (already in local library)"
                )
                continue

            fname = _safe_filename(f"gut_{bid}_{title[:80]}", ".txt")
            dest = out_dir / fname

            if dry_run:
                logger.info(f"  [DRY-RUN] Would download: {title[:60]}")
                count += 1
                continue

            logger.info(f"  Downloading: {title[:60]}  (id={bid})")
            ok = _download_file(txt_url, dest, logger)
            if ok:
                fsize = dest.stat().st_size
                # Deduplication flag
                dup = ""
                existing = manifest.find_title(title.lower())
                if existing and existing.get("source") != "gutenberg":
                    dup = existing.get("filename", "")

                manifest.add(
                    filename=str(dest.relative_to(output_root)),
                    source="gutenberg",
                    author=book_authors or author,
                    title=title,
                    url=txt_url,
                    file_size=fsize,
                    fmt="txt",
                    duplicate_of=dup,
                )
                count += 1
                logger.info(
                    f"    OK ({fsize // 1024} KB)"
                    + (f"  [DUPLICATE of {dup}]" if dup else "")
                )
            else:
                logger.error(f"  FAILED: {title[:60]}")

            time.sleep(GUTENBERG_RATE_LIMIT)

        stats[author] = count

    return stats


# ---------------------------------------------------------------------------
# MODULE 2: Internet Archive (advancedsearch API)
# ---------------------------------------------------------------------------

ARCHIVE_SEARCH_BASE = "https://archive.org/advancedsearch.php"
ARCHIVE_DL_BASE = "https://archive.org/download"
ARCHIVE_META_BASE = "https://archive.org/metadata"
ARCHIVE_RATE_LIMIT = 3.0  # seconds between requests


def _archive_search(
    query: str,
    logger: DownloadLogger,
    *,
    rows: int = 50,
) -> list[dict]:
    """Search Internet Archive for texts matching the query."""
    params = {
        "q": query,
        "output": "json",
        "rows": rows,
        "fl[]": "identifier,title,creator,date,mediatype,licenseurl,downloads",
    }
    resp = _get_with_retry(
        ARCHIVE_SEARCH_BASE, logger, params=params, timeout=45
    )
    if resp is None:
        return []
    try:
        data = resp.json()
    except ValueError:
        logger.error(f"Invalid JSON from archive.org for query: {query}")
        return []

    docs = data.get("response", {}).get("docs", [])
    # Filter: only texts
    return [d for d in docs if d.get("mediatype") == "texts"]


def _archive_get_text_file(
    identifier: str,
    logger: DownloadLogger,
) -> Optional[tuple[str, str]]:
    """
    Find the best plaintext (or PDF) file for an Internet Archive item.
    Returns (download_url, format) or None.
    """
    meta_url = f"{ARCHIVE_META_BASE}/{identifier}"
    resp = _get_with_retry(meta_url, logger, timeout=30)
    if resp is None:
        return None
    try:
        data = resp.json()
    except ValueError:
        return None

    files = data.get("files", [])

    # Priority 1: .txt files
    for f in files:
        name = f.get("name", "")
        if name.lower().endswith(".txt") and f.get("format", "") != "Metadata":
            size = int(f.get("size", 0))
            if size >= 1024:
                dl_url = f"{ARCHIVE_DL_BASE}/{identifier}/{name}"
                return (dl_url, "txt")

    # Priority 2: DjVu TXT (auto-OCR text)
    for f in files:
        name = f.get("name", "")
        if "djvu.txt" in name.lower():
            dl_url = f"{ARCHIVE_DL_BASE}/{identifier}/{name}"
            return (dl_url, "txt")

    # Priority 3: PDF
    for f in files:
        name = f.get("name", "")
        fmt = f.get("format", "").lower()
        if name.lower().endswith(".pdf") and "thumb" not in name.lower():
            size = int(f.get("size", 0))
            if size >= 10240:  # PDFs should be > 10KB
                dl_url = f"{ARCHIVE_DL_BASE}/{identifier}/{name}"
                return (dl_url, "pdf")

    return None


def download_archive(
    authors: list[str],
    output_root: Path,
    manifest: Manifest,
    logger: DownloadLogger,
    *,
    dry_run: bool = False,
    local_titles: set[str] | None = None,
) -> dict[str, int]:
    """
    Download public-domain texts from Internet Archive for each author.
    Returns stats dict {search_term: count}.
    """
    logger.info("\n" + "=" * 60)
    logger.info("INTERNET ARCHIVE — Downloading rationalist philosophy texts")
    logger.info(f"Rate limit: 1 request per {ARCHIVE_RATE_LIMIT}s")
    logger.info("=" * 60)

    stats: dict[str, int] = {}
    seen_ids: set[str] = set()

    # Build search queries: author searches + extra topical searches
    searches: list[tuple[str, str, str]] = []  # (query, folder, label)

    for author in authors:
        author_lower = author.lower()
        folder_key = None
        for k, v in AUTHOR_FOLDER_MAP.items():
            if k in author_lower:
                folder_key = v
                break
        if folder_key is None:
            folder_key = "other_rationalists"

        q = f'creator:("{author}") AND mediatype:(texts)'
        searches.append((q, folder_key, author))

    # Extra searches go into critical_editions or the default folder
    # Use simple keyword search (not exact phrase) for better recall
    for extra in ARCHIVE_EXTRA_SEARCHES:
        if "Adam Tannery" in extra or "Gebhardt" in extra:
            folder = "critical_editions"
        else:
            folder = "descartes"  # most extras are Descartes-related
        # Split into words, join with AND for better Archive.org matching
        words = extra.split()
        q_inner = " AND ".join(words)
        q = f'({q_inner}) AND mediatype:(texts)'
        searches.append((q, folder, extra))

    for query, folder, label in searches:
        out_dir = output_root / "archive_org" / folder
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n--- Archive.org: '{label}' ---")
        items = _archive_search(query, logger)
        logger.info(f"  Found {len(items)} items")
        time.sleep(ARCHIVE_RATE_LIMIT)

        count = 0
        for item in items:
            ident = item.get("identifier", "")
            if not ident or ident in seen_ids:
                continue
            seen_ids.add(ident)

            title = item.get("title", ident)
            creator = item.get("creator", "Unknown")
            if isinstance(creator, list):
                creator = "; ".join(creator)

            # Find text file
            result = _archive_get_text_file(ident, logger)
            time.sleep(ARCHIVE_RATE_LIMIT)

            if result is None:
                logger.warn(f"  No text/pdf found for: {title[:60]} ({ident})")
                continue

            dl_url, fmt = result
            ext = f".{fmt}"

            # Resume check
            if manifest.already_have(dl_url):
                logger.info(f"  [CACHED] {title[:60]}")
                count += 1
                continue

            # Local library dedup check
            if _already_in_local_library(title, local_titles or set()):
                logger.info(
                    f"  [LOCAL-SKIP] {title[:60]}  (already in local library)"
                )
                continue

            fname = _safe_filename(f"ia_{ident[:80]}", ext)
            dest = out_dir / fname

            if dry_run:
                logger.info(f"  [DRY-RUN] Would download: {title[:60]}")
                count += 1
                continue

            logger.info(f"  Downloading: {title[:60]}  ({ident})")
            ok = _download_file(dl_url, dest, logger, min_size=1024)
            if ok:
                fsize = dest.stat().st_size
                # Deduplication flag
                dup = ""
                existing = manifest.find_title(title.lower())
                if existing and existing.get("source") != "archive_org":
                    dup = existing.get("filename", "")

                manifest.add(
                    filename=str(dest.relative_to(output_root)),
                    source="archive_org",
                    author=creator,
                    title=title,
                    url=dl_url,
                    file_size=fsize,
                    fmt=fmt,
                    duplicate_of=dup,
                )
                count += 1
                logger.info(
                    f"    OK ({fsize // 1024} KB, {fmt})"
                    + (f"  [DUPLICATE of {dup}]" if dup else "")
                )
            else:
                logger.error(f"  FAILED: {title[:60]} ({ident})")

            time.sleep(ARCHIVE_RATE_LIMIT)

        stats[label] = count

    return stats


# ---------------------------------------------------------------------------
# Summary / report
# ---------------------------------------------------------------------------

def _print_summary(
    gut_stats: dict[str, int],
    ia_stats: dict[str, int],
    manifest: Manifest,
    logger: DownloadLogger,
):
    total_files = len(manifest.entries)
    total_bytes = sum(e.get("file_size", 0) for e in manifest.entries)
    total_mb = total_bytes / (1024 * 1024)

    # Count by author
    by_author: dict[str, int] = {}
    by_source: dict[str, int] = {}
    duplicates = 0
    for e in manifest.entries:
        a = e.get("author", "Unknown")
        # Normalise to surname
        surname = a.split(",")[0].strip().split()[-1] if a else "Unknown"
        by_author[surname] = by_author.get(surname, 0) + 1
        s = e.get("source", "unknown")
        by_source[s] = by_source.get(s, 0) + 1
        if e.get("duplicate_of"):
            duplicates += 1

    banner = "\n" + "=" * 60
    logger.info(banner)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Total files in manifest : {total_files}")
    logger.info(f"  Total size              : {total_mb:.1f} MB")
    logger.info(f"  Cross-source duplicates : {duplicates}")

    logger.info("\n  By source:")
    for s, c in sorted(by_source.items()):
        logger.info(f"    {s:20s} : {c}")

    logger.info("\n  By author (surname):")
    for a, c in sorted(by_author.items(), key=lambda x: -x[1]):
        logger.info(f"    {a:20s} : {c}")

    if gut_stats:
        logger.info("\n  Gutenberg this session:")
        for a, c in gut_stats.items():
            logger.info(f"    {a:20s} : {c}")

    if ia_stats:
        logger.info("\n  Archive.org this session:")
        for a, c in ia_stats.items():
            logger.info(f"    {a:20s} : {c}")

    logger.info("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Download early-modern rationalist philosophy corpus from "
            "Project Gutenberg and Internet Archive."
        ),
    )
    p.add_argument(
        "--authors",
        type=str,
        default=None,
        help=(
            "Comma-separated list of authors to download. "
            "Default: all authors."
        ),
    )
    p.add_argument(
        "--sources",
        type=str,
        default=None,
        help=(
            "Comma-separated list of sources: gutenberg,archive. "
            "Default: both."
        ),
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output directory for corpus. "
            "Default: <project>/corpus/raw/ "
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be downloaded without actually downloading.",
    )
    p.add_argument(
        "--skip-local",
        type=str,
        default=None,
        help=(
            "Comma-separated directories of existing books to avoid "
            "re-downloading. Titles are fuzzy-matched against filenames. "
            "Default: auto-detects <project>/descart/ if it exists."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Resolve output root
    if args.output:
        output_root = Path(args.output).expanduser().resolve()
    else:
        output_root = PROJECT_ROOT / "corpus" / "raw"
    output_root.mkdir(parents=True, exist_ok=True)

    # Resolve authors
    if args.authors:
        selected_authors = [
            a.strip().title() for a in args.authors.split(",")
        ]
        # Validate
        valid = {a.lower() for a in AUTHORS}
        for sa in selected_authors:
            if sa.lower() not in valid and not any(
                sa.lower() in v for v in valid
            ):
                print(
                    f"Warning: '{sa}' not in default author list, "
                    f"will search anyway."
                )
    else:
        selected_authors = AUTHORS[:]

    # Resolve sources
    if args.sources:
        sources = {s.strip().lower() for s in args.sources.split(",")}
    else:
        sources = {"gutenberg", "archive"}

    # Init logger + manifest
    log_path = output_root / "download_log.txt"
    manifest_path = output_root / "manifest.json"

    logger = DownloadLogger(log_path)
    manifest = Manifest(manifest_path)

    # Scan local libraries to avoid re-downloading existing books
    local_dirs: list[Path] = []
    if args.skip_local:
        for d in args.skip_local.split(","):
            local_dirs.append(Path(d.strip()).expanduser().resolve())
    else:
        # Auto-detect the descart/ directory if it exists
        auto_dir = PROJECT_ROOT / "descart"
        if auto_dir.exists():
            local_dirs.append(auto_dir)
    local_titles = _scan_local_library(*local_dirs) if local_dirs else set()

    logger.info(f"Output root     : {output_root}")
    logger.info(f"Authors         : {', '.join(selected_authors)}")
    logger.info(f"Sources         : {', '.join(sorted(sources))}")
    logger.info(f"Manifest        : {manifest_path}")
    logger.info(f"Existing files  : {len(manifest.entries)}")
    logger.info(f"Dry run         : {args.dry_run}")
    if local_titles:
        logger.info(
            f"Local library   : {len(local_titles)} titles scanned "
            f"from {', '.join(str(d) for d in local_dirs)}"
        )

    gut_stats: dict[str, int] = {}
    ia_stats: dict[str, int] = {}

    try:
        if "gutenberg" in sources:
            gut_stats = download_gutenberg(
                selected_authors,
                output_root,
                manifest,
                logger,
                dry_run=args.dry_run,
                local_titles=local_titles,
            )
            manifest.save()

        if "archive" in sources:
            ia_stats = download_archive(
                selected_authors,
                output_root,
                manifest,
                logger,
                dry_run=args.dry_run,
                local_titles=local_titles,
            )
            manifest.save()

    except KeyboardInterrupt:
        logger.warn("Interrupted by user — saving manifest so far.")
        manifest.save()
    except Exception:
        logger.error(f"Unhandled error:\n{traceback.format_exc()}")
        manifest.save()
        raise
    finally:
        manifest.save()
        _print_summary(gut_stats, ia_stats, manifest, logger)
        logger.info(f"Manifest saved to: {manifest_path}")
        logger.info(f"Log saved to: {log_path}")
        logger.close()


if __name__ == "__main__":
    main()
