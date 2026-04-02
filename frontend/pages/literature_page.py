"""Literature Review page — displays markdown notes from docs/papers/ as a blog-style feed.

The RSP science advisor agent writes biorxiv scan notes to docs/papers/ in date-based
subfolders (e.g. docs/papers/2026-04/biorxiv-scan-2026-04-02.md). This page scans
that directory recursively and renders each file newest first.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import streamlit as st

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PAPERS_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "papers"

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _extract_date(path: Path) -> str:
    """Extract a sortable date string from filename or parent folder name.

    Parameters
    ----------
    path:
        Path to a markdown file under docs/papers/.

    Returns
    -------
    str
        ISO date string (YYYY-MM-DD) if found, otherwise the empty string.
    """
    # Try filename first, then parent directory name
    for text in (path.stem, path.parent.name):
        m = _DATE_RE.search(text)
        if m:
            return m.group(1)
    return ""


def _collect_entries() -> list[tuple[str, Path]]:
    """Return all markdown files sorted newest first.

    Returns
    -------
    list of (date_str, path)
        Sorted descending by date string, then by filename.
    """
    if not _PAPERS_DIR.exists():
        return []

    entries: list[tuple[str, Path]] = []
    for md_path in sorted(_PAPERS_DIR.rglob("*.md")):
        date = _extract_date(md_path)
        entries.append((date, md_path))

    # Sort: primary key = date descending, secondary = path descending (newest filename)
    entries.sort(key=lambda x: (x[0], str(x[1])), reverse=True)
    return entries


def _friendly_label(date: str, path: Path) -> str:
    """Build a human-readable label for use in expander headers.

    Parameters
    ----------
    date:
        ISO date string or empty string.
    path:
        Path to the file.

    Returns
    -------
    str
        Label combining date and filename stem.
    """
    stem = path.stem.replace("-", " ").replace("_", " ")
    if date:
        return f"{date} — {stem}"
    return stem


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("Literature Review")
st.caption(
    "Markdown notes from periodic bioRxiv and literature scans, rendered newest first. "
    "Files are read from `docs/papers/` in the repository."
)

entries = _collect_entries()

if not entries:
    st.info(
        "No literature notes found in `docs/papers/`. "
        "The science advisor agent will populate this directory when it runs a scan."
    )
    st.stop()

st.write(f"{len(entries)} note{'s' if len(entries) != 1 else ''} found.")
st.markdown("---")

for date, path in entries:
    label = _friendly_label(date, path)
    content = path.read_text(encoding="utf-8")

    # Count lines to decide whether to collapse by default
    line_count = content.count("\n")
    collapse_by_default = line_count > 40

    with st.expander(label, expanded=not collapse_by_default):
        if date:
            st.caption(f"Date: {date}  |  File: `{path.relative_to(_PAPERS_DIR.parent.parent)}`")
        else:
            st.caption(f"File: `{path.relative_to(_PAPERS_DIR.parent.parent)}`")
        st.markdown(content)
