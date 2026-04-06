"""Literature & Papers — reference papers, biorxiv scans, and research notes.

Displays:
1. Biorxiv scan blog (newest first) from papers/biorxiv-scans/
2. Reference papers catalogue from docs/reference-papers.md
3. Research landscape from docs/research-landscape.md
4. Maze exploration ideas from docs/maze-exploration-ideas.md
5. PDF inventory from papers/ (with subfolder organisation)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parent.parent.parent
_PAPERS_DIR = _REPO / "papers"
_SCANS_DIR = _PAPERS_DIR / "biorxiv-scans"
_DOCS_DIR = _REPO / "docs"

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_date(path: Path) -> str:
    """Extract ISO date from filename or parent folder."""
    for text in (path.stem, path.parent.name):
        m = _DATE_RE.search(text)
        if m:
            return m.group(1)
    return ""


_REVIEWS_DIR = _PAPERS_DIR / "reviews"


def _collect_reviews() -> list[tuple[str, Path]]:
    """Collect paper review/summary markdown files from papers/reviews/."""
    if not _REVIEWS_DIR.exists():
        return []
    entries = []
    for md in _REVIEWS_DIR.glob("*-summary.md"):
        entries.append((md.stem.replace("-summary", ""), md))
    entries.sort(key=lambda x: x[0])
    return entries


def _collect_scans() -> list[tuple[str, Path]]:
    """Collect biorxiv scan markdown files, newest first."""
    if not _SCANS_DIR.exists():
        return []
    entries = []
    for md in _SCANS_DIR.rglob("*.md"):
        entries.append((_extract_date(md), md))
    entries.sort(key=lambda x: (x[0], str(x[1])), reverse=True)
    return entries


def _collect_pdfs() -> dict[str, list[Path]]:
    """Collect PDFs from papers/ grouped by subfolder."""
    if not _PAPERS_DIR.exists():
        return {}
    groups: dict[str, list[Path]] = {}
    for pdf in sorted(_PAPERS_DIR.rglob("*.pdf")):
        rel = pdf.relative_to(_PAPERS_DIR)
        folder = str(rel.parent) if rel.parent != Path(".") else "General"
        groups.setdefault(folder, []).append(pdf)
    return groups


def _read_doc(name: str) -> str | None:
    """Read a docs/ markdown file, return content or None."""
    path = _DOCS_DIR / name
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("Literature & Papers")

tab_scans, tab_reviews, tab_methods, tab_refs, tab_neuropil, tab_maze, tab_pdfs, tab_landscape = st.tabs([
    "Biorxiv Scans",
    "Paper Reviews",
    "Methods",
    "Reference Papers",
    "Neuropil",
    "Maze & Navigation",
    "PDF Library",
    "Pipeline Landscape",
])

# ── Tab 1: Biorxiv scans (blog feed) ─────────────────────────────────────

with tab_scans:
    st.header("Biorxiv Scan Feed")
    st.caption(
        "Periodic literature scans by the RSP science advisor. "
        "Newest entries first."
    )

    scans = _collect_scans()
    if not scans:
        st.info(
            "No biorxiv scan notes yet. "
            "The science advisor agent writes these to `papers/biorxiv-scans/`."
        )
    else:
        st.write(f"{len(scans)} scan{'s' if len(scans) != 1 else ''} found.")
        for date, path in scans:
            label = date if date else path.stem
            content = path.read_text(encoding="utf-8")
            line_count = content.count("\n")
            with st.expander(
                f"{label} — {path.stem.replace('-', ' ')}",
                expanded=len(scans) == 1 or line_count < 40,
            ):
                st.caption(f"File: `papers/biorxiv-scans/{path.name}`")
                st.markdown(content)

# ── Tab 2: Paper reviews ─────────────────────────────────────────────────

with tab_reviews:
    st.header("Paper Reviews")
    st.caption(
        "Summaries of key papers written by the RSP science advisor. "
        "Source PDFs are in `papers/reviews/`."
    )

    reviews = _collect_reviews()
    if not reviews:
        st.info(
            "No paper review summaries yet. "
            "Add PDFs to `papers/reviews/` and the RSP agent will summarise them."
        )
    else:
        st.write(f"{len(reviews)} review{'s' if len(reviews) != 1 else ''} found.")
        for name, path in reviews:
            content = path.read_text(encoding="utf-8")
            # Use first markdown heading as label, fallback to filename
            first_line = content.strip().split("\n")[0].lstrip("# ").strip()
            label = first_line if first_line else name
            with st.expander(label, expanded=len(reviews) == 1):
                st.markdown(content)

# ── Tab 3: Methods documentation ──────────────────────────────────────────

with tab_methods:
    st.header("Methods Documentation")
    st.caption("Technical documentation for pipeline methods.")

    methods_docs = [
        ("hd-computation.md", "Head Direction Computation"),
    ]
    for filename, title in methods_docs:
        content = _read_doc(filename)
        if content:
            with st.expander(title, expanded=True):
                st.markdown(content)
        else:
            st.info(f"No `docs/{filename}` found.")

# ── Tab 4: Reference papers ──────────────────────────────────────────────

with tab_refs:
    st.header("Reference Papers")
    st.caption("Papers used in or relevant to the hm2p pipeline.")

    content = _read_doc("reference-papers.md")
    if content:
        st.markdown(content)
    else:
        st.info("No `docs/reference-papers.md` found.")

# ── Tab 4: Neuropil literature review ───────────────────────────────────

with tab_neuropil:
    st.header("Neuropil Contamination in Two-Photon Calcium Imaging")
    st.caption(
        "Literature review of neuropil signals, correction methods, and "
        "implications for HD tuning analysis in RSP."
    )

    neuropil_content = _read_doc("neuropil-literature-review.md")
    if neuropil_content:
        st.markdown(neuropil_content)
    else:
        st.info("No `docs/neuropil-literature-review.md` found.")

# ── Tab 5: Maze & navigation ideas ──────────────────────────────────────

with tab_maze:
    st.header("Maze Exploration & Navigation")
    st.caption(
        "Literature review and analysis ideas connecting maze behaviour to RSP activity."
    )

    content = _read_doc("maze-exploration-ideas.md")
    if content:
        st.markdown(content)
    else:
        st.info("No `docs/maze-exploration-ideas.md` found.")

# ── Tab 6: PDF library ──────────────────────────────────────────────────

with tab_pdfs:
    st.header("PDF Library")
    st.caption(
        "Papers stored in `papers/`. Organised by subfolder."
    )

    pdf_groups = _collect_pdfs()
    if not pdf_groups:
        st.info("No PDFs found in `papers/`.")
    else:
        total = sum(len(v) for v in pdf_groups.values())
        st.write(f"{total} PDFs across {len(pdf_groups)} folder{'s' if len(pdf_groups) != 1 else ''}.")

        for folder, pdfs in sorted(pdf_groups.items()):
            with st.expander(f"{folder} ({len(pdfs)} papers)", expanded=True):
                for pdf in pdfs:
                    st.markdown(f"- **{pdf.stem}**")

# ── Tab 7: Pipeline landscape ──────────────────────────────────────────

with tab_landscape:
    st.header("Pipeline Landscape")
    st.caption("Survey of related neuroscience pipelines and tools.")

    content = _read_doc("research-landscape.md")
    if content:
        st.markdown(content)
    else:
        st.info("No `docs/research-landscape.md` found.")
