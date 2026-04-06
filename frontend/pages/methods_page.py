"""Methods — documentation for computational methods used in the pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

_DOCS_DIR = Path(__file__).resolve().parent.parent.parent / "docs"

st.title("Methods")
st.caption("Technical documentation for computational methods in the hm2p pipeline.")


def _read_doc(name: str) -> str | None:
    path = _DOCS_DIR / name
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


# Auto-discover method docs (files starting with specific prefixes or all in docs/)
METHOD_DOCS = [
    ("hd-computation.md", "Head Direction & Posture"),
    ("dlc-retraining.md", "DLC Retraining Workflow"),
    ("navigraph-evaluation.md", "NaviGraph Evaluation"),
    ("stats-strategy.md", "Statistical Strategy"),
]

for filename, title in METHOD_DOCS:
    content = _read_doc(filename)
    if content:
        with st.expander(title, expanded=False):
            st.markdown(content)
