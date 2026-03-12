"""Shared colour constants for plots and frontend.

Consistent with the legacy pipeline colour scheme (old-pipeline/utils/plot.py).
All RGBA tuples use 0–1 floats. Named colours use matplotlib CSS4 names.
"""

from __future__ import annotations

# ── Cell types ──────────────────────────────────────────────────────────
COLOR_PENK = (0, 0, 1, 1)          # blue
COLOR_NONPENK = (1, 0, 0, 1)       # red

# Hex equivalents for Plotly / Streamlit (no alpha channel)
HEX_PENK = "#0000FF"
HEX_NONPENK = "#FF0000"

# ── ROI types ───────────────────────────────────────────────────────────
COLOR_SOMA = "turquoise"
COLOR_DEND = "darkorchid"

# ── Tuning curves ──────────────────────────────────────────────────────
COLOUR_TUNE_CURVE = "deepskyblue"

# ── Light conditions ───────────────────────────────────────────────────
COLOUR_LIGHT = "orange"
COLOUR_DARK = "black"

# ── Movement state ─────────────────────────────────────────────────────
COLOUR_ACTIVE = "limegreen"
COLOUR_INACTIVE = "saddlebrown"

# ── Neural events ──────────────────────────────────────────────────────
COLOUR_EVENTS = "midnightblue"

# ── Decoder halves ─────────────────────────────────────────────────────
COLOR_DEC1 = (0.2, 0.6, 1.0, 1)
COLOR_DEC2 = (1.0, 0.4, 0.4, 1)

HEX_DEC1 = "#3399FF"
HEX_DEC2 = "#FF6666"

# ── Convenience mapping ────────────────────────────────────────────────
CELLTYPE_COLOR = {
    "penk": COLOR_PENK,
    "nonpenk": COLOR_NONPENK,
}

CELLTYPE_HEX = {
    "penk": HEX_PENK,
    "nonpenk": HEX_NONPENK,
}

CELLTYPE_LABEL = {
    "penk": "Penk+",
    "nonpenk": "CamKII+",
}

LIGHT_HEX = {
    "light": "#FFA500",   # orange
    "dark": "#000000",    # black
}

ROI_TYPE_HEX = {
    "soma": "#40E0D0",    # turquoise
    "dendrite": "#9932CC", # darkorchid
}
