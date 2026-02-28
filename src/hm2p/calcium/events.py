"""Stage 4 — Voigts & Harnett calcium event detection (fallback).

Retained as a fallback and for comparison with CASCADE spike inference.
The V&H algorithm threshold-detects transient events in dF/F0 traces and
assigns a per-frame event probability.

Reference: Voigts & Harnett 2020, Neuron.
"""

from __future__ import annotations

import numpy as np


def detect_events(
    dff: np.ndarray,
    fps: float,
    threshold_z: float = 2.0,
    min_duration_s: float = 0.1,
) -> np.ndarray:
    """Detect calcium transient events using Voigts & Harnett threshold method.

    Args:
        dff: (n_rois, n_frames) float32 — dF/F0 traces.
        fps: Imaging frame rate (Hz).
        threshold_z: Z-score threshold for event onset detection.
        min_duration_s: Minimum event duration (seconds) to be counted.

    Returns:
        (n_rois, n_frames) float32 — per-frame event probability in [0, 1].
    """
    raise NotImplementedError
