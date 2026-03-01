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

    Per-ROI algorithm:
      1. Estimate noise std from frames below the median (quiet baseline).
      2. Compute Z-score trace: z = dff / noise_std.
      3. Find contiguous runs where z > threshold_z.
      4. Discard events shorter than min_duration_s.
      5. Return binary (0/1) event mask.

    Args:
        dff: (n_rois, n_frames) float32 — dF/F0 traces.
        fps: Imaging frame rate (Hz).
        threshold_z: Z-score threshold for event onset detection.
        min_duration_s: Minimum event duration (seconds) to be counted.

    Returns:
        (n_rois, n_frames) float32 — per-frame event mask (1 during event, 0 outside).
    """
    n_rois, n_frames = dff.shape
    min_frames = max(1, int(min_duration_s * fps))
    events = np.zeros((n_rois, n_frames), dtype=np.float32)

    for i in range(n_rois):
        trace = dff[i].astype(np.float64)

        # Noise std from below-median frames (quiet baseline, robust to transients).
        # Fall back to the trace std if all frames are at/above the median (e.g.
        # a zero-baseline trace with only positive transients).
        median = np.median(trace)
        below = trace[trace < median]
        if len(below) >= 2:
            noise_std = below.std()
        else:
            noise_std = trace.std()
        if noise_std <= 0.0:
            noise_std = np.finfo(np.float64).eps

        z = trace / noise_std
        above = z > threshold_z

        # Find contiguous runs and apply min-duration filter
        mask = np.zeros(n_frames, dtype=np.float32)
        in_event = False
        start = 0
        for j in range(n_frames):
            if above[j] and not in_event:
                in_event = True
                start = j
            elif not above[j] and in_event:
                in_event = False
                if j - start >= min_frames:
                    mask[start:j] = 1.0
        # Handle event running to end of trace
        if in_event and n_frames - start >= min_frames:
            mask[start:n_frames] = 1.0

        events[i] = mask

    return events
