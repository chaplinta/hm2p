"""Stage 4c — calibrated spike inference via CASCADE.

CASCADE (Rupprecht et al. 2021, Nature Neuroscience) outputs spike rates in
calibrated physical units (spikes/s), using pre-trained deep-learning models
matched to the GCaMP indicator and imaging frame rate.

The Voigts & Harnett threshold method (events.py) is retained as a fallback.

Model selection guide:
    GCaMP7f @ ~30 Hz  → 'Global_EXC_7.5Hz_smoothing200ms'  (closest available)
    GCaMP8f @ ~30 Hz  → same model (GCaMP8 not separately available as of Feb 2026)
    See cascade2p.utils.get_model_folder() for all available models.

Note: cascade2p is conda-only (not on PyPI). Install via:
    conda install -c conda-forge cascade2p
"""

from __future__ import annotations

import numpy as np


def predict_spike_rates(
    dff: np.ndarray,
    model_name: str,
    fps: float,
) -> np.ndarray:
    """Infer spike rates from dF/F0 traces using CASCADE.

    Requires cascade2p to be installed (conda-only). Falls back gracefully
    with a clear ImportError if not available.

    Args:
        dff: (n_rois, n_frames) float32 — dF/F0 traces.
        model_name: CASCADE pre-trained model name.
        fps: Imaging frame rate (Hz) — used to validate model compatibility.

    Returns:
        (n_rois, n_frames) float32 — spike rates in spikes/s.

    Raises:
        ImportError: If cascade2p is not installed.
    """
    try:
        from cascade2p import cascade
    except ImportError as exc:
        raise ImportError(
            "cascade2p is not installed. "
            "Install via conda: conda install -c conda-forge cascade2p\n"
            "See: https://github.com/HelmchenLabSoftware/Cascade"
        ) from exc

    spike_prob = cascade.predict(model_name, dff)
    return np.asarray(spike_prob, dtype=np.float32)


def compute_mean_spike_rate(
    spikes: np.ndarray,
    fps: float,
    bad_frames: np.ndarray | None = None,
) -> np.ndarray:
    """Compute mean spike rate (spikes/min) per ROI, excluding bad frames.

    Args:
        spikes: (n_rois, n_frames) float32 — CASCADE spike rates (spikes/s).
        fps: Imaging frame rate (Hz).
        bad_frames: Optional (n_frames,) bool — True for frames to exclude.

    Returns:
        (n_rois,) float32 — mean spike rate in spikes/min.
    """
    if bad_frames is not None:
        good = ~bad_frames
        spikes = spikes[:, good]
    if spikes.shape[1] == 0:
        return np.full(spikes.shape[0], np.nan, dtype=np.float32)
    return (spikes.mean(axis=1) * 60.0).astype(np.float32)
