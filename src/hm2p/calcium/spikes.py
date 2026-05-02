"""Stage 4c — calibrated spike inference via CASCADE.

CASCADE (Rupprecht et al. 2021, Nature Neuroscience) outputs spike rates in
calibrated physical units (spikes/s), using pre-trained deep-learning models
matched to the GCaMP indicator and imaging frame rate.

The Voigts & Harnett threshold method (events.py) is retained as a fallback.

Model selection guide:
    hm2p @ ~9.6 Hz   → 'Global_EXC_10Hz_smoothing200ms'  (closest available)
    See cascade2p.utils.get_model_folder() for all available models.
    Full model list: https://github.com/HelmchenLabSoftware/Cascade/blob/master/Pretrained_models/available_models.yaml

Note: cascade2p is conda-only (not on PyPI). Install via:
    conda install -c conda-forge cascade2p

References:
    Rupprecht et al. 2021. "Optimized spiking detection by deep learning."
    Nature Neuroscience 24:1165–1175. doi:10.1038/s41593-021-00895-5
    https://github.com/HelmchenLabSoftware/Cascade
"""

from __future__ import annotations

import re
import warnings

import numpy as np

# Maximum permissible absolute difference (Hz) between the model's nominal
# frame rate (parsed from its name) and the session fps before a warning is
# issued.  At 1.5 Hz tolerance, a 10 Hz model used at 9.6 Hz is silent.
_FPS_MISMATCH_THRESHOLD_HZ: float = 1.5


def _parse_model_fps(model_name: str) -> float | None:
    """Parse the nominal frame rate from a CASCADE model name.

    Looks for a pattern like ``10Hz`` or ``7.5Hz`` anywhere in the name.

    Args:
        model_name: CASCADE pre-trained model name string.

    Returns:
        Nominal frame rate in Hz, or ``None`` if no pattern found.
    """
    # Lookbehind ensures the leading digit is not preceded by another
    # alnum, so e.g. ``v100Hz`` would not parse as 100 Hz (QA 1.13).
    match = re.search(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)Hz", model_name)
    if match:
        return float(match.group(1))
    return None


def predict_spike_rates(
    dff: np.ndarray,
    model_name: str,
    fps: float,
) -> np.ndarray:
    """Infer spike rates from dF/F0 traces using CASCADE.

    Requires cascade2p to be installed (conda-only). Falls back gracefully
    with a clear ImportError if not available.

    A ``UserWarning`` is raised when the model's nominal frame rate (parsed
    from ``model_name``) differs from ``fps`` by more than
    ``_FPS_MISMATCH_THRESHOLD_HZ`` Hz, indicating a potential model–data
    mismatch.

    Args:
        dff: (n_rois, n_frames) float32 — dF/F0 traces.
        model_name: CASCADE pre-trained model name.
        fps: Imaging frame rate (Hz). Used to check model compatibility.

    Returns:
        (n_rois, n_frames) float32 — spike rates in spikes/s.

    Raises:
        ImportError: If cascade2p is not installed.

    References:
        Rupprecht et al. 2021. "Optimized spiking detection by deep learning."
        Nature Neuroscience 24:1165–1175. doi:10.1038/s41593-021-00895-5
    """
    # Check model/fps compatibility before loading cascade2p.
    model_fps = _parse_model_fps(model_name)
    if model_fps is not None:
        delta = abs(model_fps - fps)
        if delta > _FPS_MISMATCH_THRESHOLD_HZ:
            warnings.warn(
                f"CASCADE model '{model_name}' was trained at {model_fps:.1f} Hz "
                f"but session fps is {fps:.2f} Hz (difference {delta:.2f} Hz > "
                f"{_FPS_MISMATCH_THRESHOLD_HZ} Hz threshold). "
                "Spike inference accuracy may be reduced. Select a model whose "
                "rate matches the session fps.",
                UserWarning,
                stacklevel=2,
            )

    try:
        from cascade2p import cascade
    except ImportError as exc:
        raise ImportError(
            "cascade2p is not installed. "
            "Install via conda: conda install -c conda-forge cascade2p\n"
            "See: https://github.com/HelmchenLabSoftware/Cascade"
        ) from exc

    result = cascade.predict(model_name, dff)
    # CASCADE returns a list [spike_prob_array]; extract the array.
    if isinstance(result, (list, tuple)):
        spike_prob = result[0]
    else:
        spike_prob = result
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
