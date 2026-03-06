"""Stage 1 — run Suite2p on raw TIFF stacks.

Wraps ``suite2p.run_s2p()`` to produce the standard plane0/ output directory
containing F.npy, Fneu.npy, iscell.npy, stat.npy, and ops.npy.

Suite2p 1.0+ API: ``run_s2p(db=..., settings=...)``.

Suite2p is an optional dependency (GPU recommended). Install via:
    pip install suite2p
    # or conda install -c conda-forge suite2p
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def default_settings(
    fps: float = 29.97,
    classifier_path: Path | None = None,
) -> dict[str, Any]:
    """Return default Suite2p settings for hm2p single-plane GCaMP imaging.

    Starts from Suite2p's built-in defaults and applies hm2p-specific
    parameters matching the legacy pipeline (``sourcedata/trackers/suite2p/ops_default.npy``):
    - Single-plane recordings (~30 Hz)
    - GCaMP7f indicators (tau=1.0)
    - diameter=12, nonrigid registration, block_size=(128,128)
    - Custom soma classifier from ``sourcedata/trackers/suite2p/classifier_soma.npy``

    Args:
        fps: Imaging frame rate in Hz.
        classifier_path: Path to a custom Suite2p classifier .npy file.
            If None, looks for ``sourcedata/trackers/suite2p/classifier_soma.npy``
            relative to the repo root. Falls back to Suite2p's builtin classifier.

    Returns:
        Dict of Suite2p settings suitable for passing to ``suite2p.run_s2p(settings=...)``.
    """
    try:
        from suite2p import default_settings as s2p_defaults
    except ImportError:
        return {"fs": fps, "tau": 1.0}

    settings = s2p_defaults()

    # Core imaging parameters (from legacy ops_default.npy)
    settings["fs"] = fps
    settings["tau"] = 1.0  # GCaMP7f decay time ~1s
    settings["diameter"] = [12.0, 12.0]

    # Pipeline control
    settings["run"]["do_deconvolution"] = False  # CASCADE handles spikes in Stage 4

    # IO
    settings["io"]["delete_bin"] = True

    # Registration (matching legacy)
    settings["registration"]["nonrigid"] = True
    settings["registration"]["block_size"] = (128, 128)
    settings["registration"]["batch_size"] = 100
    settings["registration"]["maxregshift"] = 0.1
    settings["registration"]["smooth_sigma"] = 1.15
    settings["registration"]["th_badframes"] = 1.0
    settings["registration"]["subpixel"] = 10

    # Detection (matching legacy)
    settings["detection"]["threshold_scaling"] = 1.0
    settings["detection"]["max_overlap"] = 0.75
    settings["detection"]["sparsery_settings"]["highpass_neuropil"] = 25

    # Extraction (matching legacy)
    settings["extraction"]["batch_size"] = 500
    settings["extraction"]["neuropil_extract"] = True
    settings["extraction"]["neuropil_coefficient"] = 0.7
    settings["extraction"]["inner_neuropil_radius"] = 2
    settings["extraction"]["min_neuropil_pixels"] = 350
    settings["extraction"]["allow_overlap"] = False

    # Classification — use custom soma classifier if available
    if classifier_path is None:
        candidate = Path("sourcedata/trackers/suite2p/classifier_soma.npy")
        if candidate.exists():
            classifier_path = candidate

    if classifier_path is not None and classifier_path.exists():
        settings["classification"]["classifier_path"] = str(classifier_path.resolve())
        settings["classification"]["use_builtin_classifier"] = False
        log.info("Using custom classifier: %s", classifier_path)
    else:
        settings["classification"]["use_builtin_classifier"] = True
        log.info("Using Suite2p builtin classifier (no custom classifier found)")

    return settings


# Keep backward-compatible alias
def default_ops(fps: float = 29.97) -> dict[str, Any]:
    """Return default ops dict (backward-compatible alias for default_settings)."""
    return default_settings(fps=fps)


def _patch_sparsedetect_mode_bug() -> None:
    """Patch Suite2p 1.0 bug where scipy.stats.mode returns an ndarray.

    ``estimate_spatial_scale`` calls ``mode(..., keepdims=True)`` which returns
    a numpy array. Downstream code does ``int(3 * 2**scale)`` which fails on
    arrays with more than 0 dimensions. This wraps ``find_best_scale`` to
    ensure ``scale`` is always a Python int.
    """
    try:
        import suite2p.detection.sparsedetect as sd
    except ImportError:
        return

    if getattr(sd.find_best_scale, "_hm2p_patched", False):
        return

    import numpy as np

    _orig = sd.find_best_scale

    def _patched(I, spatial_scale):  # noqa: ANN001, ANN202, N803
        scale, mode = _orig(I, spatial_scale)
        if isinstance(scale, np.ndarray):
            scale = int(scale.item())
        return scale, mode

    _patched._hm2p_patched = True  # type: ignore[attr-defined]
    sd.find_best_scale = _patched


def _deep_update(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base dict."""
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], val)
        else:
            base[key] = val
    return base


def run_suite2p(
    tiff_dir: Path,
    output_dir: Path,
    ops_overrides: dict[str, Any] | None = None,
    fps: float = 29.97,
) -> Path:
    """Run Suite2p on a directory of TIFF stacks.

    Args:
        tiff_dir: Directory containing raw TIFF imaging files (*_XYT.tif).
        output_dir: Directory where Suite2p output (plane0/) will be written.
            Suite2p creates its own subdirectories under this path.
        ops_overrides: Optional dict of Suite2p settings to override defaults.
            Can contain nested keys matching the Suite2p 1.0 settings structure.
        fps: Imaging frame rate (Hz). Used to set ``settings["fs"]``.

    Returns:
        Path to the suite2p output directory containing plane0/.

    Raises:
        ImportError: If suite2p is not installed.
        FileNotFoundError: If ``tiff_dir`` does not exist or has no TIFFs.
        RuntimeError: If Suite2p fails during processing.
    """
    if not tiff_dir.exists():
        raise FileNotFoundError(f"TIFF directory not found: {tiff_dir}")

    tiff_files = sorted(tiff_dir.glob("*.tif")) + sorted(tiff_dir.glob("*.tiff"))
    if not tiff_files:
        raise FileNotFoundError(f"No TIFF files found in {tiff_dir}")

    try:
        import suite2p
    except ImportError as exc:
        raise ImportError(
            "suite2p is not installed. "
            "Install via: pip install suite2p (GPU recommended)\n"
            "See: https://suite2p.readthedocs.io/"
        ) from exc

    log.info("Found %d TIFF file(s) in %s", len(tiff_files), tiff_dir)

    # Patch Suite2p 1.0 bug: scipy.stats.mode returns an array, not a scalar,
    # causing int() to fail in sparsedetect.sparsery. Fixed upstream in
    # https://github.com/MouseLand/suite2p — remove once suite2p >1.0.0.1.
    _patch_sparsedetect_mode_bug()

    # Build settings (Suite2p 1.0 API)
    settings = default_settings(fps=fps)
    if ops_overrides:
        _deep_update(settings, ops_overrides)

    # db dict — input/output paths
    db = {
        "data_path": [str(tiff_dir)],
        "save_path0": str(output_dir),
        "nplanes": 1,
        "nchannels": 1,
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Running Suite2p (fs=%.2f Hz, %d TIFFs)...", fps, len(tiff_files))
    suite2p.run_s2p(db=db, settings=settings)

    # Suite2p writes output to save_path0/suite2p/plane0/
    suite2p_dir = output_dir / "suite2p"
    plane0 = suite2p_dir / "plane0"

    if not plane0.exists():
        raise RuntimeError(
            f"Suite2p completed but plane0 directory not found at {plane0}. "
            "Check Suite2p logs for errors."
        )

    # Verify required output files
    for name in ("F.npy", "Fneu.npy", "iscell.npy", "stat.npy", "ops.npy"):
        if not (plane0 / name).exists():
            raise RuntimeError(f"Suite2p output file missing: {plane0 / name}")

    log.info("Suite2p complete. Output: %s", suite2p_dir)
    return suite2p_dir
