"""Stage 1 — run Suite2p on raw TIFF stacks.

Wraps ``suite2p.run_s2p()`` to produce the standard plane0/ output directory
containing F.npy, Fneu.npy, iscell.npy, stat.npy, and ops.npy.

Suite2p 1.0+ API: ``run_s2p(db=..., settings=...)``.

Suite2p is an optional dependency (GPU recommended). Install via:
    pip install suite2p
    # or conda install -c conda-forge suite2p

Architecture note — Cellpose 3 anatomical prior:
    hm2p single-plane recordings contain both somatic and dendritic ROIs in
    the same imaging plane; activity-based detection alone cannot reliably
    distinguish them at the detection stage. The default extraction mode uses
    Suite2p's ``anatomical_only=2`` setting, which seeds ROI detection from a
    Cellpose 3 segmentation of the mean/max projection before refining with
    activity statistics. This biases initial candidates toward compact,
    round-soma morphologies. Shape-based post-hoc classification in
    ``extraction/suite2p.py`` then separates retained soma and dendrite ROIs.
    See ARCHITECTURE.md for the full rationale.

References:
    Pachitariu et al. 2017. "Suite2p: beyond 10,000 neurons with standard
    two-photon microscopy." bioRxiv. doi:10.1101/061507.
    https://github.com/MouseLand/suite2p

    Stringer & Pachitariu 2025. "Cellpose3: one-click image restoration for
    improved cellular segmentation." Nature Methods.
    doi:10.1038/s41592-025-02595-5.
    https://github.com/MouseLand/cellpose
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# GCaMP indicator → GCaMP decay time constant (seconds).
# Used to set Suite2p's tau parameter, which affects temporal filtering of
# ROI detection statistics. Mismatched tau biases detection toward slower or
# faster transients. Values from Zhang et al. 2023.
#
# Reference:
#   Zhang et al. 2023. "Fast and sensitive GCaMP calcium indicators for
#   imaging neural populations." Nature 615:884–891.
#   doi:10.1038/s41586-023-05828-9
INDICATOR_TAU: dict[str, float] = {
    "GCaMP6f": 0.4,
    "GCaMP6s": 1.5,
    "GCaMP7f": 1.0,
    "GCaMP8s": 0.7,
    "GCaMP8m": 0.4,
    "GCaMP8f": 0.2,
}

_INDICATOR_TAU_DEFAULT: float = 1.0


def tau_for_indicator(indicator: str) -> float:
    """Return the Suite2p tau decay constant (s) for a GCaMP indicator name.

    Unknown indicators fall back to ``1.0`` s with a logged warning.

    Args:
        indicator: GCaMP indicator name, e.g. ``"GCaMP6f"``.

    Returns:
        Decay time constant in seconds.

    References:
        Zhang et al. 2023. "Fast and sensitive GCaMP calcium indicators for
        imaging neural populations." Nature 615:884–891.
        doi:10.1038/s41586-023-05828-9
    """
    if indicator in INDICATOR_TAU:
        return INDICATOR_TAU[indicator]
    log.warning(
        "Unknown GCaMP indicator %r — using default tau=%.1f s. "
        "Add to INDICATOR_TAU in extraction/run_suite2p.py if this is a new indicator.",
        indicator,
        _INDICATOR_TAU_DEFAULT,
    )
    return _INDICATOR_TAU_DEFAULT


def fps_from_timestamps(timestamps_h5: Path) -> float:
    """Compute imaging frame rate from a Stage 0 timestamps.h5 file.

    Computes ``mean(1.0 / diff(frame_times_imaging))`` from the timestamps
    file. Falls back to 29.97 Hz (with a warning) if the file is absent or
    has fewer than 2 frames.

    Args:
        timestamps_h5: Path to Stage 0 timestamps file.

    Returns:
        Estimated imaging frame rate in Hz.
    """
    import numpy as np

    _FALLBACK_FPS = 29.97

    if not timestamps_h5.exists():
        log.warning(
            "timestamps.h5 not found at %s — using fallback fps=%.2f Hz. "
            "Re-run Stage 0 to generate per-session timestamps.",
            timestamps_h5,
            _FALLBACK_FPS,
        )
        return _FALLBACK_FPS

    try:
        from hm2p.io.hdf5 import read_h5

        ts = read_h5(timestamps_h5)
        frame_times = ts.get("frame_times_imaging")
        if frame_times is None or len(frame_times) < 2:
            raise ValueError("frame_times_imaging missing or has fewer than 2 entries")
        fps = float(np.mean(1.0 / np.diff(frame_times)))
        log.info(
            "Measured imaging fps=%.4f Hz from %s (%d frames)",
            fps,
            timestamps_h5,
            len(frame_times),
        )
        return fps
    except Exception as exc:
        log.warning(
            "Failed to read fps from %s (%s) — using fallback fps=%.2f Hz.",
            timestamps_h5,
            exc,
            _FALLBACK_FPS,
        )
        return _FALLBACK_FPS


def default_settings(
    fps: float = 9.6,
    tau: float = _INDICATOR_TAU_DEFAULT,
    classifier_path: Path | None = None,
    anatomical_only: int = 2,
) -> dict[str, Any]:
    """Return default Suite2p settings for hm2p single-plane GCaMP imaging.

    Starts from Suite2p's built-in defaults and applies hm2p-specific
    parameters matching the legacy pipeline (``sourcedata/trackers/suite2p/ops_default.npy``):
    - Single-plane recordings (~9.6 Hz)
    - Per-session fps and tau — caller should pass values derived from
      ``fps_from_timestamps()`` and ``tau_for_indicator()``
    - diameter=12, nonrigid registration, block_size=(96, 96)
    - Custom soma classifier from ``sourcedata/trackers/suite2p/classifier_soma.npy``
    - Cellpose 3 anatomical prior (``anatomical_only=2``, default)

    Motion correction parameters are tuned for freely-moving recordings:
    - ``block_size=(96, 96)`` — smaller blocks for local distortion correction;
      freely-moving RSP recordings show cranial-window flexure that head-fixed
      defaults (128×128) cannot resolve
    - ``maxregshift=0.15`` — relaxed shift limit for the higher translation
      amplitudes expected from unrestrained movement

    Cellpose 3 anatomical prior (``anatomical_only`` parameter):
        Suite2p's ``anatomical_only`` setting controls whether Cellpose is used
        to seed ROI detection from a static image (mean/max projection) before
        refining with activity statistics. Values:

        - 0: activity-only detection (legacy behaviour; no Cellpose)
        - 1: Cellpose segmentation only, no activity refinement
        - 2: Cellpose seeds refined by activity (recommended default)
        - 3: stricter Cellpose-then-activity with higher activity weighting

        Default 2 is the recommended setting for freely-moving RSP recordings
        where the imaging plane contains both somatic and dendritic processes.
        Cellpose 3 must be installed (``pip install cellpose>=3.0``) when
        ``anatomical_only >= 1``. An ImportError is raised at run time if
        Cellpose is absent (see ``run_suite2p()``).

    Args:
        fps: Imaging frame rate in Hz. Should be derived from timestamps.h5
            via ``fps_from_timestamps()``. Default 9.6 Hz is typical for
            single-plane RSP recordings but may differ per session.
        tau: GCaMP decay time constant (s). Use ``tau_for_indicator()`` to
            look this up from the ``indicator`` column of experiments.csv.
        classifier_path: Path to a custom Suite2p classifier .npy file.
            If None, looks for ``sourcedata/trackers/suite2p/classifier_soma.npy``
            relative to the repo root. Falls back to Suite2p's builtin classifier.
        anatomical_only: Cellpose anatomical prior mode (0–3). Default 2.
            See parameter description above.

    Returns:
        Dict of Suite2p settings suitable for passing to ``suite2p.run_s2p(settings=...)``.

    References:
        Pachitariu et al. 2017. "Suite2p: beyond 10,000 neurons with standard
        two-photon microscopy." bioRxiv. doi:10.1101/061507.
        https://github.com/MouseLand/suite2p

        Stringer & Pachitariu 2025. "Cellpose3: one-click image restoration for
        improved cellular segmentation." Nature Methods.
        doi:10.1038/s41592-025-02595-5.
        https://github.com/MouseLand/cellpose

        Zhang et al. 2023. "Fast and sensitive GCaMP calcium indicators for
        imaging neural populations." Nature 615:884–891.
        doi:10.1038/s41586-023-05828-9
    """
    try:
        from suite2p import default_settings as s2p_defaults
    except ImportError:
        return {"fs": fps, "tau": tau, "anatomical_only": anatomical_only}

    settings = s2p_defaults()

    # Core imaging parameters
    settings["fs"] = fps
    settings["tau"] = tau
    settings["diameter"] = [12.0, 12.0]

    # Pipeline control
    settings["run"]["do_deconvolution"] = False  # CASCADE handles spikes in Stage 4

    # IO
    settings["io"]["delete_bin"] = True

    # Registration — tuned for freely-moving prep.
    # RSP cranial-window flexure produces more local distortion than head-fixed
    # preparations; smaller blocks and a relaxed shift limit improve correction.
    settings["registration"]["nonrigid"] = True
    settings["registration"]["block_size"] = (96, 96)
    settings["registration"]["batch_size"] = 100
    settings["registration"]["maxregshift"] = 0.15
    settings["registration"]["smooth_sigma"] = 1.15
    settings["registration"]["th_badframes"] = 1.0
    settings["registration"]["subpixel"] = 10

    # Detection (matching legacy)
    settings["detection"]["threshold_scaling"] = 1.0
    settings["detection"]["max_overlap"] = 0.75
    settings["detection"]["sparsery_settings"]["highpass_neuropil"] = 25

    # Cellpose 3 anatomical prior.
    # anatomical_only=2: Cellpose seeds candidate ROIs from a static image
    # (mean/max projection), then activity statistics refine them.  This is the
    # recommended setting for hm2p because the imaging plane contains both
    # somatic and dendritic processes; an anatomical prior biases initial
    # detection toward compact soma morphologies without discarding dendritic
    # ROIs entirely — they remain in the candidate set for shape-based
    # post-hoc classification in extraction/suite2p.py.
    # Reference: Stringer & Pachitariu 2025. doi:10.1038/s41592-025-02595-5
    settings["detection"]["anatomical_only"] = anatomical_only

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
def default_ops(fps: float = 9.6, tau: float = _INDICATOR_TAU_DEFAULT) -> dict[str, Any]:
    """Return default ops dict (backward-compatible alias for default_settings)."""
    return default_settings(fps=fps, tau=tau)


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

    def _patched(I, spatial_scale):  # noqa: ANN001, ANN202, E741, N803
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
    fps: float | None = None,
    timestamps_h5: Path | None = None,
    indicator: str = "GCaMP6s",
    anatomical_only: int = 2,
) -> Path:
    """Run Suite2p on a directory of TIFF stacks.

    The frame rate (``fs``) and GCaMP decay constant (``tau``) are determined
    per-session from ``timestamps_h5`` and ``indicator`` respectively, so that
    Suite2p's temporal filters match the actual recording conditions.

    When ``anatomical_only >= 1``, Cellpose 3 is used to seed ROI detection
    from a static mean/max projection before activity-based refinement.
    Cellpose must be installed (``pip install cellpose>=3.0``) for this mode.
    A pre-flight ImportError is raised with install instructions if it is
    absent, rather than letting Suite2p fail later with a less actionable error.

    Args:
        tiff_dir: Directory containing raw TIFF imaging files (*_XYT.tif).
        output_dir: Directory where Suite2p output (plane0/) will be written.
            Suite2p creates its own subdirectories under this path.
        ops_overrides: Optional dict of Suite2p settings to override defaults.
            Can contain nested keys matching the Suite2p 1.0 settings structure.
        fps: Imaging frame rate (Hz). When ``None`` (default), the rate is
            computed from ``timestamps_h5`` via ``fps_from_timestamps()``.
            Pass an explicit value only when timestamps.h5 is unavailable.
        timestamps_h5: Path to the Stage 0 timestamps file. Used to compute
            ``fps`` per-session when ``fps`` is ``None``. Ignored when ``fps``
            is supplied explicitly.
        indicator: GCaMP indicator name (e.g. ``"GCaMP6s"``). Used to look up
            the Suite2p tau decay constant via ``tau_for_indicator()``.
            Defaults to ``"GCaMP6s"`` — should be overridden from the
            ``indicator`` column of experiments.csv for each session.
        anatomical_only: Cellpose anatomical prior mode (0–3). Passed to
            ``default_settings(anatomical_only=...)``. Default 2 (Cellpose
            seeds + activity refinement). 0 = activity-only (legacy).

    Returns:
        Path to the suite2p output directory containing plane0/.

    Raises:
        ImportError: If suite2p is not installed, or if ``anatomical_only >= 1``
            and cellpose is not installed.
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

    # Pre-flight check: Cellpose must be importable when anatomical_only >= 1.
    # Without this check, Suite2p raises an opaque error deep inside its
    # detection code, which is harder to diagnose.
    if anatomical_only >= 1:
        try:
            import cellpose  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                f"anatomical_only={anatomical_only} requires Cellpose >= 3.0, "
                "but cellpose is not installed.\n"
                "Install via: pip install 'cellpose>=3.0'\n"
                "See: https://github.com/MouseLand/cellpose\n"
                "Reference: Stringer & Pachitariu 2025. doi:10.1038/s41592-025-02595-5\n"
                "To use activity-only detection without Cellpose, pass anatomical_only=0."
            ) from exc

    # Resolve fps from timestamps.h5 when not explicitly supplied.
    if fps is None:
        if timestamps_h5 is not None:
            fps = fps_from_timestamps(timestamps_h5)
        else:
            fps = fps_from_timestamps(Path("timestamps.h5"))  # best-effort fallback

    # Resolve tau from indicator name.
    tau = tau_for_indicator(indicator)

    log.info(
        "Found %d TIFF file(s) in %s (fps=%.4f Hz, indicator=%s, tau=%.2f s)",
        len(tiff_files),
        tiff_dir,
        fps,
        indicator,
        tau,
    )

    # Patch Suite2p 1.0 bug: scipy.stats.mode returns an array, not a scalar,
    # causing int() to fail in sparsedetect.sparsery. Fixed upstream in
    # https://github.com/MouseLand/suite2p — remove once suite2p >1.0.0.1.
    _patch_sparsedetect_mode_bug()

    # Build settings (Suite2p 1.0 API)
    settings = default_settings(fps=fps, tau=tau, anatomical_only=anatomical_only)
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

    log.info(
        "Running Suite2p (fs=%.2f Hz, tau=%.2f s, anatomical_only=%d, %d TIFFs)...",
        fps,
        tau,
        anatomical_only,
        len(tiff_files),
    )
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

    # Compute max projection from the registered binary while it's still on disk.
    # Suite2p only stores meanImg and meanImgE (enhanced mean), not a true max.
    _compute_max_projection(plane0)

    log.info("Suite2p complete. Output: %s", suite2p_dir)
    return suite2p_dir


def _compute_max_projection(plane0: Path) -> None:
    """Compute max projection from Suite2p's registered binary and save to ops.

    Reads data.bin in chunks to avoid loading the entire movie into memory.
    Saves as ops["max_proj"] in ops.npy.
    """
    import numpy as np

    ops_path = plane0 / "ops.npy"
    bin_path = plane0 / "data.bin"

    if not bin_path.exists() or not ops_path.exists():
        log.warning("Cannot compute max projection: missing data.bin or ops.npy")
        return

    try:
        ops = np.load(ops_path, allow_pickle=True).item()
        ly = ops.get("Ly", 0)
        lx = ops.get("Lx", 0)
        if ly == 0 or lx == 0:
            log.warning("Cannot compute max projection: Ly/Lx not in ops")
            return

        # data.bin uses cropped dimensions (yrange/xrange), not full Ly×Lx
        yrange = ops.get("yrange", [0, ly])
        xrange = ops.get("xrange", [0, lx])
        crop_ly = yrange[1] - yrange[0]
        crop_lx = xrange[1] - xrange[0]

        n_frames = ops.get("nframes", 0)
        if n_frames == 0:
            file_size = bin_path.stat().st_size
            n_frames = file_size // (crop_ly * crop_lx * 2)  # int16

        log.info(
            "Computing max projection (%d frames, full %dx%d, crop %dx%d)...",
            n_frames,
            ly,
            lx,
            crop_ly,
            crop_lx,
        )

        # Read in chunks of 1000 frames to limit memory
        chunk_size = 1000
        max_proj_crop = None

        with open(bin_path, "rb") as f:
            for start in range(0, n_frames, chunk_size):
                n_read = min(chunk_size, n_frames - start)
                chunk = np.fromfile(f, dtype=np.int16, count=n_read * crop_ly * crop_lx)
                if chunk.size != n_read * crop_ly * crop_lx:
                    break
                chunk = chunk.reshape(n_read, crop_ly, crop_lx).astype(np.float32)
                chunk_max = chunk.max(axis=0)
                if max_proj_crop is None:
                    max_proj_crop = chunk_max
                else:
                    max_proj_crop = np.maximum(max_proj_crop, chunk_max)

        if max_proj_crop is not None:
            # Embed into full frame (same size as meanImg)
            max_proj = np.zeros((ly, lx), dtype=np.float32)
            max_proj[yrange[0] : yrange[1], xrange[0] : xrange[1]] = max_proj_crop
            ops["max_proj"] = max_proj
            np.save(ops_path, ops)
            log.info("Max projection saved to ops.npy (max_proj key)")

    except Exception as exc:
        log.warning("Failed to compute max projection: %s", exc)
