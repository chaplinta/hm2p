"""Z-drift estimation from serial2p z-stacks.

Registers each imaging frame against a z-stack to track focal plane drift
over time. Uses Suite2p's phase-correlation registration when available,
with a pure scipy/numpy fallback.

References
----------
Pachitariu et al. 2017. "Suite2p: beyond 10,000 neurons with standard
two-photon microscopy." bioRxiv. doi:10.1101/061507
GitHub: https://github.com/MouseLand/suite2p
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from hm2p.io.hdf5 import read_h5, write_h5

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Suite2p imports
# ---------------------------------------------------------------------------

try:
    from suite2p.io import BinaryFile
    from suite2p.registration.zalign import register_to_zstack

    _HAS_SUITE2P = True
except ImportError:
    _HAS_SUITE2P = False


# ---------------------------------------------------------------------------
# Fallback phase-correlation (no Suite2p dependency)
# ---------------------------------------------------------------------------


def _phase_correlate_2d(
    frame: np.ndarray,
    ref: np.ndarray,
) -> float:
    """Compute normalised phase correlation between *frame* and *ref*.

    Returns the peak value of the normalised cross-power spectrum,
    which serves as a similarity score (higher = better match).

    Parameters
    ----------
    frame : 2-D array (Ly, Lx)
    ref : 2-D array (Ly, Lx), same shape as *frame*

    Returns
    -------
    float
        Peak correlation value in [0, 1].
    """
    f_frame = np.fft.fft2(frame.astype(np.float64))
    f_ref = np.fft.fft2(ref.astype(np.float64))
    cross_power = f_frame * np.conj(f_ref)
    denom = np.abs(cross_power)
    # Avoid division by zero
    denom = np.where(denom > 0, denom, 1.0)
    normalised = cross_power / denom
    correlation = np.fft.ifft2(normalised).real
    return float(np.max(correlation))


def _register_to_zstack_fallback(
    frames: np.ndarray,
    zstack: np.ndarray,
) -> np.ndarray:
    """Register frames against z-stack using phase correlation.

    Pure numpy/scipy fallback when Suite2p is not installed.

    Parameters
    ----------
    frames : (n_frames, Ly, Lx) array
    zstack : (n_zplanes, Ly, Lx) array

    Returns
    -------
    zcorr : (n_frames, n_zplanes) float64 array of correlation values.
    """
    n_frames = frames.shape[0]
    n_zplanes = zstack.shape[0]
    zcorr = np.zeros((n_frames, n_zplanes), dtype=np.float64)
    for i in range(n_frames):
        for z in range(n_zplanes):
            zcorr[i, z] = _phase_correlate_2d(frames[i], zstack[z])
    return zcorr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_zstack(zstack_path: Path) -> np.ndarray:
    """Load a z-stack TIFF file.

    Parameters
    ----------
    zstack_path : Path
        Path to a multi-page TIFF file containing the z-stack.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_zplanes, Ly, Lx)`` with dtype float32.

    Raises
    ------
    FileNotFoundError
        If *zstack_path* does not exist.
    ValueError
        If the loaded TIFF is not 3-D.
    """
    import tifffile

    zstack_path = Path(zstack_path)
    if not zstack_path.exists():
        raise FileNotFoundError(f"Z-stack TIFF not found: {zstack_path}")

    data = tifffile.imread(str(zstack_path))
    if data.ndim != 3:
        raise ValueError(
            f"Expected 3-D z-stack (n_zplanes, Ly, Lx), got shape {data.shape}"
        )
    return data.astype(np.float32)


def compute_zdrift(
    suite2p_dir: Path,
    zstack_path: Path,
    sigma: float = 2.0,
    batch_size: int = 500,
) -> dict:
    """Compute z-drift from Suite2p registered data and a z-stack.

    Registers each imaging frame against the z-stack planes using phase
    correlation, then smooths the correlation traces and extracts the
    best-matching z-plane per frame.

    Parameters
    ----------
    suite2p_dir : Path
        Path to the Suite2p ``plane0/`` directory containing ``ops.npy``
        and ``data.bin``.
    zstack_path : Path
        Path to z-stack TIFF file.
    sigma : float, optional
        Gaussian smoothing sigma (in frames) applied to per-plane
        correlation traces before taking argmax. Default ``2.0``.
    batch_size : int, optional
        Number of frames to process at once for memory management.
        Default ``500``.

    Returns
    -------
    dict
        ``zpos`` : np.ndarray (n_frames,) int — z-plane index per frame.
        ``zcorr`` : np.ndarray (n_frames, n_zplanes) float — correlation
        matrix.
        ``zpos_smooth`` : np.ndarray (n_frames,) float — smoothed
        z-position (weighted mean, not argmax).
        ``n_zplanes`` : int — number of planes in the z-stack.
        ``zstack_path`` : str — path to the z-stack file used.

    Raises
    ------
    FileNotFoundError
        If the Suite2p directory or z-stack file does not exist.
    """
    suite2p_dir = Path(suite2p_dir)
    zstack_path = Path(zstack_path)

    if not suite2p_dir.exists():
        raise FileNotFoundError(f"Suite2p directory not found: {suite2p_dir}")

    zstack = load_zstack(zstack_path)
    n_zplanes, Ly, Lx = zstack.shape

    # Load registered frames from Suite2p binary
    ops_path = suite2p_dir / "ops.npy"
    bin_path = suite2p_dir / "data.bin"

    if not ops_path.exists():
        raise FileNotFoundError(f"ops.npy not found in {suite2p_dir}")
    if not bin_path.exists():
        raise FileNotFoundError(f"data.bin not found in {suite2p_dir}")

    ops = np.load(ops_path, allow_pickle=True).item()
    n_frames = ops.get("nframes", 0)
    ly_ops = ops.get("Ly", Ly)
    lx_ops = ops.get("Lx", Lx)

    if n_frames == 0:
        raise ValueError("ops.npy reports 0 frames — cannot compute z-drift")

    # Accumulate correlation in batches
    zcorr_all = np.zeros((n_frames, n_zplanes), dtype=np.float64)

    if _HAS_SUITE2P:
        logger.info("Using Suite2p register_to_zstack")
        with BinaryFile(ly_ops, lx_ops, str(bin_path)) as bf:
            for start in range(0, n_frames, batch_size):
                end = min(start + batch_size, n_frames)
                frames = bf[start:end]  # (batch, Ly, Lx)
                # suite2p register_to_zstack returns (n_zplanes, n_batch)
                corr_batch = register_to_zstack(frames, zstack)
                zcorr_all[start:end, :] = corr_batch.T
    else:
        logger.warning(
            "Suite2p not installed — using fallback phase-correlation "
            "(slower, no sub-pixel registration)"
        )
        # Read binary directly as numpy memmap
        frames_mmap = np.memmap(
            str(bin_path),
            dtype=np.int16,
            mode="r",
            shape=(n_frames, ly_ops, lx_ops),
        )
        for start in range(0, n_frames, batch_size):
            end = min(start + batch_size, n_frames)
            batch = frames_mmap[start:end].astype(np.float32)
            zcorr_all[start:end, :] = _register_to_zstack_fallback(batch, zstack)

    # Smooth each z-plane's correlation trace over time
    zcorr_smooth = np.zeros_like(zcorr_all)
    for z in range(n_zplanes):
        zcorr_smooth[:, z] = gaussian_filter1d(zcorr_all[:, z], sigma=sigma)

    # z-position: argmax of smoothed correlations
    zpos = np.argmax(zcorr_smooth, axis=1).astype(np.int32)

    # Smooth z-position: weighted mean over z-planes (continuous estimate)
    plane_indices = np.arange(n_zplanes, dtype=np.float64)
    # Softmax-like weighting from correlation values
    zcorr_pos = zcorr_smooth - zcorr_smooth.max(axis=1, keepdims=True)
    weights = np.exp(zcorr_pos * 10.0)  # sharpen
    weights /= weights.sum(axis=1, keepdims=True)
    zpos_smooth = (weights @ plane_indices).astype(np.float64)

    return {
        "zpos": zpos,
        "zcorr": zcorr_smooth.astype(np.float32),
        "zpos_smooth": zpos_smooth.astype(np.float64),
        "n_zplanes": n_zplanes,
        "zstack_path": str(zstack_path),
    }


def save_zdrift(zdrift: dict, output_path: Path) -> None:
    """Save z-drift results to HDF5.

    Parameters
    ----------
    zdrift : dict
        Output of :func:`compute_zdrift`.
    output_path : Path
        Destination HDF5 file path.
    """
    arrays = {
        "zpos": zdrift["zpos"],
        "zcorr": zdrift["zcorr"],
        "zpos_smooth": zdrift["zpos_smooth"],
    }
    attrs = {
        "n_zplanes": zdrift["n_zplanes"],
        "zstack_path": zdrift["zstack_path"],
    }
    write_h5(Path(output_path), arrays, attrs)


def load_zdrift(path: Path) -> dict:
    """Load z-drift results from HDF5.

    Parameters
    ----------
    path : Path
        Path to the HDF5 file written by :func:`save_zdrift`.

    Returns
    -------
    dict
        Same structure as :func:`compute_zdrift` output.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    from hm2p.io.hdf5 import read_attrs

    arrays = read_h5(Path(path), keys=["zpos", "zcorr", "zpos_smooth"])
    attrs = read_attrs(Path(path))
    return {
        "zpos": arrays["zpos"],
        "zcorr": arrays["zcorr"],
        "zpos_smooth": arrays["zpos_smooth"],
        "n_zplanes": int(attrs["n_zplanes"]),
        "zstack_path": str(attrs["zstack_path"]),
    }
