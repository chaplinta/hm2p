"""Build FISSA ROI masks from Suite2p ``stat.npy``.

FISSA (Keemink et al. 2018) requires per-ROI binary spatial masks plus the raw
(motion-corrected) movie. It generates its own concentric neuropil rings around
each ROI internally, so only the somatic/ROI mask is needed here — not a
separate neuropil mask.

The masks produced here are aligned to the **full** Suite2p ROI axis (every row
of ``stat.npy``, in order), so the FISSA output traces line up one-to-one with
``F``/``Fneu``/``dff`` in ca.h5. The Stage 4 pipeline (``hm2p.calcium.run.run``)
processes all ROIs — not only ``iscell`` — so masks must cover all of them.

Suite2p axis convention: ``stat[i]["ypix"]`` indexes image rows and
``stat[i]["xpix"]`` indexes image columns, with image shape ``(Ly, Lx)``. This
matches ``hm2p.extraction.suite2p.Suite2pExtractor.get_roi_masks`` and is the
format ``fissa.Experiment`` expects (a list of 2-D bool arrays, height × width).

References
----------
Keemink SW, Lowe SC, Pakan JMP, Dylda E, van Rossum MCW, Bhatt DH. 2018.
"FISSA: A neuropil decontamination toolbox for calcium imaging signals."
Scientific Reports 8:3493. doi:10.1038/s41598-018-21640-2
https://github.com/rochefort-lab/fissa

Pachitariu M, Stringer C, Dipoppa M, et al. 2017. "Suite2p: beyond 10,000
neurons with standard two-photon microscopy." doi:10.1101/061507
https://github.com/MouseLand/suite2p
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def build_roi_mask(
    ypix: np.ndarray,
    xpix: np.ndarray,
    Ly: int,
    Lx: int,
) -> np.ndarray:
    """Build a single binary ROI mask from Suite2p pixel coordinates.

    Parameters
    ----------
    ypix : np.ndarray
        1-D integer array of row indices (Suite2p ``ypix``).
    xpix : np.ndarray
        1-D integer array of column indices (Suite2p ``xpix``). Must have the
        same length as ``ypix``.
    Ly : int
        Image height in pixels (number of rows; Suite2p ``ops['Ly']``).
    Lx : int
        Image width in pixels (number of columns; Suite2p ``ops['Lx']``).

    Returns
    -------
    np.ndarray
        ``(Ly, Lx)`` boolean array, ``True`` at every ROI pixel.

    Raises
    ------
    ValueError
        If ``Ly``/``Lx`` are not positive, if ``ypix`` and ``xpix`` differ in
        length, or if any pixel coordinate falls outside ``[0, Ly)`` × ``[0, Lx)``.
    """
    if Ly <= 0 or Lx <= 0:
        raise ValueError(f"Ly and Lx must be positive; got Ly={Ly}, Lx={Lx}")

    ypix = np.asarray(ypix)
    xpix = np.asarray(xpix)
    if ypix.shape != xpix.shape:
        raise ValueError(
            f"ypix and xpix must have the same shape; got {ypix.shape} and {xpix.shape}"
        )
    if ypix.ndim != 1:
        raise ValueError(f"ypix/xpix must be 1-D; got {ypix.ndim}-D")

    mask = np.zeros((Ly, Lx), dtype=bool)
    if ypix.size == 0:
        return mask

    yi = ypix.astype(np.int64)
    xi = xpix.astype(np.int64)
    if yi.min() < 0 or yi.max() >= Ly or xi.min() < 0 or xi.max() >= Lx:
        raise ValueError(
            "ROI pixel coordinates out of bounds for image "
            f"({Ly}x{Lx}): y in [{yi.min()}, {yi.max()}], "
            f"x in [{xi.min()}, {xi.max()}]"
        )

    mask[yi, xi] = True
    return mask


def build_roi_masks_from_stat(
    stat: Sequence[dict[str, Any]],
    Ly: int,
    Lx: int,
    roi_indices: Sequence[int] | None = None,
) -> list[np.ndarray]:
    """Build per-ROI binary masks for FISSA from a Suite2p ``stat`` list.

    Produces one ``(Ly, Lx)`` boolean mask per ROI, in the order requested.
    When ``roi_indices`` is ``None`` (the default) masks are built for **all**
    ROIs in ``stat`` order, matching the ROI axis of ``F``/``Fneu``/``dff`` in
    ca.h5 exactly. Reuses the existing detected ROIs — no re-detection — so the
    ROI set stays identical to the current outputs.

    Parameters
    ----------
    stat : sequence of dict
        Suite2p ``stat.npy`` contents (one dict per ROI, each with ``ypix`` and
        ``xpix`` integer arrays).
    Ly : int
        Image height in pixels (``ops['Ly']``).
    Lx : int
        Image width in pixels (``ops['Lx']``).
    roi_indices : sequence of int or None
        Indices into ``stat`` to build masks for, in output order. If ``None``,
        all ROIs are used in their native order.

    Returns
    -------
    list of np.ndarray
        One ``(Ly, Lx)`` boolean mask per requested ROI, ready to pass to
        ``hm2p.calcium.neuropil.subtract_fissa`` as ``roi_masks``.

    Raises
    ------
    ValueError
        If ``Ly``/``Lx`` are not positive, a requested index is out of range,
        or a ``stat`` entry lacks ``ypix``/``xpix``.
    """
    if Ly <= 0 or Lx <= 0:
        raise ValueError(f"Ly and Lx must be positive; got Ly={Ly}, Lx={Lx}")

    n_stat = len(stat)
    if roi_indices is None:
        indices: list[int] = list(range(n_stat))
    else:
        indices = [int(i) for i in roi_indices]
        for i in indices:
            if i < 0 or i >= n_stat:
                raise ValueError(f"roi index {i} out of range for stat of length {n_stat}")

    masks: list[np.ndarray] = []
    for i in indices:
        entry = stat[i]
        if "ypix" not in entry or "xpix" not in entry:
            raise ValueError(f"stat entry {i} missing ypix/xpix")
        masks.append(build_roi_mask(entry["ypix"], entry["xpix"], Ly, Lx))

    log.info("Built %d FISSA ROI mask(s) for a %dx%d field of view", len(masks), Ly, Lx)
    return masks


def crop_masks_to_window(
    masks: Sequence[np.ndarray],
    yrange: tuple[int, int],
    xrange: tuple[int, int],
) -> list[np.ndarray]:
    """Crop full-FOV ROI masks to Suite2p's registered-binary window.

    Suite2p's ``data.bin`` is written at the *cropped* dimensions
    ``(yrange[1]-yrange[0], xrange[1]-xrange[0])`` — the valid registration
    window after motion correction — while ``ypix``/``xpix`` (and therefore the
    masks from :func:`build_roi_masks_from_stat`) are in *full*-FOV coordinates.
    To feed FISSA a movie + masks that share a pixel grid, the masks must be
    cropped to the same ``yrange``/``xrange`` window as the movie frames.

    Parameters
    ----------
    masks : sequence of np.ndarray
        Full-FOV ``(Ly, Lx)`` boolean masks (output of
        :func:`build_roi_masks_from_stat`).
    yrange : tuple of int
        ``(y_start, y_stop)`` row crop window from ``ops['yrange']``.
    xrange : tuple of int
        ``(x_start, x_stop)`` column crop window from ``ops['xrange']``.

    Returns
    -------
    list of np.ndarray
        Cropped ``(y_stop-y_start, x_stop-x_start)`` boolean masks, one per
        input mask, in the same order.

    Raises
    ------
    ValueError
        If the crop window is degenerate or exceeds a mask's bounds.

    Notes
    -----
    ROI pixels that fall outside the crop window (at the very edge of the FOV)
    are silently dropped — they are absent from the registered movie too, so
    FISSA cannot use them. A warning is logged if any mask loses all its pixels.
    """
    (y0, y1), (x0, x1) = yrange, xrange
    if y1 <= y0 or x1 <= x0:
        raise ValueError(f"degenerate crop window: yrange={yrange}, xrange={xrange}")

    cropped: list[np.ndarray] = []
    for i, m in enumerate(masks):
        if y1 > m.shape[0] or x1 > m.shape[1]:
            raise ValueError(f"crop window {yrange}x{xrange} exceeds mask {i} bounds {m.shape}")
        sub = np.asarray(m, dtype=bool)[y0:y1, x0:x1].copy()
        if not sub.any():
            log.warning(
                "ROI mask %d has no pixels inside the registration crop window "
                "%s x %s — all its pixels are at the cropped FOV edge.",
                i,
                yrange,
                xrange,
            )
        cropped.append(sub)
    return cropped


def build_roi_masks_from_plane(
    plane_dir: Path,
) -> tuple[list[np.ndarray], int]:
    """Build FISSA ROI masks directly from a Suite2p ``plane0`` directory.

    Loads ``stat.npy`` and ``ops.npy`` (for ``Ly``/``Lx``) and builds one mask
    per ROI in native ``stat`` order. Convenience wrapper used by the Stage 4
    FISSA driver so the caller does not have to load the arrays itself.

    Parameters
    ----------
    plane_dir : Path
        Path to the Suite2p ``plane0`` directory containing ``stat.npy`` and
        ``ops.npy``.

    Returns
    -------
    masks : list of np.ndarray
        Per-ROI ``(Ly, Lx)`` boolean masks in ``stat`` order.
    n_rois : int
        Number of ROIs (== ``len(masks)``), returned for a cheap sanity check
        against ``F.npy``'s first axis at the call site.

    Raises
    ------
    FileNotFoundError
        If ``stat.npy`` or ``ops.npy`` is missing.
    """
    plane_dir = Path(plane_dir)
    stat_path = plane_dir / "stat.npy"
    ops_path = plane_dir / "ops.npy"
    if not stat_path.exists():
        raise FileNotFoundError(f"stat.npy not found in {plane_dir}")
    if not ops_path.exists():
        raise FileNotFoundError(f"ops.npy not found in {plane_dir}")

    stat = list(np.load(stat_path, allow_pickle=True))
    ops = np.load(ops_path, allow_pickle=True).item()
    Ly = int(ops.get("Ly", 0))
    Lx = int(ops.get("Lx", 0))
    masks = build_roi_masks_from_stat(stat, Ly, Lx)
    return masks, len(masks)
