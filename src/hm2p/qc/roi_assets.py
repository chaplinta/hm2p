"""Per-session Suite2p inspection data for the ROI quality page.

Builds the lazily loaded session asset of ``docs/qc/qc-rois.html``
(``results/qc/assets/rois/<exp_id>.json``): the Suite2p summary images
registered to the full frame, every ROI's pixel weights (``lam``), the traces
of the soma and dendrite ROIs and the registration offsets. Also the small
per-ROI Suite2p ``stat`` fields and the calcium QC thresholds, which are
embedded in the page itself.

Suite2p stores ``max_proj`` and ``Vcorr`` cropped to the region that stays
valid after registration (``ops["yrange"]``, ``ops["xrange"]``), while
``meanImg``, ``meanImgE`` and ``refImg`` and the ROI pixel coordinates in
``stat.npy`` are in full-frame coordinates; :func:`pad_to_frame` places a
cropped image back into the full frame.

Suite2p: Pachitariu M, Stringer C, Dipoppa M, et al. 2017. "Suite2p: beyond
10,000 neurons with standard two-photon microscopy." bioRxiv.
doi:10.1101/061507. https://github.com/MouseLand/suite2p
"""

from __future__ import annotations

import base64
import math
from typing import Any

import numpy as np
import numpy.typing as npt

from hm2p.qc.common import bin_reduce, encode_i16, fnum, mask_intervals

# ops.npy image key -> page label
IMAGE_KEYS = {
    "meanImg": "mean",
    "meanImgE": "enhanced mean",
    "max_proj": "max projection",
    "Vcorr": "correlation",
}
# Suite2p stat.npy scalar fields shown per ROI
STAT_FIELDS = ("npix", "radius", "aspect_ratio", "compact", "skew", "std", "footprint")
# Traces stored per ROI: asset key -> (source, description)
TRACE_KEYS = ("F", "Fneu", "F_corr", "dff", "spikes")
TRACE_BUDGET_BYTES = 2_500_000


def pad_to_frame(
    img: npt.ArrayLike,
    ly: int,
    lx: int,
    yrange: npt.ArrayLike | None = None,
    xrange: npt.ArrayLike | None = None,
) -> np.ndarray:
    """Place a (possibly cropped) Suite2p image in an ``ly`` x ``lx`` frame.

    An image already of the full-frame shape is returned unchanged (as
    float64). A cropped image must have shape
    ``(yrange[1] - yrange[0], xrange[1] - xrange[0])``; pixels outside the
    crop are NaN.

    Raises
    ------
    ValueError
        If the image is neither full-frame nor the shape of the crop.
    """
    a = np.asarray(img, dtype=np.float64)
    if a.shape == (ly, lx):
        return a
    if yrange is None or xrange is None:
        raise ValueError(f"image {a.shape} is not {ly}x{lx} and no crop range was given")
    y0, y1 = (int(v) for v in np.asarray(yrange).ravel()[:2])
    x0, x1 = (int(v) for v in np.asarray(xrange).ravel()[:2])
    if a.shape != (y1 - y0, x1 - x0) or y0 < 0 or x0 < 0 or y1 > ly or x1 > lx:
        raise ValueError(
            f"image {a.shape} does not match crop y {y0}:{y1}, x {x0}:{x1} of {ly}x{lx}"
        )
    out = np.full((ly, lx), np.nan)
    out[y0:y1, x0:x1] = a
    return out


def frame_images(ops: dict[str, Any]) -> dict[str, np.ndarray]:
    """Summary images of ``ops.npy`` in full-frame coordinates, keyed as in :data:`IMAGE_KEYS`.

    Images missing from *ops*, or whose shape matches neither the frame nor
    the crop, are left out.
    """
    ly, lx = int(ops["Ly"]), int(ops["Lx"])
    out: dict[str, np.ndarray] = {}
    for k in IMAGE_KEYS:
        if k not in ops or ops[k] is None:
            continue
        try:
            out[k] = pad_to_frame(ops[k], ly, lx, ops.get("yrange"), ops.get("xrange"))
        except ValueError:
            continue
    return out


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def encode_footprints(stat: list[dict[str, Any]]) -> dict:
    """Pixel coordinates and weights of every ROI, packed for the page.

    ``y``, ``x``: little-endian uint16; ``w``: uint8 weight, ``lam`` scaled to
    255 at each ROI's maximum; ``start``: little-endian uint32 offset of each
    ROI's first pixel (length ``n_rois + 1``). All base64.
    """
    ys, xs, ws, starts = [], [], [], [0]
    for st in stat:
        yp = np.asarray(st.get("ypix", []), dtype=np.int64).ravel()
        xp = np.asarray(st.get("xpix", []), dtype=np.int64).ravel()
        lam = np.asarray(st.get("lam", np.ones(yp.size)), dtype=np.float64).ravel()
        if lam.size != yp.size:
            lam = np.ones(yp.size)
        mx = lam.max() if lam.size else 0.0
        w = np.round(255 * np.clip(lam / mx, 0, 1)) if mx > 0 else np.zeros(lam.size)
        ys.append(yp)
        xs.append(xp)
        ws.append(w)
        starts.append(starts[-1] + yp.size)
    cat = lambda parts: np.concatenate(parts) if parts else np.zeros(0)  # noqa: E731
    return {
        "y": _b64(cat(ys).astype("<u2")),
        "x": _b64(cat(xs).astype("<u2")),
        "w": _b64(cat(ws).astype(np.uint8)),
        "start": _b64(np.asarray(starts, dtype="<u4")),
        "n": len(stat),
    }


def stat_fields(stat: list[dict[str, Any]], iscell: npt.ArrayLike | None = None) -> list[dict]:
    """Suite2p scalar ``stat`` fields (:data:`STAT_FIELDS`) and ``iscell`` probability per ROI."""
    ic = None if iscell is None else np.asarray(iscell, dtype=np.float64)
    out = []
    for i, st in enumerate(stat):
        d: dict = {}
        for k in STAT_FIELDS:
            v = st.get(k)
            if k == "npix" and v is None:
                v = len(st.get("ypix", []))
            d[k] = fnum(v, 4) if v is not None and np.ndim(v) == 0 else None
        if ic is not None and i < ic.shape[0]:
            d["iscell_prob"] = fnum(ic[i, 1] if ic.ndim == 2 else ic[i], 4)
        out.append(d)
    return out


def choose_bin(
    n_frames: int, n_rois: int, n_traces: int, budget_bytes: int = TRACE_BUDGET_BYTES
) -> int:
    """Smallest frame-binning factor keeping base64 int16 traces within *budget_bytes*.

    Each sample costs 2 bytes, times 4/3 for base64.
    """
    if n_frames <= 0 or n_rois <= 0 or n_traces <= 0:
        return 1
    per_frame = n_rois * n_traces * 2 * 4 / 3
    return max(1, math.ceil(n_frames * per_frame / budget_bytes))


def encode_traces(
    traces: dict[str, np.ndarray | None],
    rois: npt.ArrayLike,
    n_per_bin: int = 1,
) -> dict[str, list[dict | None]]:
    """int16-encoded traces of the selected ROIs, binned by ``n_per_bin`` frames (mean).

    *traces* maps a name to an ``(n_rois, n_frames)`` array (or None, which is
    skipped). Returns ``{name: [encoded trace per selected ROI]}``; an ROI index
    beyond an array's rows gives None.
    """
    idx = [int(i) for i in np.asarray(rois).ravel()]
    out: dict[str, list[dict | None]] = {}
    for name, arr in traces.items():
        if arr is None:
            continue
        a = np.asarray(arr)
        if a.ndim != 2:
            continue
        out[name] = [
            encode_i16(bin_reduce(a[i], n_per_bin) if n_per_bin > 1 else a[i])
            if 0 <= i < a.shape[0]
            else None
            for i in idx
        ]
    return out


def binned_dark_intervals(light_on: npt.ArrayLike, n_per_bin: int = 1) -> list[list[int]]:
    """``[start, stop)`` intervals, in binned samples, where the light is off.

    A bin counts as dark when more than half of its frames are dark.
    """
    lo = np.asarray(light_on, dtype=np.float64).ravel()
    if lo.size == 0:
        return []
    frac = bin_reduce(lo, n_per_bin) if n_per_bin > 1 else lo
    return mask_intervals(np.nan_to_num(frac, nan=1.0) < 0.5)


def registration(ops: dict[str, Any]) -> dict:
    """Rigid registration offsets (``xoff``, ``yoff``, px) and ``corrXY`` per frame (int16)."""
    out: dict = {}
    for k in ("xoff", "yoff", "corrXY"):
        v = ops.get(k)
        if v is not None and np.asarray(v).size:
            a = np.asarray(v, dtype=np.float64).ravel()
            out[k] = encode_i16(a)
            out[f"{k}_abs_max" if k != "corrXY" else "corrXY_min"] = fnum(
                np.nanmax(np.abs(a)) if k != "corrXY" else np.nanmin(a), 4
            )
    return out


def roi_qc_thresholds() -> dict:
    """Calcium QC thresholds from :mod:`hm2p.calcium.qc` (as used by the Streamlit ROI viewer).

    Each entry gives the pass range ``[lo, hi]`` (None = unbounded).
    """
    from hm2p.calcium import qc

    return {
        "snr_event": [qc.SNR_MIN, None],
        "decay_tau_s": [qc.TAU_MIN_S, qc.TAU_MAX_S],
        "fneu_dff_corr": [None, qc.FNEU_CORR_MAX],
        "bleach_slope": [qc.BLEACH_MAX_LOSS, None],
        "active_fraction": [qc.ACTIVE_FRAC_MIN, None],
    }


def session_asset(
    ops: dict[str, Any],
    stat: list[dict[str, Any]],
    traces: dict[str, np.ndarray | None],
    trace_rois: npt.ArrayLike,
    fps: float,
    light_on: npt.ArrayLike | None = None,
    budget_bytes: int = TRACE_BUDGET_BYTES,
) -> dict:
    """Complete session asset (JSON-ready).

    Parameters
    ----------
    ops, stat
        Suite2p ``ops.npy`` (dict) and ``stat.npy`` (list).
    traces
        ``(n_rois, n_frames)`` arrays by name (:data:`TRACE_KEYS`), rows in
        ``stat`` order; None entries are skipped.
    trace_rois
        ROI indices whose traces are stored (the page states which).
    fps
        Imaging frame rate (Hz), for the time axis.
    light_on
        Per-imaging-frame light state, for dark-epoch shading.
    """
    from hm2p.qc.rois import image_png_b64

    # full resolution (never downscaled), so one PNG pixel is one imaging pixel
    images = {k: image_png_b64(v, max_side=max(v.shape)) for k, v in frame_images(ops).items()}
    sel = np.asarray(trace_rois, dtype=np.int64).ravel()
    present = {k: v for k, v in traces.items() if v is not None and np.ndim(v) == 2}
    n_frames = max((int(np.shape(v)[1]) for v in present.values()), default=0)
    k = choose_bin(n_frames, sel.size, len(present), budget_bytes)
    asset: dict = {
        "images": images,
        "image_shape": [int(ops["Ly"]), int(ops["Lx"])],
        "footprints": encode_footprints(stat),
        "traces": encode_traces(present, sel, k),
        "trace_rois": [int(i) for i in sel],
        "n_frames": n_frames,
        "bin": k,
        "fps": fnum(fps, 4),
        "registration": registration(ops),
    }
    if light_on is not None:
        asset["dark"] = binned_dark_intervals(light_on, k)
    return asset
