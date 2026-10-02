"""Egocentric coding measures: boundary vectors, head-body angle, wall distance.

Pure numpy/scipy functions (no I/O) for testing whether neurons encode the
position of maze walls relative to the animal (egocentric boundary vector
coding), the angle of the head relative to the body, or the distance to the
nearest wall.

Angle convention
----------------
All functions in this module that take a ``heading_deg`` or return a bearing
use the mathematical convention of the coordinate frame they are given:
0 deg points along +x and angles increase from +x towards +y. Egocentric
angles are measured from the heading in the same rotational sense, so an
egocentric angle of +90 deg is the direction obtained by rotating the heading
from +x towards +y by a quarter turn.

The pipeline ``hd_deg`` uses a different convention (see
``hm2p.kinematics.compute._vector_angle_deg``): for a facing vector with
pixel-frame angle ``phi = atan2(dy, dx)``, ``hd_deg = 180 - phi``. Convert
with :func:`hd_to_frame_heading` before passing pipeline head direction to
the functions below. Whether +90 deg egocentric is the animal's left or right
depends on whether the camera image is mirrored and is not assumed here.

References
----------
Alexander et al. 2020. "Egocentric boundary vector tuning of the
retrosplenial cortex." Science Advances 6:eaaz2322.
doi:10.1126/sciadv.aaz2322

Hinman et al. 2019. "Neuronal representation of environmental boundaries in
egocentric coordinates." Nature Communications 10:2772.
doi:10.1038/s41467-019-10722-y

Muller et al. 1987. "The effects of changes in the environment on the
spatial firing of hippocampal complex-spike cells." J Neurosci
7(7):1951-1968. doi:10.1523/JNEUROSCI.07-07-01951.1987 (circular-shift null)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import sparse, stats

from hm2p.maze.topology import RoseMaze, build_rose_maze

log = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

DEFAULT_MAX_DIST = 3.0
"""Default largest egocentric boundary distance (maze grid units)."""

DEFAULT_N_DIST_BINS = 12
"""Default number of distance bins between 0 and ``DEFAULT_MAX_DIST``."""

DEFAULT_FPS = 9.6
"""Imaging frame rate used to convert ``min_shift_s`` to frames when not given."""

_RAY_CHUNK = 2048
"""Frames per chunk in the vectorised ray/segment computations (memory bound)."""

_EPS = 1e-12

_SIDES = (
    # (dcol, drow), segment endpoints relative to cell origin (col, row)
    ((1, 0), (1.0, 0.0, 1.0, 1.0)),
    ((-1, 0), (0.0, 0.0, 0.0, 1.0)),
    ((0, 1), (0.0, 1.0, 1.0, 1.0)),
    ((0, -1), (0.0, 0.0, 1.0, 0.0)),
)


# ---------------------------------------------------------------------------
# Conventions and geometry
# ---------------------------------------------------------------------------


def wrap_deg(angle_deg: npt.ArrayLike) -> FloatArray:
    """Wrap angles to the half-open interval (-180, 180].

    Parameters
    ----------
    angle_deg : array_like
        Angles in degrees (NaN propagates).

    Returns
    -------
    ndarray of float64
        Wrapped angles.
    """
    a = np.asarray(angle_deg, dtype=np.float64)
    out = 180.0 - np.mod(180.0 - a, 360.0)
    return np.asarray(out, dtype=np.float64)


def hd_to_frame_heading(hd_deg: npt.ArrayLike) -> FloatArray:
    """Convert pipeline ``hd_deg`` to the mathematical heading of its frame.

    The pipeline computes heading from a facing vector ``(dx, dy)`` in pixel
    (and hence mm and maze) coordinates as ``180 + atan2(dx, dy) - 90``,
    which equals ``180 - atan2(dy, dx)``. Inverting gives
    ``phi = 180 - hd_deg``. ``x_maze``/``y_maze`` are an affine map of the same
    frame (TL->TR is +x, TL->BL is +y), so ``phi`` is also the heading in maze
    coordinates.

    Parameters
    ----------
    hd_deg : array_like
        Pipeline head direction in degrees (wrapped or unwrapped).

    Returns
    -------
    ndarray of float64
        Heading in degrees, wrapped to [0, 360); NaN preserved.
    """
    hd = np.asarray(hd_deg, dtype=np.float64)
    return np.asarray(np.mod(180.0 - hd, 360.0), dtype=np.float64)


def maze_wall_segments(maze: RoseMaze | None = None) -> FloatArray:
    """Wall segments of a grid maze in maze grid units.

    A wall is an edge of an accessible unit cell that is not shared with an
    adjacent cell in ``maze.adj``. Adjacency in ``hm2p.maze.topology`` is
    4-connectivity between accessible cells minus explicitly blocked edges,
    so this includes the outer boundary and internal walls between two
    accessible cells. A wall between two accessible cells is listed once.

    Parameters
    ----------
    maze : RoseMaze or None
        Maze topology; the q-rose maze is built when None.

    Returns
    -------
    ndarray, shape (n_walls, 4)
        Rows ``(x0, y0, x1, y1)``, sorted for reproducibility.
    """
    if maze is None:
        maze = build_rose_maze()
    segs: set[tuple[float, float, float, float]] = set()
    for cell in maze.cells:
        col, row = cell
        neighbours = set(maze.adj.get(cell, []))
        for (dc, dr), (ax, ay, bx, by) in _SIDES:
            if (col + dc, row + dr) in neighbours:
                continue
            segs.add((col + ax, row + ay, col + bx, row + by))
    if not segs:
        return np.empty((0, 4), dtype=np.float64)
    return np.array(sorted(segs), dtype=np.float64)


def _as_walls(walls: npt.ArrayLike) -> FloatArray:
    w = np.asarray(walls, dtype=np.float64)
    if w.ndim != 2 or w.shape[1] != 4:
        raise ValueError(f"walls must have shape (n_walls, 4), got {w.shape}")
    return w


def ray_wall_distances(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    heading_deg: npt.ArrayLike,
    walls: npt.ArrayLike,
    angles_deg: npt.ArrayLike,
) -> FloatArray:
    """Distance along egocentric rays to the nearest wall.

    For each frame the ray starts at ``(x, y)`` with direction
    ``heading + angle`` and the distance is the smallest positive ray
    parameter at which the ray intersects a wall segment (standard
    ray-segment intersection by 2-D cross products).

    Parameters
    ----------
    x, y : array_like, shape (n_frames,)
        Position.
    heading_deg : array_like, shape (n_frames,)
        Heading in the mathematical convention of the (x, y) frame.
    walls : array_like, shape (n_walls, 4)
        Wall segments ``(x0, y0, x1, y1)``.
    angles_deg : array_like, shape (n_angles,)
        Egocentric ray angles relative to the heading.

    Returns
    -------
    ndarray, shape (n_frames, n_angles)
        Distances in the units of (x, y). NaN when no wall is hit or any input
        for the frame is non-finite.
    """
    xs = np.asarray(x, dtype=np.float64).ravel()
    ys = np.asarray(y, dtype=np.float64).ravel()
    hd = np.asarray(heading_deg, dtype=np.float64).ravel()
    ang = np.asarray(angles_deg, dtype=np.float64).ravel()
    w = _as_walls(walls)
    if not (xs.size == ys.size == hd.size):
        raise ValueError("x, y and heading_deg must have the same length")
    n, m = xs.size, ang.size
    out = np.full((n, m), np.nan)
    if n == 0 or m == 0 or w.shape[0] == 0:
        return out
    ax, ay = w[:, 0], w[:, 1]
    ex, ey = w[:, 2] - ax, w[:, 3] - ay
    for start in range(0, n, _RAY_CHUNK):
        sl = slice(start, min(start + _RAY_CHUNK, n))
        theta = np.deg2rad(hd[sl, None] + ang[None, :])  # (c, m)
        dx, dy = np.cos(theta)[..., None], np.sin(theta)[..., None]  # (c, m, 1)
        px, py = xs[sl, None, None], ys[sl, None, None]
        qx, qy = ax[None, None, :] - px, ay[None, None, :] - py  # (c, 1, w)
        denom = dx * ey - dy * ex  # (c, m, w)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (qx * ey - qy * ex) / denom
            u = (qx * dy - qy * dx) / denom
        hit = (np.abs(denom) > _EPS) & (t > _EPS) & (u >= -_EPS) & (u <= 1.0 + _EPS)
        t = np.where(hit, t, np.inf)
        best = t.min(axis=2)
        best[~np.isfinite(best)] = np.nan
        out[sl] = best
    return out


def wall_distance(x: npt.ArrayLike, y: npt.ArrayLike, walls: npt.ArrayLike) -> FloatArray:
    """Euclidean distance from each position to the nearest wall segment.

    Parameters
    ----------
    x, y : array_like, shape (n_frames,)
        Position.
    walls : array_like, shape (n_walls, 4)
        Wall segments ``(x0, y0, x1, y1)``.

    Returns
    -------
    ndarray, shape (n_frames,)
        Distance in the units of (x, y); NaN where the position is non-finite
        or no walls are given.
    """
    xs = np.asarray(x, dtype=np.float64).ravel()
    ys = np.asarray(y, dtype=np.float64).ravel()
    w = _as_walls(walls)
    if xs.size != ys.size:
        raise ValueError("x and y must have the same length")
    out = np.full(xs.size, np.nan)
    if xs.size == 0 or w.shape[0] == 0:
        return out
    ax, ay = w[:, 0], w[:, 1]
    ex, ey = w[:, 2] - ax, w[:, 3] - ay
    len2 = ex * ex + ey * ey
    safe_len2 = np.where(len2 > 0, len2, 1.0)
    for start in range(0, xs.size, _RAY_CHUNK):
        sl = slice(start, min(start + _RAY_CHUNK, xs.size))
        qx = xs[sl, None] - ax[None, :]
        qy = ys[sl, None] - ay[None, :]
        s = np.clip((qx * ex + qy * ey) / safe_len2, 0.0, 1.0)
        s = np.where(len2 > 0, s, 0.0)
        d = np.hypot(qx - s * ex, qy - s * ey)
        out[sl] = d.min(axis=1)
    out[~(np.isfinite(xs) & np.isfinite(ys))] = np.nan
    return out


def head_body_angle(
    heading_deg: npt.ArrayLike,
    x_head: npt.ArrayLike,
    y_head: npt.ArrayLike,
    x_body: npt.ArrayLike,
    y_body: npt.ArrayLike,
) -> FloatArray:
    """Signed angle of the head direction relative to the body axis.

    ``heading - bearing(body -> head)``, wrapped to (-180, 180], where the
    bearing is ``atan2(y_head - y_body, x_head - x_body)``. Both angles are in
    the mathematical convention of the (x, y) frame, so positive values mean
    the head is rotated from the body axis towards +y of a +x-facing body.

    Parameters
    ----------
    heading_deg : array_like, shape (n_frames,)
        Head direction in the mathematical convention of the (x, y) frame
        (convert pipeline ``hd_deg`` with :func:`hd_to_frame_heading`).
    x_head, y_head, x_body, y_body : array_like, shape (n_frames,)
        Head and body positions in the same frame and units.

    Returns
    -------
    ndarray, shape (n_frames,)
        Angle in degrees; NaN where any input is non-finite or head and body
        coincide.
    """
    hd = np.asarray(heading_deg, dtype=np.float64)
    dx = np.asarray(x_head, dtype=np.float64) - np.asarray(x_body, dtype=np.float64)
    dy = np.asarray(y_head, dtype=np.float64) - np.asarray(y_body, dtype=np.float64)
    bearing = np.degrees(np.arctan2(dy, dx))
    out = wrap_deg(hd - bearing)
    bad = ~np.isfinite(hd) | ~np.isfinite(dx) | ~np.isfinite(dy) | (np.hypot(dx, dy) == 0)
    out[bad] = np.nan
    return out


# ---------------------------------------------------------------------------
# Shuffle machinery shared by all significance tests
# ---------------------------------------------------------------------------


def _shift_offsets(
    n_frames: int,
    n_shuffles: int,
    min_shift_frames: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.int64]:
    """Random circular-shift offsets in ``[min_shift, n_frames - min_shift]``.

    Falls back to ``[1, n_frames - 1]`` when the session is too short for the
    requested minimum shift.
    """
    lo = max(int(min_shift_frames), 1)
    hi = n_frames - lo
    if hi <= lo:
        lo, hi = 1, max(n_frames - 1, 2)
    return rng.integers(lo, hi, size=n_shuffles, endpoint=True).astype(np.int64)


def _shifted_stack(signal: FloatArray, offsets: npt.NDArray[np.int64]) -> FloatArray:
    """Column 0 is *signal*; column k+1 is ``np.roll(signal, offsets[k])``."""
    n = signal.size
    shifts = np.concatenate([[0], offsets])
    idx = (np.arange(n)[:, None] - shifts[None, :]) % n
    return np.asarray(signal[idx], dtype=np.float64)


def _min_shift_frames(min_shift_s: float, fps: float) -> int:
    return int(round(float(min_shift_s) * float(fps)))


def _indicator(bin_idx: npt.NDArray[np.int64], n_bins: int, n_frames: int) -> sparse.csr_matrix:
    """Sparse (n_bins, n_frames) indicator; entries with bin < 0 are dropped.

    ``bin_idx`` is (n_frames,) or (n_frames, k) for frames that fall in
    several bins (one per egocentric angle).
    """
    b = np.asarray(bin_idx).reshape(n_frames, -1)
    frames = np.broadcast_to(np.arange(n_frames)[:, None], b.shape)
    keep = b >= 0
    data = np.ones(int(keep.sum()), dtype=np.float64)
    return sparse.csr_matrix((data, (b[keep], frames[keep])), shape=(n_bins, n_frames))


def _null_summary(observed: float, null: FloatArray, two_sided: bool = False) -> dict[str, float]:
    """Conservative shuffle p-value, 95th percentile and z of *observed*."""
    null = null[np.isfinite(null)]
    if not np.isfinite(observed) or null.size == 0:
        return {"observed": float(observed), "null_95": np.nan, "p": np.nan, "z": np.nan}
    if two_sided:
        n_ge = int(np.sum(np.abs(null) >= abs(observed)))
        null_95 = float(np.percentile(np.abs(null), 95))
    else:
        n_ge = int(np.sum(null >= observed))
        null_95 = float(np.percentile(null, 95))
    sd = float(np.std(null))
    z = (observed - float(np.mean(null))) / sd if sd > 0 else np.nan
    return {
        "observed": float(observed),
        "null_95": null_95,
        "p": (n_ge + 1) / (null.size + 1),
        "z": float(z),
    }


# ---------------------------------------------------------------------------
# Egocentric boundary vector ratemaps
# ---------------------------------------------------------------------------


def default_dist_edges() -> FloatArray:
    """Distance bin edges 0..3 grid units in 0.25-unit steps."""
    return np.linspace(0.0, DEFAULT_MAX_DIST, DEFAULT_N_DIST_BINS + 1)


def ebc_angle_centres(n_angle_bins: int = 24) -> FloatArray:
    """Egocentric angle bin centres in [0, 360) (degrees)."""
    width = 360.0 / n_angle_bins
    return np.asarray(np.arange(n_angle_bins) * width + width / 2.0, dtype=np.float64)


def _ebc_bins(
    ray_dist: FloatArray,
    valid: BoolArray,
    dist_edges: FloatArray,
) -> npt.NDArray[np.int64]:
    """Flat (angle, distance) bin per frame and ray; -1 when outside/invalid."""
    n_frames, n_ang = ray_dist.shape
    n_dist = dist_edges.size - 1
    d = np.where(np.isfinite(ray_dist), ray_dist, -1.0)
    dbin = np.searchsorted(dist_edges, d, side="right") - 1
    ok = (dbin >= 0) & (dbin < n_dist) & np.isfinite(ray_dist) & valid[:, None]
    flat = np.arange(n_ang)[None, :] * n_dist + dbin
    return np.where(ok, flat, -1).astype(np.int64)


def _ebc_setup(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    heading_deg: npt.ArrayLike,
    walls: npt.ArrayLike,
    mask: npt.ArrayLike,
    n_angle_bins: int,
    dist_edges: npt.ArrayLike | None,
    ray_dist: FloatArray | None,
) -> tuple[sparse.csr_matrix, FloatArray, FloatArray, FloatArray, int]:
    """Indicator matrix, occupancy, bin centres and valid-frame count for a ratemap."""
    edges = default_dist_edges() if dist_edges is None else np.asarray(dist_edges, float)
    angles = ebc_angle_centres(n_angle_bins)
    xs = np.asarray(x, dtype=np.float64).ravel()
    ys = np.asarray(y, dtype=np.float64).ravel()
    hd = np.asarray(heading_deg, dtype=np.float64).ravel()
    m = np.asarray(mask, dtype=bool).ravel()
    if ray_dist is None:
        ray_dist = ray_wall_distances(xs, ys, hd, walls, angles)
    elif ray_dist.shape != (xs.size, n_angle_bins):
        raise ValueError(f"ray_dist shape {ray_dist.shape} != {(xs.size, n_angle_bins)}")
    valid = m & np.isfinite(xs) & np.isfinite(ys) & np.isfinite(hd)
    bins = _ebc_bins(ray_dist, valid, edges)
    n_dist = edges.size - 1
    ind = _indicator(bins, n_angle_bins * n_dist, xs.size)
    occ = np.asarray(ind.sum(axis=1)).ravel().reshape(n_angle_bins, n_dist)
    dist_centres = (edges[:-1] + edges[1:]) / 2.0
    return ind, occ, angles, dist_centres, int(valid.sum())


def _ratemaps_from_sums(sums: FloatArray, occ: FloatArray, min_occ: int) -> FloatArray:
    """(n_bins, k) sums / occupancy, NaN below *min_occ*; returns (k, n_ang, n_dist)."""
    n_ang, n_dist = occ.shape
    flat_occ = occ.ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        rm = sums / flat_occ[:, None]
    rm[flat_occ < max(min_occ, 1)] = np.nan
    return np.asarray(rm.T.reshape(-1, n_ang, n_dist), dtype=np.float64)


def egocentric_boundary_ratemap(
    signal: npt.ArrayLike,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    heading_deg: npt.ArrayLike,
    walls: npt.ArrayLike,
    mask: npt.ArrayLike,
    n_angle_bins: int = 24,
    dist_edges: npt.ArrayLike | None = None,
    min_occupancy_frames: int = 5,
    ray_dist: FloatArray | None = None,
) -> dict[str, Any]:
    """Egocentric boundary ratemap (angle x distance) of a neural signal.

    For each valid frame and each egocentric angle bin, the distance to the
    nearest wall along the ray at the bin centre is assigned to a distance
    bin; occupancy (frames) and summed signal are accumulated per
    (angle, distance) bin and the ratemap is their ratio. This follows the
    egocentric boundary ratemap of Alexander et al. 2020 and Hinman et al.
    2019, without spatial smoothing.

    Alexander et al. 2020. "Egocentric boundary vector tuning of the
    retrosplenial cortex." Science Advances 6:eaaz2322.
    doi:10.1126/sciadv.aaz2322

    Hinman et al. 2019. "Neuronal representation of environmental boundaries
    in egocentric coordinates." Nature Communications 10:2772.
    doi:10.1038/s41467-019-10722-y

    Parameters
    ----------
    signal : array_like, shape (n_frames,)
        Neural signal (dF/F, event mask or spike rate).
    x, y : array_like, shape (n_frames,)
        Position (maze grid units for the default distance edges).
    heading_deg : array_like, shape (n_frames,)
        Heading, mathematical convention of the (x, y) frame.
    walls : array_like, shape (n_walls, 4)
        Wall segments.
    mask : array_like of bool, shape (n_frames,)
        Frames to include (non-finite position/heading/signal also excluded).
    n_angle_bins : int
        Number of egocentric angle bins covering 360 deg.
    dist_edges : array_like or None
        Distance bin edges; default 0..3 units in 0.25 steps.
    min_occupancy_frames : int
        Bins with fewer frames are NaN in the ratemap.
    ray_dist : ndarray or None
        Precomputed ``ray_wall_distances(x, y, heading, walls,
        ebc_angle_centres(n_angle_bins))`` to avoid recomputation.

    Returns
    -------
    dict
        ``ratemap`` (n_angle_bins, n_dist), ``occupancy`` (frames, same
        shape), ``angle_centres`` (deg), ``dist_centres``.
    """
    sig = np.asarray(signal, dtype=np.float64).ravel()
    m = np.asarray(mask, dtype=bool).ravel() & np.isfinite(sig)
    ind, occ, angles, dists, _ = _ebc_setup(
        x, y, heading_deg, walls, m, n_angle_bins, dist_edges, ray_dist
    )
    sums = np.asarray(ind @ np.where(np.isfinite(sig), sig, 0.0)).reshape(-1, 1)
    rm = _ratemaps_from_sums(sums, occ, min_occupancy_frames)[0]
    return {"ratemap": rm, "occupancy": occ, "angle_centres": angles, "dist_centres": dists}


def ebc_score(
    ratemap: npt.ArrayLike,
    angle_centres: npt.ArrayLike,
    dist_centres: npt.ArrayLike | None = None,
) -> dict[str, float]:
    """Mean resultant length of an egocentric boundary ratemap.

    The ratemap is collapsed over distance by taking the maximum in each
    angle bin (Alexander et al. 2020); the angular profile is rectified
    (negative dF/F bins clipped to 0, as in ``hm2p.analysis.tuning``) and its
    mean resultant length and direction are computed. The preferred distance
    is the distance bin with the largest value in the angle bin nearest the
    preferred angle.

    Alexander et al. 2020. "Egocentric boundary vector tuning of the
    retrosplenial cortex." Science Advances 6:eaaz2322.
    doi:10.1126/sciadv.aaz2322

    Parameters
    ----------
    ratemap : array_like, shape (n_angle_bins, n_dist)
        Ratemap (may contain NaN for unvisited bins).
    angle_centres : array_like, shape (n_angle_bins,)
        Egocentric angle bin centres (deg).
    dist_centres : array_like or None
        Distance bin centres; ``preferred_distance`` is the bin index as float
        when None.

    Returns
    -------
    dict
        ``mrl`` in [0, 1], ``preferred_angle`` in (-180, 180] and
        ``preferred_distance``; all NaN when the profile has no positive mass.
    """
    rm = np.asarray(ratemap, dtype=np.float64)
    ang = np.asarray(angle_centres, dtype=np.float64)
    nan_res = {"mrl": np.nan, "preferred_angle": np.nan, "preferred_distance": np.nan}
    if rm.ndim != 2 or rm.shape[0] != ang.size or rm.size == 0:
        return nan_res
    has = np.isfinite(rm).any(axis=1)
    if not has.any():
        return nan_res
    profile = np.full(ang.size, np.nan)
    profile[has] = np.nanmax(rm[has], axis=1)
    w = np.clip(profile[has], 0.0, None)
    total = float(w.sum())
    if total <= 0:
        return nan_res
    theta = np.deg2rad(ang[has])
    resultant = np.sum(w * np.exp(1j * theta)) / total
    pref = float(np.degrees(np.angle(resultant)))
    diff = np.abs(wrap_deg(ang - pref))
    diff[~has] = np.inf
    a_idx = int(np.argmin(diff))
    d_idx = int(np.nanargmax(rm[a_idx]))
    if dist_centres is None:
        pref_d = float(d_idx)
    else:
        pref_d = float(np.asarray(dist_centres, dtype=np.float64)[d_idx])
    return {
        "mrl": float(np.abs(resultant)),
        "preferred_angle": float(wrap_deg(pref)),
        "preferred_distance": pref_d,
    }


def ebc_significance(
    signal: npt.ArrayLike,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    heading_deg: npt.ArrayLike,
    walls: npt.ArrayLike,
    mask: npt.ArrayLike,
    n_shuffles: int = 200,
    min_shift_s: float = 10.0,
    fps: float = DEFAULT_FPS,
    rng: np.random.Generator | None = None,
    n_angle_bins: int = 24,
    dist_edges: npt.ArrayLike | None = None,
    min_occupancy_frames: int = 5,
    ray_dist: FloatArray | None = None,
) -> dict[str, Any]:
    """EBC mean resultant length with a circular-shift null.

    The signal is circularly shifted relative to the behaviour by at least
    ``min_shift_s`` seconds (Muller et al. 1987), the ratemap and MRL are
    recomputed for each shift, and the conservative p-value
    ``(#null >= observed + 1) / (n_shuffles + 1)`` is returned. Occupancy is
    fixed across shuffles, so all shuffles are evaluated with one sparse
    matrix product.

    Alexander et al. 2020. "Egocentric boundary vector tuning of the
    retrosplenial cortex." Science Advances 6:eaaz2322.
    doi:10.1126/sciadv.aaz2322

    Parameters
    ----------
    signal, x, y, heading_deg, walls, mask, n_angle_bins, dist_edges,
    min_occupancy_frames, ray_dist
        As in :func:`egocentric_boundary_ratemap`.
    n_shuffles : int
        Number of circular shifts.
    min_shift_s : float
        Minimum shift in seconds.
    fps : float
        Frame rate used to convert ``min_shift_s`` to frames.
    rng : numpy.random.Generator or None
        Random generator.

    Returns
    -------
    dict
        ``observed`` (MRL), ``null_95``, ``p``, ``z``, ``preferred_angle``,
        ``preferred_distance``, ``ratemap``, ``angle_centres``,
        ``dist_centres``, ``n_frames``.
    """
    rng = np.random.default_rng() if rng is None else rng
    sig = np.asarray(signal, dtype=np.float64).ravel()
    m = np.asarray(mask, dtype=bool).ravel() & np.isfinite(sig)
    ind, occ, angles, dists, n_valid = _ebc_setup(
        x, y, heading_deg, walls, m, n_angle_bins, dist_edges, ray_dist
    )
    n_frames = sig.size
    offsets = _shift_offsets(n_frames, n_shuffles, _min_shift_frames(min_shift_s, fps), rng)
    stack = _shifted_stack(np.where(np.isfinite(sig), sig, 0.0), offsets)
    sums = np.asarray(ind @ stack)
    rms = _ratemaps_from_sums(sums, occ, min_occupancy_frames)
    scores = [ebc_score(rms[k], angles, dists) for k in range(rms.shape[0])]
    mrls = np.array([s["mrl"] for s in scores], dtype=np.float64)
    res: dict[str, Any] = _null_summary(float(mrls[0]), mrls[1:])
    res.update(
        {
            "preferred_angle": scores[0]["preferred_angle"],
            "preferred_distance": scores[0]["preferred_distance"],
            "ratemap": rms[0],
            "angle_centres": angles,
            "dist_centres": dists,
            "n_frames": n_valid,
        }
    )
    return res


# ---------------------------------------------------------------------------
# 1-D tuning: signed angles (head-body angle) and wall distance
# ---------------------------------------------------------------------------


def _angle_bins(angle_deg: FloatArray, valid: BoolArray, n_bins: int) -> npt.NDArray[np.int64]:
    edges = np.linspace(-180.0, 180.0, n_bins + 1)
    a = wrap_deg(angle_deg)
    b = np.searchsorted(edges, np.where(np.isfinite(a), a, 0.0), side="left") - 1
    b = np.clip(b, 0, n_bins - 1)
    return np.where(valid & np.isfinite(a), b, -1).astype(np.int64)


def _modulation(curve: FloatArray, sd: float) -> float:
    if not np.isfinite(curve).any() or not np.isfinite(sd) or sd <= 0:
        return np.nan
    return float((np.nanmax(curve) - np.nanmin(curve)) / sd)


def _curve_rho(curve: FloatArray, centres: FloatArray) -> float:
    ok = np.isfinite(curve)
    if ok.sum() < 3 or np.ptp(curve[ok]) == 0:
        return np.nan
    return float(stats.spearmanr(centres[ok], curve[ok]).statistic)


def angle_tuning(
    signal: npt.ArrayLike,
    angle_deg: npt.ArrayLike,
    mask: npt.ArrayLike,
    n_bins: int = 18,
    min_occupancy_frames: int = 5,
) -> dict[str, Any]:
    """Tuning curve of a signal to a signed angle in (-180, 180].

    ``modulation_index = (max(curve) - min(curve)) / SD(signal)`` over the
    included frames: the range of the binned mean in units of the signal's
    standard deviation, which is independent of the dF/F baseline offset.
    ``slope_sign`` is the sign of the Spearman rank correlation between bin
    centre and binned mean (+1: signal increases towards positive angles).

    Parameters
    ----------
    signal : array_like, shape (n_frames,)
        Neural signal.
    angle_deg : array_like, shape (n_frames,)
        Signed angle per frame (wrapped internally).
    mask : array_like of bool, shape (n_frames,)
        Frames to include.
    n_bins : int
        Number of equal-width bins over (-180, 180].
    min_occupancy_frames : int
        Bins with fewer frames are NaN.

    Returns
    -------
    dict
        ``curve``, ``centres``, ``occupancy``, ``modulation_index``,
        ``slope_rho``, ``slope_sign`` (-1, 0, 1; 0 when undefined).
    """
    sig = np.asarray(signal, dtype=np.float64).ravel()
    a = np.asarray(angle_deg, dtype=np.float64).ravel()
    valid = np.asarray(mask, dtype=bool).ravel() & np.isfinite(sig)
    bins = _angle_bins(a, valid, n_bins)
    keep = bins >= 0
    occ = np.bincount(bins[keep], minlength=n_bins).astype(np.float64)
    sums = np.bincount(bins[keep], weights=sig[keep], minlength=n_bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        curve = sums / occ
    curve[occ < max(min_occupancy_frames, 1)] = np.nan
    width = 360.0 / n_bins
    centres = np.asarray(-180.0 + width / 2.0 + width * np.arange(n_bins), dtype=np.float64)
    sd = float(np.std(sig[keep])) if keep.any() else np.nan
    rho = _curve_rho(curve, centres)
    return {
        "curve": curve,
        "centres": centres,
        "occupancy": occ,
        "modulation_index": _modulation(curve, sd),
        "slope_rho": rho,
        "slope_sign": int(np.sign(rho)) if np.isfinite(rho) else 0,
    }


def angle_tuning_significance(
    signal: npt.ArrayLike,
    angle_deg: npt.ArrayLike,
    mask: npt.ArrayLike,
    n_bins: int = 18,
    n_shuffles: int = 200,
    min_shift_s: float = 10.0,
    fps: float = DEFAULT_FPS,
    rng: np.random.Generator | None = None,
    min_occupancy_frames: int = 5,
) -> dict[str, Any]:
    """Circular-shift significance of the angle-tuning modulation index.

    The signal is circularly shifted by at least ``min_shift_s`` seconds
    (Muller et al. 1987) and :func:`angle_tuning`'s modulation index is
    recomputed. One-sided conservative p-value and z against the null.

    Parameters
    ----------
    signal, angle_deg, mask, n_bins, min_occupancy_frames
        As in :func:`angle_tuning`.
    n_shuffles : int
        Number of shifts.
    min_shift_s, fps : float
        Minimum shift (s) and frame rate.
    rng : numpy.random.Generator or None
        Random generator.

    Returns
    -------
    dict
        ``observed``, ``null_95``, ``p``, ``z`` plus the observed tuning
        (``curve``, ``centres``, ``slope_rho``, ``slope_sign``).
    """
    rng = np.random.default_rng() if rng is None else rng
    sig = np.asarray(signal, dtype=np.float64).ravel()
    a = np.asarray(angle_deg, dtype=np.float64).ravel()
    m = np.asarray(mask, dtype=bool).ravel()
    obs = angle_tuning(sig, a, m, n_bins, min_occupancy_frames)
    offsets = _shift_offsets(sig.size, n_shuffles, _min_shift_frames(min_shift_s, fps), rng)
    stack = _shifted_stack(sig, offsets)[:, 1:]
    finite = np.isfinite(stack)
    valid = m & np.isfinite(a)
    bins = _angle_bins(a, valid, n_bins)
    ind = _indicator(bins, n_bins, sig.size)
    stack0 = np.where(finite, stack, 0.0)
    occ = np.asarray(ind @ finite.astype(np.float64))
    sums = np.asarray(ind @ stack0)
    with np.errstate(divide="ignore", invalid="ignore"):
        curves = sums / occ
    curves[occ < max(min_occupancy_frames, 1)] = np.nan
    sel = valid[:, None] & finite
    null = np.full(stack.shape[1], np.nan)
    for k in range(stack.shape[1]):
        col = stack[sel[:, k], k]
        sd = float(np.std(col)) if col.size else np.nan
        null[k] = _modulation(curves[:, k], sd)
    res: dict[str, Any] = _null_summary(obs["modulation_index"], null)
    res.update({k: obs[k] for k in ("curve", "centres", "slope_rho", "slope_sign")})
    return res


def _rank_columns(a: FloatArray) -> FloatArray:
    return np.asarray(stats.rankdata(a, axis=0), dtype=np.float64)


def _col_corr(r: FloatArray, v: FloatArray) -> FloatArray:
    """Pearson correlation of each column of *r* with vector *v* (ranks in, Spearman out)."""
    rc = r - r.mean(axis=0, keepdims=True)
    vc = v - v.mean()
    num = vc @ rc
    den = np.sqrt((rc * rc).sum(axis=0) * float(vc @ vc))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    return np.asarray(np.where(den > 0, out, np.nan), dtype=np.float64)


def distance_tuning(
    signal: npt.ArrayLike,
    d: npt.ArrayLike,
    mask: npt.ArrayLike,
    n_bins: int = 10,
    n_shuffles: int = 200,
    min_shift_s: float = 10.0,
    fps: float = DEFAULT_FPS,
    rng: np.random.Generator | None = None,
    min_occupancy_frames: int = 5,
) -> dict[str, Any]:
    """Tuning of a signal to a scalar distance, with Spearman rho and shuffle p.

    The curve is the mean signal in ``n_bins`` equal-width bins over the
    range of ``d`` in included frames. ``rho`` is the frame-level Spearman
    rank correlation between ``d`` and the signal; ``p`` is two-sided against
    a circular-shift null of rho (shifts of at least ``min_shift_s`` s; the
    analytic Spearman p-value would ignore autocorrelation).

    Parameters
    ----------
    signal : array_like, shape (n_frames,)
        Neural signal.
    d : array_like, shape (n_frames,)
        Distance per frame (e.g. :func:`wall_distance`).
    mask : array_like of bool, shape (n_frames,)
        Frames to include.
    n_bins : int
        Number of distance bins for the curve.
    n_shuffles : int
        Number of circular shifts (0 gives ``p = NaN``).
    min_shift_s, fps : float
        Minimum shift (s) and frame rate.
    rng : numpy.random.Generator or None
        Random generator.
    min_occupancy_frames : int
        Bins with fewer frames are NaN.

    Returns
    -------
    dict
        ``curve``, ``centres``, ``rho``, ``p``, ``null_95`` (of |rho|),
        ``n_frames``.
    """
    rng = np.random.default_rng() if rng is None else rng
    sig = np.asarray(signal, dtype=np.float64).ravel()
    dd = np.asarray(d, dtype=np.float64).ravel()
    m = np.asarray(mask, dtype=bool).ravel() & np.isfinite(dd)
    valid = m & np.isfinite(sig)
    n_valid = int(valid.sum())
    curve = np.full(n_bins, np.nan)
    centres = np.full(n_bins, np.nan)
    out: dict[str, Any] = {
        "curve": curve,
        "centres": centres,
        "rho": np.nan,
        "p": np.nan,
        "null_95": np.nan,
        "n_frames": n_valid,
    }
    if n_valid < 3:
        return out
    lo, hi = float(dd[valid].min()), float(dd[valid].max())
    edges = np.linspace(lo, hi if hi > lo else lo + 1.0, n_bins + 1)
    out["centres"] = (edges[:-1] + edges[1:]) / 2.0
    b = np.clip(np.searchsorted(edges, dd[valid], side="right") - 1, 0, n_bins - 1)
    occ = np.bincount(b, minlength=n_bins).astype(np.float64)
    sums = np.bincount(b, weights=sig[valid], minlength=n_bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        curve = sums / occ
    curve[occ < max(min_occupancy_frames, 1)] = np.nan
    out["curve"] = curve

    offsets = _shift_offsets(sig.size, n_shuffles, _min_shift_frames(min_shift_s, fps), rng)
    if n_shuffles <= 0:
        offsets = offsets[:0]
    # Frames used are those valid for the *observed* signal; shifted copies
    # are evaluated on frames where both d and the shifted signal are finite.
    stack = _shifted_stack(sig, offsets)[m]
    dm = dd[m]
    rhos = np.full(stack.shape[1], np.nan)
    all_finite = np.isfinite(stack).all(axis=0)
    if all_finite.any():
        cols = np.flatnonzero(all_finite)
        rhos[cols] = _col_corr(_rank_columns(stack[:, cols]), _rank_columns(dm[:, None])[:, 0])
    for k in np.flatnonzero(~all_finite):
        ok = np.isfinite(stack[:, k])
        if ok.sum() >= 3:
            rhos[k] = _col_corr(
                _rank_columns(stack[ok, k : k + 1]), _rank_columns(dm[ok, None])[:, 0]
            )[0]
    out["rho"] = float(rhos[0])
    summ = _null_summary(float(rhos[0]), rhos[1:], two_sided=True)
    out["p"] = summ["p"]
    out["null_95"] = summ["null_95"]
    return out
