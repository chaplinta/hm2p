"""Per-syllable visualisations: grid movies, average pose trajectories, occupancy.

Pure functions used by ``scripts/make_qc_reports.py`` to show what each
keypoint-MoSeq syllable looks like:

* **grid movie** - up to nine example bouts of one syllable, each cropped
  around the animal, rotated so that the body axis (tail base -> neck) at bout
  onset points up, and tiled 3 x 3; the tile border is drawn while the bout
  is in progress;
* **average trajectory** - keypoint positions in the egocentric frame of the
  bout onset (body centre at the origin, body axis along +y), averaged over
  all bouts;
* **occupancy** - where in the maze the syllable is used.

These follow the grid movies and typical-trajectory plots of keypoint-MoSeq:
Weinreb C, Pearl JE, Lin S, et al. 2024. "Keypoint-MoSeq: parsing behavior
by linking point tracking to pose dynamics." Nature Methods 21:1329-1339.
doi:10.1038/s41592-024-02318-2. https://github.com/dattalab/keypoint-moseq

Coordinate conventions
----------------------
Image (pixel) coordinates have y pointing down. The egocentric frame used
for trajectories has y pointing forward (along the body axis) and x pointing
to the animal's right when seen from above, so it can be plotted with y up.
Angles (``heading``) are measured in the image frame, ``atan2(dy, dx)``.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from hm2p.qc.common import run_lengths

TRAJ_KEYPOINTS = (
    "nose_tip",
    "left_ear",
    "right_ear",
    "head_midpoint",
    "neck",
    "mid_back",
    "mouse_center",
    "tail_base",
)
KEYPOINT_ALIASES = {"head_midpoint": ("implant_base_rear",)}
SKELETON = (
    ("nose_tip", "left_ear"),
    ("nose_tip", "right_ear"),
    ("left_ear", "head_midpoint"),
    ("right_ear", "head_midpoint"),
    ("head_midpoint", "neck"),
    ("neck", "mid_back"),
    ("mid_back", "mouse_center"),
    ("mouse_center", "tail_base"),
)
GRID_ROWS = 3
GRID_COLS = 3
TILE_PX = 120
CLIP_PRE = 10
CLIP_POST = 10
CLIP_MAX_FRAMES = 75
TRAJ_PRE = 5
TRAJ_POST = 30
BORDER_PX = 4
BORDER_BGR = (0, 140, 255)


# ---------------------------------------------------------------------------
# Bouts
# ---------------------------------------------------------------------------


def syllable_bouts(syllable_id: npt.ArrayLike) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Bout onsets and lengths per syllable: ``{id: (starts, lengths)}``.

    Negative ids (unassigned frames) are skipped.
    """
    vals, starts, lens = run_lengths(np.asarray(syllable_id).astype(np.int64))
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for k in np.unique(vals):
        if k < 0:
            continue
        m = vals == k
        out[int(k)] = (starts[m], lens[m])
    return out


def window_ok(
    first: npt.ArrayLike,
    last: npt.ArrayLike,
    bad: npt.ArrayLike,
    tracked: npt.ArrayLike,
    min_tracked_frac: float = 0.9,
    n_max: int | None = None,
) -> np.ndarray:
    """Which frame windows ``[first, last)`` are usable as examples.

    A window is usable when it lies inside ``[0, n_max)`` (default: the
    length of *bad*), contains no frame flagged in *bad* (e.g. the animal
    stuck on the fibre), and at least *min_tracked_frac* of its frames are
    flagged in *tracked*.
    """
    f = np.asarray(first, dtype=np.int64)
    la = np.asarray(last, dtype=np.int64)
    b = np.asarray(bad, dtype=bool)
    t = np.asarray(tracked, dtype=bool)
    n = b.size if n_max is None else min(int(n_max), b.size)
    cb = np.concatenate([[0], np.cumsum(b)])
    ct = np.concatenate([[0], np.cumsum(t)])
    inside = (f >= 0) & (la <= n) & (la > f)
    fi, li = np.clip(f, 0, b.size), np.clip(la, 0, b.size)
    n_bad = cb[li] - cb[fi]
    width = np.maximum(li - fi, 1)
    frac = (ct[li] - ct[fi]) / width
    return inside & (n_bad == 0) & (frac >= min_tracked_frac)


def select_bouts(
    candidates: Sequence[tuple[str, int, int]],
    target_len: float,
    n: int = GRID_ROWS * GRID_COLS,
    rng: np.random.Generator | None = None,
    tol_frac: float = 0.25,
) -> list[tuple[str, int, int]]:
    """Pick up to *n* example bouts, near the typical duration, from different sessions.

    Parameters
    ----------
    candidates
        ``(session, start, length)`` for every usable bout of one syllable.
    target_len
        Typical (median) bout length in frames.
    n
        Number of bouts to return.
    rng
        Random generator (seed it for reproducible selections).
    tol_frac
        Bouts whose length is within ``max(1, tol_frac * target_len)`` frames
        of *target_len* are preferred; if fewer than *n* such bouts exist,
        the closest remaining bouts are added.

    Returns
    -------
    list
        Selected bouts, ordered so that consecutive picks come from different
        sessions as long as possible (one per session first, then a second
        round, ...).
    """
    if not candidates:
        return []
    rng = np.random.default_rng(0) if rng is None else rng
    lens = np.array([c[2] for c in candidates], dtype=np.float64)
    dist = np.abs(lens - target_len)
    tol = max(1.0, tol_frac * target_len)
    jitter = rng.random(len(candidates))
    near = dist <= tol
    if near.sum() >= n:
        order = np.flatnonzero(near)[np.argsort(jitter[near])]
    else:  # closest first, random among ties
        order = np.lexsort((jitter, dist))
    pool = [candidates[i] for i in order]
    by_ses: dict[str, list[tuple[str, int, int]]] = {}
    for c in pool:
        by_ses.setdefault(c[0], []).append(c)
    sessions = list(by_ses)
    sessions = [sessions[i] for i in rng.permutation(len(sessions))]
    out: list[tuple[str, int, int]] = []
    depth = 0
    while len(out) < n and any(len(by_ses[s]) > depth for s in sessions):
        for s in sessions:
            if len(by_ses[s]) > depth and len(out) < n:
                out.append(by_ses[s][depth])
        depth += 1
    return out


def clip_window(
    start: int,
    length: int,
    pre: int = CLIP_PRE,
    post: int = CLIP_POST,
    max_frames: int = CLIP_MAX_FRAMES,
) -> tuple[int, int, int, int]:
    """Frames to cut for one example bout.

    Returns ``(first, last, bout_from, bout_to)``: the clip covers frames
    ``[first, last)`` of the session, and the bout occupies clip frames
    ``[bout_from, bout_to)``. Bouts longer than ``max_frames - pre - post``
    are shown truncated.
    """
    shown = max(1, min(int(length), max_frames - pre - post))
    first = int(start) - pre
    return first, int(start) + shown + post, pre, pre + shown


# ---------------------------------------------------------------------------
# Egocentric alignment
# ---------------------------------------------------------------------------


def heading_angle(tail_xy: npt.ArrayLike, head_xy: npt.ArrayLike) -> np.ndarray:
    """Image-frame angle (rad) of the vector from *tail_xy* to *head_xy*, shape (n,)."""
    t = np.asarray(tail_xy, dtype=np.float64)
    h = np.asarray(head_xy, dtype=np.float64)
    d = h - t
    return np.arctan2(d[..., 1], d[..., 0])


def egocentric_points(
    points: npt.ArrayLike,
    centre: npt.ArrayLike,
    heading: float,
    mirror: bool = False,
) -> np.ndarray:
    """Image-frame points (..., 2) in the egocentric frame of (*centre*, *heading*).

    The output has the centre at the origin, the heading along +y and the
    animal's right along +x (seen from above, y up). ``mirror`` flips x, for
    videos whose image is mirrored.
    """
    p = np.asarray(points, dtype=np.float64) - np.asarray(centre, dtype=np.float64)
    c, s = np.cos(heading), np.sin(heading)
    fwd = p[..., 0] * c + p[..., 1] * s  # along heading
    lat = -p[..., 0] * s + p[..., 1] * c  # 90 deg clockwise of heading in the image = right
    x = -lat if mirror else lat
    return np.stack([x, fwd], axis=-1)


def tile_affine(
    centre: npt.ArrayLike,
    heading: float,
    src_px: float,
    tile_px: int = TILE_PX,
    mirror: bool = False,
) -> np.ndarray:
    """2 x 3 affine (for ``cv2.warpAffine``) from image to an egocentric tile.

    A square of side *src_px* image pixels around *centre* is mapped onto a
    *tile_px* square, with the heading pointing up and the animal's right to
    the right (unless *mirror*).
    """
    s = tile_px / float(src_px)
    c, sn = np.cos(heading), np.sin(heading)
    # egocentric (x right, y forward) = R @ (p - centre); tile = s * (x, -y) + tile_px/2
    r = np.array([[-sn, c], [c, sn]])  # rows: lateral (right), forward
    if mirror:
        r[0] = -r[0]
    a = np.array([[s, 0.0], [0.0, -s]]) @ r
    cxy = np.asarray(centre, dtype=np.float64)
    b = np.array([tile_px / 2.0, tile_px / 2.0]) - a @ cxy
    return np.hstack([a, b[:, None]])


def ego_windows(
    pos: npt.ArrayLike,
    centre: npt.ArrayLike,
    heading: npt.ArrayLike,
    onsets: npt.ArrayLike,
    offsets: npt.ArrayLike,
    mirror: bool = False,
) -> np.ndarray:
    """Keypoints around each onset, in the egocentric frame of that onset.

    Parameters
    ----------
    pos
        (n, K, 2) image-frame keypoint positions.
    centre, heading
        (n, 2) body centre and (n,) heading per frame.
    onsets
        (B,) onset frames; ``onsets + offsets`` must be valid frame indices.
    offsets
        (T,) frame offsets relative to onset (e.g. ``range(-5, 31)``).

    Returns
    -------
    numpy.ndarray
        (B, T, K, 2) egocentric positions (same units as *pos*).
    """
    p = np.asarray(pos, dtype=np.float64)
    c = np.asarray(centre, dtype=np.float64)
    h = np.asarray(heading, dtype=np.float64)
    on = np.asarray(onsets, dtype=np.int64)
    off = np.asarray(offsets, dtype=np.int64)
    if on.size == 0:
        return np.zeros((0, off.size, p.shape[1], 2))
    idx = on[:, None] + off[None, :]
    win = p[idx] - c[on][:, None, None, :]  # (B, T, K, 2)
    hh = h[on][:, None, None]
    cs, sn = np.cos(hh), np.sin(hh)
    fwd = win[..., 0] * cs + win[..., 1] * sn
    lat = -win[..., 0] * sn + win[..., 1] * cs
    return np.stack([-lat if mirror else lat, fwd], axis=-1)


def trajectory_sums(windows: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """NaN-aware sum and count over bouts of (B, T, K, 2) windows -> (T, K, 2), (T, K)."""
    w = np.asarray(windows, dtype=np.float64)
    ok = np.isfinite(w).all(axis=-1)
    return np.where(ok[..., None], w, 0.0).sum(axis=0), ok.sum(axis=0)


def mean_trajectory(sums: npt.ArrayLike, counts: npt.ArrayLike) -> np.ndarray:
    """Mean (T, K, 2) from :func:`trajectory_sums` outputs; NaN where count is 0."""
    s = np.asarray(sums, dtype=np.float64)
    n = np.asarray(counts, dtype=np.float64)[..., None]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / np.where(n > 0, n, 1.0), np.nan)


def left_on_left_frac(left_ego_x: npt.ArrayLike, right_ego_x: npt.ArrayLike) -> float | None:
    """Fraction of frames where the left ear lies left of the right ear in the egocentric frame.

    About 1 for a correctly oriented (non-mirrored) video; about 0 if the
    image is mirrored.
    """
    lx = np.asarray(left_ego_x, dtype=np.float64)
    rx = np.asarray(right_ego_x, dtype=np.float64)
    ok = np.isfinite(lx) & np.isfinite(rx) & (lx != rx)
    return float(np.mean(lx[ok] < rx[ok])) if ok.any() else None


def signed_turn_rate(heading: npt.ArrayLike, fps: float) -> np.ndarray:
    """Angular velocity (deg/s) of an image-frame heading; + = counter-clockwise seen from above.

    Image y points down, so a counter-clockwise turn (a left turn) decreases
    the image angle; the sign is flipped to give the usual convention.
    """
    h = np.asarray(heading, dtype=np.float64)
    out = np.full(h.shape, np.nan)
    ok = np.isfinite(h)
    if ok.sum() < 2:
        return out
    idx = np.flatnonzero(ok)
    u = np.unwrap(h[idx])
    d = np.gradient(u, idx) if idx.size > 1 else np.zeros(1)
    out[idx] = -np.degrees(d) * fps
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def draw_border(
    frames: npt.ArrayLike,
    on: npt.ArrayLike,
    width: int = BORDER_PX,
    colour: tuple[int, int, int] = BORDER_BGR,
) -> np.ndarray:
    """Copy of (T, H, W, 3) uint8 *frames* with a border on frames where *on* is True."""
    f = np.array(frames, dtype=np.uint8, copy=True)
    m = np.asarray(on, dtype=bool)
    if width <= 0 or not m.any():
        return f
    col = np.asarray(colour, dtype=np.uint8)
    f[m, :width] = col
    f[m, -width:] = col
    f[m, :, :width] = col
    f[m, :, -width:] = col
    return f


def tile_grid(
    tiles: Sequence[np.ndarray | None],
    rows: int = GRID_ROWS,
    cols: int = GRID_COLS,
    gap: int = 2,
    dim_after_end: float = 0.4,
) -> np.ndarray:
    """Tile clips (each (T_i, h, w, 3) uint8, or None for an empty slot) into a grid.

    All clips must share h and w. Shorter clips hold their last frame,
    dimmed by *dim_after_end*, until the longest clip ends. Returns
    (T, rows*h + (rows-1)*gap, cols*w + (cols-1)*gap, 3) uint8.
    """
    real = [t for t in tiles if t is not None]
    if not real:
        raise ValueError("no clips to tile")
    if len(tiles) > rows * cols:
        raise ValueError(f"{len(tiles)} clips do not fit a {rows}x{cols} grid")
    h, w = real[0].shape[1:3]
    if any(t.shape[1:3] != (h, w) for t in real):
        raise ValueError("clips must share frame size")
    n_t = max(t.shape[0] for t in real)
    out = np.zeros((n_t, rows * h + (rows - 1) * gap, cols * w + (cols - 1) * gap, 3), np.uint8)
    for i, t in enumerate(tiles):
        if t is None or t.shape[0] == 0:
            continue
        r, c = divmod(i, cols)
        y0, x0 = r * (h + gap), c * (w + gap)
        k = t.shape[0]
        out[:k, y0 : y0 + h, x0 : x0 + w] = t
        if k < n_t:
            out[k:, y0 : y0 + h, x0 : x0 + w] = (t[-1].astype(np.float32) * dim_after_end).astype(
                np.uint8
            )
    return out


def encode_h264(
    frames: npt.ArrayLike,
    path: Path,
    fps: float = 30.0,
    crf: int = 28,
    ffmpeg: str | None = None,
) -> Path:
    """Write (T, H, W, 3) BGR uint8 frames as browser-playable H.264 mp4 (yuv420p, faststart).

    Odd frame sizes are padded by one pixel (yuv420p needs even sizes).
    """
    f = np.ascontiguousarray(np.asarray(frames, dtype=np.uint8))
    if f.ndim != 4 or f.shape[-1] != 3 or f.shape[0] == 0:
        raise ValueError("frames must be (T, H, W, 3) with T > 0")
    ph, pw = f.shape[1] % 2, f.shape[2] % 2
    if ph or pw:
        f = np.pad(f, ((0, 0), (0, ph), (0, pw), (0, 0)))
    if ffmpeg is None:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{f.shape[2]}x{f.shape[1]}",
        "-r",
        f"{fps:g}",
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    proc = subprocess.run(cmd, input=f.tobytes(), capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode(errors='replace')[-500:]}")
    return Path(path)


# ---------------------------------------------------------------------------
# Pooling helpers
# ---------------------------------------------------------------------------


def hist_median(counts: npt.ArrayLike, lo: float, hi: float) -> float | None:
    """Median of a histogram with equal-width bins over ``[lo, hi)`` (linear within the bin)."""
    c = np.asarray(counts, dtype=np.float64)
    tot = c.sum()
    if tot <= 0:
        return None
    w = (hi - lo) / c.size
    cum = np.cumsum(c)
    i = int(np.searchsorted(cum, tot / 2.0))
    before = cum[i - 1] if i > 0 else 0.0
    frac = (tot / 2.0 - before) / c[i] if c[i] > 0 else 0.0
    return float(lo + (i + frac) * w)


def occupancy_counts(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    mask: npt.ArrayLike,
    x_edges: npt.ArrayLike,
    y_edges: npt.ArrayLike,
) -> np.ndarray:
    """2-D histogram (len(y_edges)-1, len(x_edges)-1) of the frames in *mask*."""
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    m = np.asarray(mask, dtype=bool) & np.isfinite(xx) & np.isfinite(yy)
    h, _, _ = np.histogram2d(yy[m], xx[m], bins=[np.asarray(y_edges), np.asarray(x_edges)])
    return h.astype(np.int64)


def syllables_covering(usage: dict[int, float], frac: float = 0.95, cap: int = 30) -> list[int]:
    """Most-used syllables, by usage, until *frac* of frames are covered (at most *cap*)."""
    items = sorted(usage.items(), key=lambda kv: -kv[1])
    tot = sum(v for _, v in items)
    out: list[int] = []
    acc = 0.0
    for k, v in items:
        if len(out) >= cap or (tot > 0 and acc / tot >= frac - 1e-12):
            break
        out.append(int(k))
        acc += v
    return out


def pool_media(sessions: dict[str, dict], frac: float = 0.95, cap: int = 30) -> dict:
    """Pool per-session syllable media summaries into the per-syllable page data.

    Each session dict (from ``scripts/make_qc_reports.py``) holds, per
    syllable id (string key) under ``"syllables"``: ``lengths`` (all bout
    lengths, frames), ``n_frames``, ``traj_sum`` (T, K, 2) and
    ``traj_count`` (T, K), ``occ`` (Ny, Nx) counts, ``speed_hist`` counts,
    ``turn_sum`` and ``turn_n``; plus session-level ``n_frames``, ``fps``,
    ``occ_total``, and the shared ``traj_offsets``, ``keypoints``,
    ``occ_edges`` and ``speed_range``.

    Returns the syllables covering *frac* of all frames (at most *cap*),
    each with pooled usage, median bout duration, median speed, mean signed
    turn rate, mean egocentric trajectory and occupancy.
    """
    ok = {k: v for k, v in sessions.items() if v and v.get("syllables")}
    if not ok:
        return {"syllables": [], "n_sessions": 0}
    first = next(iter(ok.values()))
    total = sum(v["n_frames"] for v in ok.values())
    usage: dict[int, float] = {}
    for v in ok.values():
        for k, r in v["syllables"].items():
            usage[int(k)] = usage.get(int(k), 0.0) + r["n_frames"]
    keep = syllables_covering(usage, frac, cap)
    fps = float(np.median([v["fps"] for v in ok.values()]))
    s_lo, s_hi = first["speed_range"]
    occ_total = np.sum([np.asarray(v["occ_total"], dtype=np.int64) for v in ok.values()], axis=0)
    out_syl = []
    for rank, k in enumerate(keep):
        rows = [r["syllables"][str(k)] for r in ok.values() if str(k) in r["syllables"]]
        lengths = np.concatenate([np.asarray(r["lengths"], dtype=np.int64) for r in rows])
        tsum = np.sum([np.asarray(r["traj_sum"], dtype=np.float64) for r in rows], axis=0)
        tcnt = np.sum([np.asarray(r["traj_count"], dtype=np.int64) for r in rows], axis=0)
        occ = np.sum([np.asarray(r["occ"], dtype=np.int64) for r in rows], axis=0)
        sh = np.sum([np.asarray(r["speed_hist"], dtype=np.int64) for r in rows], axis=0)
        tn = sum(r["turn_n"] for r in rows)
        ts = sum(r["turn_sum"] for r in rows)
        med = hist_median(sh, s_lo, s_hi)
        out_syl.append(
            {
                "id": k,
                "rank": rank,
                "usage": usage[k] / total,
                "n_frames": int(usage[k]),
                "n_bouts": int(lengths.size),
                "n_sessions": len(rows),
                "median_len": float(np.median(lengths)),
                "median_ms": float(np.median(lengths)) / fps * 1e3,
                "median_speed": None if med is None else round(med, 2),
                "mean_turn": None if tn == 0 else round(ts / tn, 1),
                "traj": np.round(mean_trajectory(tsum, tcnt), 1).tolist(),
                "traj_n": int(tcnt.max()) if tcnt.size else 0,
                "occ": occ.tolist(),
            }
        )
    return {
        "syllables": out_syl,
        "n_sessions": len(ok),
        "n_frames": int(total),
        "fps": fps,
        "covered": float(sum(usage[k] for k in keep) / total) if total else None,
        "keypoints": list(first["keypoints"]),
        "skeleton": [list(p) for p in SKELETON],
        "traj_offsets": list(first["traj_offsets"]),
        "occ_edges": first["occ_edges"],
        "occ_total": occ_total.tolist(),
    }
