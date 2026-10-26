"""Video frames for the tracking report's frame viewer.

Chooses a small set of frames per session to check by eye (lowest-confidence
stretches, ear swaps, jumps, plus random reference frames), and packs each
frame with the raw tracker keypoints and the stage 3 head / body position
and head direction mapped back into video pixels. The page draws the overlay
itself, so it can be switched off to see the unannotated frame.

Coordinate conventions follow :mod:`hm2p.kinematics.compute`: stage 3
rotates every keypoint clockwise by ``orientation_deg`` about the mean of all
keypoint positions (:func:`~hm2p.kinematics.compute.apply_orientation_rotation`)
before computing positions (``*_mm = px * scale_mm_per_px``) and head
direction (0° = towards -x of the rotated image, 90° = +y, image y down).
"""

from __future__ import annotations

import base64

import numpy as np
import numpy.typing as npt

from hm2p.pose.quality import detect_ear_swaps
from hm2p.qc.common import fnum, mask_intervals

SKELETON = (
    ("nose_tip", "head_midpoint"),
    ("head_midpoint", "neck"),
    ("neck", "mid_back"),
    ("mid_back", "mouse_center"),
    ("mouse_center", "tail_base"),
    ("left_ear", "head_midpoint"),
    ("right_ear", "head_midpoint"),
)
N_WORST = 8
N_SWAP = 2
N_JUMP = 2
N_RANDOM = 3
JPEG_QUALITY = 70
SWAP_AXES = (("nose_tip", "neck"), ("nose_tip", "head_midpoint"), ("head_midpoint", "neck"))

KeypointData = dict[str, dict[str, npt.NDArray[np.floating]]]


def likelihood_rank(kp: KeypointData, parts: list[str]) -> np.ndarray:
    """Mean over *parts* of each keypoint's within-session likelihood percentile (0 = worst).

    Converting to ranks lets keypoints with different likelihood scales count
    equally. NaN likelihoods rank lowest.
    """
    n = len(np.asarray(kp[parts[0]]["likelihood"]))
    rank = np.zeros(n, dtype=np.float64)
    for b in parts:
        lik = np.nan_to_num(np.asarray(kp[b]["likelihood"], dtype=np.float64), nan=-1.0)
        rank += np.argsort(np.argsort(lik, kind="stable"), kind="stable") / max(n - 1, 1)
    return rank / len(parts)


def max_displacement(kp: KeypointData, parts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per frame, the largest frame-to-frame keypoint displacement and which keypoint moved.

    Returns ``(disp, which)``: ``disp[i]`` is the largest ``|p[i] - p[i-1]|``
    over *parts* (px, NaN where no keypoint is finite in both frames; frame 0
    is NaN) and ``which[i]`` its index into *parts* (-1 where undefined).
    """
    d = np.stack(
        [
            np.hypot(
                np.diff(np.asarray(kp[b]["x"], dtype=np.float64), prepend=np.nan),
                np.diff(np.asarray(kp[b]["y"], dtype=np.float64), prepend=np.nan),
            )
            for b in parts
        ]
    )
    ok = np.isfinite(d).any(axis=0)
    which = np.full(d.shape[1], -1, dtype=np.int64)
    which[ok] = np.nanargmax(np.where(np.isfinite(d[:, ok]), d[:, ok], -np.inf), axis=0)
    disp = np.full(d.shape[1], np.nan)
    disp[ok] = d[which[ok], np.flatnonzero(ok)]
    return disp, which


def ear_swap_mask(kp: KeypointData) -> np.ndarray | None:
    """Frames flagged by :func:`hm2p.pose.quality.detect_ear_swaps` (axis as in the report)."""
    if "left_ear" not in kp or "right_ear" not in kp:
        return None
    for a, p in SWAP_AXES:
        if a in kp and p in kp:
            xs = [
                np.asarray(kp[b][c], dtype=np.float64)
                for b in ("left_ear", "right_ear", a, p)
                for c in ("x", "y")
            ]
            return np.asarray(detect_ear_swaps(*xs)["is_swapped"], dtype=bool)
    return None


def _far_from(f: int, taken: list[int], sep: int) -> bool:
    return all(abs(f - t) >= sep for t in taken)


def select_review_frames(
    kp: KeypointData,
    parts: list[str],
    fps: float,
    jump_px: float,
    worst_bin_starts_min: list[float],
    bin_s: float,
    n_random: int = N_RANDOM,
    seed: int = 0,
) -> list[dict]:
    """Frames worth checking by eye, sorted by frame index.

    One frame per low-confidence 10 s bin (the frame with the lowest mean
    likelihood rank inside the bin), the middle frame of the ``N_SWAP``
    longest ear-swap runs, the ``N_JUMP`` largest jumps above *jump_px*, and
    *n_random* reference frames drawn (with a fixed *seed*) from frames not in
    any of those categories. Frames are kept at least 1 s apart.

    Returns a list of ``{"frame", "reason", "detail"}``; ``reason`` is one of
    ``"low likelihood"`` (these also carry ``bin_t_min``, the bin start),
    ``"ear swap"``, ``"jump"``, ``"reference"``.
    """
    n = len(np.asarray(kp[parts[0]]["x"]))
    if n == 0:
        return []
    sep = max(1, round(fps))
    per_bin = max(1, round(bin_s * fps))
    taken: list[int] = []
    out: list[dict] = []

    def add(f: int, reason: str, detail: str, **extra: object) -> None:
        taken.append(f)
        out.append({"frame": int(f), "reason": reason, "detail": detail, **extra})

    rank = likelihood_rank(kp, parts)
    for t_min in worst_bin_starts_min:
        a = int(round(t_min * 60.0 * fps))
        if not 0 <= a < n:
            continue
        seg = rank[a : a + per_bin]
        f = a + int(np.argmin(seg))
        if _far_from(f, taken, sep):
            add(
                f,
                "low likelihood",
                f"lowest mean likelihood rank in the bin ({rank[f]:.3f})",
                bin_t_min=t_min,
            )

    swaps = ear_swap_mask(kp)
    if swaps is not None:
        runs = sorted(mask_intervals(swaps), key=lambda r: r[0] - r[1])  # longest first
        k = 0
        for s, e in runs:
            f = (s + e - 1) // 2
            if k < N_SWAP and _far_from(f, taken, sep):
                add(f, "ear swap", f"ears on the minority side for {e - s} frame(s)")
                k += 1

    disp, which = max_displacement(kp, parts)
    order = np.argsort(np.nan_to_num(-disp, nan=np.inf), kind="stable")
    k = 0
    for f in order:
        if k >= N_JUMP or not np.isfinite(disp[f]) or disp[f] <= jump_px:
            break
        if _far_from(int(f), taken, sep):
            add(int(f), "jump", f"{parts[which[f]]} moved {disp[f]:.0f} px in one frame")
            k += 1

    rng = np.random.default_rng(seed)
    flagged = np.zeros(n, dtype=bool)
    if swaps is not None:
        flagged |= swaps
    flagged |= np.nan_to_num(disp, nan=0.0) > jump_px
    pool = np.flatnonzero(~flagged)
    k = 0
    for f in rng.permutation(pool)[: max(50, 10 * n_random)]:
        if k >= n_random:
            break
        if _far_from(int(f), taken, sep):
            add(int(f), "reference", "random frame (no jump or ear swap)")
            k += 1
    return sorted(out, key=lambda r: r["frame"])


def frame_keypoints(kp: KeypointData, parts: list[str], frame: int) -> dict[str, list]:
    """``{keypoint: [x, y, likelihood]}`` at one frame (pixels; None for NaN)."""
    return {
        b: [
            fnum(kp[b]["x"][frame], 2),
            fnum(kp[b]["y"][frame], 2),
            fnum(kp[b]["likelihood"][frame], 3),
        ]
        for b in parts
    }


def _inverse_rotate(
    x: np.ndarray, y: np.ndarray, angle_deg: float, cx: float, cy: float
) -> tuple[np.ndarray, np.ndarray]:
    """Undo :func:`hm2p.kinematics.compute._rotate_xy` (clockwise by *angle_deg* about cx, cy)."""
    rad = np.deg2rad(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    dx, dy = x - cx, y - cy
    return cx + c * dx - s * dy, cy + s * dx + c * dy


def mm_to_video_px(
    x_mm: npt.ArrayLike,
    y_mm: npt.ArrayLike,
    scale_mm_per_px: float,
    orientation_deg: float,
    centre_px: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Stage 3 position (mm, rotated frame) back to video pixel coordinates.

    *centre_px* is the rotation centre stage 3 used: the mean of all keypoint
    positions of the unrotated pose file.
    """
    x = np.asarray(x_mm, dtype=np.float64) / scale_mm_per_px
    y = np.asarray(y_mm, dtype=np.float64) / scale_mm_per_px
    return _inverse_rotate(x, y, orientation_deg, centre_px[0], centre_px[1])


def hd_to_video_vector(
    hd_deg: npt.ArrayLike, orientation_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """Unit vector in video pixels (image y down) pointing along the stage 3 head direction."""
    th = np.deg2rad(np.asarray(hd_deg, dtype=np.float64))
    vx, vy = -np.cos(th), np.sin(th)  # convention of compute._vector_angle_deg
    return _inverse_rotate(vx, vy, orientation_deg, 0.0, 0.0)


def encode_jpeg(img: np.ndarray, quality: int = JPEG_QUALITY) -> str:
    """Base64 JPEG of a grey (H, W) or BGR (H, W, 3) uint8 image."""
    import cv2

    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(img), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")
