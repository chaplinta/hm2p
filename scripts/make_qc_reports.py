#!/usr/bin/env python3
"""Collect per-session data for the five standalone data-quality reports.

For every session in ``metadata/experiments.csv`` (all 26, regardless of the
``exclude`` / ``primary_exp`` flags, which are shown in the reports instead)
this downloads the Stage 2b-4 outputs from S3, summarises them with
:mod:`hm2p.qc`, and writes one JSON per report to ``results/qc/data/``:

=============  ===========================================================
tracking       champion DLC pose .h5 (``pose/{sub}/{ses}/``)
movement       ``kinematics/{sub}/{ses}/kinematics.h5``
syllables      ``kinematics/{sub}/{ses}/syllables.npz`` + provenance, model
rois           ``ca_extraction/{sub}/{ses}/suite2p/plane0/*.npy`` + ca.h5
spikes         ``calcium/{sub}/{ses}/ca.h5``
=============  ===========================================================

Per-session summaries are cached in ``results/qc/cache/{report}/`` so an
interrupted run resumes; ``--refresh`` ignores the cache. Nothing is written
to S3. ``--classifier-reference`` also computes leave-one-session-out
predictions on the manually labelled legacy sessions in ``/data/s2p``
(read-only) for the classifier report.

Usage
-----
    python scripts/make_qc_reports.py                       # all reports
    python scripts/make_qc_reports.py --reports tracking movement
    python scripts/make_qc_reports.py --sessions 20220804_13_52_02_1117646
    python scripts/make_qc_reports.py --reports rois --classifier-reference
    python scripts/build_qc_reports_html.py
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import tempfile
import time
import traceback
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from hm2p.qc import movement as qmov  # noqa: E402
from hm2p.qc import roi_assets as qra  # noqa: E402
from hm2p.qc import rois as qroi  # noqa: E402
from hm2p.qc import spikes as qspk  # noqa: E402
from hm2p.qc import syllable_media as qsm  # noqa: E402
from hm2p.qc import syllables as qsyl  # noqa: E402
from hm2p.qc import tracking as qtrk  # noqa: E402
from hm2p.qc.common import hist, span_fps  # noqa: E402

log = logging.getLogger("make_qc_reports")

BUCKET = "hm2p-derivatives"
OUT_DIR = REPO / "results" / "qc"
REPORTS = ("tracking", "movement", "syllables", "rois", "spikes")
CA_KEYS_SPIKES = (
    "dff",
    "F_corr",
    "F0_rolling",
    "F0_percentile",
    "F_raw",
    "event_masks",
    "event_masks_sd",
    "spikes",
    "roi_types",
    "frame_times",
)
KIN_SKIP_PREFIX = "bp_"

_S3: Any = None


# ---------------------------------------------------------------------------
# S3 access (read-only, with retries: the container connection is flaky)
# ---------------------------------------------------------------------------


def s3() -> Any:
    global _S3
    if _S3 is None:
        import boto3

        profile = os.environ.get("HM2P_AWS_PROFILE", "hm2p-agent")
        _S3 = boto3.Session(profile_name=profile).client("s3")
    return _S3


def get_bytes(key: str, retries: int = 4) -> bytes | None:
    """Object body, or None if it does not exist / every attempt failed."""
    for attempt in range(retries):
        try:
            return s3().get_object(Bucket=BUCKET, Key=key)["Body"].read()
        except Exception as exc:  # noqa: BLE001
            # non-ClientError exceptions (connection errors) carry response=None
            code = (getattr(exc, "response", None) or {}).get("Error", {}).get("Code")
            if code in ("NoSuchKey", "404"):
                return None
            log.debug("get %s failed (attempt %d): %s", key, attempt + 1, exc)
            time.sleep(1.5 * (attempt + 1))
    log.warning("download failed after %d attempts: %s", retries, key)
    return None


def list_keys(prefix: str) -> list[str]:
    keys: list[str] = []
    token = None
    while True:
        kw = {"Bucket": BUCKET, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        resp = s3().list_objects_v2(**kw)
        keys += [o["Key"] for o in resp.get("Contents", [])]
        if not resp.get("IsTruncated"):
            return keys
        token = resp["NextContinuationToken"]


def get_json(key: str) -> dict | None:
    b = get_bytes(key)
    return None if b is None else json.loads(b)


def read_h5(key: str, wanted: Callable[[str], bool] | None = None) -> tuple[dict, dict] | None:
    """Datasets (all, or those whose path passes *wanted*) and root attrs of an S3 HDF5."""
    b = get_bytes(key)
    return None if b is None else h5_from_bytes(b, wanted)


def h5_from_bytes(b: bytes, wanted: Callable[[str], bool] | None = None) -> tuple[dict, dict]:
    """Datasets (all, or those whose path passes *wanted*) and root attrs of an in-memory HDF5."""
    data: dict[str, np.ndarray] = {}
    with h5py.File(io.BytesIO(b), "r") as f:
        attrs = dict(f.attrs)

        def visit(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset) and (wanted is None or wanted(name)):
                data[name] = obj[()]

        f.visititems(visit)
    return data, attrs


# ---------------------------------------------------------------------------
# Session metadata
# ---------------------------------------------------------------------------


def sessions_table() -> pd.DataFrame:
    exps = pd.read_csv(REPO / "metadata" / "experiments.csv", dtype={"exp_id": str})
    animals = pd.read_csv(REPO / "metadata" / "animals.csv", dtype={"animal_id": str})
    exps["animal_id"] = exps["exp_id"].str.split("_").str[-1]
    p = exps["exp_id"].str.split("_")
    exps["sub"] = "sub-" + exps["animal_id"]
    exps["ses"] = "ses-" + p.str[0] + "T" + p.str[1] + p.str[2] + p.str[3]
    return exps.merge(animals[["animal_id", "celltype"]], on="animal_id", how="left")


def mm_per_px(sub: str, ses: str) -> float | None:
    import configparser

    path = REPO / "metadata" / "video_meta" / f"{sub}_{ses}_meta.txt"
    if not path.exists():
        return None
    cfg = configparser.ConfigParser()
    cfg.read(path)
    try:
        return float(cfg["scale"]["mm_per_pix"])
    except (KeyError, ValueError):
        return None


def session_meta(row: pd.Series) -> dict:
    return {
        "exp_id": row["exp_id"],
        "sub": row["sub"],
        "ses": row["ses"],
        "animal": row["animal_id"],
        "celltype": None if pd.isna(row.get("celltype")) else row["celltype"],
        "exclude": bool(row.get("exclude", 0) == 1),
        "primary": bool(row.get("primary_exp", 0) == 1),
        "notes": None if pd.isna(row.get("Notes")) else str(row["Notes"]),
    }


# ---------------------------------------------------------------------------
# Per-report collectors (each returns a summary dict for one session)
# ---------------------------------------------------------------------------


class Ctx:
    """Per-session download cache shared by the collectors."""

    def __init__(self, sub: str, ses: str, champion: dict | None) -> None:
        self.sub, self.ses, self.champion = sub, ses, champion
        self._kin: tuple[dict, dict] | None | bool = False
        self._ca_bytes: bytes | None | bool = False

    def kinematics(self) -> tuple[dict, dict] | None:
        if self._kin is False:
            self._kin = read_h5(
                f"kinematics/{self.sub}/{self.ses}/kinematics.h5",
                lambda n: not n.startswith(KIN_SKIP_PREFIX),
            )
        return self._kin  # type: ignore[return-value]

    def ca(self, wanted: Callable[[str], bool]) -> tuple[dict, dict] | None:
        """Selected datasets of ca.h5; the file is downloaded once per session."""
        if self._ca_bytes is False:
            self._ca_bytes = get_bytes(f"calcium/{self.sub}/{self.ses}/ca.h5")
        if self._ca_bytes is None:
            return None
        return h5_from_bytes(self._ca_bytes, wanted)  # type: ignore[arg-type]

    def ca_labels(self) -> tuple[dict, dict] | None:
        """roi_types, frame_times and roi_qc from ca.h5 (small datasets only)."""
        return self.ca(
            lambda n: n in ("roi_types", "frame_times", "iscell") or n.startswith("roi_qc/")
        )


# ---------------------------------------------------------------------------
# Tracking media: still frames with the keypoints, for the frame viewer
# ---------------------------------------------------------------------------

# options set from the command line (see main); tests switch frames off or on
TRACKING_MEDIA: dict[str, Any] = {"frames": True, "tmp_dir": None}
RAW_BUCKET = "hm2p-rawdata"
DEFAULT_POSE_CUTOFF = 0.5  # config/pipeline.yaml pose_confidence_threshold
DLC_SUBSAMPLE_FPS = 30  # DLC ran on `ffmpeg -i <overhead.mp4> -r 30` (scripts/run_dlc_retrain.py)
LABELLED_SIZE = (416, 304)  # labelled_30fps.mp4 is the 30 fps video resized to this


def _asset_path(report: str, key: str) -> Path:
    return OUT_DIR / "assets" / report / f"{key}.json"


def _write_asset(report: str, key: str, payload: dict) -> int:
    """Write ``results/qc/assets/<report>/<key>.json``; returns its size in bytes."""
    path = _asset_path(report, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(payload), separators=(",", ":")))
    return path.stat().st_size


def _ffmpeg_exe() -> str:
    import shutil

    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _download(bucket: str, key: str, dest: Path, retries: int = 4) -> bool:
    for attempt in range(retries):
        try:
            s3().download_file(bucket, key, str(dest))
            return True
        except Exception as exc:  # noqa: BLE001
            code = (getattr(exc, "response", None) or {}).get("Error", {}).get("Code")
            if code in ("NoSuchKey", "404"):
                return False
            log.debug("download %s failed (attempt %d): %s", key, attempt + 1, exc)
            time.sleep(2.0 * (attempt + 1))
    return False


def read_frames_subsampled(
    video: Path, wanted: set[int], fps: int = DLC_SUBSAMPLE_FPS
) -> tuple[dict[int, np.ndarray], int]:
    """Grey frames *wanted* of *video* resampled the way DLC's input was made.

    Runs ``ffmpeg -i video -r fps`` (constant frame rate, as for the mp4 the
    DLC retrain script wrote) and keeps only the requested output frames, so
    frame ``i`` here is frame ``i`` of the pose file. Returns the frames and
    the total number of output frames (to compare with the pose length).
    """
    import subprocess

    import cv2

    cap = cv2.VideoCapture(str(video))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if w <= 0 or h <= 0:
        raise ValueError(f"cannot read frame size of {video}")
    cmd = [_ffmpeg_exe(), "-nostdin", "-loglevel", "error", "-i", str(video), "-r", str(fps),
           "-fps_mode", "cfr", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]  # fmt: skip
    size = w * h
    got: dict[int, np.ndarray] = {}
    n = 0
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as p:
        assert p.stdout is not None
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            if n in wanted:
                got[n] = np.frombuffer(buf, dtype=np.uint8).reshape(h, w).copy()
            n += 1
    return got, n


def read_frames_seek(video: Path, wanted: set[int]) -> dict[int, np.ndarray]:
    """Grey frames by index from a video whose frames map 1:1 to the pose (labelled_30fps.mp4)."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    got: dict[int, np.ndarray] = {}
    for i in sorted(wanted):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if ok:
            got[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    cap.release()
    return got


def _video_crop_size(sub: str, ses: str) -> tuple[float, float] | None:
    """Width and height of the cropped overhead video (the pose pixel space)."""
    import configparser

    path = REPO / "metadata" / "video_meta" / f"{sub}_{ses}_meta.txt"
    if not path.exists():
        return None
    cfg = configparser.ConfigParser()
    cfg.read(path)
    try:
        return float(cfg["crop"]["width"]), float(cfg["crop"]["height"])
    except (KeyError, ValueError):
        return None


def tracking_frames_asset(
    c: Ctx,
    kp: dict,
    s: dict,
    kin: tuple[dict, dict] | None,
    centre_px: tuple[float, float],
) -> dict | None:
    """Pick review frames, read them from the video and write the frame-viewer asset.

    Frames come from the raw overhead video resampled to 30 fps exactly as for
    DLC (unannotated), when its frame count equals the pose length; otherwise
    from ``labelled_30fps.mp4`` (which has the raw DLC points drawn in).
    Returns ``{"key", "source", "frames", "bytes"}`` for the summary, or None.
    """
    from hm2p.qc import frames as qfr
    from hm2p.qc.common import fnum

    exp_id = f"{c.sub}_{c.ses}"
    fps, parts, n = s["fps"], s["bodyparts"], s["n_frames"]
    jump_px = s["jump_threshold"] / (s["mm_per_px"] or 1.0)
    worst = [b["t_min"] for b in s["worst_bins"]]
    picks = qfr.select_review_frames(kp, parts, fps, jump_px, worst, qtrk.TIMELINE_BIN_S)
    if not picks:
        return None
    imgs, source, note, size_px = _review_video_frames(c, {p["frame"] for p in picks}, n)
    if not imgs or source is None:
        return None
    k, attrs = kin if kin is not None else ({}, {})
    kin_ok = "frame_times" in k and len(k["frame_times"]) == n
    orient = float(attrs.get("orientation_deg", 0.0) or 0.0)
    kscale = float(attrs.get("scale_mm_per_px", 0.0) or 0.0)
    try:
        cutoff = float(attrs.get("confidence_threshold", DEFAULT_POSE_CUTOFF))
    except (TypeError, ValueError):
        cutoff = DEFAULT_POSE_CUTOFF
    out_frames = []
    for p in picks:
        f = p["frame"]
        if f not in imgs:
            continue
        rec = dict(p, t_s=f / fps, jpeg=qfr.encode_jpeg(imgs[f]))
        rec["kp"] = qfr.frame_keypoints(kp, parts, f)
        if kin_ok:
            rec["light"] = bool(k["light_on"][f]) if "light_on" in k else None
            rec["bad"] = bool(k["bad_behav"][f]) if "bad_behav" in k else None
            for name in ("head", "body"):
                if kscale > 0 and f"x_{name}_mm" in k:
                    x, y = qfr.mm_to_video_px(
                        k[f"x_{name}_mm"][f], k[f"y_{name}_mm"][f], kscale, orient, centre_px
                    )
                    rec[name] = [fnum(x, 2), fnum(y, 2)]
            if "hd_deg" in k and np.isfinite(k["hd_deg"][f]):
                vx, vy = qfr.hd_to_video_vector(k["hd_deg"][f], orient)
                rec["hd"] = [fnum(vx, 4), fnum(vy, 4)]
                rec["hd_deg"] = fnum(np.mod(k["hd_deg"][f], 360.0), 1)
        out_frames.append(rec)
    key = f"frames-{exp_id}"
    payload = {
        "exp_id": exp_id,
        "source": source,
        "note": note,
        "width": size_px[0],
        "height": size_px[1],
        "cutoff": cutoff,
        "kinematics_aligned": kin_ok,
        "orientation_deg": orient,
        "skeleton": [list(e) for e in qfr.SKELETON],
        "frames": out_frames,
    }
    size = _write_asset("tracking", key, payload)
    log.info("%s: %d frames (%s), %.0f kB", exp_id, len(out_frames), source, size / 1e3)
    keep = ("frame", "reason", "bin_t_min")
    return {
        "key": key,
        "source": source,
        "bytes": size,
        "frames": [{k2: v for k2, v in r.items() if k2 in keep} for r in out_frames],
    }


def _review_video_frames(
    c: Ctx, wanted: set[int], n: int
) -> tuple[dict[int, np.ndarray], str | None, str, tuple[float, float]]:
    """Frames *wanted* as ``(images, source, note, (width, height) in pose pixels)``."""
    with tempfile.TemporaryDirectory(dir=TRACKING_MEDIA["tmp_dir"]) as td:
        resp = s3().list_objects_v2(Bucket=RAW_BUCKET, Prefix=f"rawdata/{c.sub}/{c.ses}/behav/")
        vids = sorted(
            o["Key"]
            for o in resp.get("Contents", [])
            if o["Key"].endswith(".mp4") and "overhead" in o["Key"] and "side" not in o["Key"]
        )
        note = ""
        path = Path(td) / "overhead.mp4"
        if vids and _download(RAW_BUCKET, vids[0], path):
            got, n_out = read_frames_subsampled(path, wanted)
            path.unlink()
            if n_out == n and got:
                h, w = next(iter(got.values())).shape
                note = (
                    f"Unannotated frames of {vids[0].split('/')[-1]}, resampled to "
                    f"{DLC_SUBSAMPLE_FPS} fps as for DLC ({n_out} frames, as in the pose file)."
                )
                return got, "raw", note, (float(w), float(h))
            log.warning("%s %s: resampled video has %d frames, pose %d", c.sub, c.ses, n_out, n)
            note = f"The raw video resampled to 30 fps has {n_out} frames, not {n}. "
        crop = _video_crop_size(c.sub, c.ses)
        path = Path(td) / "labelled.mp4"
        if crop and _download(BUCKET, f"pose/{c.sub}/{c.ses}/labelled_30fps.mp4", path):
            note += (
                "Frames from labelled_30fps.mp4, which already has the raw DLC keypoints "
                "drawn in; the page's overlay is drawn on top of them."
            )
            return read_frames_seek(path, wanted), "labelled", note, crop
    return {}, None, note, (0.0, 0.0)


def illumination_summary(c: Ctx) -> dict | None:
    got = read_h5(f"analysis/{c.sub}/{c.ses}/illumination.h5")
    if got is None:
        return None
    return qtrk.summarise_illumination(*got)


def tracking_model_info() -> dict:
    """DLC champion history and the evaluation / promotion-gate files, when present."""
    log_b = get_bytes("dlc-champion-history/promotions.log")
    history = []
    for line in (log_b or b"").decode(errors="replace").splitlines():
        f = line.split("\t")
        if len(f) >= 2:
            history.append({"promoted_at": f[0], "champion_id": f[1],
                            "instance": f[2] if len(f) > 2 else None,
                            "git_sha": f[3] if len(f) > 3 else None})  # fmt: skip
    return {
        "history": history,
        "eval": get_json("dlc-retrain/_eval_results.json"),
        "verdict": get_json("dlc-retrain/models/_compare_verdict.json"),
    }


def collect_tracking(c: Ctx) -> dict:
    from hm2p.kinematics.compute import load_pose_dataset
    from hm2p.pose.select import select_champion_h5

    prefix = f"pose/{c.sub}/{c.ses}/"
    keys = list_keys(prefix)
    champ_id = (c.champion or {}).get("champion_id")
    if champ_id:
        key = select_champion_h5(keys, champ_id)
    else:  # no manifest: newest-looking non-filtered .h5
        cands = sorted(
            k for k in keys if k.endswith(".h5") and "_filtered" not in k and "_single" not in k
        )
        if not cands:
            raise FileNotFoundError(f"no pose .h5 under {prefix}")
        key = cands[-1]
    kin = c.kinematics()
    meta = get_json(prefix + "dlc_meta.json") or {}
    fps = float(meta.get("tracking_fps", 30.0))
    scale = mm_per_px(c.sub, c.ses)
    fps_source, scale_source = (
        "dlc_meta.json" if "tracking_fps" in meta else "default 30",
        "video_meta",
    )
    body = get_bytes(key)
    if body is None:
        raise FileNotFoundError(key)
    with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
        tmp.write(body)
        tmp.flush()
        ds = load_pose_dataset(Path(tmp.name), "dlc")
    kp = qtrk.keypoints_from_dataset(ds)
    # stage 3 rotates about the mean of all keypoint positions (apply_orientation_rotation)
    centre_px = (
        float(np.nanmean(ds["position"].sel(space="x").values)),
        float(np.nanmean(ds["position"].sel(space="y").values)),
    )
    del ds
    n_pose = len(next(iter(kp.values()))["x"])
    light = None
    if kin is not None:
        k, attrs = kin
        # kinematics.h5 is computed from this pose file, frame for frame: use its
        # frame rate and video scale so tracking QC matches stage 3
        if "frame_times" in k and len(k["frame_times"]) == n_pose:
            fps, fps_source = span_fps(k["frame_times"]), "kinematics frame_times"
        if attrs.get("scale_mm_per_px"):
            scale, scale_source = float(attrs["scale_mm_per_px"]), "kinematics.h5"
        light = k.get("light_on")
    s = qtrk.summarise_tracking(kp, fps=fps, mm_per_px=scale, light_on=light)
    s["fps_source"], s["scale_source"] = fps_source, scale_source
    s["pose_file"] = key.split("/")[-1]
    s["champion_id"] = champ_id
    s["light_from_kinematics"] = light is not None and len(light) == s["n_frames"]
    s["illumination"] = illumination_summary(c)
    s["frames_asset"] = None
    if TRACKING_MEDIA["frames"]:
        try:
            s["frames_asset"] = tracking_frames_asset(
                c, qtrk.normalise_names(kp), s, kin, centre_px
            )
        except Exception as exc:  # noqa: BLE001 - the rest of the summary is still useful
            log.warning("tracking frames %s %s failed: %s", c.sub, c.ses, exc)
            log.debug(traceback.format_exc())
    return s


def collect_movement(c: Ctx) -> dict:
    kin = c.kinematics()
    if kin is None:
        raise FileNotFoundError("kinematics.h5")
    s = qmov.summarise_movement(*kin)
    cid = s["attrs"].get("dlc_champion_id")
    want = (c.champion or {}).get("champion_id")
    s["champion_current"] = None if want is None else bool(cid == want)
    key = f"traj-{c.sub}_{c.ses}"
    size = _write_asset("movement", key, qmov.trajectory_asset(kin[0]))
    s["traj_asset"] = {"key": key, "bytes": size, "step": qmov.TRAJ_STEP}
    return s


def maze_cells() -> list[list[int]]:
    """(col, row) of the accessible maze cells (maze coordinates, 1 unit per cell)."""
    from hm2p.maze.topology import build_rose_maze

    return [[int(a), int(b)] for a, b in build_rose_maze().cell_list]


def collect_syllables(c: Ctx) -> dict:
    b = get_bytes(f"kinematics/{c.sub}/{c.ses}/syllables.npz")
    if b is None:
        raise FileNotFoundError("syllables.npz")
    z = np.load(io.BytesIO(b))
    sid = z["syllable_id"] if "syllable_id" in z else z["syllable_ids"]
    kin = c.kinematics()
    k = kin[0] if kin is not None else {}
    fps = 30.0
    if "frame_times" in k and len(k["frame_times"]) == len(sid):
        fps = span_fps(k["frame_times"])
    s = qsyl.summarise_syllables(
        sid,
        fps=fps,
        speed_cm_s=k.get("speed_cm_s"),
        ahv_deg_s=k.get("ahv_deg_s"),
        light_on=k.get("light_on"),
        bad_behav=k.get("bad_behav"),
    )
    s["aligned_to_kinematics"] = "frame_times" in k and len(k["frame_times"]) == len(sid)
    s["provenance"] = get_json(f"kinematics/{c.sub}/{c.ses}/syllables.provenance.json")
    if "syllable_prob" in z:
        p = np.asarray(z["syllable_prob"], dtype=np.float64)
        s["max_prob_hist"] = hist(p.max(axis=1), 0.0, 1.0, 40) if p.ndim == 2 else None
    return s


# ---------------------------------------------------------------------------
# Syllable media: grid movies, average trajectories, occupancy (pooled)
# ---------------------------------------------------------------------------

# options set from the command line (see main); tests switch media off or on
SYLLABLE_MEDIA: dict[str, Any] = {"enabled": True, "clips": True, "tmp_dir": None}
SYL_MEDIA_VERSION = 2
SYL_CONF = 0.5  # DLC likelihood for a keypoint to count as tracked
SYL_WINDOW_MM = 180.0  # side of the square cropped around the animal for a tile
SYL_OCC_EDGES = (np.arange(0.0, 7.001, 0.25), np.arange(0.0, 5.001, 0.25))  # maze units
SYL_SPEED_RANGE = (0.0, 60.0)
SYL_SPEED_BINS = 120
SYL_VIDEO = "labelled_30fps.mp4"
SYL_VIDEO_SIZE = (416, 304)  # labelled videos are the cropped video resized to this


def _syllable_dirs() -> tuple[Path, Path]:
    return OUT_DIR / "cache" / "syllables_media", OUT_DIR / "assets" / "syllables"


def _video_scale(sub: str, ses: str) -> tuple[float, float] | None:
    """(x, y) factors from pose pixels to labelled-video pixels.

    The labelled video is the cropped video resized to :data:`SYL_VIDEO_SIZE`,
    not always with the same factor on both axes.
    """
    import configparser

    path = REPO / "metadata" / "video_meta" / f"{sub}_{ses}_meta.txt"
    if not path.exists():
        return None
    cfg = configparser.ConfigParser()
    cfg.read(path)
    try:
        w, h = float(cfg["crop"]["width"]), float(cfg["crop"]["height"])
    except (KeyError, ValueError):
        return None
    return SYL_VIDEO_SIZE[0] / w, SYL_VIDEO_SIZE[1] / h


def _syllable_pose(c: Ctx) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Champion DLC pose as (n, K, 2) positions (filtered, interpolated, median-smoothed)
    and (n, K) raw likelihoods for :data:`qsm.TRAJ_KEYPOINTS` (NaN where absent)."""
    from hm2p.kinematics.compute import (
        filter_low_confidence,
        interpolate_gaps,
        load_pose_dataset,
        median_filter_dataset,
    )
    from hm2p.pose.select import select_champion_h5

    prefix = f"pose/{c.sub}/{c.ses}/"
    keys = list_keys(prefix)
    champ_id = (c.champion or {}).get("champion_id")
    key = select_champion_h5(keys, champ_id) if champ_id else None
    if not key:
        raise FileNotFoundError(f"no champion pose .h5 under {prefix}")
    with tempfile.TemporaryDirectory(dir=SYLLABLE_MEDIA["tmp_dir"]) as td:
        h5 = Path(td) / "pose.h5"
        _download_file(key, h5)
        ds = load_pose_dataset(h5, "dlc").load()
    if ds.sizes.get("individuals", 1) > 1:
        ds = ds.isel(individuals=[0])
    names = [str(k) for k in ds["keypoints"].values]
    pick = []
    for kp in qsm.TRAJ_KEYPOINTS:
        cands = (kp, *qsm.KEYPOINT_ALIASES.get(kp, ()))
        pick.append(next((n for n in cands if n in names), None))
    ds = ds.sel(keypoints=[p for p in pick if p is not None])
    conf_raw = ds["confidence"].transpose("time", "keypoints", "individuals").values[..., 0]
    ds = median_filter_dataset(interpolate_gaps(filter_low_confidence(ds, SYL_CONF), 5), 5)
    p = ds["position"].transpose("time", "keypoints", "space", "individuals").values[..., 0]
    n = p.shape[0]
    pos = np.full((n, len(pick), 2), np.nan)
    conf = np.full((n, len(pick)), np.nan)
    j = 0
    for i, name in enumerate(pick):
        if name is not None:
            pos[:, i], conf[:, i] = p[:, j], conf_raw[:, j]
            j += 1
    return pos, conf, list(qsm.TRAJ_KEYPOINTS)


def syllable_session_media(c: Ctx) -> tuple[dict, dict[str, np.ndarray]]:
    """Per-session inputs for the pooled syllable visualisations.

    Returns a JSON-ready dict (per-syllable bout lengths, usable example
    bouts, egocentric trajectory sums, occupancy and speed histograms, turn
    rate sums) and arrays needed later to render clips (body centre in
    labelled-video pixels and heading per frame).
    """
    b = get_bytes(f"kinematics/{c.sub}/{c.ses}/syllables.npz")
    if b is None:
        raise FileNotFoundError("syllables.npz")
    z = np.load(io.BytesIO(b))
    sid = np.asarray(z["syllable_id"] if "syllable_id" in z else z["syllable_ids"], np.int64)
    kin = c.kinematics()
    if kin is None:
        raise FileNotFoundError("kinematics.h5")
    k, attrs = kin
    if len(k.get("frame_times", [])) != sid.size:
        raise ValueError("syllables.npz and kinematics.h5 differ in length")
    fps = span_fps(k["frame_times"])
    scale = attrs.get("scale_mm_per_px") or mm_per_px(c.sub, c.ses)
    if not scale:
        raise ValueError("no mm-per-pixel scale")
    scale = float(scale)
    pos, conf, kps = _syllable_pose(c)
    n = min(sid.size, pos.shape[0])
    sid, pos, conf = sid[:n], pos[:n], conf[:n]
    ki = {name: i for i, name in enumerate(kps)}
    centre = pos[:, ki["mouse_center"]]
    heading = qsm.heading_angle(pos[:, ki["tail_base"]], pos[:, ki["neck"]])
    axis = [ki["neck"], ki["tail_base"], ki["mouse_center"]]
    tracked = (np.nan_to_num(conf[:, axis], nan=0.0) >= SYL_CONF).all(axis=1)
    tracked &= np.isfinite(centre).all(axis=1) & np.isfinite(heading)
    bad = np.asarray(k.get("bad_behav", np.zeros(n)), dtype=bool)[:n]
    # handedness check: left ear should be on the animal's left in the image
    ego = qsm.egocentric_points(
        pos[:, [ki["left_ear"], ki["right_ear"]]], centre[:, None, :], heading[:, None]
    )
    lr = qsm.left_on_left_frac(ego[tracked, 0, 0], ego[tracked, 1, 0])
    mirror = lr is not None and lr < 0.2
    turn = qsm.signed_turn_rate(qsm.heading_angle(pos[:, ki["neck"]], pos[:, ki["nose_tip"]]), fps)
    if mirror:
        turn = -turn
    speed = np.asarray(k.get("speed_cm_s", np.full(n, np.nan)), dtype=np.float64)[:n]
    xm = np.asarray(k.get("x_maze", np.full(n, np.nan)), dtype=np.float64)[:n]
    ym = np.asarray(k.get("y_maze", np.full(n, np.nan)), dtype=np.float64)[:n]
    good = ~bad
    offsets = np.arange(-qsm.TRAJ_PRE, qsm.TRAJ_POST + 1)
    shown_max = qsm.CLIP_MAX_FRAMES - qsm.CLIP_PRE - qsm.CLIP_POST
    per: dict[str, dict] = {}
    for syl, (starts, lens) in qsm.syllable_bouts(sid).items():
        m = (sid == syl) & good
        clip_ok = qsm.window_ok(
            starts - qsm.CLIP_PRE,
            starts + np.minimum(lens, shown_max) + qsm.CLIP_POST,
            bad,
            tracked,
            0.9,
            n_max=n - 1,  # the labelled video is one frame shorter than the pose
        )
        traj_ok = (
            qsm.window_ok(starts - qsm.TRAJ_PRE, starts + qsm.TRAJ_POST + 1, bad, tracked, 0.8)
            & tracked[np.clip(starts, 0, n - 1)]
        )
        win = qsm.ego_windows(pos, centre, heading, starts[traj_ok], offsets, mirror) * scale
        tsum, tcnt = qsm.trajectory_sums(win)
        tv = turn[m]
        tv = tv[np.isfinite(tv)]
        per[str(syl)] = {
            "n_frames": int((sid == syl).sum()),
            "lengths": lens.tolist(),
            "clips": [
                [int(s), int(ln)] for s, ln in zip(starts[clip_ok], lens[clip_ok], strict=True)
            ],
            "traj_sum": np.round(tsum, 2).tolist(),
            "traj_count": tcnt.tolist(),
            "occ": qsm.occupancy_counts(xm, ym, m, *SYL_OCC_EDGES).tolist(),
            "speed_hist": np.histogram(
                speed[m & np.isfinite(speed)], bins=SYL_SPEED_BINS, range=SYL_SPEED_RANGE
            )[0].tolist(),
            "turn_sum": float(tv.sum()),
            "turn_n": int(tv.size),
        }
    vs = _video_scale(c.sub, c.ses)
    out = {
        "version": SYL_MEDIA_VERSION,
        "n_frames": int(n),
        "fps": fps,
        "scale_mm_per_px": scale,
        "video_scale": vs,
        "left_on_left_frac": lr,
        "mirrored": bool(mirror),
        "frac_tracked": float(tracked.mean()),
        "keypoints": kps,
        "traj_offsets": offsets.tolist(),
        "occ_edges": [e.tolist() for e in SYL_OCC_EDGES],
        "occ_total": qsm.occupancy_counts(xm, ym, good, *SYL_OCC_EDGES).tolist(),
        "speed_range": list(SYL_SPEED_RANGE),
        "syllables": per,
    }
    sxy = np.asarray(vs if vs else (np.nan, np.nan))
    arrays = {
        "centre_vid": (centre * sxy).astype(np.float32),
        "heading_vid": qsm.heading_angle(
            pos[:, ki["tail_base"]] * sxy, pos[:, ki["neck"]] * sxy
        ).astype(np.float32),
        "light_on": np.asarray(k.get("light_on", np.zeros(n)), dtype=bool)[:n],
    }
    return out, arrays


def syllable_media_cached(c: Ctx, exp_id: str, refresh: bool) -> dict | None:
    """:func:`syllable_session_media` with a per-session cache; None if it failed."""
    cache_dir, _ = _syllable_dirs()
    js, npz = cache_dir / f"{exp_id}.json", cache_dir / f"{exp_id}.npz"
    if not refresh and js.exists() and npz.exists():
        d = json.loads(js.read_text())
        if d.get("version") == SYL_MEDIA_VERSION:
            return d
    t0 = time.time()
    try:
        d, arrays = syllable_session_media(c)
    except Exception as exc:  # noqa: BLE001 - the session is left out of the pooled media
        log.warning("syllable media %s failed: %s", exp_id, exc)
        log.debug(traceback.format_exc())
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, **arrays)
    js.write_text(json.dumps(_clean(d), separators=(",", ":")))
    log.info("syllable media %s ok (%.0f s)", exp_id, time.time() - t0)
    return d


def _bout_selection(pooled: dict, media: dict[str, dict]) -> dict[int, list[tuple[str, int, int]]]:
    """Example bouts per pooled syllable (seeded by syllable id, so reruns pick the same)."""
    out: dict[int, list[tuple[str, int, int]]] = {}
    for row in pooled["syllables"]:
        k = row["id"]
        cands = [
            (e, s, ln)
            for e, d in media.items()
            if d.get("video_scale")
            for s, ln in d["syllables"].get(str(k), {}).get("clips", [])
        ]
        rng = np.random.default_rng(1000 + int(k))
        out[k] = qsm.select_bouts(cands, row["median_len"], rng=rng)
    return out


def _download_file(key: str, path: Path) -> None:
    """Write one large S3 object to *path* (chunked download, see :func:`get_large_bytes`)."""
    body = get_large_bytes(key)
    if body is None:
        raise FileNotFoundError(key)
    path.write_bytes(body)


def _render_tiles(
    sub: str, ses: str, bouts: list[tuple[int, int, int]], arrays: dict, d: dict, tile_dir: Path
) -> None:
    """Cut, align and save tile clips (``.npy``) for *bouts* = [(syllable, start, len)]."""
    import cv2

    tmp_root = SYLLABLE_MEDIA["tmp_dir"]
    with tempfile.TemporaryDirectory(dir=tmp_root) as td:
        vid = Path(td) / "video.mp4"
        t0 = time.time()
        _download_file(f"pose/{sub}/{ses}/{SYL_VIDEO}", vid)
        log.info(
            "  %s downloaded (%.1f MB, %.0f s)",
            SYL_VIDEO,
            vid.stat().st_size / 1e6,
            time.time() - t0,
        )
        cap = cv2.VideoCapture(str(vid))
        n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_px = SYL_WINDOW_MM / d["scale_mm_per_px"] * float(np.mean(d["video_scale"]))
        # Read the video sequentially (grab is cheap; seeking decodes from the
        # previous keyframe and is very slow in some of these files) and keep
        # frames only for the clips in progress.
        jobs = []
        for syl, start, ln in sorted(bouts, key=lambda b: b[1]):
            first, last, b0, b1 = qsm.clip_window(start, ln)
            if first < 0 or last > n_vid:
                continue
            m = qsm.tile_affine(
                arrays["centre_vid"][start],
                float(arrays["heading_vid"][start]),
                src_px,
                qsm.TILE_PX,
                d["mirrored"],
            )
            jobs.append(
                {
                    "syl": syl,
                    "start": start,
                    "first": first,
                    "last": last,
                    "b0": b0,
                    "b1": b1,
                    "m": m,
                    "frames": [],
                }
            )
        try:
            if not jobs:
                return
            end = max(j["last"] for j in jobs)
            nxt = 0  # index of the first job not yet started
            active: list[dict] = []
            for f in range(end):
                if not cap.grab():
                    break
                while nxt < len(jobs) and jobs[nxt]["first"] == f:
                    active.append(jobs[nxt])
                    nxt += 1
                if not active:
                    continue
                ok, im = cap.retrieve()
                if not ok:
                    break
                for j in active:
                    j["frames"].append(
                        cv2.warpAffine(
                            im, j["m"], (qsm.TILE_PX, qsm.TILE_PX), flags=cv2.INTER_LINEAR
                        )
                    )
                done = [j for j in active if f + 1 >= j["last"]]
                for j in done:
                    active.remove(j)
                    on = np.zeros(len(j["frames"]), dtype=bool)
                    on[j["b0"] : j["b1"]] = True
                    np.save(
                        tile_dir / f"{j['syl']}_{sub}_{ses}_{j['start']}.npy",
                        qsm.draw_border(np.stack(j["frames"]), on),
                    )
                    j["frames"] = []
        finally:
            cap.release()


def render_syllable_grids(
    selection: dict[int, list[tuple[str, int, int]]],
    media: dict[str, dict],
    metas: dict[str, dict],
    refresh: bool,
) -> dict[str, dict]:
    """Grid movie assets (``results/qc/assets/syllables/syl_<id>.json``) per syllable.

    Sessions are processed one at a time (one video on disk, a few hundred
    frames in memory); a syllable whose asset already exists for the same
    selection is not rendered again. Returns per-syllable asset info.
    """
    import base64

    cache_dir, asset_dir = _syllable_dirs()
    asset_dir.mkdir(parents=True, exist_ok=True)
    info: dict[str, dict] = {}
    todo: dict[int, list[tuple[str, int, int]]] = {}
    for k, sel in selection.items():
        path = asset_dir / f"syl_{k}.json"
        sig = [list(b) for b in sel]
        if not sel:
            continue
        if not refresh and path.exists():
            old = json.loads(path.read_text())
            if old.get("selection") == sig and old.get("version") == SYL_MEDIA_VERSION:
                info[str(k)] = {
                    "key": f"syl_{k}",
                    "n_tiles": len(old["tiles"]),
                    "bytes": path.stat().st_size,
                }
                continue
        todo[k] = sel
    if not todo:
        return info
    tile_dir = Path(tempfile.mkdtemp(prefix="syl_tiles_", dir=SYLLABLE_MEDIA["tmp_dir"]))
    try:
        by_ses: dict[str, list[tuple[int, int, int]]] = {}
        for k, sel in todo.items():
            for e, s, ln in sel:
                by_ses.setdefault(e, []).append((k, s, ln))
        for i, (e, bouts) in enumerate(by_ses.items()):
            arrays = dict(np.load(cache_dir / f"{e}.npz"))
            t0 = time.time()
            try:
                _render_tiles(metas[e]["sub"], metas[e]["ses"], bouts, arrays, media[e], tile_dir)
            except Exception as exc:  # noqa: BLE001 - the tiles of that session stay empty
                log.warning("syllable clips %s failed: %s", e, exc)
            log.info(
                "syllable clips %s: %d bouts (%d/%d, %.0f s)",
                e,
                len(bouts),
                i + 1,
                len(by_ses),
                time.time() - t0,
            )
        fps = float(np.median([media[e]["fps"] for e in by_ses]))
        for k, sel in todo.items():
            tiles, meta_tiles = [], []
            for e, s, ln in sel:
                p = tile_dir / f"{k}_{metas[e]['sub']}_{metas[e]['ses']}_{s}.npy"
                if not p.exists():
                    continue
                tiles.append(np.load(p))
                light = np.load(cache_dir / f"{e}.npz")["light_on"]
                meta_tiles.append(
                    {
                        "exp_id": e,
                        "start": int(s),
                        "len": int(ln),
                        "t_s": round(s / media[e]["fps"], 2),
                        "light_on": bool(light[s]) if s < light.size else None,
                    }
                )
            if not tiles:
                continue
            grid = qsm.tile_grid(tiles)
            mp4 = tile_dir / f"grid_{k}.mp4"
            qsm.encode_h264(grid, mp4, fps=fps)
            del grid, tiles
            payload = {
                "version": SYL_MEDIA_VERSION,
                "syllable": k,
                "selection": [list(b) for b in sel],
                "tiles": meta_tiles,
                "fps": fps,
                "pre": qsm.CLIP_PRE,
                "mp4": base64.b64encode(mp4.read_bytes()).decode("ascii"),
            }
            path = asset_dir / f"syl_{k}.json"
            path.write_text(json.dumps(payload, separators=(",", ":")))
            info[str(k)] = {
                "key": f"syl_{k}",
                "n_tiles": len(meta_tiles),
                "bytes": path.stat().st_size,
            }
    finally:
        import shutil

        shutil.rmtree(tile_dir, ignore_errors=True)
    return info


def collect_syllable_media(media: dict[str, dict], metas: dict[str, dict], refresh: bool) -> dict:
    """Pooled per-syllable page data, rendering grid movies if enabled."""
    pooled = qsm.pool_media(media)
    pooled["sessions"] = {
        e: {
            "mirrored": d["mirrored"],
            "left_on_left_frac": d["left_on_left_frac"],
            "frac_tracked": d["frac_tracked"],
            "has_video_scale": bool(d.get("video_scale")),
        }
        for e, d in media.items()
    }
    pooled["clip_params"] = {
        "pre": qsm.CLIP_PRE,
        "post": qsm.CLIP_POST,
        "max_frames": qsm.CLIP_MAX_FRAMES,
        "window_mm": SYL_WINDOW_MM,
        "tile_px": qsm.TILE_PX,
    }
    pooled["assets"] = {}
    if SYLLABLE_MEDIA["clips"] and pooled["syllables"]:
        sel = _bout_selection(pooled, media)
        pooled["assets"] = render_syllable_grids(sel, media, metas, refresh)
    return pooled


_MODEL: Any = None


def collect_rois(c: Ctx) -> dict:
    """Classifier QC for one session.

    Stored labels and probabilities come from ``roi_class.npy`` /
    ``roi_class_prob.npy`` when present, else from ca.h5 (``roi_types`` and
    ``roi_qc/p_*``; the EC2 stage 1 run classified inline and kept them only
    there). The classifier is then re-applied to features recomputed from the
    Suite2p outputs twice: at the frame rate the pipeline used
    (``ops["fs"]``), to check the stored labels can be reproduced, and at the
    imaging frame rate from ca.h5 (the rate of the training data), whose
    label changes measure the effect of the frame rate given to
    ``classify_session``.
    """
    from hm2p.extraction.roi_classify import load_model
    from hm2p.extraction.soma_features import FEATURE_COLUMNS, extract_soma_features

    global _MODEL
    base = f"ca_extraction/{c.sub}/{c.ses}/suite2p/plane0/"

    def npy(name: str, pickle: bool = False) -> Any:
        b = get_bytes(base + name)
        return None if b is None else np.load(io.BytesIO(b), allow_pickle=pickle)

    stat = npy("stat.npy", True)
    if stat is None:
        raise FileNotFoundError("stat.npy")
    stat = list(stat)
    if c._ca_bytes is False:  # first use this session: chunked download (see get_large_bytes)
        c._ca_bytes = get_large_bytes(f"calcium/{c.sub}/{c.ses}/ca.h5")
    ca = c.ca_labels()
    if ca is None:
        # every session has a ca.h5; None means the download failed, and a summary
        # without it (no roi_qc, no traces) must not be cached as a success
        raise FileNotFoundError(f"calcium/{c.sub}/{c.ses}/ca.h5 (missing or download failed)")
    ca_d, ca_attrs = ca
    roi_qc = {k.split("/", 1)[1]: v for k, v in ca_d.items() if k.startswith("roi_qc/")}
    labels, probs = npy("roi_class.npy"), npy("roi_class_prob.npy")
    ca_types = ca_d.get("roi_types")
    if labels is not None and probs is not None:
        source = "roi_class.npy"
    elif ca_types is not None and all(f"p_{k}" in roi_qc for k in qroi.LABELS):
        source = "ca.h5 (roi_types, roi_qc/p_*)"
        labels = np.array([qroi.CA_TO_CLASS.get(int(v), -1) for v in ca_types])
        probs = np.column_stack([roi_qc[f"p_{k}"] for k in qroi.LABELS])
        ca_types = None  # same source, so the consistency check would be trivial
    else:
        raise FileNotFoundError(
            "no classifier output (roi_class.npy, or ca.h5 roi_types + roi_qc/p_*)"
        )
    ops = npy("ops.npy", True)
    ops = ops.item() if ops is not None else {}
    F, Fneu = npy("F.npy"), npy("Fneu.npy")
    pipeline_fps = float(ops.get("fs", np.nan))
    imaging_fps = float(ca_attrs.get("fps_imaging", 0) or 0)
    if imaging_fps <= 0 and "frame_times" in ca_d:
        imaging_fps = span_fps(ca_d["frame_times"])
    feats = recomputed = None
    repro: dict = {
        "pipeline_fps": pipeline_fps,
        "imaging_fps": imaging_fps,
        "label_source": source,
    }
    if F is not None and Fneu is not None:
        if _MODEL is None:
            _MODEL = load_model()
        model, medians, _meta = _MODEL
        fill = pd.Series(medians, index=list(FEATURE_COLUMNS))

        def predict(fps: float) -> tuple[pd.DataFrame, np.ndarray]:
            f = extract_soma_features(stat, F.astype(np.float32), Fneu.astype(np.float32), fps=fps)
            return f, model.predict_proba(f.fillna(fill).values)

        if np.isfinite(pipeline_fps) and pipeline_fps > 0:
            _f, p_pipe = predict(pipeline_fps)
            repro["pipeline_max_abs_diff"] = (
                float(np.abs(p_pipe - probs).max()) if len(probs) else 0.0
            )
            repro["pipeline_label_changes"] = int((p_pipe.argmax(axis=1) != labels).sum())
        if imaging_fps > 0:
            feats, recomputed = predict(imaging_fps)
    asset_key = write_roi_asset(c, ops, stat, labels, F, Fneu, imaging_fps or pipeline_fps)
    s = qroi.summarise_rois(
        labels,
        probs,
        stat=stat,
        features=feats,
        # the full-resolution images are in the session asset; inline only as a fallback
        mean_img=None if asset_key else ops.get("meanImg"),
        ca_roi_types=ca_types,
        recomputed_probs=recomputed,
        roi_qc=roi_qc,
    )
    s["repro"] = repro
    s["asset"] = asset_key
    fields = qra.stat_fields(stat, npy("iscell.npy"))
    s["stat_fields"] = {
        k: [f.get(k) for f in fields] for k in (*qra.STAT_FIELDS, "iscell_prob") if fields
    }
    s["qc_thresholds"] = qra.roi_qc_thresholds()
    if recomputed is not None:
        s["imaging_fps_counts"] = {
            k: int((recomputed.argmax(axis=1) == i).sum()) for i, k in enumerate(qroi.LABELS)
        }
    return s


ROI_ASSET_CA_KEYS = ("F_corr", "dff", "spikes", "frame_times")
RANGE_CHUNK = 32 * 1024 * 1024


def get_large_bytes(key: str, chunk: int = RANGE_CHUNK, retries: int = 6) -> bytes | None:
    """Object body fetched in byte-range chunks, each retried on its own.

    ca.h5 is ~0.4 GB; a single ``get_object`` stream over the flaky container
    connection often fails part-way, and :func:`get_bytes` then restarts from
    zero. Returns None if the object does not exist or a chunk keeps failing.
    """
    try:
        size = int(s3().head_object(Bucket=BUCKET, Key=key)["ContentLength"])
    except Exception as exc:  # noqa: BLE001
        log.warning("head %s failed: %s", key, exc)
        return None
    buf = bytearray()
    for start in range(0, size, chunk):
        rng = f"bytes={start}-{min(start + chunk, size) - 1}"
        for attempt in range(retries):
            try:
                buf += s3().get_object(Bucket=BUCKET, Key=key, Range=rng)["Body"].read()
                break
            except Exception as exc:  # noqa: BLE001
                log.debug("get %s %s failed (attempt %d): %s", key, rng, attempt + 1, exc)
                time.sleep(2.0 * (attempt + 1))
        else:
            log.warning("ranged download failed: %s %s", key, rng)
            return None
    return bytes(buf) if len(buf) == size else None


def write_roi_asset(
    c: Ctx,
    ops: dict,
    stat: list,
    labels: np.ndarray,
    F: np.ndarray | None,
    Fneu: np.ndarray | None,
    fps: float,
) -> str | None:
    """Write the lazily loaded Suite2p inspection asset of one session.

    ``results/qc/assets/rois/<sub>_<ses>.json``: summary images, ROI pixel
    weights, registration offsets and the F / Fneu / F_corr / dF/F / CASCADE
    traces of the soma and dendrite ROIs (binned if needed to stay within
    ``qra.TRACE_BUDGET_BYTES``). Returns the asset key, or None when
    ``ops.npy`` lacks the frame size.
    """
    if "Ly" not in ops or "Lx" not in ops:
        return None
    n = len(stat)
    got = c.ca(lambda k: k in ROI_ASSET_CA_KEYS)
    ca_d = got[0] if got is not None else {}
    traces: dict[str, np.ndarray | None] = {"F": F, "Fneu": Fneu}
    for k in ("F_corr", "dff", "spikes"):
        traces[k] = ca_d.get(k)
    traces = {k: v for k, v in traces.items() if v is not None and np.ndim(v) == 2 and len(v) == n}
    light = None
    kin = c.kinematics()
    if kin is not None and "frame_times" in ca_d and "light_on" in kin[0]:
        kt, kl = kin[0]["frame_times"], kin[0]["light_on"].astype(float)
        light = kl[np.clip(np.searchsorted(kt, ca_d["frame_times"]), 0, kt.size - 1)]
    sel = np.flatnonzero(np.isin(np.asarray(labels), (1, 2)))  # soma, dendrite
    asset = qra.session_asset(ops, stat, traces, sel, fps=fps, light_on=light)
    key = f"{c.sub}_{c.ses}"
    path = OUT_DIR / "assets" / "rois" / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(asset), separators=(",", ":")))
    log.info("rois asset %s: %.2f MB (bin %d)", path.name, path.stat().st_size / 1e6, asset["bin"])
    del traces, ca_d
    return key


def collect_spikes(c: Ctx) -> dict:
    got = c.ca(lambda n: n in CA_KEYS_SPIKES or n.startswith("roi_qc/"))
    if got is None:
        raise FileNotFoundError("ca.h5")
    ca, attrs = got
    light = None
    kin = c.kinematics()
    if kin is not None and "frame_times" in ca and "light_on" in kin[0]:
        kt, kl = kin[0]["frame_times"], kin[0]["light_on"].astype(float)
        idx = np.clip(np.searchsorted(kt, ca["frame_times"]), 0, kt.size - 1)
        light = kl[idx]
    return qspk.summarise_spikes(ca, attrs, light_on=light)


COLLECTORS: dict[str, tuple[Callable[[Ctx], dict], Callable[[dict], dict]]] = {
    "tracking": (collect_tracking, qtrk.overview_row),
    "movement": (collect_movement, qmov.overview_row),
    "syllables": (collect_syllables, qsyl.overview_row),
    "rois": (collect_rois, qroi.overview_row),
    "spikes": (collect_spikes, qspk.overview_row),
}


# ---------------------------------------------------------------------------
# Project-level extras
# ---------------------------------------------------------------------------


def syllable_model_info() -> dict:
    out = {"summary": get_json("kinematics/kpms_summary.json")}
    for name in ("pca_variance.json", "fit_info.json", "convergence.json"):
        out[name.removesuffix(".json")] = get_json(f"kinematics/kpms_model/{name}")
    return out


def classifier_info(with_reference: bool) -> dict:
    meta_path = REPO / "sourcedata" / "trackers" / "suite2p" / "roi_classifier_xgb.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    out: dict = {"model": meta}
    ref_path = OUT_DIR / "cache" / "rois" / "_reference.json"
    if with_reference:
        from hm2p.extraction.roi_training_data import load_all_sessions

        X, y, groups = load_all_sessions()
        ref = qroi.cv_reference(X, y, groups, meta.get("best_params", {}), n_jobs=-1)
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        ref_path.write_text(json.dumps(ref))
    if ref_path.exists():
        out["reference"] = json.loads(ref_path.read_text())
    return out


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run(reports: list[str], only: set[str] | None, refresh: bool, with_reference: bool) -> None:
    table = sessions_table()
    if only:
        table = table[table["exp_id"].isin(only)]
    champion = get_json("dlc-champion.json")
    log.info("champion: %s", (champion or {}).get("champion_id"))
    results: dict[str, list[dict]] = {r: [] for r in reports}
    syl_media: dict[str, dict] = {}
    syl_metas: dict[str, dict] = {}
    for _, row in table.iterrows():
        meta = session_meta(row)
        ctx = Ctx(meta["sub"], meta["ses"], champion)
        if "syllables" in reports and SYLLABLE_MEDIA["enabled"]:
            d = syllable_media_cached(ctx, meta["exp_id"], refresh)
            if d is not None:
                syl_media[meta["exp_id"]], syl_metas[meta["exp_id"]] = d, meta
        for rep in reports:
            cache = OUT_DIR / "cache" / rep / f"{meta['exp_id']}.json"
            entry: dict = dict(meta)
            cached = json.loads(cache.read_text()) if cache.exists() and not refresh else None
            if cached is not None and cached.get("error") is None:
                entry.update(cached)
            else:  # no cache, --refresh, or a cached failure (retried: S3 errors are often transient)
                fn, ov = COLLECTORS[rep]
                t0 = time.time()
                try:
                    s = fn(ctx)
                    entry_data = {"summary": s, "overview": ov(s), "error": None}
                    log.info("%s %s ok (%.0f s)", rep, meta["exp_id"], time.time() - t0)
                except Exception as exc:  # noqa: BLE001 - recorded in the report
                    log.warning("%s %s failed: %s", rep, meta["exp_id"], exc)
                    log.debug(traceback.format_exc())
                    entry_data = {
                        "summary": None,
                        "overview": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(_clean(entry_data)))
                entry.update(entry_data)
            results[rep].append(entry)
    for rep in reports:
        doc: dict = {
            "report": rep,
            "generated": date.today().isoformat(),
            "champion": champion,
            "sessions": results[rep],
        }
        if rep == "syllables":
            doc["model"] = syllable_model_info()
            if SYLLABLE_MEDIA["enabled"]:
                doc["media"] = collect_syllable_media(syl_media, syl_metas, refresh)
        if rep == "rois":
            doc["classifier"] = classifier_info(with_reference)
        if rep == "tracking":
            doc["model"] = tracking_model_info()
        if rep == "movement":
            doc["maze_cells"] = maze_cells()
        path = OUT_DIR / "data" / f"{rep}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_clean(doc), separators=(",", ":")))
        n_ok = sum(1 for e in results[rep] if e.get("error") is None)
        log.info(
            "wrote %s (%d/%d sessions ok, %.1f MB)",
            path,
            n_ok,
            len(results[rep]),
            path.stat().st_size / 1e6,
        )


def _clean(o: Any) -> Any:
    """Make numpy scalars and non-finite floats JSON-safe."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, bytes):
        return o.decode(errors="replace")
    return o


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - S3 entry point
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reports", nargs="+", choices=REPORTS, default=list(REPORTS))
    ap.add_argument("--sessions", nargs="+", help="exp_id values (default: all)")
    ap.add_argument("--refresh", action="store_true", help="ignore cached per-session summaries")
    ap.add_argument(
        "--classifier-reference", action="store_true", help="LOSO CV on /data/s2p manual labels"
    )
    ap.add_argument(
        "--no-syllable-media",
        action="store_true",
        help="skip the per-syllable trajectories, occupancy and grid movies",
    )
    ap.add_argument(
        "--no-syllable-clips", action="store_true", help="skip rendering the grid movies"
    )
    ap.add_argument("--tmp-dir", help="directory for downloaded videos and tile clips")
    ap.add_argument(
        "--no-tracking-frames",
        action="store_true",
        help="skip the tracking frame viewer (downloads each raw overhead video)",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    TRACKING_MEDIA.update(frames=not a.no_tracking_frames, tmp_dir=a.tmp_dir)
    SYLLABLE_MEDIA.update(
        enabled=not a.no_syllable_media, clips=not a.no_syllable_clips, tmp_dir=a.tmp_dir
    )
    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO, format="%(levelname)s %(message)s"
    )
    run(a.reports, set(a.sessions) if a.sessions else None, a.refresh, a.classifier_reference)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
