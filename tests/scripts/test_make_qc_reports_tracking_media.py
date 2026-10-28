"""Tests for the tracking / movement media helpers in scripts/make_qc_reports.py.

S3 is replaced by in-memory synthetic files; videos are small synthetic mp4s.
"""

from __future__ import annotations

import io
import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import make_qc_reports as mqr  # noqa: E402

from tests.qc.test_movement import _kin  # noqa: E402
from tests.qc.test_tracking import _mouse  # noqa: E402


def _video(path: Path, n: int, fps: float, size=(64, 48)) -> None:
    """Synthetic grey video: frame i has mean brightness 2 * i (mod 256)."""
    import cv2

    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(n):
        w.write(np.full((size[1], size[0], 3), (2 * i) % 256, np.uint8))
    w.release()


def _h5_bytes(datasets: dict, attrs: dict) -> bytes:
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        for k, v in datasets.items():
            f.create_dataset(k, data=v)
        for k, v in attrs.items():
            f.attrs[k] = v
    return buf.getvalue()


def test_read_frames_subsampled_counts_and_selects(tmp_path):
    vid = tmp_path / "v.mp4"
    _video(vid, n=50, fps=100.0)  # 0.5 s at 100 fps -> 15 frames at 30 fps
    got, n = mqr.read_frames_subsampled(vid, {0, 5}, fps=30)
    assert 15 <= n <= 17 and set(got) == {0, 5}  # ffmpeg may pad the last frame's duration
    assert got[0].shape == (48, 64) and got[0].dtype == np.uint8
    # later output frames come from later input frames (brightness rises 2 per input frame)
    assert float(got[5].mean()) > float(got[0].mean()) + 10


def test_read_frames_seek(tmp_path):
    vid = tmp_path / "v.mp4"
    _video(vid, n=10, fps=30.0)
    got = mqr.read_frames_seek(vid, {3, 99})
    assert set(got) == {3} and abs(float(got[3].mean()) - 6) < 6


def test_write_asset_and_model_info(monkeypatch, tmp_path):
    monkeypatch.setattr(mqr, "OUT_DIR", tmp_path)
    size = mqr._write_asset("tracking", "k", {"a": np.float32(1.0), "b": float("nan")})
    assert size > 0
    assert json.loads((tmp_path / "assets/tracking/k.json").read_text()) == {"a": 1.0, "b": None}
    store = {
        "dlc-champion-history/promotions.log": b"2026-01-01T00:00:00Z\tdlc-a\ti-1\tabc\nbad\n",
        "dlc-retrain/_eval_results.json": b'{"champion_id": "dlc-a"}',
    }
    monkeypatch.setattr(mqr, "get_bytes", lambda k, retries=4: store.get(k))
    info = mqr.tracking_model_info()
    assert info["history"] == [
        {"promoted_at": "2026-01-01T00:00:00Z", "champion_id": "dlc-a", "instance": "i-1",
         "git_sha": "abc"}
    ]  # fmt: skip
    assert info["eval"]["champion_id"] == "dlc-a" and info["verdict"] is None


def test_maze_cells_and_illumination(monkeypatch):
    cells = mqr.maze_cells()
    assert len(cells) == 23 and [0, 0] in cells
    t = np.arange(0.0, 200.0)
    d = {"frame_times": t, "intensities": np.ones(t.size), "light_on_times": np.array([60.0])}
    store = {"analysis/s/t/illumination.h5": _h5_bytes(d, {"decay_pct": 1.0})}
    monkeypatch.setattr(mqr, "get_bytes", lambda k, retries=4: store.get(k))
    assert mqr.illumination_summary(mqr.Ctx("s", "t", None))["on_step_median"] == 0.0
    assert mqr.illumination_summary(mqr.Ctx("x", "y", None)) is None


def test_collect_movement_writes_trajectory_asset(monkeypatch, tmp_path):
    monkeypatch.setattr(mqr, "OUT_DIR", tmp_path)
    store = {"kinematics/s/t/kinematics.h5": _h5_bytes(_kin(n=300), {"dlc_champion_id": "c"})}
    monkeypatch.setattr(mqr, "get_bytes", lambda k, retries=4: store.get(k))
    s = mqr.collect_movement(mqr.Ctx("s", "t", {"champion_id": "c"}))
    a = json.loads((tmp_path / "assets/movement/traj-s_t.json").read_text())
    assert s["traj_asset"]["key"] == "traj-s_t" and a["t_s"]["n"] == 100
    assert "bouts" in s


class _FakeS3:
    def __init__(self, files: dict[tuple[str, str], Path]):
        self.files = files

    def list_objects_v2(self, Bucket, Prefix):  # noqa: N803 - boto3 signature
        return {
            "Contents": [{"Key": k} for b, k in self.files if b == Bucket and k.startswith(Prefix)]
        }

    def download_file(self, bucket, key, dest):
        if (bucket, key) not in self.files:
            raise FileNotFoundError(key)
        shutil.copy(self.files[(bucket, key)], dest)


@pytest.mark.parametrize(
    "match", [True, False]
)  # pose length equal to / unlike the resampled video
def test_tracking_frames_asset_raw_and_fallback(monkeypatch, tmp_path, match):
    from hm2p.qc.tracking import summarise_tracking

    monkeypatch.setattr(mqr, "OUT_DIR", tmp_path)
    monkeypatch.setitem(mqr.TRACKING_MEDIA, "tmp_dir", str(tmp_path))
    raw, lab = tmp_path / "raw.mp4", tmp_path / "lab.mp4"
    _video(raw, n=50, fps=100.0)
    n = mqr.read_frames_subsampled(raw, set())[1] + (0 if match else 3)
    _video(lab, n=n, fps=30.0, size=(32, 24))
    sub, ses = "sub-1", "ses-2"
    s3 = _FakeS3({
        ("hm2p-rawdata", f"rawdata/{sub}/{ses}/behav/x_overhead.camera-cropped.mp4"): raw,
        ("hm2p-derivatives", f"pose/{sub}/{ses}/labelled_30fps.mp4"): lab,
    })  # fmt: skip
    monkeypatch.setattr(mqr, "s3", lambda: s3)
    monkeypatch.setattr(mqr, "_video_crop_size", lambda a, b: (64.0, 48.0))
    kp = _mouse(n=n)
    s = summarise_tracking(kp, fps=30.0, mm_per_px=0.5)
    s["worst_bins"] = [{"t_min": 0.0, "rank": 0.1}]
    kin = _kin(n=n)
    kin.update(x_head_mm=np.full(n, 10.0), y_head_mm=np.full(n, 5.0))
    out = mqr.tracking_frames_asset(
        mqr.Ctx(sub, ses, None), kp, s, (kin, {"scale_mm_per_px": 0.5, "orientation_deg": 0.0,
                                               "confidence_threshold": "0.5"}), (32.0, 24.0)
    )  # fmt: skip
    assert out is not None and out["key"] == "frames-sub-1_ses-2"
    a = json.loads((tmp_path / "assets/tracking/frames-sub-1_ses-2.json").read_text())
    assert a["source"] == ("raw" if match else "labelled")
    assert (a["width"], a["height"]) == (64.0, 48.0) and a["cutoff"] == 0.5
    f0 = a["frames"][0]
    assert f0["jpeg"] and set(f0["kp"]) >= {"nose_tip", "tail_base"}
    assert f0["head"] == [20.0, 10.0]  # 10 mm / 0.5 mm per px, no rotation
    assert len(f0["hd"]) == 2 and "light" in f0
    assert out["frames"][0]["frame"] == f0["frame"]
