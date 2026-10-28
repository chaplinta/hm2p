"""Tests for the syllable-media collectors in scripts/make_qc_reports.py (no S3)."""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import make_qc_reports as mqr  # noqa: E402

from hm2p.qc import syllable_media as qsm  # noqa: E402

N = 600


class FakeCtx:
    sub, ses, champion = "sub-1", "ses-20990101T100000", {"champion_id": "x"}

    def __init__(self, kin):
        self._kin = kin

    def kinematics(self):
        return self._kin


def _pose(n: int = N) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Animal walking along +x in the image, facing +x."""
    kps = list(qsm.TRAJ_KEYPOINTS)
    offs = {  # (forward, right) in px relative to the centre; right of +x heading is +y
        "nose_tip": (40, 0),
        "left_ear": (28, -8),
        "right_ear": (28, 8),
        "head_midpoint": (28, 0),
        "neck": (20, 0),
        "mid_back": (5, 0),
        "mouse_center": (0, 0),
        "tail_base": (-30, 0),
    }
    pos = np.zeros((n, len(kps), 2))
    for i, k in enumerate(kps):
        pos[:, i, 0] = 100 + 0.5 * np.arange(n) + offs[k][0]
        pos[:, i, 1] = 200 + offs[k][1]
    return pos, np.ones((n, len(kps))), kps


def _kin(n: int = N) -> tuple[dict, dict]:
    bad = np.zeros(n, bool)
    bad[500:520] = True
    k = {
        "frame_times": np.arange(n) / 30.0,
        "bad_behav": bad,
        "speed_cm_s": np.full(n, 3.0),
        "x_maze": np.full(n, 1.1),
        "y_maze": np.full(n, 2.2),
        "light_on": np.arange(n) % 200 < 100,
    }
    return k, {"scale_mm_per_px": 0.5}


@pytest.fixture
def media_env(monkeypatch, tmp_path):
    sid = np.repeat(np.arange(N // 20) % 3, 20).astype(np.int16)
    npz = io.BytesIO()
    np.savez(npz, syllable_id=sid)
    store = {f"kinematics/{FakeCtx.sub}/{FakeCtx.ses}/syllables.npz": npz.getvalue()}
    monkeypatch.setattr(mqr, "OUT_DIR", tmp_path)
    monkeypatch.setattr(mqr, "get_bytes", lambda key, retries=4: store.get(key))
    monkeypatch.setattr(mqr, "_syllable_pose", lambda c: _pose())
    monkeypatch.setattr(mqr, "_video_scale", lambda sub, ses: (0.5, 0.5))
    return FakeCtx(_kin())


def test_video_scale_reads_crop(monkeypatch, tmp_path):
    d = tmp_path / "metadata" / "video_meta"
    d.mkdir(parents=True)
    (d / "sub-1_ses-2_meta.txt").write_text("[crop]\nwidth = 832\nheight = 640\n")
    (d / "sub-1_ses-3_meta.txt").write_text("[scale]\nmm_per_pix = 1\n")
    monkeypatch.setattr(mqr, "REPO", tmp_path)
    assert mqr._video_scale("sub-1", "ses-2") == pytest.approx((0.5, 0.475))
    assert mqr._video_scale("sub-1", "ses-3") is None
    assert mqr._video_scale("sub-1", "ses-4") is None


def test_download_file(monkeypatch, tmp_path):
    monkeypatch.setattr(mqr, "get_large_bytes", lambda key, **kw: b"abc" if key == "k" else None)
    mqr._download_file("k", tmp_path / "f")
    assert (tmp_path / "f").read_bytes() == b"abc"
    with pytest.raises(FileNotFoundError):
        mqr._download_file("missing", tmp_path / "g")


def test_syllable_session_media(media_env):
    d, arrays = mqr.syllable_session_media(media_env)
    assert d["n_frames"] == N and not d["mirrored"]
    assert d["left_on_left_frac"] == pytest.approx(1.0)
    assert set(d["syllables"]) == {"0", "1", "2"}
    s0 = d["syllables"]["0"]
    assert sum(s0["lengths"]) == s0["n_frames"] == 200
    # no clip window overlaps the bad_behav frames 500-519 or leaves the session
    for start, ln in s0["clips"]:
        first, last, _, _ = qsm.clip_window(start, ln)
        assert first >= 0 and last <= N - 1 and (last <= 500 or first >= 520)
    # trajectory: at onset the nose is 40 px = 20 mm straight ahead, tail base 15 mm behind
    t = np.asarray(s0["traj_sum"]) / np.asarray(s0["traj_count"])[..., None]
    i0 = d["traj_offsets"].index(0)
    np.testing.assert_allclose(t[i0, 0], [0.0, 20.0], atol=1e-6)
    np.testing.assert_allclose(t[i0, 7], [0.0, -15.0], atol=1e-6)
    np.testing.assert_allclose(t[i0, 2], [4.0, 14.0], atol=1e-6)  # right ear on the right
    assert s0["turn_n"] > 0 and abs(s0["turn_sum"]) < 1e-6
    assert np.asarray(s0["occ"]).sum() <= 200
    assert arrays["centre_vid"].shape == (N, 2)
    np.testing.assert_allclose(arrays["centre_vid"][0], [50.0, 100.0])
    np.testing.assert_allclose(arrays["heading_vid"], 0.0, atol=1e-9)


def test_syllable_session_media_length_mismatch(media_env):
    k, a = _kin(N - 1)
    media_env._kin = (k, a)
    with pytest.raises(ValueError):
        mqr.syllable_session_media(media_env)


def test_syllable_media_cached(media_env, tmp_path):
    d = mqr.syllable_media_cached(media_env, "e1", refresh=False)
    assert d is not None and (tmp_path / "cache" / "syllables_media" / "e1.npz").exists()
    media_env._kin = None  # would fail now, but the cache is used
    again = mqr.syllable_media_cached(media_env, "e1", refresh=False)
    assert again == json.loads(json.dumps(mqr._clean(d)))
    assert mqr.syllable_media_cached(media_env, "e1", refresh=True) is None


def test_collect_syllable_media_renders_grids(media_env, monkeypatch, tmp_path):
    for e in ("e1", "e2"):
        mqr.syllable_media_cached(media_env, e, refresh=False)
    media = {
        e: json.loads((tmp_path / "cache" / "syllables_media" / f"{e}.json").read_text())
        for e in ("e1", "e2")
    }
    metas = {e: {"sub": f"sub-{e}", "ses": "ses-x"} for e in media}
    calls = []

    def fake_tiles(sub, ses, bouts, arrays, d, tile_dir):
        calls.append(sub)
        for syl, start, _ln in bouts:
            np.save(
                tile_dir / f"{syl}_{sub}_{ses}_{start}.npy", np.full((6, 8, 8, 3), 90, np.uint8)
            )

    monkeypatch.setattr(mqr, "_render_tiles", fake_tiles)
    monkeypatch.setitem(mqr.SYLLABLE_MEDIA, "clips", True)
    out = mqr.collect_syllable_media(media, metas, refresh=False)
    assert [r["id"] for r in out["syllables"]] == [0, 1, 2]
    assert out["sessions"]["e1"]["has_video_scale"]
    assert set(out["assets"]) == {"0", "1", "2"} and sorted(calls) == ["sub-e1", "sub-e2"]
    a = json.loads((tmp_path / "assets" / "syllables" / "syl_0.json").read_text())
    assert a["tiles"] and b"ftyp" in base64.b64decode(a["mp4"])[:64]
    assert all(t["exp_id"] in media for t in a["tiles"])
    # same selection: nothing is rendered again
    calls.clear()
    again = mqr.collect_syllable_media(media, metas, refresh=False)
    assert calls == [] and set(again["assets"]) == {"0", "1", "2"}


def test_collect_syllable_media_without_clips(media_env, monkeypatch, tmp_path):
    d = mqr.syllable_media_cached(media_env, "e1", refresh=False)
    monkeypatch.setitem(mqr.SYLLABLE_MEDIA, "clips", False)
    out = mqr.collect_syllable_media({"e1": d}, {"e1": {"sub": "s", "ses": "t"}}, refresh=False)
    assert out["assets"] == {} and out["syllables"]
    assert mqr.collect_syllable_media({}, {}, refresh=False)["syllables"] == []


def test_render_tiles_from_video(monkeypatch, tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    n = 40
    frames = np.zeros((n, 64, 64, 3), np.uint8)
    frames[:, 28:36, 28:36] = 255  # bright square at the centre
    src = qsm.encode_h264(frames, tmp_path / "v.mp4", fps=30, crf=10)
    monkeypatch.setattr(
        mqr, "_download_file", lambda key, path: path.write_bytes(src.read_bytes())
    )
    arrays = {"centre_vid": np.full((n, 2), 32.0), "heading_vid": np.full(n, -np.pi / 2)}
    d = {"scale_mm_per_px": 1.0, "video_scale": [1.0, 1.0], "mirrored": False}
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    # (syllable, start, len): two overlapping clips, one that runs past the video end
    mqr._render_tiles("s", "t", [(1, 12, 5), (2, 14, 3), (3, 35, 5)], arrays, d, tiles)
    a = np.load(tiles / "1_s_t_12.npy")
    assert a.shape == (5 + qsm.CLIP_PRE + qsm.CLIP_POST, qsm.TILE_PX, qsm.TILE_PX, 3)
    c = qsm.TILE_PX // 2
    assert a[0, c, c].mean() > 200  # the centre of the frame lands at the tile centre
    assert a[qsm.CLIP_PRE, 0, c].tolist() == list(qsm.BORDER_BGR)  # border during the bout
    assert a[0, 0, c].tolist() != list(qsm.BORDER_BGR)
    assert (tiles / "2_s_t_14.npy").exists()
    assert not (tiles / "3_s_t_35.npy").exists()
