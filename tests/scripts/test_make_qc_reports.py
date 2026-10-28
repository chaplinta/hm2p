"""Tests for scripts/make_qc_reports.py with S3 replaced by in-memory synthetic files."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import make_qc_reports as mqr  # noqa: E402

from tests.qc.test_movement import _kin  # noqa: E402
from tests.qc.test_spikes import _ca  # noqa: E402


def _h5_bytes(datasets: dict, attrs: dict) -> bytes:
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        for k, v in datasets.items():
            f.create_dataset(k, data=v)
        for k, v in attrs.items():
            f.attrs[k] = v
    return buf.getvalue()


@pytest.fixture
def fake_s3(monkeypatch):
    monkeypatch.setitem(mqr.SYLLABLE_MEDIA, "enabled", False)  # tested in *_syllable_media.py
    sub, ses = "sub-0000001", "ses-20990101T100000"
    kin = _kin(n=3000)
    kin["bp_nose_tip_x_maze"] = np.zeros(3000)  # skipped by the reader
    ca = _ca()
    ca["frame_times"] = np.arange(3000) / 10.0
    sid = np.repeat(np.arange(300) % 25, 10).astype(np.int16)
    npz = io.BytesIO()
    np.savez(npz, syllable_id=sid)
    store = {
        f"kinematics/{sub}/{ses}/kinematics.h5": _h5_bytes(
            kin, {"dlc_champion_id": "dlc-x-snap1"}
        ),
        f"calcium/{sub}/{ses}/ca.h5": _h5_bytes(ca, {"fps_imaging": 10.0, "f0_method": "rolling"}),
        f"kinematics/{sub}/{ses}/syllables.npz": npz.getvalue(),
        f"kinematics/{sub}/{ses}/syllables.provenance.json": json.dumps(
            {"kappa": 1e6, "num_pcs": 4}
        ).encode(),
        "dlc-champion.json": json.dumps({"champion_id": "dlc-x-snap1"}).encode(),
    }
    monkeypatch.setattr(mqr, "get_bytes", lambda key, retries=4: store.get(key))
    monkeypatch.setattr(mqr, "get_large_bytes", lambda key, **kw: store.get(key))
    return sub, ses


def test_clean_handles_numpy_nan_and_bytes():
    out = mqr._clean({"a": np.float32(1.5), "b": float("nan"), "c": [np.int64(2), b"x"], 3: (1,)})
    assert out == {"a": 1.5, "b": None, "c": [2, "x"], "3": [1]}
    json.dumps(out, allow_nan=False)


def test_session_meta_flags():
    row = pd.Series(
        {
            "exp_id": "e",
            "sub": "s",
            "ses": "t",
            "animal_id": "1",
            "celltype": "penk",
            "exclude": 1,
            "primary_exp": 0,
            "Notes": float("nan"),
        }
    )
    m = mqr.session_meta(row)
    assert m["exclude"] is True and m["primary"] is False and m["notes"] is None


def test_read_h5_filters_and_missing(fake_s3):
    sub, ses = fake_s3
    data, attrs = mqr.read_h5(
        f"kinematics/{sub}/{ses}/kinematics.h5", lambda n: not n.startswith("bp_")
    )
    assert "hd_deg" in data and "bp_nose_tip_x_maze" not in data
    assert attrs["dlc_champion_id"] == "dlc-x-snap1"
    assert mqr.read_h5("nope.h5") is None


def test_collectors_movement_spikes_syllables(fake_s3):
    sub, ses = fake_s3
    ctx = mqr.Ctx(sub, ses, {"champion_id": "dlc-x-snap1"})
    mv = mqr.collect_movement(ctx)
    assert mv["champion_current"] is True and mv["n_frames"] == 3000
    sp = mqr.collect_spikes(ctx)
    assert sp["n_rois"] == 3 and "light" in sp["population"]
    sy = mqr.collect_syllables(ctx)
    assert sy["aligned_to_kinematics"] and sy["provenance"]["num_pcs"] == 4
    assert sy["fps"] == pytest.approx(30.0)


def _npy(a, pickle=False):
    b = io.BytesIO()
    np.save(b, a, allow_pickle=pickle)
    return b.getvalue()


def test_collect_rois_recomputes_with_model(monkeypatch, fake_s3):
    from tests.qc.test_rois import _disk

    rng = np.random.default_rng(0)
    n, T = 6, 600
    stat = np.array(
        [
            dict(
                _disk(20 + 15 * i, 30, 4),
                radius=4.0,
                compact=1.0,
                aspect_ratio=1.0,
                skew=1.0,
                std=1.0,
                npix_norm=1.0,
                lam=np.ones(len(_disk(0, 0, 4)["ypix"])),
                overlap=np.zeros(len(_disk(0, 0, 4)["ypix"]), bool),
                soma_crop=np.ones(len(_disk(0, 0, 4)["ypix"]), bool),
            )
            for i in range(n)
        ],
        dtype=object,
    )
    F = (100 + rng.normal(0, 5, (n, T))).astype(np.float32)
    probs = np.tile([0.2, 0.7, 0.1], (n, 1)).astype(np.float32)
    base = "ca_extraction/sub-r/ses-r/suite2p/plane0/"
    ca = _h5_bytes(
        {
            "roi_types": np.zeros(n, dtype=np.uint8),  # soma
            "roi_qc/p_artefact": probs[:, 0],
            "roi_qc/p_soma": probs[:, 1],
            "roi_qc/p_dend": probs[:, 2],
        },
        {"fps_imaging": 9.77},
    )
    store = {
        base + "stat.npy": _npy(stat, True),
        base + "ops.npy": _npy(
            np.array({"fs": 29.97, "meanImg": rng.normal(size=(128, 64))}, dtype=object), True
        ),
        base + "F.npy": _npy(F),
        base + "Fneu.npy": _npy((F * 0.5).astype(np.float32)),
        "calcium/sub-r/ses-r/ca.h5": ca,
    }
    monkeypatch.setattr(mqr, "get_bytes", lambda key, retries=4: store.get(key))
    monkeypatch.setattr(mqr, "get_large_bytes", lambda key, **kw: store.get(key))
    # labels from ca.h5 when roi_class.npy is absent (the EC2 stage 1 run)
    s = mqr.collect_rois(mqr.Ctx("sub-r", "ses-r", None))
    assert s["n_rois"] == n and len(s["feature_names"]) == 26
    assert s["counts"]["soma"] == n
    assert s["repro"]["label_source"].startswith("ca.h5")
    assert s["repro"]["pipeline_fps"] == 29.97 and s["repro"]["imaging_fps"] == 9.77
    assert s["repro"]["pipeline_label_changes"] is not None
    assert s["recomputed_label_changes"] is not None and "ca_label_mismatch" not in s
    assert sum(s["imaging_fps_counts"].values()) == n
    assert s["image"]["width"] == 64 and s["rois"][0]["outline"]
    # roi_class.npy takes precedence and enables the ca.h5 consistency check
    store[base + "roi_class.npy"] = _npy(np.ones(n, dtype=np.int8))
    store[base + "roi_class_prob.npy"] = _npy(probs)
    s2 = mqr.collect_rois(mqr.Ctx("sub-r", "ses-r", None))
    assert s2["repro"]["label_source"] == "roi_class.npy" and s2["ca_label_mismatch"] == 0
    assert s["asset"] is None  # ops.npy without Ly / Lx: no asset, inline mean image kept
    assert s["stat_fields"]["radius"] == [4.0] * n and s["qc_thresholds"]["snr_event"][0] == 3.0


def test_collect_rois_writes_session_asset(monkeypatch, tmp_path):
    from hm2p.qc.common import decode_i16
    from tests.qc.test_rois import _disk

    rng = np.random.default_rng(1)
    n, T = 3, 400
    stat = np.array(
        [dict(_disk(10 + 12 * i, 20, 3), lam=np.ones(29)) for i in range(n)], dtype=object
    )
    F = (100 + rng.normal(0, 5, (n, T))).astype(np.float32)
    base = "ca_extraction/sub-a/ses-a/suite2p/plane0/"
    ops = {
        "Ly": 48,
        "Lx": 40,
        "yrange": [2, 46],
        "xrange": [1, 39],
        "fs": 9.6,
        "meanImg": rng.normal(size=(48, 40)),
        "max_proj": rng.normal(size=(44, 38)),
        "xoff": rng.integers(-2, 3, T),
        "yoff": rng.integers(-2, 3, T),
    }
    ca = _h5_bytes(
        {
            "dff": rng.normal(size=(n, T)).astype(np.float32),
            "F_corr": F * 0.9,
            "spikes": rng.uniform(0, 1, (n, T)).astype(np.float32),
            "frame_times": np.arange(T) / 9.6,
        },
        {"fps_imaging": 9.6},
    )
    kin = _h5_bytes(
        {"frame_times": np.arange(T * 3) / 28.8, "light_on": np.repeat([1, 0], T * 3 // 2)}, {}
    )
    store = {
        base + "stat.npy": _npy(stat, True),
        base + "ops.npy": _npy(np.array(ops, dtype=object), True),
        base + "F.npy": _npy(F),
        base + "Fneu.npy": _npy(F * 0.5),
        base + "iscell.npy": _npy(np.array([[1, 0.9], [0, 0.2], [1, 0.7]])),
        base + "roi_class.npy": _npy(np.array([1, 0, 2], dtype=np.int8)),
        base + "roi_class_prob.npy": _npy(
            np.array([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.1, 0.2, 0.7]])
        ),
        "calcium/sub-a/ses-a/ca.h5": ca,
        "kinematics/sub-a/ses-a/kinematics.h5": kin,
    }
    monkeypatch.setattr(mqr, "get_bytes", lambda key, retries=4: store.get(key))
    monkeypatch.setattr(mqr, "get_large_bytes", lambda key, **kw: store.get(key))
    monkeypatch.setattr(mqr, "OUT_DIR", tmp_path)
    ctx = mqr.Ctx("sub-a", "ses-a", None)
    key = mqr.write_roi_asset(ctx, ops, list(stat), np.array([1, 0, 2]), F, F * 0.5, 9.6)
    assert key == "sub-a_ses-a"
    a = json.loads((tmp_path / "assets" / "rois" / f"{key}.json").read_text())
    assert a["trace_rois"] == [0, 2]
    assert set(a["traces"]) == {"F", "Fneu", "F_corr", "dff", "spikes"}
    assert set(a["images"]) == {"meanImg", "max_proj"} and a["image_shape"] == [48, 40]
    np.testing.assert_allclose(decode_i16(a["traces"]["F"][1]), F[2], atol=1e-2)
    assert a["dark"] == [[T // 2, T]] and "xoff" in a["registration"]
    assert mqr.write_roi_asset(ctx, {}, list(stat), np.array([1, 0, 2]), F, F, 9.6) is None
    # a failed ca.h5 download is an error (retried next run), not a cached partial summary
    del store["calcium/sub-a/ses-a/ca.h5"]
    with pytest.raises(FileNotFoundError, match="ca.h5"):
        mqr.collect_rois(mqr.Ctx("sub-a", "ses-a", None))


def test_collect_tracking_reads_dlc_file(monkeypatch, tmp_path):
    from tests.qc.test_tracking import _mouse

    kp = _mouse(n=300)
    cols = pd.MultiIndex.from_tuples(
        [("scorer", b, c) for b in kp for c in ("x", "y", "likelihood")],
        names=["scorer", "bodyparts", "coords"],
    )
    df = pd.DataFrame(
        np.column_stack([kp[b][c] for b in kp for c in ("x", "y", "likelihood")]), columns=cols
    )
    path = tmp_path / "pose.h5"
    df.to_hdf(path, key="df_with_missing")
    key = "pose/sub-t/ses-t/videoDLC_HrnetW32_snapshot-best-7.h5"
    store = {key: path.read_bytes(), "pose/sub-t/ses-t/dlc_meta.json": b'{"tracking_fps": 30}'}
    monkeypatch.setattr(mqr, "get_bytes", lambda k, retries=4: store.get(k))
    monkeypatch.setattr(
        mqr, "list_keys", lambda prefix: [k for k in store if k.startswith(prefix)]
    )
    s = mqr.collect_tracking(
        mqr.Ctx("sub-t", "ses-t", {"champion_id": "dlc-20990101-hrnetw32-snap7"})
    )
    assert s["n_frames"] == 300 and s["pose_file"].endswith("snapshot-best-7.h5")
    assert set(s["bodyparts"]) >= {"nose_tip", "left_ear", "tail_base"}
    assert s["light_from_kinematics"] is False


def test_get_bytes_retries_connection_errors(monkeypatch):
    class ConnErr(Exception):
        response = None  # botocore connection errors have no response dict

    calls = []

    class FakeS3:
        def get_object(self, **kw):
            calls.append(kw)
            raise ConnErr("could not connect")

    monkeypatch.setattr(mqr, "s3", lambda: FakeS3())
    monkeypatch.setattr(mqr.time, "sleep", lambda s: None)
    assert mqr.get_bytes("k", retries=3) is None
    assert len(calls) == 3


def test_failed_sessions_are_retried_not_cached(fake_s3, tmp_path, monkeypatch):
    table = pd.DataFrame(
        [
            {
                "exp_id": "e1",
                "sub": fake_s3[0],
                "ses": fake_s3[1],
                "animal_id": "1",
                "celltype": "penk",
                "exclude": 0,
                "primary_exp": 0,
                "Notes": None,
            }
        ]
    )
    monkeypatch.setattr(mqr, "sessions_table", lambda: table)
    monkeypatch.setattr(mqr, "OUT_DIR", tmp_path)
    real = mqr.get_bytes
    monkeypatch.setattr(mqr, "get_bytes", lambda key, retries=4: None)  # S3 down
    mqr.run(["movement"], None, refresh=False, with_reference=False)
    assert json.loads((tmp_path / "data" / "movement.json").read_text())["sessions"][0]["error"]
    monkeypatch.setattr(mqr, "get_bytes", real)  # S3 back
    mqr.run(["movement"], None, refresh=False, with_reference=False)
    assert (
        json.loads((tmp_path / "data" / "movement.json").read_text())["sessions"][0]["error"]
        is None
    )


def test_collector_missing_file_raises(fake_s3):
    ctx = mqr.Ctx("sub-x", "ses-y", None)
    with pytest.raises(FileNotFoundError):
        mqr.collect_movement(ctx)


def test_run_writes_report_with_errors_recorded(fake_s3, tmp_path, monkeypatch):
    table = pd.DataFrame(
        [
            {
                "exp_id": "20990101_10_00_00_0000001",
                "sub": fake_s3[0],
                "ses": fake_s3[1],
                "animal_id": "0000001",
                "celltype": "penk",
                "exclude": 0,
                "primary_exp": 1,
                "Notes": None,
            },
            {
                "exp_id": "20990102_10_00_00_0000002",
                "sub": "sub-0000002",
                "ses": "ses-20990102T100000",
                "animal_id": "0000002",
                "celltype": "nonpenk",
                "exclude": 0,
                "primary_exp": 0,
                "Notes": None,
            },
        ]
    )
    monkeypatch.setattr(mqr, "sessions_table", lambda: table)
    monkeypatch.setattr(mqr, "OUT_DIR", tmp_path)
    mqr.run(["movement"], None, refresh=False, with_reference=False)
    doc = json.loads((tmp_path / "data" / "movement.json").read_text())
    ok, bad = doc["sessions"]
    assert ok["error"] is None and ok["overview"]["fps"] == pytest.approx(30.0)
    assert bad["summary"] is None and "FileNotFoundError" in bad["error"]
    assert (tmp_path / "cache" / "movement" / "20990101_10_00_00_0000001.json").exists()
    # second run uses the cache
    monkeypatch.setattr(mqr, "get_bytes", lambda key, retries=4: None)
    mqr.run(["movement"], None, refresh=False, with_reference=False)
    doc2 = json.loads((tmp_path / "data" / "movement.json").read_text())
    assert doc2["sessions"][0]["error"] is None


def test_get_large_bytes_retries_each_chunk(monkeypatch):
    body = bytes(range(256)) * 10
    calls = {"n": 0}

    class Body:
        def __init__(self, b: bytes) -> None:
            self.b = b

        def read(self) -> bytes:
            return self.b

    class Fake:
        fail = False

        def head_object(self, Bucket, Key):  # noqa: N803
            if Key == "missing":
                raise RuntimeError("404")
            return {"ContentLength": len(body)}

        def get_object(self, Bucket, Key, Range):  # noqa: N803
            calls["n"] += 1
            if self.fail or calls["n"] % 3 == 1:
                raise ConnectionError("reset")  # first attempt of some chunks fails
            a, b = (int(v) for v in Range.removeprefix("bytes=").split("-"))
            return {"Body": Body(body[a : b + 1])}

    fake = Fake()
    monkeypatch.setattr(mqr, "s3", lambda: fake)
    monkeypatch.setattr(mqr.time, "sleep", lambda s: None)
    assert mqr.get_large_bytes("k", chunk=300) == body
    assert mqr.get_large_bytes("missing") is None
    fake.fail = True
    assert mqr.get_large_bytes("k", chunk=300, retries=2) is None
