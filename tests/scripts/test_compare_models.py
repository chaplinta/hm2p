"""Tests for ``scripts/compare_models.py`` — SA fine-tune comparison CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import jsonschema
import numpy as np
import pandas as pd
import pytest

# Wire up the scripts dir for direct import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
import compare_models as cm  # noqa: E402

from hm2p.pose.finetune import HM2P_BODYPARTS, GateConfig, verdict_from_json  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "pose" / "fixtures"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def verdict_schema() -> dict:
    return json.loads((FIXTURES_DIR / "verdict.schema.json").read_text())


def _write_gt_h5(
    path: Path,
    frame_indices: list[int],
    coords: np.ndarray,
    keypoint_names: list[str],
    *,
    scorer: str = "tristan",
) -> None:
    """Write a CollectedData-style multi-index .h5 file."""
    columns = pd.MultiIndex.from_product(
        [[scorer], keypoint_names, ["x", "y"]],
        names=["scorer", "bodyparts", "coords"],
    )
    rows = [("labeled-data", "clip", f"frame_{i}.png") for i in frame_indices]
    index = pd.MultiIndex.from_tuples(rows)
    data = coords.reshape(len(frame_indices), -1)
    df = pd.DataFrame(data, index=index, columns=columns)
    df.to_hdf(path, key="df_with_missing", mode="w")


def _write_pred_h5(
    path: Path,
    n_dlc_frames: int,
    coords: np.ndarray,
    keypoint_names: list[str],
    *,
    scorer: str = "DLC_HrnetW32_test",
) -> None:
    """Write a DLC-style prediction .h5 with `n_dlc_frames` rows.

    ``coords`` has shape ``(n_dlc_frames, n_kp, 2)``.
    """
    columns = pd.MultiIndex.from_product(
        [[scorer], keypoint_names, ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"],
    )
    n = n_dlc_frames
    out = np.zeros((n, len(keypoint_names), 3))
    out[..., 0:2] = coords
    out[..., 2] = 1.0
    df = pd.DataFrame(
        out.reshape(n, -1),
        index=pd.RangeIndex(n),
        columns=columns,
    )
    df.to_hdf(path, key="df", mode="w")


@pytest.fixture
def synthetic_session(tmp_path: Path) -> dict:
    """Build a tiny synthetic session: GT + two prediction h5s.

    Designed so the candidate cleanly beats the baseline:
      - nose_tip:  60% reduction in median error
      - tail_base: 60% reduction
      - head_midpoint: 50% reduction in p90
      - other keypoints: candidate is exactly equal to baseline (no regression)
    """
    rng = np.random.default_rng(0)
    keypoint_names = list(HM2P_BODYPARTS)
    # Choose raw frames that each map to a *distinct* DLC frame so the
    # paired (gt, pred) comparison is well-defined. raw_idx = i * 4
    # gives DLC frames 0, 1, 2, 4, ... — most unique under round(i*0.3).
    raw_indices = [i * 10 for i in range(200)]  # 0, 10, 20, ... 1990
    # DLC indices: 0, 3, 6, ..., 597 -> n_dlc = 598
    n_dlc = max(cm.map_raw_to_dlc_frame(rf) for rf in raw_indices) + 1
    n = len(raw_indices)
    n_kp = len(keypoint_names)
    # Sanity: raw->dlc mapping is one-to-one for these indices.
    assert len({cm.map_raw_to_dlc_frame(rf) for rf in raw_indices}) == n

    # GT: random scattering.
    gt_xy = rng.uniform(0, 100, size=(n, n_kp, 2))

    # Per-keypoint baseline error scale (px). Heavy-tailed (exponential).
    base_scale = np.array([24.0, 5.0, 5.0, 12.0, 5.0, 5.0, 5.0, 59.0])
    # Candidate scale: 0.4× nose/tail, 0.5× head, equal elsewhere.
    cand_scale = base_scale.copy()
    cand_scale[0] *= 0.4
    cand_scale[7] *= 0.4
    cand_scale[3] *= 0.5

    baseline = np.zeros((n_dlc, n_kp, 2))
    candidate = np.zeros((n_dlc, n_kp, 2))
    for i, raw_fi in enumerate(raw_indices):
        dlc_fi = cm.map_raw_to_dlc_frame(raw_fi)
        # Direction: random unit vector per (frame, keypoint).
        thetas = rng.uniform(0, 2 * np.pi, size=n_kp)
        directions = np.stack([np.cos(thetas), np.sin(thetas)], axis=-1)
        # Magnitudes drawn from exponential — paired (same direction)
        # so the candidate is on the same ray as the baseline but
        # closer to GT for the targeted keypoints.
        b_mag = rng.exponential(scale=base_scale)
        c_mag = b_mag * (cand_scale / base_scale)
        baseline[dlc_fi, :, :] = gt_xy[i, :, :] + directions * b_mag[:, None]
        candidate[dlc_fi, :, :] = gt_xy[i, :, :] + directions * c_mag[:, None]

    # Build the labelled-data dir layout.
    labels_dir = tmp_path / "labels"
    clip_dir = labels_dir / "20210823_16_59_50_1114353-clip"
    clip_dir.mkdir(parents=True)
    gt_path = clip_dir / "CollectedData_tristan.h5"
    _write_gt_h5(gt_path, raw_indices, gt_xy, keypoint_names)

    # Build retrain_frames mapping for clip_dir_to_sub_ses.
    retrain_dir = tmp_path / "retrain_frames"
    retrain_dir.mkdir()
    (retrain_dir / "sub-1114353_ses-20210823T165950.json").write_text("{}")

    baseline_h5 = tmp_path / "baseline.h5"
    candidate_h5 = tmp_path / "candidate.h5"
    _write_pred_h5(baseline_h5, n_dlc, baseline, keypoint_names)
    _write_pred_h5(candidate_h5, n_dlc, candidate, keypoint_names)

    return {
        "labels_dir": labels_dir,
        "retrain_frames_dir": retrain_dir,
        "baseline_h5": baseline_h5,
        "candidate_h5": candidate_h5,
        "raw_indices": raw_indices,
        "gt_xy": gt_xy,
    }


def _make_fake_s3_for_sessions(
    baseline_h5: Path,
    candidate_h5: Path,
    *,
    candidate_missing: bool = False,
    baseline_missing: bool = False,
) -> MagicMock:
    """Mock S3 client whose `download_file` writes the right h5 file."""
    s3 = MagicMock()
    s3.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})

    def _download(Bucket, Key, Filename):
        if "candidate" in Bucket or "candidate" in Key:
            src = candidate_h5
        else:
            src = baseline_h5
        Path(Filename).write_bytes(src.read_bytes())

    def _put(**kwargs):
        return {}

    s3.download_file.side_effect = _download
    s3.put_object.side_effect = _put
    return s3


def _select_fn_factory(
    *,
    baseline_missing: bool = False,
    candidate_missing: bool = False,
):
    """Build a select_fn replacement that just returns a deterministic key."""

    def _selector(s3, bucket, prefix):
        if "baseline" in bucket and baseline_missing:
            return None
        if "candidate" in bucket and candidate_missing:
            return None
        return f"{prefix}{bucket}-best.h5"

    return _selector


# ---------------------------------------------------------------------------
# CLI argparse
# ---------------------------------------------------------------------------


class TestArgparse:
    def test_required_args(self):
        parser = cm._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_default_alpha(self):
        parser = cm._build_parser()
        args = parser.parse_args(["--baseline-id", "b", "--candidate-id", "c"])
        assert args.alpha == pytest.approx(6.25e-3)

    def test_default_seed(self):
        parser = cm._build_parser()
        args = parser.parse_args(["--baseline-id", "b", "--candidate-id", "c"])
        assert args.seed == 42

    def test_mode_default_predict(self):
        parser = cm._build_parser()
        args = parser.parse_args(["--baseline-id", "b", "--candidate-id", "c"])
        assert args.mode == "predict"

    def test_mode_rmse_json_accepted(self):
        parser = cm._build_parser()
        args = parser.parse_args(
            ["--baseline-id", "b", "--candidate-id", "c", "--mode", "rmse-json"]
        )
        assert args.mode == "rmse-json"

    def test_upload_s3_is_flag(self):
        parser = cm._build_parser()
        args = parser.parse_args(["--baseline-id", "b", "--candidate-id", "c", "--upload-s3"])
        assert args.upload_s3 is True

    def test_invalid_mode_rejected(self):
        parser = cm._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--baseline-id", "b", "--candidate-id", "c", "--mode", "garbage"])


# ---------------------------------------------------------------------------
# parse_s3_uri
# ---------------------------------------------------------------------------


class TestParseS3Uri:
    def test_with_prefix(self):
        b, p = cm.parse_s3_uri("s3://my-bucket/foo/bar/")
        assert b == "my-bucket"
        assert p == "foo/bar/"

    def test_bucket_only(self):
        b, p = cm.parse_s3_uri("s3://my-bucket")
        assert b == "my-bucket"
        assert p == ""

    def test_invalid_uri_raises(self):
        with pytest.raises(ValueError, match="not an S3 URI"):
            cm.parse_s3_uri("/local/path")


# ---------------------------------------------------------------------------
# Frame index mapping
# ---------------------------------------------------------------------------


def test_map_raw_to_dlc_frame():
    assert cm.map_raw_to_dlc_frame(0) == 0
    assert cm.map_raw_to_dlc_frame(100) == 30
    assert cm.map_raw_to_dlc_frame(200) == 60
    # Round-half: 50 raw -> 15.0 dlc -> 15.
    assert cm.map_raw_to_dlc_frame(50) == 15


# ---------------------------------------------------------------------------
# load_gt_keypoints + load_predictions_from_h5
# ---------------------------------------------------------------------------


class TestLoadGT:
    def test_loads_coords_and_indices(self, synthetic_session):
        loaded = cm.load_gt_keypoints(
            synthetic_session["labels_dir"]
            / "20210823_16_59_50_1114353-clip"
            / "CollectedData_tristan.h5",
            list(HM2P_BODYPARTS),
        )
        assert loaded is not None
        coords, raw_idx = loaded
        assert coords.shape == (
            len(synthetic_session["raw_indices"]),
            len(HM2P_BODYPARTS),
            2,
        )
        assert raw_idx == synthetic_session["raw_indices"]

    def test_missing_file_returns_none(self, tmp_path):
        # A non-existent path -> pd.read_hdf raises -> function returns None.
        loaded = cm.load_gt_keypoints(tmp_path / "nope.h5", list(HM2P_BODYPARTS))
        assert loaded is None

    def test_legacy_alias(self, tmp_path):
        # GT with `implant_base_rear` only — load_gt_keypoints should
        # transparently map it onto head_midpoint.
        legacy_names = [
            "nose_tip",
            "left_ear",
            "right_ear",
            "implant_base_rear",
            "neck",
            "mid_back",
            "mouse_center",
            "tail_base",
        ]
        n = 10
        gt_xy = np.full((n, len(legacy_names), 2), 5.0)
        path = tmp_path / "alias.h5"
        _write_gt_h5(path, list(range(n)), gt_xy, legacy_names)
        loaded = cm.load_gt_keypoints(path, list(HM2P_BODYPARTS))
        assert loaded is not None
        coords, _ = loaded
        # head_midpoint column should have the legacy data, not NaN.
        head_idx = HM2P_BODYPARTS.index("head_midpoint")
        assert np.all(np.isfinite(coords[:, head_idx, :]))


class TestLoadPredictions:
    def test_round_trip(self, synthetic_session):
        keypoint_names = list(HM2P_BODYPARTS)
        n_raw = len(synthetic_session["raw_indices"])
        # Sanity: load the baseline predictions and confirm the shape.
        out = cm.load_predictions_from_h5(
            synthetic_session["baseline_h5"],
            keypoint_names,
            synthetic_session["raw_indices"],
        )
        assert out.shape == (n_raw, len(keypoint_names), 2)

    def test_negative_index_yields_nan(self, synthetic_session):
        out = cm.load_predictions_from_h5(
            synthetic_session["baseline_h5"],
            list(HM2P_BODYPARTS),
            [-1, 0, 1],
        )
        assert np.all(np.isnan(out[0]))


# ---------------------------------------------------------------------------
# clip_dir_to_sub_ses
# ---------------------------------------------------------------------------


class TestClipDirToSubSes:
    def test_basic_match(self, tmp_path):
        retrain = tmp_path / "retrain_frames"
        retrain.mkdir()
        (retrain / "sub-1114353_ses-20210823T165950.json").write_text("{}")
        result = cm.clip_dir_to_sub_ses("20210823_16_59_50_1114353-clip", retrain)
        assert result == ("sub-1114353", "ses-20210823T165950")

    def test_no_candidates_returns_none(self, tmp_path):
        retrain = tmp_path / "retrain_frames"
        retrain.mkdir()
        result = cm.clip_dir_to_sub_ses("20210823_16_59_50_1114353-clip", retrain)
        assert result is None

    def test_short_name_returns_none(self):
        assert cm.clip_dir_to_sub_ses("too_short") is None

    def test_missing_dir_returns_none(self, tmp_path):
        result = cm.clip_dir_to_sub_ses(
            "20210823_16_59_50_1114353-clip", tmp_path / "does-not-exist"
        )
        assert result is None

    def test_invalid_time_returns_none(self, tmp_path):
        retrain = tmp_path / "retrain_frames"
        retrain.mkdir()
        # Bad clip-time portion -> unparseable.
        assert cm.clip_dir_to_sub_ses("20210823_xx_yy_zz_1114353-clip", retrain) is None


# ---------------------------------------------------------------------------
# fetch_prediction_h5
# ---------------------------------------------------------------------------


def test_fetch_prediction_h5_returns_path(tmp_path, synthetic_session):
    s3 = _make_fake_s3_for_sessions(
        synthetic_session["baseline_h5"], synthetic_session["candidate_h5"]
    )
    select_fn = _select_fn_factory()
    out = cm.fetch_prediction_h5(
        s3,
        "baseline-bucket",
        "pose-archive/",
        "sub-x",
        "ses-y",
        select_fn=select_fn,
    )
    assert out is not None
    assert out.exists()
    out.unlink(missing_ok=True)


def test_fetch_prediction_h5_returns_none_when_missing(tmp_path, synthetic_session):
    s3 = _make_fake_s3_for_sessions(
        synthetic_session["baseline_h5"], synthetic_session["candidate_h5"]
    )
    select_fn = _select_fn_factory(baseline_missing=True)
    out = cm.fetch_prediction_h5(
        s3,
        "baseline-bucket",
        "pose-archive/",
        "sub-x",
        "ses-y",
        select_fn=select_fn,
    )
    assert out is None


# ---------------------------------------------------------------------------
# collect_paired_errors
# ---------------------------------------------------------------------------


class TestCollectPairedErrors:
    def test_clear_winner_session(self, synthetic_session):
        s3 = _make_fake_s3_for_sessions(
            synthetic_session["baseline_h5"], synthetic_session["candidate_h5"]
        )
        e_b, e_c, hd_b, hd_c, hd_g, skipped = cm.collect_paired_errors(
            synthetic_session["labels_dir"],
            s3,
            "baseline-bucket",
            "pose-archive/",
            "candidate-bucket",
            "pose/",
            select_fn=_select_fn_factory(),
            retrain_frames_dir=synthetic_session["retrain_frames_dir"],
        )
        assert e_b.shape[0] > 0
        assert e_b.shape == e_c.shape
        # Candidate should be smaller on nose (col 0) on average.
        nose_b_med = np.nanmedian(e_b[:, 0])
        nose_c_med = np.nanmedian(e_c[:, 0])
        assert nose_c_med < nose_b_med
        # HD signal computed from ear pair -> non-None.
        assert hd_b is not None
        assert hd_c is not None
        assert hd_g is not None
        assert skipped == []

    def test_skips_missing_candidate(self, tmp_path, synthetic_session):
        s3 = _make_fake_s3_for_sessions(
            synthetic_session["baseline_h5"], synthetic_session["candidate_h5"]
        )
        e_b, e_c, *_, skipped = cm.collect_paired_errors(
            synthetic_session["labels_dir"],
            s3,
            "baseline-bucket",
            "pose-archive/",
            "candidate-bucket",
            "pose/",
            select_fn=_select_fn_factory(candidate_missing=True),
            retrain_frames_dir=synthetic_session["retrain_frames_dir"],
        )
        assert e_b.shape[0] == 0
        assert any("no_candidate_prediction" in s for s in skipped)

    def test_no_gt_returns_empty(self, tmp_path):
        # Empty labels dir -> nothing collected.
        labels = tmp_path / "labels"
        labels.mkdir()
        s3 = MagicMock()
        e_b, e_c, *_, skipped = cm.collect_paired_errors(
            labels,
            s3,
            "b",
            "p1/",
            "c",
            "p2/",
            select_fn=_select_fn_factory(),
            retrain_frames_dir=tmp_path / "retrain",
        )
        assert e_b.shape == (0, 8)
        assert skipped == []

    def test_no_clip_match_skipped(self, tmp_path, synthetic_session):
        # Empty retrain_frames_dir -> clip cannot be mapped.
        empty_retrain = tmp_path / "empty_retrain"
        empty_retrain.mkdir()
        s3 = _make_fake_s3_for_sessions(
            synthetic_session["baseline_h5"], synthetic_session["candidate_h5"]
        )
        e_b, *_, skipped = cm.collect_paired_errors(
            synthetic_session["labels_dir"],
            s3,
            "b-bucket",
            "p1/",
            "c-bucket",
            "p2/",
            select_fn=_select_fn_factory(),
            retrain_frames_dir=empty_retrain,
        )
        assert e_b.shape[0] == 0
        assert any("no_match" in s for s in skipped)


# ---------------------------------------------------------------------------
# Verdict end-to-end via main()
# ---------------------------------------------------------------------------


def _run_main_with_session(
    session: dict,
    output: Path,
    *,
    extra_argv: list[str] | None = None,
    monkeypatch: pytest.MonkeyPatch,
    upload_s3: bool = False,
    candidate_missing: bool = False,
    baseline_missing: bool = False,
) -> tuple[int, MagicMock]:
    s3 = _make_fake_s3_for_sessions(session["baseline_h5"], session["candidate_h5"])
    monkeypatch.setattr(cm, "_make_s3_client", lambda region: s3)
    # Patch select_fn at the module level via fetch_prediction_h5 path:
    # we don't have one — instead, patch `select_best_dlc_h5_s3`.
    monkeypatch.setattr(
        cm,
        "fetch_prediction_h5",
        lambda s3, bucket, prefix, sub, ses, select_fn=None: (
            Path(session["baseline_h5"])
            if "baseline" in bucket and not baseline_missing
            else Path(session["candidate_h5"])
            if "candidate" in bucket and not candidate_missing
            else None
        ),
    )
    monkeypatch.setattr(
        cm,
        "clip_dir_to_sub_ses",
        lambda name, retrain=None: ("sub-x", "ses-y"),
    )

    argv = [
        "--mode",
        "predict",
        "--baseline-id",
        "dlc-20260430-hrnetw32-snap110",
        "--candidate-id",
        "dlc-20260501-hrnetw32-snap60",
        "--labels-dir",
        str(session["labels_dir"]),
        "--baseline-h5-prefix",
        "s3://baseline-bucket/pose-archive/",
        "--candidate-h5-prefix",
        "s3://candidate-bucket/pose/",
        "--output",
        str(output),
    ]
    if upload_s3:
        argv.append("--upload-s3")
    if extra_argv:
        argv.extend(extra_argv)
    code = cm.main(argv)
    return code, s3


def test_main_clear_winner_returns_zero(synthetic_session, tmp_path, monkeypatch):
    out = tmp_path / "verdict.json"
    code, s3 = _run_main_with_session(synthetic_session, out, monkeypatch=monkeypatch)
    # The candidate is constructed to beat baseline cleanly.
    assert code == 0
    assert out.exists()
    v = verdict_from_json(out.read_text())
    assert v.overall_pass is True
    assert v.fail_reasons == ()
    s3.put_object.assert_not_called()


def test_main_clear_loser_returns_two(tmp_path, synthetic_session, monkeypatch):
    """Construct a session where candidate is uniformly worse -> exit 2."""
    # Mutate candidate pred file to make candidate worse: scale by 3x.
    df = pd.read_hdf(synthetic_session["candidate_h5"])
    cols = df.columns
    # Scale the x/y columns (not the likelihood).
    for c in cols:
        if c[-1] in ("x", "y"):
            df[c] = df[c] * 3.0
    df.to_hdf(synthetic_session["candidate_h5"], key="df", mode="w")

    out = tmp_path / "verdict.json"
    code, _ = _run_main_with_session(synthetic_session, out, monkeypatch=monkeypatch)
    # Either 2 (gate failed) or 0 if some fluke; with deterministic
    # synthetic data and 3x scale this should fail.
    assert code == 2
    v = verdict_from_json(out.read_text())
    assert v.overall_pass is False
    assert len(v.fail_reasons) >= 1


def test_main_no_overlap_returns_three(tmp_path, monkeypatch):
    """No labelled-data dirs -> exit 3 with meta.error populated."""
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    out = tmp_path / "verdict.json"
    s3 = MagicMock()
    monkeypatch.setattr(cm, "_make_s3_client", lambda region: s3)
    code = cm.main(
        [
            "--mode",
            "predict",
            "--baseline-id",
            "b",
            "--candidate-id",
            "c",
            "--labels-dir",
            str(labels_dir),
            "--output",
            str(out),
        ]
    )
    assert code == 3
    v = verdict_from_json(out.read_text())
    assert "no overlapping" in v.meta.get("error", "")


def test_main_upload_s3_calls_put_object(synthetic_session, tmp_path, monkeypatch):
    out = tmp_path / "verdict.json"
    code, s3 = _run_main_with_session(
        synthetic_session, out, upload_s3=True, monkeypatch=monkeypatch
    )
    assert code in (0, 2, 3)
    s3.put_object.assert_called_once()
    kwargs = s3.put_object.call_args.kwargs
    assert kwargs["Key"] == cm.VERDICT_S3_KEY
    assert kwargs["ContentType"] == "application/json"


def test_main_predict_requires_labels_dir(monkeypatch, capsys):
    # No --labels-dir -> exit 1.
    monkeypatch.setattr(cm, "_make_s3_client", lambda region: MagicMock())
    code = cm.main(
        [
            "--mode",
            "predict",
            "--baseline-id",
            "b",
            "--candidate-id",
            "c",
        ]
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "labels-dir" in err


def test_main_rmse_json_mode(tmp_path, monkeypatch):
    """Triage mode with two pre-aggregated JSON files."""
    base = {
        "bodyparts": {
            "nose_tip": {"median": 24.0, "n": 100},
            "tail_base": {"median": 59.0, "n": 100},
        }
    }
    cand = {
        "bodyparts": {
            "nose_tip": {"median": 9.6, "n": 100},
            "tail_base": {"median": 24.0, "n": 100},
        }
    }
    b_path = tmp_path / "base.json"
    c_path = tmp_path / "cand.json"
    b_path.write_text(json.dumps(base))
    c_path.write_text(json.dumps(cand))

    out = tmp_path / "verdict.json"
    code = cm.main(
        [
            "--mode",
            "rmse-json",
            "--baseline-id",
            "b",
            "--candidate-id",
            "c",
            "--baseline-rmse-json",
            str(b_path),
            "--candidate-rmse-json",
            str(c_path),
            "--output",
            str(out),
        ]
    )
    assert code == 0
    v = verdict_from_json(out.read_text())
    # Per-keypoint p_value_wilcoxon is NaN in this mode.
    assert all(np.isnan(kp.p_value_wilcoxon) for kp in v.keypoints)
    assert "rmse-json" in v.meta.get("error", "")


def test_main_rmse_json_requires_both_files(tmp_path, monkeypatch, capsys):
    code = cm.main(
        [
            "--mode",
            "rmse-json",
            "--baseline-id",
            "b",
            "--candidate-id",
            "c",
        ]
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "baseline-rmse-json" in err


# ---------------------------------------------------------------------------
# Verdict schema
# ---------------------------------------------------------------------------


def test_verdict_passes_schema(synthetic_session, tmp_path, monkeypatch, verdict_schema):
    out = tmp_path / "verdict.json"
    _run_main_with_session(synthetic_session, out, monkeypatch=monkeypatch)
    d = json.loads(out.read_text())
    jsonschema.validate(d, verdict_schema)


# ---------------------------------------------------------------------------
# Helper-only tests
# ---------------------------------------------------------------------------


def test_build_descriptive_verdict_marks_meta():
    v = cm.build_descriptive_verdict_from_rmse(
        {"bodyparts": {}},
        {"bodyparts": {}},
        baseline_id="b",
        candidate_id="c",
    )
    assert "rmse-json" in v.meta["mode"]
    assert "descriptive" in v.meta["error"]


def test_build_gate_from_args_uses_alpha():
    g = cm._build_gate_from_args(0.001)
    assert g.alpha == pytest.approx(0.001)
    # Other thresholds keep defaults.
    assert g.nose_required_pct_reduction == GateConfig().nose_required_pct_reduction


def test_list_gt_session_dirs_missing(tmp_path):
    assert cm.list_gt_session_dirs(tmp_path / "missing") == []


def test_list_gt_session_dirs_only_files(tmp_path):
    (tmp_path / "a-file").write_text("hi")
    assert cm.list_gt_session_dirs(tmp_path) == []


def test_load_gt_keypoints_unparseable_index_yields_neg_one(tmp_path):
    """Frame index column without `frame_<N>.png` -> -1 entry."""
    keypoint_names = list(HM2P_BODYPARTS)
    n = 5
    gt_xy = np.full((n, len(keypoint_names), 2), 1.0)
    columns = pd.MultiIndex.from_product(
        [["scorer1"], keypoint_names, ["x", "y"]],
        names=["scorer", "bodyparts", "coords"],
    )
    rows = [("labeled-data", "clip", f"unparseable_{i}.tif") for i in range(n)]
    df = pd.DataFrame(
        gt_xy.reshape(n, -1),
        index=pd.MultiIndex.from_tuples(rows),
        columns=columns,
    )
    path = tmp_path / "weird.h5"
    df.to_hdf(path, key="df_with_missing", mode="w")
    loaded = cm.load_gt_keypoints(path, keypoint_names)
    assert loaded is not None
    _, raw_idx = loaded
    assert all(r == -1 for r in raw_idx)


def test_load_pred_dlc_index_out_of_range(tmp_path):
    """raw_fi mapping past the end of pred -> NaN row."""
    keypoint_names = list(HM2P_BODYPARTS)
    n_dlc = 3
    pred_xy = np.zeros((n_dlc, len(keypoint_names), 2))
    path = tmp_path / "pred.h5"
    _write_pred_h5(path, n_dlc, pred_xy, keypoint_names)
    # raw_fi=999 maps past n_dlc=3.
    out = cm.load_predictions_from_h5(path, keypoint_names, [999])
    assert np.all(np.isnan(out))


def test_load_pred_legacy_alias(tmp_path):
    legacy_names = [
        "nose_tip",
        "left_ear",
        "right_ear",
        "implant_base_rear",
        "neck",
        "mid_back",
        "mouse_center",
        "tail_base",
    ]
    n_dlc = 5
    pred_xy = np.full((n_dlc, len(legacy_names), 2), 7.0)
    path = tmp_path / "legacy.h5"
    _write_pred_h5(path, n_dlc, pred_xy, legacy_names)
    out = cm.load_predictions_from_h5(path, list(HM2P_BODYPARTS), [0, 1, 2])
    head_idx = HM2P_BODYPARTS.index("head_midpoint")
    assert np.all(out[:, head_idx, :] == 7.0)


def test_load_pred_missing_keypoint_remains_nan(tmp_path):
    """A pred file that lacks one keypoint entirely -> NaN column."""
    short_names = list(HM2P_BODYPARTS)[:7]  # drop tail_base
    n_dlc = 5
    pred_xy = np.zeros((n_dlc, len(short_names), 2))
    path = tmp_path / "short.h5"
    _write_pred_h5(path, n_dlc, pred_xy, short_names)
    out = cm.load_predictions_from_h5(path, list(HM2P_BODYPARTS), [0, 1, 2])
    tail_idx = HM2P_BODYPARTS.index("tail_base")
    # Missing column -> still NaN.
    assert np.all(np.isnan(out[:, tail_idx, :]))


def test_collect_paired_errors_no_clip_dir_label_file(tmp_path):
    """Clip dir without a CollectedData_*.h5 -> skipped silently."""
    labels = tmp_path / "labels"
    clip = labels / "clipname"
    clip.mkdir(parents=True)
    s3 = MagicMock()
    e_b, *_, skipped = cm.collect_paired_errors(
        labels,
        s3,
        "b",
        "p1/",
        "c",
        "p2/",
        select_fn=_select_fn_factory(),
        retrain_frames_dir=tmp_path / "retrain",
    )
    # No skipped reason because the file simply isn't there.
    assert e_b.shape[0] == 0
    assert skipped == []


def test_collect_paired_errors_no_gt_loadable(tmp_path):
    """A clip dir with an unreadable label file -> skipped with reason."""
    labels = tmp_path / "labels"
    clip = labels / "20210823_16_59_50_1114353-clip"
    clip.mkdir(parents=True)
    # Write a non-h5 file with the right name so load_gt_keypoints fails.
    (clip / "CollectedData_tristan.h5").write_text("not an h5")
    s3 = MagicMock()
    e_b, *_, skipped = cm.collect_paired_errors(
        labels,
        s3,
        "b",
        "p1/",
        "c",
        "p2/",
        select_fn=_select_fn_factory(),
        retrain_frames_dir=tmp_path / "retrain",
    )
    assert e_b.shape[0] == 0
    assert any("no_gt" in s for s in skipped)


def test_summarise_to_stdout_handles_nan(synthetic_session, tmp_path, monkeypatch, capsys):
    """The stdout summary must tolerate NaN p-values gracefully."""
    out = tmp_path / "verdict.json"
    _run_main_with_session(synthetic_session, out, monkeypatch=monkeypatch)
    captured = capsys.readouterr().out
    assert "SA fine-tune verdict" in captured
    assert "per-keypoint" in captured
