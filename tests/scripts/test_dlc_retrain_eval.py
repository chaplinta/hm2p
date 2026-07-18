"""Tests for DLC evaluation helpers in ``scripts/run_dlc_retrain.py``.

Covers the per-bodypart RMSE pipeline:
- ``_index_to_frame_id`` — tuple/string/empty index conversion
- ``_match_indices_by_filename`` — GT/pred alignment by filename stem
- ``_compute_per_bodypart_rmse`` — RMSE, per_frame array, edge cases
- ``_upload_eval_results_json`` — JSON structure, missing CSV handling
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import yaml

# Pre-stub heavy imports before loading the module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
sys.modules.setdefault("deeplabcut", MagicMock())
sys.modules.setdefault("dlclibrary", MagicMock())

import run_dlc_retrain as rdr  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════
# Helpers — build synthetic DLC-style DataFrames
# ═══════════════════════════════════════════════════════════════════════


def _make_dlc_df(
    scorer: str,
    bodyparts: list[str],
    coords: dict[str, list[tuple[float, float]]],
    index: list | None = None,
) -> pd.DataFrame:
    """Build a DLC-style MultiIndex DataFrame.

    Parameters
    ----------
    scorer : str
        Scorer name (top-level column).
    bodyparts : list[str]
        List of bodypart names.
    coords : dict[str, list[tuple[float, float]]]
        Mapping from bodypart name to list of (x, y) pairs.
    index : list | None
        DataFrame index entries (e.g. tuples or strings).
    """
    arrays = []
    col_tuples = []
    for bp in bodyparts:
        xy = coords.get(bp, [(np.nan, np.nan)] * len(next(iter(coords.values()))))
        for x, y in xy:
            pass  # just need length
        xs = [c[0] for c in coords.get(bp, [(np.nan, np.nan)])]
        ys = [c[1] for c in coords.get(bp, [(np.nan, np.nan)])]
        arrays.extend([xs, ys])
        col_tuples.extend([(scorer, bp, "x"), (scorer, bp, "y")])

    n_rows = len(next(iter(coords.values())))
    data = {}
    for i, ct in enumerate(col_tuples):
        bp = ct[1]
        coord = ct[2]
        vals = coords.get(bp, [(np.nan, np.nan)] * n_rows)
        if coord == "x":
            data[ct] = [v[0] for v in vals]
        else:
            data[ct] = [v[1] for v in vals]

    columns = pd.MultiIndex.from_tuples(col_tuples)
    df = pd.DataFrame(data, columns=columns)
    if index is not None:
        df.index = index
    return df


# ═══════════════════════════════════════════════════════════════════════
# _index_to_frame_id
# ═══════════════════════════════════════════════════════════════════════


class TestIndexToFrameId:
    def test_tuple_index_returns_last_element(self):
        idx = ("labeled-data", "clip_001", "frame_000123.png")
        assert rdr._index_to_frame_id(idx) == "frame_000123.png"

    def test_string_index_returns_as_is(self):
        idx = "path/to/frame.png"
        assert rdr._index_to_frame_id(idx) == "path/to/frame.png"

    def test_empty_tuple_returns_str_repr(self):
        idx = ()
        assert rdr._index_to_frame_id(idx) == str(idx)

    def test_single_element_tuple(self):
        idx = ("frame_0.png",)
        assert rdr._index_to_frame_id(idx) == "frame_0.png"

    def test_integer_index(self):
        idx = 42
        assert rdr._index_to_frame_id(idx) == "42"

    def test_nested_tuple_returns_deepest(self):
        idx = ("a", "b", ("c", "d"))
        result = rdr._index_to_frame_id(idx)
        # Last element is the inner tuple — str(("c", "d"))
        assert result == str(("c", "d"))


# ═══════════════════════════════════════════════════════════════════════
# _match_indices_by_filename
# ═══════════════════════════════════════════════════════════════════════


class TestMatchIndicesByFilename:
    def test_matching_frames_returns_gt_pred_pairs(self):
        gt = _make_dlc_df(
            "scorer_gt",
            ["bp1"],
            {"bp1": [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]},
            index=[
                ("labeled-data", "clip1", "frame_001.png"),
                ("labeled-data", "clip1", "frame_002.png"),
                ("labeled-data", "clip2", "frame_003.png"),
            ],
        )
        pred = _make_dlc_df(
            "scorer_pred",
            ["bp1"],
            {"bp1": [(1.1, 2.1), (5.1, 6.1)]},
            index=[
                ("different-path", "x", "frame_001.png"),
                ("different-path", "y", "frame_003.png"),
            ],
        )
        matched = rdr._match_indices_by_filename(gt, pred)
        assert len(matched) == 2
        # Each entry is a (gt_idx, pred_idx) pair
        gt_idx_0, pred_idx_0 = matched[0]
        gt_idx_1, pred_idx_1 = matched[1]
        assert gt_idx_0 == ("labeled-data", "clip1", "frame_001.png")
        assert pred_idx_0 == ("different-path", "x", "frame_001.png")
        assert gt_idx_1 == ("labeled-data", "clip2", "frame_003.png")
        assert pred_idx_1 == ("different-path", "y", "frame_003.png")

    def test_no_matches_returns_empty(self):
        gt = _make_dlc_df(
            "scorer_gt",
            ["bp1"],
            {"bp1": [(1.0, 2.0)]},
            index=[("labeled-data", "clip1", "frame_001.png")],
        )
        pred = _make_dlc_df(
            "scorer_pred",
            ["bp1"],
            {"bp1": [(1.1, 2.1)]},
            index=[("labeled-data", "clip1", "frame_999.png")],
        )
        matched = rdr._match_indices_by_filename(gt, pred)
        assert len(matched) == 0

    def test_partial_matches(self):
        gt = _make_dlc_df(
            "scorer_gt",
            ["bp1"],
            {"bp1": [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]},
            index=["frame_001.png", "frame_002.png", "frame_003.png"],
        )
        pred = _make_dlc_df(
            "scorer_pred",
            ["bp1"],
            {"bp1": [(10.0, 20.0)]},
            index=["frame_002.png"],
        )
        matched = rdr._match_indices_by_filename(gt, pred)
        assert len(matched) == 1
        gt_idx, pred_idx = matched[0]
        assert gt_idx == "frame_002.png"
        assert pred_idx == "frame_002.png"

    def test_string_indices_matched_by_stem(self):
        """Paths with different directories but same filename stem match."""
        gt = _make_dlc_df(
            "s",
            ["bp1"],
            {"bp1": [(1.0, 1.0)]},
            index=["/path/a/frame_001.png"],
        )
        pred = _make_dlc_df(
            "s",
            ["bp1"],
            {"bp1": [(2.0, 2.0)]},
            index=["/path/b/frame_001.png"],
        )
        matched = rdr._match_indices_by_filename(gt, pred)
        assert len(matched) == 1
        gt_idx, pred_idx = matched[0]
        assert gt_idx == "/path/a/frame_001.png"
        assert pred_idx == "/path/b/frame_001.png"

    def test_gt_tuple_pred_string_cross_format_match(self):
        """GT tuples matched against pred string paths (DLC 3.x format)."""
        gt = _make_dlc_df(
            "s",
            ["bp1"],
            {"bp1": [(1.0, 1.0), (2.0, 2.0)]},
            index=[
                ("labeled-data", "clip1", "frame_001.png"),
                ("labeled-data", "clip1", "frame_002.png"),
            ],
        )
        pred = _make_dlc_df(
            "s",
            ["bp1"],
            {"bp1": [(10.0, 10.0), (20.0, 20.0)]},
            index=[
                "/tmp/dlc-retrain/labeled-data/clip1/frame_001.png",
                "/tmp/dlc-retrain/labeled-data/clip1/frame_002.png",
            ],
        )
        matched = rdr._match_indices_by_filename(gt, pred)
        assert len(matched) == 2
        gt_idx_0, pred_idx_0 = matched[0]
        assert gt_idx_0 == ("labeled-data", "clip1", "frame_001.png")
        assert pred_idx_0 == "/tmp/dlc-retrain/labeled-data/clip1/frame_001.png"


# ═══════════════════════════════════════════════════════════════════════
# _compute_per_bodypart_rmse
# ═══════════════════════════════════════════════════════════════════════


def _setup_rmse_project(
    tmp_path: Path,
    bodyparts: list[str],
    gt_coords: dict[str, list[tuple[float, float]]],
    pred_coords: dict[str, list[tuple[float, float]]],
    index: list | None = None,
    write_pred: bool = True,
) -> tuple[Path, Path]:
    """Create a minimal DLC project with GT and prediction files."""
    work = tmp_path / "dlc-retrain"
    work.mkdir()

    # config.yaml
    cfg_path = work / "config.yaml"
    cfg_path.write_text(
        yaml.dump(
            {
                "bodyparts": bodyparts,
                "project_path": str(work),
            }
        )
    )

    n_rows = len(next(iter(gt_coords.values())))
    if index is None:
        index = [f"frame_{i:03d}.png" for i in range(n_rows)]

    # Ground-truth CollectedData
    gt_dir = work / "labeled-data" / "clip1"
    gt_dir.mkdir(parents=True)
    gt_df = _make_dlc_df("scorer_gt", bodyparts, gt_coords, index=index)
    gt_df.to_hdf(gt_dir / "CollectedData_scorer_gt.h5", key="df_with_missing")

    # Prediction file
    if write_pred:
        eval_dir = work / "evaluation-results-pytorch" / "iteration-0" / "test"
        eval_dir.mkdir(parents=True)
        pred_df = _make_dlc_df("DLC_scorer", bodyparts, pred_coords, index=index)
        pred_df.to_hdf(eval_dir / "DLC_scorer.h5", key="df_with_missing")

    return work, cfg_path


class TestComputePerBodypartRmse:
    def test_correct_rmse_single_bodypart(self, tmp_path):
        """RMSE should be sqrt(mean(errors^2)). With constant error, RMSE == error."""
        # GT: (0, 0), (0, 0); Pred: (3, 4), (3, 4) => error = 5.0 each
        work, cfg = _setup_rmse_project(
            tmp_path,
            bodyparts=["bp1"],
            gt_coords={"bp1": [(0.0, 0.0), (0.0, 0.0)]},
            pred_coords={"bp1": [(3.0, 4.0), (3.0, 4.0)]},
        )
        s3 = MagicMock()
        rdr._compute_per_bodypart_rmse(s3, work, cfg)

        # Check the output JSON was written
        result_path = work / "_per_bodypart_eval.json"
        assert result_path.exists()

        result = json.loads(result_path.read_text())
        bp_data = result["bodyparts"]["bp1"]
        assert bp_data["rmse"] == pytest.approx(5.0, abs=0.01)
        assert bp_data["mean_error"] == pytest.approx(5.0, abs=0.01)
        assert bp_data["median_error"] == pytest.approx(5.0, abs=0.01)
        assert bp_data["n"] == 2
        # PCK@10: 5 <= 10 is True for both, so 100%
        assert bp_data["pck_10"] == pytest.approx(100.0)
        # PCK@5: 5 <= 5 is True for both
        assert bp_data["pck_5"] == pytest.approx(100.0)

    def test_correct_rmse_multiple_bodyparts(self, tmp_path):
        """Each bodypart has independent RMSE."""
        work, cfg = _setup_rmse_project(
            tmp_path,
            bodyparts=["ear", "nose"],
            gt_coords={
                "ear": [(0.0, 0.0)],
                "nose": [(10.0, 10.0)],
            },
            pred_coords={
                "ear": [(3.0, 4.0)],  # error = 5
                "nose": [(10.0, 22.0)],  # error = 12
            },
        )
        s3 = MagicMock()
        rdr._compute_per_bodypart_rmse(s3, work, cfg)

        result = json.loads((work / "_per_bodypart_eval.json").read_text())
        assert result["bodyparts"]["ear"]["rmse"] == pytest.approx(5.0, abs=0.01)
        assert result["bodyparts"]["nose"]["rmse"] == pytest.approx(12.0, abs=0.01)

    def test_per_frame_array_structure(self, tmp_path):
        """per_frame entries have frame_id, split, errors, gt, pred keys."""
        work, cfg = _setup_rmse_project(
            tmp_path,
            bodyparts=["bp1"],
            gt_coords={"bp1": [(0.0, 0.0)]},
            pred_coords={"bp1": [(3.0, 4.0)]},
            index=["frame_042.png"],
        )
        s3 = MagicMock()
        rdr._compute_per_bodypart_rmse(s3, work, cfg)

        result = json.loads((work / "_per_bodypart_eval.json").read_text())
        per_frame = result["per_frame"]
        assert len(per_frame) == 1
        frame = per_frame[0]
        assert frame["frame_id"] == "frame_042.png"
        assert "split" in frame
        assert "errors" in frame
        assert "gt" in frame
        assert "pred" in frame
        # gt and pred are [x, y] rounded
        assert frame["gt"]["bp1"] == [0.0, 0.0]
        assert frame["pred"]["bp1"] == [3.0, 4.0]
        assert frame["errors"]["bp1"] == pytest.approx(5.0, abs=0.01)

    def test_nan_gt_values_skipped(self, tmp_path):
        """Frames where GT has NaN are skipped for that bodypart."""
        work, cfg = _setup_rmse_project(
            tmp_path,
            bodyparts=["bp1"],
            gt_coords={"bp1": [(np.nan, np.nan), (0.0, 0.0)]},
            pred_coords={"bp1": [(1.0, 1.0), (3.0, 4.0)]},
        )
        s3 = MagicMock()
        rdr._compute_per_bodypart_rmse(s3, work, cfg)

        result = json.loads((work / "_per_bodypart_eval.json").read_text())
        # Only 1 frame should contribute (the second one)
        assert result["bodyparts"]["bp1"]["n"] == 1
        assert result["bodyparts"]["bp1"]["rmse"] == pytest.approx(5.0, abs=0.01)

    def test_nan_pred_values_skipped(self, tmp_path):
        """Frames where predictions have NaN are skipped."""
        work, cfg = _setup_rmse_project(
            tmp_path,
            bodyparts=["bp1"],
            gt_coords={"bp1": [(0.0, 0.0), (0.0, 0.0)]},
            pred_coords={"bp1": [(np.nan, np.nan), (3.0, 4.0)]},
        )
        s3 = MagicMock()
        rdr._compute_per_bodypart_rmse(s3, work, cfg)

        result = json.loads((work / "_per_bodypart_eval.json").read_text())
        assert result["bodyparts"]["bp1"]["n"] == 1

    def test_missing_bodypart_in_predictions(self, tmp_path):
        """Bodypart in config but not in pred DataFrame gets rmse=None."""
        work = tmp_path / "dlc-retrain"
        work.mkdir()
        cfg_path = work / "config.yaml"
        cfg_path.write_text(
            yaml.dump(
                {
                    "bodyparts": ["bp1", "bp_missing"],
                    "project_path": str(work),
                }
            )
        )

        gt_dir = work / "labeled-data" / "clip1"
        gt_dir.mkdir(parents=True)
        gt_df = _make_dlc_df(
            "scorer_gt",
            ["bp1", "bp_missing"],
            {"bp1": [(0.0, 0.0)], "bp_missing": [(1.0, 1.0)]},
            index=["frame_000.png"],
        )
        gt_df.to_hdf(gt_dir / "CollectedData_scorer_gt.h5", key="df_with_missing")

        eval_dir = work / "evaluation-results-pytorch" / "iter" / "test"
        eval_dir.mkdir(parents=True)
        # Prediction only has bp1, not bp_missing
        pred_df = _make_dlc_df(
            "DLC_scorer",
            ["bp1"],
            {"bp1": [(3.0, 4.0)]},
            index=["frame_000.png"],
        )
        pred_df.to_hdf(eval_dir / "DLC_scorer.h5", key="df_with_missing")

        s3 = MagicMock()
        rdr._compute_per_bodypart_rmse(s3, work, cfg_path)

        result = json.loads((work / "_per_bodypart_eval.json").read_text())
        assert result["bodyparts"]["bp1"]["rmse"] is not None
        assert result["bodyparts"]["bp_missing"]["rmse"] is None
        assert result["bodyparts"]["bp_missing"]["n"] == 0

    def test_no_gt_files_returns_early(self, tmp_path, capsys):
        """No CollectedData files -> prints warning and returns."""
        work = tmp_path / "dlc-retrain"
        work.mkdir()
        cfg_path = work / "config.yaml"
        cfg_path.write_text(yaml.dump({"bodyparts": ["bp1"]}))

        s3 = MagicMock()
        rdr._compute_per_bodypart_rmse(s3, work, cfg_path)

        out = capsys.readouterr().out
        assert "No ground-truth files found" in out
        assert not (work / "_per_bodypart_eval.json").exists()

    def test_uploads_to_s3(self, tmp_path):
        """Result JSON is uploaded to the correct S3 key."""
        work, cfg = _setup_rmse_project(
            tmp_path,
            bodyparts=["bp1"],
            gt_coords={"bp1": [(0.0, 0.0)]},
            pred_coords={"bp1": [(3.0, 4.0)]},
        )
        s3 = MagicMock()
        rdr._compute_per_bodypart_rmse(s3, work, cfg)

        s3.upload_file.assert_called_once()
        call_args = s3.upload_file.call_args
        assert call_args[0][1] == rdr.DERIVATIVES_BUCKET
        assert call_args[0][2] == f"{rdr.RETRAIN_PREFIX}/models/_per_bodypart_eval.json"

    def test_pck_thresholds_correct(self, tmp_path):
        """PCK values correctly count predictions within threshold."""
        # Error = 7.0 for all 4 frames
        # PCK@5: 0%, PCK@10: 100%, PCK@20: 100%
        work, cfg = _setup_rmse_project(
            tmp_path,
            bodyparts=["bp1"],
            gt_coords={"bp1": [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]},
            pred_coords={"bp1": [(0.0, 7.0), (0.0, 7.0), (0.0, 7.0), (0.0, 7.0)]},
        )
        s3 = MagicMock()
        rdr._compute_per_bodypart_rmse(s3, work, cfg)

        result = json.loads((work / "_per_bodypart_eval.json").read_text())
        bp = result["bodyparts"]["bp1"]
        assert bp["pck_5"] == pytest.approx(0.0)
        assert bp["pck_10"] == pytest.approx(100.0)
        assert bp["pck_20"] == pytest.approx(100.0)

    def test_n_total_matched_is_sum_across_bodyparts(self, tmp_path):
        """n_total_matched counts all matched bodypart-frame pairs."""
        work, cfg = _setup_rmse_project(
            tmp_path,
            bodyparts=["bp1", "bp2"],
            gt_coords={
                "bp1": [(0.0, 0.0), (0.0, 0.0)],
                "bp2": [(1.0, 1.0), (1.0, 1.0)],
            },
            pred_coords={
                "bp1": [(1.0, 0.0), (1.0, 0.0)],
                "bp2": [(2.0, 1.0), (2.0, 1.0)],
            },
        )
        s3 = MagicMock()
        rdr._compute_per_bodypart_rmse(s3, work, cfg)

        result = json.loads((work / "_per_bodypart_eval.json").read_text())
        # 2 bodyparts x 2 frames = 4
        assert result["n_total_matched"] == 4

    def test_string_index_in_per_frame(self, tmp_path):
        """String indices are preserved as frame IDs in per_frame output."""
        index = ["path/to/frame_001.png", "path/to/frame_002.png"]
        work, cfg = _setup_rmse_project(
            tmp_path,
            bodyparts=["bp1"],
            gt_coords={"bp1": [(0.0, 0.0), (1.0, 1.0)]},
            pred_coords={"bp1": [(1.0, 0.0), (2.0, 1.0)]},
            index=index,
        )
        s3 = MagicMock()
        rdr._compute_per_bodypart_rmse(s3, work, cfg)

        result = json.loads((work / "_per_bodypart_eval.json").read_text())
        frame_ids = [f["frame_id"] for f in result["per_frame"]]
        assert "path/to/frame_001.png" in frame_ids
        assert "path/to/frame_002.png" in frame_ids

    def test_tuple_index_produces_valid_rmse(self, tmp_path):
        """Tuple indices (DLC's native format) are handled correctly.

        DLC DataFrames often have tuple indices like
        ``("labeled-data", "clip_name", "frame_000123.png")``.  The
        function must not skip these frames due to pandas ``.loc``
        multi-level indexing ambiguity.
        """
        index = [
            ("labeled-data", "clip1", "frame_001.png"),
            ("labeled-data", "clip1", "frame_002.png"),
        ]
        work, cfg = _setup_rmse_project(
            tmp_path,
            bodyparts=["bp1"],
            gt_coords={"bp1": [(0.0, 0.0), (0.0, 0.0)]},
            pred_coords={"bp1": [(3.0, 4.0), (3.0, 4.0)]},
            index=index,
        )
        s3 = MagicMock()
        rdr._compute_per_bodypart_rmse(s3, work, cfg)

        result_path = work / "_per_bodypart_eval.json"
        assert result_path.exists(), "JSON output must be produced for tuple indices"

        result = json.loads(result_path.read_text())
        assert result["bodyparts"]["bp1"]["rmse"] == pytest.approx(5.0, abs=0.01)
        assert result["bodyparts"]["bp1"]["n"] == 2

        # per_frame entries should reference the last tuple element as frame_id
        frame_ids = [f["frame_id"] for f in result["per_frame"]]
        assert "frame_001.png" in frame_ids
        assert "frame_002.png" in frame_ids

    def test_no_prediction_file_prints_warning(self, tmp_path, capsys):
        """When no prediction H5 is found and DLC is not available, returns early."""
        work, cfg = _setup_rmse_project(
            tmp_path,
            bodyparts=["bp1"],
            gt_coords={"bp1": [(0.0, 0.0)]},
            pred_coords={"bp1": [(1.0, 1.0)]},
            write_pred=False,
        )
        # Remove the fallback DLC module to prevent inference attempt
        dlc_mock = MagicMock()
        dlc_mock.analyze_time_lapse_frames = MagicMock(side_effect=AttributeError("not available"))

        s3 = MagicMock()
        rdr._compute_per_bodypart_rmse(s3, work, cfg)

        out = capsys.readouterr().out
        assert "No predictions available" in out or "No evaluation prediction" in out

    def test_zero_error_produces_zero_rmse(self, tmp_path):
        """Perfect predictions should give RMSE = 0."""
        work, cfg = _setup_rmse_project(
            tmp_path,
            bodyparts=["bp1"],
            gt_coords={"bp1": [(10.0, 20.0), (30.0, 40.0)]},
            pred_coords={"bp1": [(10.0, 20.0), (30.0, 40.0)]},
        )
        s3 = MagicMock()
        rdr._compute_per_bodypart_rmse(s3, work, cfg)

        result = json.loads((work / "_per_bodypart_eval.json").read_text())
        assert result["bodyparts"]["bp1"]["rmse"] == pytest.approx(0.0, abs=1e-6)
        assert result["bodyparts"]["bp1"]["pck_5"] == pytest.approx(100.0)

    def test_std_correct_for_varying_errors(self, tmp_path):
        """Standard deviation is computed correctly for mixed errors."""
        # Error = 5.0 for frame 1, 13.0 for frame 2
        work, cfg = _setup_rmse_project(
            tmp_path,
            bodyparts=["bp1"],
            gt_coords={"bp1": [(0.0, 0.0), (0.0, 0.0)]},
            pred_coords={"bp1": [(3.0, 4.0), (5.0, 12.0)]},
        )
        s3 = MagicMock()
        rdr._compute_per_bodypart_rmse(s3, work, cfg)

        result = json.loads((work / "_per_bodypart_eval.json").read_text())
        expected_std = float(np.std([5.0, 13.0]))
        assert result["bodyparts"]["bp1"]["std"] == pytest.approx(expected_std, abs=0.01)

    def test_cross_format_gt_tuple_pred_string(self, tmp_path):
        """GT with tuple indices matches pred with string path indices.

        DLC 3.x evaluate_network stores predictions with full file path
        indices (strings), while GT CollectedData uses tuple indices like
        ``('labeled-data', 'clip_name', 'frame_000123.png')``. The
        function must fall back to filename-stem matching and correctly
        look up each row using its own DataFrame's index format.
        """
        bodyparts = ["bp1"]
        gt_index = [
            ("labeled-data", "clip1", "frame_001.png"),
            ("labeled-data", "clip1", "frame_002.png"),
        ]
        pred_index = [
            "/tmp/dlc-retrain/labeled-data/clip1/frame_001.png",
            "/tmp/dlc-retrain/labeled-data/clip1/frame_002.png",
        ]

        work = tmp_path / "dlc-retrain"
        work.mkdir()
        cfg_path = work / "config.yaml"
        cfg_path.write_text(
            yaml.dump(
                {
                    "bodyparts": bodyparts,
                    "project_path": str(work),
                }
            )
        )

        # Ground-truth with tuple indices
        gt_dir = work / "labeled-data" / "clip1"
        gt_dir.mkdir(parents=True)
        gt_df = _make_dlc_df(
            "scorer_gt",
            bodyparts,
            {"bp1": [(0.0, 0.0), (0.0, 0.0)]},
            index=gt_index,
        )
        gt_df.to_hdf(gt_dir / "CollectedData_scorer_gt.h5", key="df_with_missing")

        # Predictions with string path indices (DLC 3.x format)
        eval_dir = work / "evaluation-results-pytorch" / "iteration-0" / "test"
        eval_dir.mkdir(parents=True)
        pred_df = _make_dlc_df(
            "DLC_scorer",
            bodyparts,
            {"bp1": [(3.0, 4.0), (3.0, 4.0)]},
            index=pred_index,
        )
        pred_df.to_hdf(eval_dir / "DLC_scorer.h5", key="df_with_missing")

        s3 = MagicMock()
        rdr._compute_per_bodypart_rmse(s3, work, cfg_path)

        result_path = work / "_per_bodypart_eval.json"
        assert result_path.exists(), "Cross-format matching must produce output"

        result = json.loads(result_path.read_text())
        assert result["bodyparts"]["bp1"]["rmse"] == pytest.approx(5.0, abs=0.01)
        assert result["bodyparts"]["bp1"]["n"] == 2
        assert len(result["per_frame"]) == 2

    def test_pck_15_present_in_output(self, tmp_path):
        """PCK@15 must be included in the output JSON for the frontend."""
        work, cfg = _setup_rmse_project(
            tmp_path,
            bodyparts=["bp1"],
            gt_coords={"bp1": [(0.0, 0.0), (0.0, 0.0)]},
            pred_coords={"bp1": [(0.0, 7.0), (0.0, 7.0)]},
        )
        s3 = MagicMock()
        rdr._compute_per_bodypart_rmse(s3, work, cfg)

        result = json.loads((work / "_per_bodypart_eval.json").read_text())
        bp = result["bodyparts"]["bp1"]
        assert "pck_15" in bp, "pck_15 must be present in output"
        # Error = 7.0, which is <= 15, so PCK@15 should be 100%
        assert bp["pck_15"] == pytest.approx(100.0)


# ═══════════════════════════════════════════════════════════════════════
# _upload_eval_results_json
# ═══════════════════════════════════════════════════════════════════════


class TestUploadEvalResultsJson:
    def _make_eval_csv(self, work: Path, data: dict) -> Path:
        """Write a minimal DLC evaluation CSV.

        The filename must match the glob ``*-results.csv`` used by
        ``_upload_eval_results_json`` for file discovery.
        """
        csv_path = work / "evaluation-results" / "DLC-evaluation-results.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        import csv

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data.keys()))
            writer.writeheader()
            writer.writerow(data)
        return csv_path

    def test_correct_json_structure(self, tmp_path):
        """Output JSON has expected keys: champion_id, train, test, etc."""
        work = tmp_path / "dlc-retrain"
        work.mkdir()
        cfg_path = work / "config.yaml"
        cfg_path.write_text(
            yaml.dump(
                {
                    "TrainingFraction": [0.8],
                    "bodyparts": ["bp1"],
                }
            )
        )

        self._make_eval_csv(
            work,
            {
                "Training epochs": 100,
                "train rmse": 3.5,
                "train rmse_pcutoff": 2.1,
                "train mAP": 85.0,
                "train mAR": 80.0,
                "test rmse": 5.2,
                "test rmse_pcutoff": 3.8,
                "test mAP": 72.0,
                "test mAR": 68.0,
            },
        )

        s3 = MagicMock()
        # Mock champion JSON lookup
        s3.get_object.return_value = {
            "Body": MagicMock(
                read=MagicMock(return_value=json.dumps({"champion_id": "champ_123"}).encode())
            )
        }

        rdr._upload_eval_results_json(s3, work, cfg_path, epochs=100)

        # Verify put_object was called
        s3.put_object.assert_called_once()
        call_kwargs = s3.put_object.call_args.kwargs
        assert call_kwargs["Key"] == "dlc-retrain/_eval_results.json"
        assert call_kwargs["ContentType"] == "application/json"

        body = json.loads(call_kwargs["Body"])
        assert body["champion_id"] == "champ_123"
        assert body["training_fraction"] == 0.8
        assert body["best_epoch"] == 100
        assert body["total_epochs"] == 100
        assert body["train"]["rmse"] == pytest.approx(3.5)
        assert body["train"]["mAP"] == pytest.approx(85.0)
        assert body["test"]["rmse"] == pytest.approx(5.2)
        assert body["test"]["mAP"] == pytest.approx(72.0)

    def test_missing_csv_prints_warning(self, tmp_path, capsys):
        """When no eval CSV exists, prints warning and returns."""
        work = tmp_path / "dlc-retrain"
        work.mkdir()
        cfg_path = work / "config.yaml"
        cfg_path.write_text(
            yaml.dump(
                {
                    "TrainingFraction": [0.8],
                }
            )
        )

        s3 = MagicMock()
        rdr._upload_eval_results_json(s3, work, cfg_path, epochs=100)

        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "no evaluation CSV" in out.lower() or "eval" in out.lower()
        s3.put_object.assert_not_called()

    def test_empty_csv_prints_warning(self, tmp_path, capsys):
        """Empty CSV triggers a warning and no upload."""
        work = tmp_path / "dlc-retrain"
        work.mkdir()
        cfg_path = work / "config.yaml"
        cfg_path.write_text(
            yaml.dump(
                {
                    "TrainingFraction": [0.8],
                }
            )
        )

        csv_path = work / "evaluation-results" / "CombinedEvaluation-results.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        # Write header-only CSV
        csv_path.write_text("col1,col2\n")

        s3 = MagicMock()
        rdr._upload_eval_results_json(s3, work, cfg_path, epochs=100)

        out = capsys.readouterr().out
        assert "empty" in out.lower() or "WARNING" in out
        s3.put_object.assert_not_called()

    def test_champion_lookup_failure_uses_unknown(self, tmp_path):
        """When champion JSON is not on S3, champion_id falls back to 'unknown'."""
        work = tmp_path / "dlc-retrain"
        work.mkdir()
        cfg_path = work / "config.yaml"
        cfg_path.write_text(
            yaml.dump(
                {
                    "TrainingFraction": [0.8],
                }
            )
        )

        self._make_eval_csv(
            work,
            {
                "Training epochs": 50,
                "train rmse": 4.0,
                "train rmse_pcutoff": 3.0,
                "train mAP": 70.0,
                "train mAR": 65.0,
                "test rmse": 6.0,
                "test rmse_pcutoff": 5.0,
                "test mAP": 60.0,
                "test mAR": 55.0,
            },
        )

        s3 = MagicMock()
        # Both get_object calls fail
        s3.get_object.side_effect = Exception("NoSuchKey")

        rdr._upload_eval_results_json(s3, work, cfg_path, epochs=50)

        body = json.loads(s3.put_object.call_args.kwargs["Body"])
        assert body["champion_id"] == "unknown"
        assert body["previous_champion"] == {}

    def test_previous_champion_included_when_available(self, tmp_path):
        """Previous eval results are included in the output JSON."""
        work = tmp_path / "dlc-retrain"
        work.mkdir()
        cfg_path = work / "config.yaml"
        cfg_path.write_text(
            yaml.dump(
                {
                    "TrainingFraction": [0.8],
                }
            )
        )

        self._make_eval_csv(
            work,
            {
                "Training epochs": 100,
                "train rmse": 3.0,
                "train rmse_pcutoff": 2.0,
                "train mAP": 90.0,
                "train mAR": 85.0,
                "test rmse": 4.0,
                "test rmse_pcutoff": 3.0,
                "test mAP": 80.0,
                "test mAR": 75.0,
            },
        )

        prev_eval = {
            "champion_id": "old_champ",
            "train": {"rmse": 5.0, "mAP": 70.0},
            "n_labeled_frames": 100,
            "training_fraction": 0.8,
        }

        def mock_get_object(Bucket, Key):
            if Key == "dlc-champion.json":
                return {
                    "Body": MagicMock(
                        read=MagicMock(
                            return_value=json.dumps({"champion_id": "new_champ"}).encode()
                        )
                    )
                }
            elif Key == "dlc-retrain/_eval_results.json":
                return {
                    "Body": MagicMock(read=MagicMock(return_value=json.dumps(prev_eval).encode()))
                }
            raise Exception("NoSuchKey")

        s3 = MagicMock()
        s3.get_object.side_effect = mock_get_object

        rdr._upload_eval_results_json(s3, work, cfg_path, epochs=100)

        body = json.loads(s3.put_object.call_args.kwargs["Body"])
        assert body["previous_champion"]["champion_id"] == "old_champ"
        assert body["previous_champion"]["train_rmse"] == pytest.approx(5.0)
        assert body["previous_champion"]["train_mAP"] == pytest.approx(70.0)

    def test_training_fraction_from_config(self, tmp_path):
        """Training fraction is read from config.yaml's TrainingFraction."""
        work = tmp_path / "dlc-retrain"
        work.mkdir()
        cfg_path = work / "config.yaml"
        cfg_path.write_text(
            yaml.dump(
                {
                    "TrainingFraction": [0.95],
                }
            )
        )

        self._make_eval_csv(
            work,
            {
                "Training epochs": 200,
                "train rmse": 2.0,
                "train rmse_pcutoff": 1.5,
                "train mAP": 95.0,
                "train mAR": 92.0,
                "test rmse": 3.0,
                "test rmse_pcutoff": 2.5,
                "test mAP": 88.0,
                "test mAR": 84.0,
            },
        )

        s3 = MagicMock()
        s3.get_object.side_effect = Exception("NoSuchKey")

        rdr._upload_eval_results_json(s3, work, cfg_path, epochs=200)

        body = json.loads(s3.put_object.call_args.kwargs["Body"])
        assert body["training_fraction"] == pytest.approx(0.95)

    def test_labeled_frame_count(self, tmp_path):
        """n_labeled_frames counts rows across all CollectedData H5 files."""
        work = tmp_path / "dlc-retrain"
        work.mkdir()
        cfg_path = work / "config.yaml"
        cfg_path.write_text(
            yaml.dump(
                {
                    "TrainingFraction": [0.8],
                }
            )
        )

        # Create two CollectedData files with different frame counts
        for clip, n_frames in [("clip1", 3), ("clip2", 5)]:
            gt_dir = work / "labeled-data" / clip
            gt_dir.mkdir(parents=True)
            gt_df = _make_dlc_df(
                "scorer",
                ["bp1"],
                {"bp1": [(float(i), float(i)) for i in range(n_frames)]},
            )
            gt_df.to_hdf(gt_dir / "CollectedData_scorer.h5", key="df_with_missing")

        self._make_eval_csv(
            work,
            {
                "Training epochs": 50,
                "train rmse": 3.0,
                "train rmse_pcutoff": 2.0,
                "train mAP": 80.0,
                "train mAR": 75.0,
                "test rmse": 4.0,
                "test rmse_pcutoff": 3.0,
                "test mAP": 70.0,
                "test mAR": 65.0,
            },
        )

        s3 = MagicMock()
        s3.get_object.side_effect = Exception("NoSuchKey")

        rdr._upload_eval_results_json(s3, work, cfg_path, epochs=50)

        body = json.loads(s3.put_object.call_args.kwargs["Body"])
        assert body["n_labeled_frames"] == 8  # 3 + 5
