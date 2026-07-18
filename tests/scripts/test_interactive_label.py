"""Tests for interactive_label.py H5 validation and session scanning."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers to create DLC-format H5 files
# ---------------------------------------------------------------------------

BODYPARTS = [
    "nose_tip",
    "left_ear",
    "right_ear",
    "head_midpoint",
    "neck",
    "mid_back",
    "mouse_center",
    "tail_base",
]
SCORER = "tristan"


def _make_dlc_dataframe(
    clip_name: str,
    n_frames: int,
    fill_labels: bool = True,
) -> pd.DataFrame:
    """Create a DLC-format DataFrame with proper 3-level MultiIndex."""
    columns = pd.MultiIndex.from_tuples(
        [(SCORER, bp, coord) for bp in BODYPARTS for coord in ("x", "y")],
        names=["scorer", "bodyparts", "coords"],
    )
    index = pd.MultiIndex.from_tuples(
        [("labeled-data", clip_name, f"frame_{i:06d}.png") for i in range(n_frames)]
    )
    if fill_labels:
        data = np.random.rand(n_frames, len(BODYPARTS) * 2) * 800
    else:
        data = np.full((n_frames, len(BODYPARTS) * 2), np.nan)
    return pd.DataFrame(data, index=index, columns=columns)


def _write_dlc_h5(path: Path, df: pd.DataFrame, key: str = "keypoints") -> None:
    """Write a DataFrame to HDF5 with the given key."""
    df.to_hdf(path, key=key, mode="w")


# ---------------------------------------------------------------------------
# Tests for _validate_h5
# ---------------------------------------------------------------------------


class TestValidateH5:
    """Test the _validate_h5 pre-flight check."""

    def _get_validate(self):
        import sys

        sys.path.insert(0, "scripts")
        from interactive_label import _validate_h5

        return _validate_h5

    def test_valid_h5_kept(self, tmp_path):
        """A properly formatted H5 with labels should be kept."""
        validate = self._get_validate()
        clip = "20210823_test_clip"
        session_dir = tmp_path / clip
        session_dir.mkdir()
        h5 = session_dir / "CollectedData_tristan.h5"

        df = _make_dlc_dataframe(clip, n_frames=5, fill_labels=True)
        _write_dlc_h5(h5, df)

        validate(session_dir)
        assert h5.exists(), "Valid H5 should not be deleted"

    def test_all_nan_h5_removed(self, tmp_path):
        """An H5 with only NaN values (no labels) should be removed."""
        validate = self._get_validate()
        clip = "20210823_test_clip"
        session_dir = tmp_path / clip
        session_dir.mkdir()
        h5 = session_dir / "CollectedData_tristan.h5"

        df = _make_dlc_dataframe(clip, n_frames=3, fill_labels=False)
        _write_dlc_h5(h5, df)

        validate(session_dir)
        assert not h5.exists(), "All-NaN H5 should be deleted"

    def test_flat_index_h5_removed(self, tmp_path):
        """An H5 with a flat string index should be removed."""
        validate = self._get_validate()
        clip = "20210823_test_clip"
        session_dir = tmp_path / clip
        session_dir.mkdir()
        h5 = session_dir / "CollectedData_tristan.h5"

        df = _make_dlc_dataframe(clip, n_frames=3, fill_labels=True)
        # Flatten the index to simulate the bug
        df.index = pd.Index(["/".join(t) for t in df.index])
        _write_dlc_h5(h5, df)

        validate(session_dir)
        assert not h5.exists(), "Flat-index H5 should be deleted"

    def test_wrong_key_h5_removed(self, tmp_path):
        """An H5 with wrong internal key (no axis1_level0) should be removed."""
        validate = self._get_validate()
        clip = "20210823_test_clip"
        session_dir = tmp_path / clip
        session_dir.mkdir()
        h5 = session_dir / "CollectedData_tristan.h5"

        df = _make_dlc_dataframe(clip, n_frames=3, fill_labels=True)
        # Flatten index so to_hdf writes axis1 as flat array (no axis1_level0)
        df.index = pd.Index(["/".join(t) for t in df.index])
        df.to_hdf(h5, key="df_with_missing", mode="w")

        validate(session_dir)
        assert not h5.exists(), "H5 with wrong key structure should be deleted"

    def test_missing_h5_no_error(self, tmp_path):
        """No error when H5 doesn't exist."""
        validate = self._get_validate()
        session_dir = tmp_path / "no_h5_here"
        session_dir.mkdir()

        validate(session_dir)  # Should not raise

    def test_empty_h5_kept(self, tmp_path):
        """An H5 with 0 rows should be kept (DLC handles it)."""
        validate = self._get_validate()
        clip = "20210823_test_clip"
        session_dir = tmp_path / clip
        session_dir.mkdir()
        h5 = session_dir / "CollectedData_tristan.h5"

        # Create empty DataFrame with proper column structure but no rows
        columns = pd.MultiIndex.from_tuples(
            [(SCORER, bp, coord) for bp in BODYPARTS for coord in ("x", "y")],
            names=["scorer", "bodyparts", "coords"],
        )
        df = pd.DataFrame(columns=columns)
        df.to_hdf(h5, key="keypoints", mode="w")

        validate(session_dir)
        assert h5.exists(), "Empty (0-row) H5 should be kept"

    def test_csv_preserved_when_h5_removed(self, tmp_path):
        """CSV should be preserved even when corrupt H5 is removed."""
        validate = self._get_validate()
        clip = "20210823_test_clip"
        session_dir = tmp_path / clip
        session_dir.mkdir()
        h5 = session_dir / "CollectedData_tristan.h5"
        csv = session_dir / "CollectedData_tristan.csv"

        df = _make_dlc_dataframe(clip, n_frames=3, fill_labels=False)
        _write_dlc_h5(h5, df)
        csv.write_text("placeholder csv")

        validate(session_dir)
        assert not h5.exists(), "Corrupt H5 should be deleted"
        assert csv.exists(), "CSV should be preserved"


# ---------------------------------------------------------------------------
# Tests for DLC H5 format correctness
# ---------------------------------------------------------------------------


class TestDLCH5Format:
    """Test that our H5 writing produces DLC-compatible files."""

    def test_correct_key_name(self, tmp_path):
        """H5 should use 'keypoints' key, not 'df_with_missing'."""
        h5 = tmp_path / "test.h5"
        df = _make_dlc_dataframe("test_clip", 5)
        _write_dlc_h5(h5, df, key="keypoints")

        with h5py.File(h5, "r") as f:
            assert "keypoints" in f
            assert "df_with_missing" not in f

    def test_3_level_row_index(self, tmp_path):
        """H5 should have axis1 with 3 levels (labeled-data, clip, frame)."""
        h5 = tmp_path / "test.h5"
        df = _make_dlc_dataframe("test_clip", 5)
        _write_dlc_h5(h5, df)

        with h5py.File(h5, "r") as f:
            assert "keypoints/axis1_level0" in f
            assert "keypoints/axis1_level1" in f
            assert "keypoints/axis1_level2" in f

    def test_3_level_column_index(self, tmp_path):
        """H5 should have axis0 with 3 levels (scorer, bodyparts, coords)."""
        h5 = tmp_path / "test.h5"
        df = _make_dlc_dataframe("test_clip", 5)
        _write_dlc_h5(h5, df)

        with h5py.File(h5, "r") as f:
            assert "keypoints/axis0_level0" in f
            assert "keypoints/axis0_level1" in f
            assert "keypoints/axis0_level2" in f

    def test_roundtrip_preserves_structure(self, tmp_path):
        """Reading and writing should preserve the DLC format."""
        h5 = tmp_path / "test.h5"
        df_orig = _make_dlc_dataframe("test_clip", 5)
        _write_dlc_h5(h5, df_orig)

        df_read = pd.read_hdf(h5)
        assert df_read.index.nlevels == 3
        assert df_read.columns.nlevels == 3
        assert df_read.shape == df_orig.shape
        np.testing.assert_array_almost_equal(df_read.values, df_orig.values)

    def test_bodypart_rename_preserves_format(self, tmp_path):
        """Renaming bodyparts should not break the H5 format."""
        h5 = tmp_path / "test.h5"
        df = _make_dlc_dataframe("test_clip", 5)
        _write_dlc_h5(h5, df)

        # Simulate the rename operation
        df2 = pd.read_hdf(h5)
        cols = df2.columns.tolist()
        new_cols = [(s, bp.replace("head_midpoint", "renamed_bp"), c) for s, bp, c in cols]
        df2.columns = pd.MultiIndex.from_tuples(new_cols, names=df2.columns.names)
        df2.to_hdf(h5, key="keypoints", mode="w")

        # Verify structure intact
        with h5py.File(h5, "r") as f:
            assert "keypoints/axis1_level0" in f
            assert "keypoints/axis0_level0" in f

        df3 = pd.read_hdf(h5)
        assert df3.index.nlevels == 3
        assert df3.columns.nlevels == 3
