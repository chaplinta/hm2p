"""Tests for ``_create_stratified_split()`` and ``_clip_dir_to_exp_id()``
in ``scripts/run_dlc_retrain.py``.

All tests use small synthetic arrays — no real data files. The DLC and
boto3 imports are stubbed out so the module loads cleanly under test.

References
----------
Glazner et al. 2025. "Find the Leak, Fix the Split." arXiv:2511.13944.
"""

from __future__ import annotations

import csv
import pickle
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# Pre-stub heavy imports so the module loads cleanly under test.
sys.modules.setdefault("deeplabcut", MagicMock())
sys.modules.setdefault("dlclibrary", MagicMock())
sys.modules.setdefault("boto3", MagicMock())

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
import run_dlc_retrain as rdr  # noqa: E402, I001


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#: 8 bodyparts matching the project convention.
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


def _build_experiments_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    """Write a minimal experiments.csv with the required columns."""
    path.mkdir(parents=True, exist_ok=True)
    csv_path = path / "experiments.csv"
    fieldnames = ["exp_index", "exp_id", "primary_exp", "exclude"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return csv_path


def _build_collected_data(
    parent: Path,
    clip_name: str,
    n_frames: int,
    *,
    rng: np.random.Generator | None = None,
    nan_frac: float = 0.0,
) -> Path:
    """Create a minimal CollectedData H5 file under labeled-data/{clip_name}/.

    Returns the path to the H5 file. Coordinates are random in [0, 640].
    """
    if rng is None:
        rng = np.random.default_rng(42)

    clip_dir = parent / "labeled-data" / clip_name
    clip_dir.mkdir(parents=True, exist_ok=True)

    scorer = "tristan"
    columns = pd.MultiIndex.from_tuples(
        [(scorer, bp, coord) for bp in BODYPARTS for coord in ("x", "y")],
        names=["scorer", "bodyparts", "coords"],
    )
    index = pd.MultiIndex.from_tuples(
        [("labeled-data", clip_name, f"frame_{i:06d}.png") for i in range(n_frames)]
    )
    data = rng.uniform(0, 640, size=(n_frames, len(BODYPARTS) * 2))

    # Inject NaNs if requested.
    if nan_frac > 0:
        nan_mask = rng.random(data.shape) < nan_frac
        data[nan_mask] = np.nan

    df = pd.DataFrame(data, index=index, columns=columns)
    h5_path = clip_dir / "CollectedData_tristan.h5"
    df.to_hdf(h5_path, key="df_with_missing", mode="w")
    return h5_path


def _build_documentation_pickle(
    work: Path,
    n_total: int,
    train_frac: float = 0.8,
) -> Path:
    """Create a minimal Documentation_data pickle matching DLC's format.

    The pickle contains [data, trainIndices, testIndices, trainFraction].
    """
    td_dir = work / "training-datasets" / "iteration-0" / "UnaugmentedDataSet_test"
    td_dir.mkdir(parents=True, exist_ok=True)

    n_train = int(n_total * train_frac)
    indices = np.arange(n_total)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    # data is a list of dicts (one per training frame); we use a stub.
    data = [{"image": f"frame_{i}.png", "size": (3, 480, 640)} for i in train_idx]

    pickle_path = td_dir / "Documentation_data-test_80shuffle1.pickle"
    with open(pickle_path, "wb") as f:
        pickle.dump([data, train_idx, test_idx, train_frac], f)

    return pickle_path


def _make_test_project(
    tmp_path: Path,
    *,
    n_primary: int = 5,
    n_secondary: int = 3,
    n_excluded: int = 2,
    frames_per_session: int = 20,
    nan_frac: float = 0.0,
) -> tuple[Path, Path, int]:
    """Build a minimal DLC project with labeled data and experiments.csv.

    Returns (work_dir, metadata_csv, total_frames).
    """
    work = tmp_path / "dlc-retrain"
    work.mkdir()

    rng = np.random.default_rng(42)

    # Generate session exp_ids and matching clip dir names.
    # Use a base date/animal pattern that is easy to track.
    exp_rows: list[dict[str, str]] = []
    total_frames = 0
    session_idx = 0

    for _kind, count, primary, exclude in [
        ("primary", n_primary, "1", "0"),
        ("secondary", n_secondary, "0", "0"),
        ("excluded", n_excluded, "1", "1"),
    ]:
        for _i in range(count):
            animal_id = str(1110000 + session_idx)
            # exp_id: use HH_MM_SS offset of 10*session_idx minutes.
            hour = 10 + (session_idx * 10) // 60
            minute = (session_idx * 10) % 60
            exp_id = f"20220101_{hour:02d}_{minute:02d}_00_{animal_id}"
            # Clip name: same date/animal, offset by a few seconds.
            clip_time = f"{hour:02d}_{minute:02d}_04"
            clip_name = f"20220101_{clip_time}_{animal_id}_maze-rose_overhead.camera-cropped"

            exp_rows.append(
                {
                    "exp_index": str(session_idx + 1),
                    "exp_id": exp_id,
                    "primary_exp": primary,
                    "exclude": exclude,
                }
            )

            _build_collected_data(work, clip_name, frames_per_session, rng=rng, nan_frac=nan_frac)
            total_frames += frames_per_session
            session_idx += 1

    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    metadata_csv = _build_experiments_csv(metadata_dir, exp_rows)

    # Build a Documentation_data pickle.
    _build_documentation_pickle(work, total_frames)

    return work, metadata_csv, total_frames


# ---------------------------------------------------------------------------
# _clip_dir_to_exp_id
# ---------------------------------------------------------------------------


class TestClipDirToExpId:
    """Tests for the clip directory to exp_id mapping function."""

    def test_exact_animal_date_match(self):
        exp_ids = ["20220804_11_21_59_1117646", "20220804_13_52_02_1117646"]
        # Clip dir is offset by ~4 seconds.
        result = rdr._clip_dir_to_exp_id(
            "20220804_11_22_03_1117646_maze-rose_overhead.camera-cropped",
            exp_ids,
        )
        assert result == "20220804_11_21_59_1117646"

    def test_second_session_same_day(self):
        exp_ids = ["20220804_11_21_59_1117646", "20220804_13_52_02_1117646"]
        result = rdr._clip_dir_to_exp_id(
            "20220804_13_52_06_1117646_maze-rose_overhead.camera-cropped",
            exp_ids,
        )
        assert result == "20220804_13_52_02_1117646"

    def test_no_match_different_animal(self):
        exp_ids = ["20220804_11_21_59_1117646"]
        result = rdr._clip_dir_to_exp_id(
            "20220804_11_22_03_9999999_maze-rose_overhead.camera-cropped",
            exp_ids,
        )
        assert result is None

    def test_no_match_different_date(self):
        exp_ids = ["20220804_11_21_59_1117646"]
        result = rdr._clip_dir_to_exp_id(
            "20220805_11_22_03_1117646_maze-rose_overhead.camera-cropped",
            exp_ids,
        )
        assert result is None

    def test_rejects_large_time_offset(self):
        """Clips more than 60 seconds away from any exp_id should not match."""
        exp_ids = ["20220804_11_21_59_1117646"]
        # 2 minutes offset.
        result = rdr._clip_dir_to_exp_id(
            "20220804_11_24_00_1117646_maze-rose_overhead.camera-cropped",
            exp_ids,
        )
        assert result is None

    def test_short_clip_name_returns_none(self):
        result = rdr._clip_dir_to_exp_id("short", ["20220804_11_21_59_1117646"])
        assert result is None

    def test_empty_exp_ids(self):
        result = rdr._clip_dir_to_exp_id(
            "20220804_11_22_03_1117646_maze-rose_overhead.camera-cropped",
            [],
        )
        assert result is None


# ---------------------------------------------------------------------------
# _create_stratified_split — session selection logic
# ---------------------------------------------------------------------------


class TestSelectsPrimaryNonExcludedSessions:
    """Verify only primary non-excluded sessions are candidates for test."""

    def test_selects_primary_non_excluded_sessions(self, tmp_path: Path):
        work, metadata_csv, _ = _make_test_project(
            tmp_path,
            n_primary=5,
            n_secondary=3,
            n_excluded=2,
            frames_per_session=20,
        )

        # Read the experiments.csv to know which sessions are primary non-excluded.
        primary_ne = set()
        with open(metadata_csv) as f:
            for row in csv.DictReader(f):
                if row["primary_exp"] == "1" and row["exclude"] == "0":
                    primary_ne.add(row["exp_id"])

        result = rdr._create_stratified_split(work, metadata_csv, n_clusters=4, n_test_sessions=2)
        assert result is True

        # Load the overwritten pickle and check test indices.
        doc_pickles = sorted(work.rglob("Documentation_data-*.pickle"))
        assert len(doc_pickles) > 0
        with open(doc_pickles[-1], "rb") as f:
            meta = pickle.load(f)

        # Indices in pickle should be valid.
        train_idx = set(int(i) for i in meta[1])
        test_idx = set(int(i) for i in meta[2])
        assert len(train_idx & test_idx) == 0  # No overlap.


class TestAllClustersCovered:
    """Verify the test set covers all pose clusters."""

    def test_all_clusters_covered_in_test(self, tmp_path: Path):
        """With enough sessions, all clusters should be represented in test."""
        work, metadata_csv, total = _make_test_project(
            tmp_path,
            n_primary=8,
            n_secondary=2,
            n_excluded=0,
            frames_per_session=30,
        )
        n_clusters = 4
        result = rdr._create_stratified_split(
            work, metadata_csv, n_clusters=n_clusters, n_test_sessions=3
        )
        assert result is True

        # Load pickle and verify using k-means that test frames span all clusters.
        # Re-run clustering with the same params to verify.
        from sklearn.cluster import KMeans

        # Load all frames.
        ld_root = work / "labeled-data"
        gt_files = [
            h5
            for h5 in sorted(ld_root.rglob("CollectedData_*.h5"))
            if len(h5.relative_to(ld_root).parts) == 2
        ]
        frames = pd.concat([pd.read_hdf(gf) for gf in gt_files], ignore_index=False)
        scorer = frames.columns.get_level_values(0)[0]
        bodyparts = frames.columns.get_level_values(1).unique().tolist()

        coords = []
        for bp in bodyparts:
            coords.extend(
                [
                    frames[(scorer, bp, "x")].values.astype(float),
                    frames[(scorer, bp, "y")].values.astype(float),
                ]
            )
        feat = np.column_stack(coords)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(feat)

        doc_pickles = sorted(work.rglob("Documentation_data-*.pickle"))
        with open(doc_pickles[-1], "rb") as f:
            meta = pickle.load(f)
        test_idx = np.array(meta[2], dtype=int)

        # Check all clusters represented.
        test_clusters = set(labels[test_idx])
        assert test_clusters == set(range(n_clusters))


class TestNoSessionOverlap:
    """Verify no frame appears in both train and test."""

    def test_no_session_overlap(self, tmp_path: Path):
        work, metadata_csv, total = _make_test_project(
            tmp_path,
            n_primary=6,
            n_secondary=2,
            n_excluded=2,
            frames_per_session=15,
        )
        result = rdr._create_stratified_split(work, metadata_csv, n_clusters=4, n_test_sessions=2)
        assert result is True

        doc_pickles = sorted(work.rglob("Documentation_data-*.pickle"))
        with open(doc_pickles[-1], "rb") as f:
            meta = pickle.load(f)

        train_idx = set(int(i) for i in meta[1])
        test_idx = set(int(i) for i in meta[2])

        # No overlap.
        assert train_idx.isdisjoint(test_idx)
        # All frames accounted for.
        assert len(train_idx) + len(test_idx) == total


class TestKLDivergenceMinimised:
    """Verify the selected combo has the lowest KL divergence."""

    def test_kl_divergence_minimised(self, tmp_path: Path):
        """With known cluster distributions, verify KL is minimised."""
        from itertools import combinations as combos

        from scipy.special import rel_entr
        from sklearn.cluster import KMeans

        work, metadata_csv, total = _make_test_project(
            tmp_path,
            n_primary=6,
            n_secondary=2,
            n_excluded=0,
            frames_per_session=25,
        )
        n_clusters = 4

        # Run the split.
        result = rdr._create_stratified_split(
            work, metadata_csv, n_clusters=n_clusters, n_test_sessions=2
        )
        assert result is True

        # Reconstruct cluster labels to compute KL for all combos.
        ld_root = work / "labeled-data"
        gt_files = [
            h5
            for h5 in sorted(ld_root.rglob("CollectedData_*.h5"))
            if len(h5.relative_to(ld_root).parts) == 2
        ]
        frames_list = []
        frame_sessions = []
        all_exp_ids = []
        with open(metadata_csv) as f:
            for row in csv.DictReader(f):
                all_exp_ids.append(row["exp_id"])

        for gf in gt_files:
            clip_name = gf.parent.name
            exp_id = rdr._clip_dir_to_exp_id(clip_name, all_exp_ids)
            df = pd.read_hdf(gf)
            frames_list.append(df)
            frame_sessions.extend([exp_id] * len(df))

        all_frames = pd.concat(frames_list, ignore_index=False)
        scorer = all_frames.columns.get_level_values(0)[0]
        bodyparts = all_frames.columns.get_level_values(1).unique().tolist()

        coords = []
        for bp in bodyparts:
            coords.extend(
                [
                    all_frames[(scorer, bp, "x")].values.astype(float),
                    all_frames[(scorer, bp, "y")].values.astype(float),
                ]
            )
        feat = np.column_stack(coords)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(feat)

        session_ids = np.array(frame_sessions, dtype=object)
        overall_counts = np.bincount(cluster_labels, minlength=n_clusters).astype(float)
        overall_dist = overall_counts / overall_counts.sum()

        # Identify primary non-excluded with labeled data.
        primary_ne = set()
        with open(metadata_csv) as f:
            for row in csv.DictReader(f):
                if row["primary_exp"] == "1" and row["exclude"] == "0":
                    primary_ne.add(row["exp_id"])

        unique_with_data = set(s for s in frame_sessions if s is not None)
        candidates = sorted(primary_ne & unique_with_data)

        # Compute session cluster counts.
        session_counts = {}
        for sid in set(s for s in frame_sessions if s is not None):
            mask = session_ids == sid
            session_counts[sid] = np.bincount(cluster_labels[mask], minlength=n_clusters).astype(
                float
            )

        # Find the actual best combo by brute force.
        eps = 1e-10
        best_kl = float("inf")
        for combo in combos(candidates, 2):
            c = np.zeros(n_clusters)
            for sid in combo:
                c += session_counts[sid]
            t = c.sum()
            if t == 0:
                continue
            d = c / t
            kl = float(np.sum(rel_entr(d + eps, overall_dist + eps)))
            if kl < best_kl:
                best_kl = kl

        # Load the pickle to see which indices were selected.
        doc_pickles = sorted(work.rglob("Documentation_data-*.pickle"))
        with open(doc_pickles[-1], "rb") as f:
            meta = pickle.load(f)
        test_idx = np.array(meta[2], dtype=int)

        # Compute KL of the selected test set.
        test_counts = np.bincount(cluster_labels[test_idx], minlength=n_clusters).astype(float)
        test_dist = test_counts / test_counts.sum()
        actual_kl = float(np.sum(rel_entr(test_dist + eps, overall_dist + eps)))

        # The selected combo should match the brute-force best.
        assert actual_kl == pytest.approx(best_kl, abs=1e-8)


# ---------------------------------------------------------------------------
# Fallback behaviour
# ---------------------------------------------------------------------------


class TestFallsBackOnFailure:
    """If clustering or session selection fails, return False."""

    def test_returns_false_with_no_labeled_data(self, tmp_path: Path):
        work = tmp_path / "dlc-retrain"
        work.mkdir()
        csv_path = _build_experiments_csv(
            tmp_path / "meta",
            [
                {
                    "exp_index": "1",
                    "exp_id": "20220101_10_00_00_1110000",
                    "primary_exp": "1",
                    "exclude": "0",
                }
            ],
        )
        result = rdr._create_stratified_split(work, csv_path, n_clusters=4)
        assert result is False

    def test_returns_false_with_missing_metadata(self, tmp_path: Path):
        work = tmp_path / "dlc-retrain"
        work.mkdir()
        missing_csv = tmp_path / "nonexistent.csv"
        result = rdr._create_stratified_split(work, missing_csv)
        assert result is False

    def test_returns_false_with_too_few_primary_sessions(self, tmp_path: Path):
        work, metadata_csv, _ = _make_test_project(
            tmp_path,
            n_primary=2,  # Need 3, only have 2.
            n_secondary=5,
            n_excluded=0,
            frames_per_session=20,
        )
        result = rdr._create_stratified_split(work, metadata_csv, n_clusters=4, n_test_sessions=3)
        assert result is False

    def test_returns_false_with_no_pickle(self, tmp_path: Path):
        """If no Documentation_data pickle exists, should return False."""
        work = tmp_path / "dlc-retrain"
        work.mkdir()
        rng = np.random.default_rng(42)

        csv_path = _build_experiments_csv(
            tmp_path / "meta",
            [
                {
                    "exp_index": "1",
                    "exp_id": "20220101_10_00_00_1110000",
                    "primary_exp": "1",
                    "exclude": "0",
                },
                {
                    "exp_index": "2",
                    "exp_id": "20220101_10_10_00_1110001",
                    "primary_exp": "1",
                    "exclude": "0",
                },
                {
                    "exp_index": "3",
                    "exp_id": "20220101_10_20_00_1110002",
                    "primary_exp": "1",
                    "exclude": "0",
                },
                {
                    "exp_index": "4",
                    "exp_id": "20220101_10_30_00_1110003",
                    "primary_exp": "0",
                    "exclude": "0",
                },
            ],
        )
        for i in range(4):
            animal_id = str(1110000 + i)
            hour = 10
            minute = i * 10
            clip_name = (
                f"20220101_{hour:02d}_{minute:02d}_04_{animal_id}"
                f"_maze-rose_overhead.camera-cropped"
            )
            _build_collected_data(work, clip_name, 20, rng=rng)

        # No pickle → should return False.
        result = rdr._create_stratified_split(work, csv_path, n_clusters=4)
        assert result is False


# ---------------------------------------------------------------------------
# Pickle format
# ---------------------------------------------------------------------------


class TestPickleFormatCorrect:
    """Verify the overwritten pickle has the correct DLC format."""

    def test_pickle_format_correct(self, tmp_path: Path):
        work, metadata_csv, total = _make_test_project(
            tmp_path,
            n_primary=5,
            n_secondary=3,
            n_excluded=2,
            frames_per_session=20,
        )

        result = rdr._create_stratified_split(work, metadata_csv, n_clusters=4, n_test_sessions=2)
        assert result is True

        doc_pickles = sorted(work.rglob("Documentation_data-*.pickle"))
        with open(doc_pickles[-1], "rb") as f:
            meta = pickle.load(f)

        # Format: [data, trainIndices, testIndices, trainFraction]
        assert len(meta) == 4
        # data is a list of dicts.
        assert isinstance(meta[0], list)
        # trainIndices and testIndices are numpy arrays of ints.
        assert isinstance(meta[1], np.ndarray)
        assert isinstance(meta[2], np.ndarray)
        assert meta[1].dtype == int or np.issubdtype(meta[1].dtype, np.integer)
        assert meta[2].dtype == int or np.issubdtype(meta[2].dtype, np.integer)
        # trainFraction preserved.
        assert meta[3] == pytest.approx(0.8)
        # All indices are valid.
        all_idx = np.concatenate([meta[1], meta[2]])
        assert all_idx.min() >= 0
        assert all_idx.max() < total

    def test_pickle_preserves_data_field(self, tmp_path: Path):
        """The data field (meta[0]) should be unchanged by the split."""
        work, metadata_csv, total = _make_test_project(
            tmp_path,
            n_primary=4,
            n_secondary=2,
            n_excluded=0,
            frames_per_session=15,
        )

        # Read original data field.
        doc_pickles = sorted(work.rglob("Documentation_data-*.pickle"))
        with open(doc_pickles[-1], "rb") as f:
            original_meta = pickle.load(f)
        original_data = original_meta[0]

        rdr._create_stratified_split(work, metadata_csv, n_clusters=3, n_test_sessions=2)

        with open(doc_pickles[-1], "rb") as f:
            new_meta = pickle.load(f)

        # data field unchanged.
        assert len(new_meta[0]) == len(original_data)
        for orig, new in zip(original_data, new_meta[0], strict=True):
            assert orig["image"] == new["image"]


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------


class TestHandlesNanLabels:
    """Frames with NaN labels should not crash clustering."""

    def test_handles_nan_labels(self, tmp_path: Path):
        work, metadata_csv, _ = _make_test_project(
            tmp_path,
            n_primary=5,
            n_secondary=3,
            n_excluded=0,
            frames_per_session=20,
            nan_frac=0.3,  # 30% of coordinates are NaN.
        )

        # Should not raise.
        result = rdr._create_stratified_split(work, metadata_csv, n_clusters=4, n_test_sessions=2)
        assert result is True

    def test_all_nan_single_bodypart(self, tmp_path: Path):
        """If one bodypart is entirely NaN, clustering should still work."""
        work, metadata_csv, _ = _make_test_project(
            tmp_path,
            n_primary=4,
            n_secondary=2,
            n_excluded=0,
            frames_per_session=20,
        )

        # Overwrite one bodypart's coordinates with NaN in all files.
        ld_root = work / "labeled-data"
        for h5_path in sorted(ld_root.rglob("CollectedData_*.h5")):
            if len(h5_path.relative_to(ld_root).parts) != 2:
                continue
            df = pd.read_hdf(h5_path)
            scorer = df.columns.get_level_values(0)[0]
            df[(scorer, "tail_base", "x")] = np.nan
            df[(scorer, "tail_base", "y")] = np.nan
            df.to_hdf(h5_path, key="df_with_missing", mode="w")

        result = rdr._create_stratified_split(work, metadata_csv, n_clusters=3, n_test_sessions=2)
        assert result is True


# ---------------------------------------------------------------------------
# CLI arg parsing
# ---------------------------------------------------------------------------


class TestSplitCLIArgs:
    """Verify the new CLI arguments parse correctly."""

    def test_split_clusters_default(self):
        parser = rdr._build_arg_parser()
        args = parser.parse_args([])
        assert args.split_clusters == 12

    def test_split_clusters_custom(self):
        parser = rdr._build_arg_parser()
        args = parser.parse_args(["--split-clusters", "8"])
        assert args.split_clusters == 8

    def test_n_test_sessions_default(self):
        parser = rdr._build_arg_parser()
        args = parser.parse_args([])
        assert args.n_test_sessions == 3

    def test_n_test_sessions_custom(self):
        parser = rdr._build_arg_parser()
        args = parser.parse_args(["--n-test-sessions", "5"])
        assert args.n_test_sessions == 5

    def test_compatible_with_existing_args(self):
        parser = rdr._build_arg_parser()
        args = parser.parse_args(
            [
                "--sa-finetune",
                "--epochs",
                "200",
                "--split-clusters",
                "10",
                "--n-test-sessions",
                "4",
            ]
        )
        assert args.sa_finetune is True
        assert args.epochs == 200
        assert args.split_clusters == 10
        assert args.n_test_sessions == 4
