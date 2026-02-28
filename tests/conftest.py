"""Shared pytest fixtures for all tests.

All tests use synthetic data only — no real experimental data is loaded.
Fixtures are defined here so they are available across all test modules.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hm2p.session import Session

# ---------------------------------------------------------------------------
# Synthetic session fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def penk_session() -> Session:
    """A Penk+ RSP session (ADD3 virus, Cre-ON, GCaMP7f)."""
    return Session(
        session_id="20220804_13_52_02_1117646",
        animal_id="1117646",
        celltype="penk",
        gcamp="GCaMP7f",
        virus_id="ADD3",
        extractor="suite2p",
        tracker="dlc",
        orientation=15.0,
        bad_behav_times="02:30-03:00, 07:15-07:45",
    )


@pytest.fixture
def nonpenk_session() -> Session:
    """A non-Penk CamKII+ RSP session (virus 344, Cre-OFF, GCaMP7f)."""
    return Session(
        session_id="20221015_10_00_00_1116663",
        animal_id="1116663",
        celltype="nonpenk",
        gcamp="GCaMP8f",
        virus_id="344",
        extractor="suite2p",
        tracker="dlc",
        orientation=0.0,
        bad_behav_times="",
    )


# ---------------------------------------------------------------------------
# Synthetic array fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def n_frames() -> int:
    """Number of imaging frames in synthetic datasets."""
    return 1000


@pytest.fixture
def n_rois() -> int:
    """Number of ROIs in synthetic datasets."""
    return 50


@pytest.fixture
def fps_imaging() -> float:
    """Synthetic imaging frame rate."""
    return 29.97


@pytest.fixture
def fps_camera() -> float:
    """Synthetic camera frame rate."""
    return 100.0


@pytest.fixture
def synthetic_dff(n_rois: int, n_frames: int, rng: np.random.Generator) -> np.ndarray:
    """Synthetic (n_rois, n_frames) dF/F0 traces with sparse transients."""
    dff = rng.standard_normal((n_rois, n_frames)).astype(np.float32) * 0.05
    # Add sparse calcium transients
    for roi in range(n_rois):
        n_events = rng.integers(0, 10)
        for _ in range(n_events):
            onset = rng.integers(0, n_frames - 20)
            amplitude = rng.uniform(0.5, 3.0)
            tau = rng.uniform(0.3, 1.0)
            t = np.arange(20)
            dff[roi, onset : onset + 20] += amplitude * np.exp(-t / (tau * fps_imaging))
    return dff


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded NumPy random generator for reproducible synthetic data."""
    return np.random.default_rng(42)


@pytest.fixture
def tmp_h5(tmp_path: Path) -> Path:
    """Temporary path for HDF5 file output."""
    return tmp_path / "test_output.h5"
