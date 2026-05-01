"""Shared fixtures for pose tests (SA fine-tune work)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hm2p.pose.finetune import (
    HM2P_BODYPARTS,
    GateConfig,
    Verdict,
    evaluate_promotion_gate,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# RNG
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic RNG for synthetic-array fixtures."""
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Synthetic per-frame error fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_per_frame_errors_small(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0, 50, size=(20, 8))


@pytest.fixture
def synthetic_per_frame_errors_medium(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0, 50, size=(200, 8))


@pytest.fixture
def synthetic_errors_with_nan(rng: np.random.Generator) -> np.ndarray:
    arr = rng.uniform(0, 50, size=(50, 8))
    mask = rng.uniform(0, 1, size=arr.shape) < 0.10
    arr[mask] = np.nan
    return arr


@pytest.fixture
def synthetic_errors_all_equal() -> np.ndarray:
    return np.full((50, 8), 5.0)


@pytest.fixture
def synthetic_errors_n1() -> np.ndarray:
    return np.full((1, 8), 5.0)


# ---------------------------------------------------------------------------
# Paired (baseline, candidate) fixtures
# ---------------------------------------------------------------------------


def _baseline_grid(rng: np.random.Generator, n: int = 200) -> np.ndarray:
    """Plausible baseline error grid: large on nose/tail, small elsewhere."""
    medians = np.array([24.0, 5.0, 5.0, 12.0, 5.0, 5.0, 5.0, 59.0])
    # Use exponential noise so distributions are heavy-tailed (matches v2 §4.5).
    out = np.empty((n, 8))
    for k, m in enumerate(medians):
        # Exponential mean = m gives median = m * ln(2). Adjust scale so
        # column median lands close to m.
        scale = m / np.log(2)
        out[:, k] = rng.exponential(scale, size=n)
    return out


@pytest.fixture
def synthetic_clear_winner_pair(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Baseline as above; candidate is 0.4× on nose/tail and ~0.95× elsewhere."""
    baseline = _baseline_grid(rng, n=200)
    candidate = baseline.copy()
    candidate[:, 0] *= 0.4  # nose_tip
    candidate[:, 7] *= 0.4  # tail_base
    candidate[:, 3] *= 0.7  # head_midpoint — strong p90 reduction
    # Slight improvement everywhere else.
    for k in (1, 2, 4, 5, 6):
        candidate[:, k] *= 0.95
    return baseline, candidate


@pytest.fixture
def synthetic_clear_loser_pair(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Candidate worse on nose by 20%."""
    baseline = _baseline_grid(rng, n=200)
    candidate = baseline.copy()
    candidate[:, 0] *= 1.2
    return baseline, candidate


@pytest.fixture
def synthetic_mixed_pair(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Nose/tail beat baseline but mid_back regresses 15%."""
    baseline = _baseline_grid(rng, n=200)
    candidate = baseline.copy()
    candidate[:, 0] *= 0.4
    candidate[:, 7] *= 0.4
    candidate[:, 3] *= 0.7
    candidate[:, 5] *= 1.15  # mid_back regression
    return baseline, candidate


@pytest.fixture
def synthetic_insufficient_pair(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """5 frames only — below default min_pairs=10."""
    baseline = _baseline_grid(rng, n=5)
    candidate = baseline.copy() * 0.5
    return baseline, candidate


# ---------------------------------------------------------------------------
# Detector probe fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_sa_detector_probe_pass() -> list[bool]:
    return [True] * 10


@pytest.fixture
def mock_sa_detector_probe_partial() -> list[bool]:
    """One out of ten — fails the 90% threshold."""
    return [True] + [False] * 9


@pytest.fixture
def mock_sa_detector_probe_zero() -> list[bool]:
    return [False] * 10


# ---------------------------------------------------------------------------
# Verdict fixtures (full Verdict instances + JSON twins)
# ---------------------------------------------------------------------------


def _make_verdict(
    rng: np.random.Generator,
    pair: tuple[np.ndarray, np.ndarray],
    *,
    baseline_id: str = "dlc-20260430-hrnetw32-snap110",
    candidate_id: str = "dlc-20260501-hrnetw32-snap60",
) -> Verdict:
    e_b, e_c = pair
    return evaluate_promotion_gate(
        e_b,
        e_c,
        list(HM2P_BODYPARTS),
        hd_baseline_rad=None,
        hd_candidate_rad=None,
        hd_gt_rad=None,
        baseline_id=baseline_id,
        candidate_id=candidate_id,
        gate=GateConfig(),
        rng=rng,
    )


@pytest.fixture
def verdict_pass_fixture(
    rng: np.random.Generator,
    synthetic_clear_winner_pair: tuple[np.ndarray, np.ndarray],
) -> Verdict:
    return _make_verdict(rng, synthetic_clear_winner_pair)


@pytest.fixture
def verdict_fail_fixture(
    rng: np.random.Generator,
    synthetic_clear_loser_pair: tuple[np.ndarray, np.ndarray],
) -> Verdict:
    return _make_verdict(rng, synthetic_clear_loser_pair)


@pytest.fixture
def verdict_mixed_fixture(
    rng: np.random.Generator,
    synthetic_mixed_pair: tuple[np.ndarray, np.ndarray],
) -> Verdict:
    return _make_verdict(rng, synthetic_mixed_pair)


@pytest.fixture
def verdict_schema() -> dict:
    """Load the verdict JSON schema from disk."""
    return json.loads((FIXTURES_DIR / "verdict.schema.json").read_text())
