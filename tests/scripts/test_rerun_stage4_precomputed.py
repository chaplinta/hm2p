"""Tests for scripts/rerun_stage4_precomputed.py pure helpers (synthetic arrays)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from rerun_stage4_precomputed import (  # noqa: E402
    looks_like_fissa,
    neuropil_linear_r2,
    parse_exp_id,
)


def test_parse_exp_id():
    assert parse_exp_id("20220804_13_52_02_1117646") == ("sub-1117646", "ses-20220804T135202")


def test_looks_like_fissa():
    rng = np.random.default_rng(0)
    F_raw = 100 + rng.normal(0, 2, (3, 200))
    Fneu = 70 + rng.normal(0, 2, (3, 200))
    # separated signal not linear in Fneu (even with a residual baseline)
    assert looks_like_fissa(10 + np.abs(rng.normal(0, 1, (3, 200))), F_raw, Fneu)
    assert not looks_like_fissa(F_raw - 0.3 * Fneu, F_raw, Fneu)  # coefficient subtraction
    assert neuropil_linear_r2(F_raw - 0.3 * Fneu, F_raw, Fneu) == pytest.approx([1, 1, 1])
