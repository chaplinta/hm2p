"""Extra tests for hm2p.analysis.classify.

Covers the default-RNG paths and every grade branch of the summary table.
Uses small synthetic signals so the shuffle-based classifier stays fast.
"""

from __future__ import annotations

import numpy as np

from hm2p.analysis.classify import (
    classification_summary_table,
    classify_population,
    classify_single_cell,
)


def _hd_and_mask(n: int = 300):
    """Uniformly-sampled head direction over [0, 360) with all frames valid."""
    hd = np.linspace(0.0, 360.0, n, endpoint=False)
    mask = np.ones(n, dtype=bool)
    return hd, mask


def test_classify_single_cell_default_rng() -> None:
    """rng=None takes the default-generator branch and returns full dict."""
    hd, mask = _hd_and_mask()
    # Noise signal — not expected to be HD, but must classify cleanly.
    signal = np.random.default_rng(0).normal(size=hd.size)
    result = classify_single_cell(signal, hd, mask, n_shuffles=20)
    for key in (
        "is_hd",
        "mvl",
        "p_value",
        "reliability",
        "mi",
        "preferred_direction",
        "criteria_passed",
    ):
        assert key in result
    assert isinstance(result["criteria_passed"], dict)


def test_classify_population_default_rng() -> None:
    """rng=None default path for the population wrapper."""
    hd, mask = _hd_and_mask()
    rng = np.random.default_rng(1)
    signals = rng.normal(size=(2, hd.size))
    pop = classify_population(signals, hd, mask, n_shuffles=20)
    assert pop["n_hd"] + pop["n_non_hd"] == 2
    assert 0.0 <= pop["fraction_hd"] <= 1.0
    assert len(pop["cells"]) == 2


def test_classify_population_empty() -> None:
    """Zero cells → fraction_hd is 0.0 (guards division by zero)."""
    hd, mask = _hd_and_mask()
    signals = np.zeros((0, hd.size))
    pop = classify_population(signals, hd, mask, n_shuffles=5)
    assert pop["fraction_hd"] == 0.0
    assert pop["n_hd"] == 0


def _cell(is_hd: bool, mvl: float, reliability: float) -> dict:
    return {
        "is_hd": is_hd,
        "mvl": mvl,
        "p_value": 0.01,
        "reliability": reliability,
        "mi": 0.1,
        "preferred_direction": 90.0,
    }


def test_summary_table_all_grades() -> None:
    """Each grade branch A/B/C/D is exercised."""
    pop_result = {
        "cells": [
            _cell(True, 0.5, 0.9),  # A — strong MVL and reliability
            _cell(True, 0.3, 0.6),  # B — moderate MVL (>=0.25)
            _cell(True, 0.18, 0.55),  # C — weak MVL (<0.25)
            _cell(False, 0.05, 0.1),  # D — non-HD
        ]
    }
    rows = classification_summary_table(pop_result)
    grades = [r["grade"] for r in rows]
    assert grades == ["A", "B", "C", "D"]
    # Row bookkeeping.
    assert [r["cell"] for r in rows] == [0, 1, 2, 3]
    assert rows[0]["preferred_direction"] == 90.0


def test_summary_table_hd_but_not_grade_a() -> None:
    """HD cell with high MVL but low reliability drops out of grade A."""
    pop_result = {"cells": [_cell(True, 0.5, 0.5)]}  # reliability < 0.8
    rows = classification_summary_table(pop_result)
    assert rows[0]["grade"] == "B"
