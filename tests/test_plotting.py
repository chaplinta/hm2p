"""Tests for hm2p.plotting — standardized comparison plots."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

from hm2p.plotting import (
    celltype_comparison_box,
    format_pvalue,
    paired_condition_scatter,
)


# ---------------------------------------------------------------------------
# format_pvalue
# ---------------------------------------------------------------------------


class TestFormatPvalue:
    def test_small_pvalue(self):
        assert format_pvalue(0.0001) == "p < 0.001"

    def test_normal_pvalue(self):
        assert format_pvalue(0.042) == "p = 0.042"

    def test_exact_threshold(self):
        assert format_pvalue(0.001) == "p = 0.001"

    def test_nan_pvalue(self):
        assert format_pvalue(float("nan")) == "p = NaN"

    def test_one(self):
        assert format_pvalue(1.0) == "p = 1.000"

    def test_zero(self):
        assert format_pvalue(0.0) == "p < 0.001"


# ---------------------------------------------------------------------------
# celltype_comparison_box
# ---------------------------------------------------------------------------


class TestCelltypeComparisonBox:
    def test_returns_figure_and_dict(self):
        penk = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        nonpenk = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
        fig, result = celltype_comparison_box(penk, nonpenk, "MVL")
        assert isinstance(fig, go.Figure)
        assert isinstance(result, dict)

    def test_stat_dict_keys(self):
        penk = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        nonpenk = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
        _, result = celltype_comparison_box(penk, nonpenk, "MVL")
        assert set(result.keys()) == {
            "test", "U", "p", "n_penk", "n_nonpenk", "measure"
        }
        assert result["test"] == "Mann-Whitney U"
        assert result["n_penk"] == 5
        assert result["n_nonpenk"] == 5
        assert result["measure"] == "MVL"

    def test_pvalue_annotation_present(self):
        penk = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        nonpenk = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
        fig, result = celltype_comparison_box(penk, nonpenk, "MVL")
        # Check that annotations contain p-value text
        annotations = fig.layout.annotations
        assert len(annotations) >= 1
        p_texts = [a.text for a in annotations]
        assert any("p" in t for t in p_texts)

    def test_two_box_traces(self):
        penk = np.array([1.0, 2.0, 3.0])
        nonpenk = np.array([4.0, 5.0, 6.0])
        fig, _ = celltype_comparison_box(penk, nonpenk, "Test")
        box_traces = [t for t in fig.data if isinstance(t, go.Box)]
        assert len(box_traces) == 2

    def test_empty_penk(self):
        """Should handle empty Penk array without raising."""
        penk = np.array([])
        nonpenk = np.array([1.0, 2.0, 3.0])
        fig, result = celltype_comparison_box(penk, nonpenk, "Test")
        assert isinstance(fig, go.Figure)
        assert np.isnan(result["U"])
        assert np.isnan(result["p"])
        assert result["n_penk"] == 0

    def test_empty_both(self):
        """Should handle both arrays empty."""
        fig, result = celltype_comparison_box(np.array([]), np.array([]), "X")
        assert isinstance(fig, go.Figure)
        assert np.isnan(result["p"])

    def test_identical_values(self):
        """Identical distributions — should not raise."""
        vals = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        fig, result = celltype_comparison_box(vals, vals.copy(), "Const")
        assert isinstance(fig, go.Figure)
        assert result["p"] == 1.0  # identical distributions

    def test_single_value_each(self):
        """Single value per group — Mann-Whitney still runs."""
        fig, result = celltype_comparison_box(
            np.array([1.0]), np.array([2.0]), "Single"
        )
        assert isinstance(fig, go.Figure)
        assert result["n_penk"] == 1
        assert result["n_nonpenk"] == 1

    def test_nans_filtered(self):
        """NaN values should be stripped before testing."""
        penk = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        nonpenk = np.array([6.0, 7.0, np.nan, 9.0, 10.0])
        _, result = celltype_comparison_box(penk, nonpenk, "WithNaN")
        assert result["n_penk"] == 4
        assert result["n_nonpenk"] == 4

    def test_custom_labels(self):
        penk = np.array([1.0, 2.0])
        nonpenk = np.array([3.0, 4.0])
        fig, _ = celltype_comparison_box(
            penk, nonpenk, "X",
            penk_label="Group A", nonpenk_label="Group B",
        )
        names = [t.name for t in fig.data if isinstance(t, go.Box)]
        assert "Group A" in names
        assert "Group B" in names


# ---------------------------------------------------------------------------
# paired_condition_scatter
# ---------------------------------------------------------------------------


class TestPairedConditionScatter:
    def test_returns_figure_and_dict(self):
        c1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        c2 = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5])
        fig, result = paired_condition_scatter(c1, c2, "Light", "Dark", "PD")
        assert isinstance(fig, go.Figure)
        assert isinstance(result, dict)

    def test_stat_dict_keys(self):
        c1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        c2 = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
        _, result = paired_condition_scatter(c1, c2, "Light", "Dark", "PD")
        assert set(result.keys()) == {"test", "W", "p", "n", "measure"}
        assert result["test"] == "Wilcoxon signed-rank"
        assert result["n"] == 10
        assert result["measure"] == "PD"

    def test_pvalue_annotation_present(self):
        c1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        c2 = np.array([2.0, 3.0, 1.0, 5.0, 4.0, 7.0, 6.0, 9.0, 8.0, 11.0])
        fig, _ = paired_condition_scatter(c1, c2, "Light", "Dark", "PD")
        annotations = fig.layout.annotations
        assert len(annotations) >= 1
        assert any("p" in a.text for a in annotations)

    def test_unity_line_present(self):
        c1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        c2 = c1 + 0.5
        fig, _ = paired_condition_scatter(c1, c2, "A", "B", "X")
        # Unity line is the second trace (mode="lines", dash)
        line_traces = [t for t in fig.data if t.mode == "lines"]
        assert len(line_traces) >= 1
        # Check it is diagonal: x and y should be equal
        lt = line_traces[0]
        np.testing.assert_array_equal(lt.x, lt.y)

    def test_square_aspect(self):
        c1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        c2 = c1 * 1.1
        fig, _ = paired_condition_scatter(c1, c2, "A", "B", "X")
        assert fig.layout.width == 500
        assert fig.layout.height == 500
        assert fig.layout.yaxis.scaleanchor == "x"
        assert fig.layout.yaxis.scaleratio == 1

    def test_axis_labels(self):
        c1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        c2 = c1 + 1
        fig, _ = paired_condition_scatter(c1, c2, "Light", "Dark", "MVL")
        assert fig.layout.xaxis.title.text == "MVL (Light)"
        assert fig.layout.yaxis.title.text == "MVL (Dark)"

    def test_empty_arrays(self):
        fig, result = paired_condition_scatter(
            np.array([]), np.array([]), "A", "B", "X"
        )
        assert isinstance(fig, go.Figure)
        assert np.isnan(result["W"])
        assert np.isnan(result["p"])
        assert result["n"] == 0

    def test_identical_values(self):
        """All differences zero — Wilcoxon returns W=0, p=NaN."""
        vals = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
        fig, result = paired_condition_scatter(vals, vals.copy(), "A", "B", "X")
        assert isinstance(fig, go.Figure)
        # All-identical values: no differences to rank, returns NaN
        assert np.isnan(result["W"])
        assert np.isnan(result["p"])

    def test_small_sample(self):
        """Fewer than 10 paired values — should still work."""
        c1 = np.array([1.0, 2.0, 3.0])
        c2 = np.array([4.0, 5.0, 6.0])
        fig, result = paired_condition_scatter(c1, c2, "A", "B", "X")
        assert isinstance(fig, go.Figure)
        assert result["n"] == 3
        assert not np.isnan(result["p"])

    def test_nans_filtered_pairwise(self):
        """NaN in either array removes the pair."""
        c1 = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        c2 = np.array([2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
        _, result = paired_condition_scatter(c1, c2, "A", "B", "X")
        assert result["n"] == 8  # two pairs removed

    def test_single_value(self):
        """Single paired value — Wilcoxon cannot run."""
        fig, result = paired_condition_scatter(
            np.array([1.0]), np.array([2.0]), "A", "B", "X"
        )
        assert isinstance(fig, go.Figure)
        assert result["n"] == 1
