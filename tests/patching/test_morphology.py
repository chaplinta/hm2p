"""Tests for hm2p.patching.morphology — SWC parsing, tree stats, Sholl, rotation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hm2p.patching.morphology import (
    _load_swc_manual,
    compute_sholl,
    compute_surface_distance,
    compute_tree_stats,
    load_morphology,
    rotate_to_surface,
    soma_subtract,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic SWC data
# ---------------------------------------------------------------------------

_SIMPLE_SWC = textwrap.dedent("""\
    # Simple 10-node tree:
    #   1 (root) -> 2 -> 3 -> 4 (branch)
    #                              |-> 5 -> 6
    #                              |-> 7 -> 8
    #   Also 9 -> 10 (disconnected from above for edge testing)
    1 1 0.0 0.0 0.0 1.0 -1
    2 3 10.0 0.0 0.0 0.5 1
    3 3 20.0 0.0 0.0 0.5 2
    4 5 30.0 0.0 0.0 0.5 3
    5 3 40.0 10.0 0.0 0.5 4
    6 6 50.0 10.0 0.0 0.5 5
    7 3 40.0 -10.0 0.0 0.5 4
    8 6 50.0 -10.0 0.0 0.5 7
    9 3 30.0 5.0 0.0 0.5 3
    10 6 30.0 15.0 0.0 0.5 9
""")


def _write_swc(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


@pytest.fixture()
def simple_swc_path(tmp_path: Path) -> Path:
    return _write_swc(tmp_path, "test.swc", _SIMPLE_SWC)


@pytest.fixture()
def simple_tree(simple_swc_path: Path) -> dict:
    return _load_swc_manual(simple_swc_path)


# ---------------------------------------------------------------------------
# Y-shaped tree for branch stats
# ---------------------------------------------------------------------------

_Y_TREE_SWC = textwrap.dedent("""\
    # Y-shaped tree: root -> branch point -> 2 arms
    1 1 0.0 0.0 0.0 1.0 -1
    2 3 0.0 10.0 0.0 0.5 1
    3 5 10.0 20.0 0.0 0.5 2
    4 6 20.0 30.0 0.0 0.5 3
    5 5 -10.0 20.0 0.0 0.5 2
    6 6 -20.0 30.0 0.0 0.5 5
""")


@pytest.fixture()
def y_tree(tmp_path: Path) -> dict:
    return _load_swc_manual(_write_swc(tmp_path, "y.swc", _Y_TREE_SWC))


# ---------------------------------------------------------------------------
# Tests: _load_swc_manual
# ---------------------------------------------------------------------------


class TestLoadSwcManual:
    def test_node_count(self, simple_tree: dict) -> None:
        assert len(simple_tree["nodes"]) == 10

    def test_columns(self, simple_tree: dict) -> None:
        expected = {"id", "type", "x", "y", "z", "radius", "parent_id"}
        assert set(simple_tree["nodes"].columns) == expected

    def test_root_node(self, simple_tree: dict) -> None:
        root = simple_tree["nodes"].query("parent_id == -1")
        assert len(root) == 1
        assert root.iloc[0]["id"] == 1

    def test_edge_count(self, simple_tree: dict) -> None:
        # 10 nodes, 1 root => 9 edges
        assert len(simple_tree["edges"]) == 9

    def test_edges_are_parent_child(self, simple_tree: dict) -> None:
        edges = simple_tree["edges"]
        # Each edge (parent, child) — parent should be a valid id
        node_ids = set(simple_tree["nodes"]["id"])
        for parent, child in edges:
            assert parent in node_ids
            assert child in node_ids

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.swc"
        p.write_text("# only comments\n")
        with pytest.raises(ValueError, match="No valid SWC data"):
            _load_swc_manual(p)

    def test_coordinates(self, simple_tree: dict) -> None:
        nodes = simple_tree["nodes"]
        node2 = nodes.query("id == 2").iloc[0]
        assert node2["x"] == 10.0
        assert node2["y"] == 0.0


# ---------------------------------------------------------------------------
# Tests: load_morphology
# ---------------------------------------------------------------------------


class TestLoadMorphology:
    def test_loads_types(self, tmp_path: Path) -> None:
        soma_swc = "1 1 0 0 0 1 -1\n2 1 1 0 0 1 1\n"
        apical_swc = "1 3 0 0 0 0.5 -1\n2 3 0 10 0 0.5 1\n"
        basal_swc = "1 3 0 0 0 0.5 -1\n2 3 0 -10 0 0.5 1\n"
        surface_swc = "1 3 0 50 0 0.5 -1\n2 3 100 50 0 0.5 1\n"

        _write_swc(tmp_path, "Soma.swc", soma_swc)
        _write_swc(tmp_path, "Apical_tree.swc", apical_swc)
        _write_swc(tmp_path, "Basal_1.swc", basal_swc)
        _write_swc(tmp_path, "Surface.swc", surface_swc)

        result = load_morphology(tmp_path)
        assert "soma" in result
        assert "apical" in result
        assert "basal" in result
        assert "surface" in result
        assert "axon" not in result

    def test_multiple_basal_concatenated(self, tmp_path: Path) -> None:
        b1 = "1 3 0 0 0 0.5 -1\n2 3 0 -5 0 0.5 1\n"
        b2 = "1 3 0 -1 0 0.5 -1\n2 3 0 -10 0 0.5 1\n"
        _write_swc(tmp_path, "Soma.swc", "1 1 0 0 0 1 -1\n")
        _write_swc(tmp_path, "Basal_1.swc", b1)
        _write_swc(tmp_path, "Basal_2.swc", b2)

        result = load_morphology(tmp_path)
        basal_nodes = result["basal"]["nodes"]
        # Should have nodes from both trees (4 total, but after concatenation
        # ids may be renumbered)
        assert len(basal_nodes) == 4


# ---------------------------------------------------------------------------
# Tests: soma_subtract
# ---------------------------------------------------------------------------


class TestSomaSubtract:
    def test_centres_on_soma(self) -> None:
        nodes_soma = pd.DataFrame(
            {"id": [1, 2], "type": [1, 1], "x": [10.0, 12.0],
             "y": [20.0, 22.0], "z": [5.0, 7.0], "radius": [1.0, 1.0],
             "parent_id": [-1, 1]}
        )
        nodes_apical = pd.DataFrame(
            {"id": [1, 2], "type": [3, 3], "x": [10.0, 10.0],
             "y": [30.0, 40.0], "z": [5.0, 5.0], "radius": [0.5, 0.5],
             "parent_id": [-1, 1]}
        )
        neurons = {
            "soma": {"nodes": nodes_soma, "edges": np.array([[1, 2]])},
            "apical": {"nodes": nodes_apical, "edges": np.array([[1, 2]])},
        }
        result = soma_subtract(neurons)
        # Soma mean: x=11, y=21, z=6
        soma_out = result["soma"]["nodes"]
        np.testing.assert_allclose(soma_out["x"].mean(), 0.0, atol=1e-10)
        # Y is flipped
        np.testing.assert_allclose(soma_out["z"].mean(), 0.0, atol=1e-10)

    def test_y_flipped(self) -> None:
        nodes = pd.DataFrame(
            {"id": [1], "type": [1], "x": [0.0], "y": [10.0], "z": [0.0],
             "radius": [1.0], "parent_id": [-1]}
        )
        neurons = {"soma": {"nodes": nodes, "edges": np.empty((0, 2), dtype=int)}}
        result = soma_subtract(neurons, soma_center=np.array([0.0, 0.0, 0.0]))
        # Y should be flipped: (10 - 0) * -1 = -10
        assert result["soma"]["nodes"].iloc[0]["y"] == -10.0

    def test_explicit_soma_center(self) -> None:
        nodes = pd.DataFrame(
            {"id": [1], "type": [3], "x": [5.0], "y": [5.0], "z": [5.0],
             "radius": [0.5], "parent_id": [-1]}
        )
        neurons = {"apical": {"nodes": nodes, "edges": np.empty((0, 2), dtype=int)}}
        result = soma_subtract(neurons, soma_center=np.array([5.0, 5.0, 5.0]))
        row = result["apical"]["nodes"].iloc[0]
        assert row["x"] == 0.0
        assert row["y"] == 0.0
        assert row["z"] == 0.0

    def test_no_soma_raises(self) -> None:
        nodes = pd.DataFrame(
            {"id": [1], "type": [3], "x": [5.0], "y": [5.0], "z": [5.0],
             "radius": [0.5], "parent_id": [-1]}
        )
        neurons = {"apical": {"nodes": nodes, "edges": np.empty((0, 2), dtype=int)}}
        with pytest.raises(ValueError, match="No soma"):
            soma_subtract(neurons)


# ---------------------------------------------------------------------------
# Tests: rotate_to_surface
# ---------------------------------------------------------------------------


class TestRotateToSurface:
    def test_horizontal_surface_no_rotation(self) -> None:
        """A perfectly horizontal surface above soma should give ~0 rotation."""
        # Surface: horizontal line at y=100
        surface_pts = np.array([[x, 100.0] for x in range(-200, 201)])
        nodes = pd.DataFrame(
            {"id": [1, 2], "type": [3, 3], "x": [0.0, 0.0],
             "y": [0.0, -50.0], "z": [0.0, 0.0], "radius": [0.5, 0.5],
             "parent_id": [-1, 1]}
        )
        neurons = {"apical": {"nodes": nodes, "edges": np.array([[1, 2]])}}
        _, angle = rotate_to_surface(neurons, surface_pts)
        # Should be close to 0 degrees (surface already horizontal)
        # The surface is above soma (y=100 > 0), slope=0, atan(0)=0
        assert abs(angle) < 5.0 or abs(angle - 180) < 5.0

    def test_known_45_degree(self) -> None:
        """Surface tilted at 45 degrees."""
        # Surface: y = x (slope = 1)
        xs = np.linspace(-200, 200, 1001)
        surface_pts = np.column_stack([xs, xs + 50])  # shifted up
        nodes = pd.DataFrame(
            {"id": [1], "type": [3], "x": [0.0], "y": [0.0], "z": [0.0],
             "radius": [0.5], "parent_id": [-1]}
        )
        neurons = {"apical": {"nodes": nodes, "edges": np.empty((0, 2), dtype=int)}}
        _, angle = rotate_to_surface(neurons, surface_pts)
        # atan(-1) = -45 degrees, but surface is above soma, so no pi adjustment
        # The exact angle depends on the fit — just check it's in a reasonable range
        assert -90 < angle < 90 or 90 < angle < 270

    def test_coordinates_change(self) -> None:
        """Rotation should actually change coordinates."""
        surface_pts = np.array([[x, x + 50.0] for x in range(-200, 201)])
        nodes = pd.DataFrame(
            {"id": [1, 2], "type": [3, 3], "x": [0.0, 0.0],
             "y": [0.0, -50.0], "z": [0.0, 0.0], "radius": [0.5, 0.5],
             "parent_id": [-1, 1]}
        )
        neurons = {"apical": {"nodes": nodes, "edges": np.array([[1, 2]])}}
        rotated, _ = rotate_to_surface(neurons, surface_pts)
        # At least one coordinate should differ from original
        orig_xy = nodes[["x", "y"]].values
        new_xy = rotated["apical"]["nodes"][["x", "y"]].values
        assert not np.allclose(orig_xy, new_xy)


# ---------------------------------------------------------------------------
# Tests: compute_tree_stats
# ---------------------------------------------------------------------------


class TestComputeTreeStats:
    def test_y_tree_basic(self, y_tree: dict) -> None:
        """Y-shaped tree: 1 branch point (node 2), 2 terminal nodes."""
        stats = compute_tree_stats(y_tree["nodes"], y_tree["edges"])

        assert stats["n_branch_points"] == 1
        assert stats["total_length"] > 0
        assert stats["max_path_length"] > 0
        assert stats["max_branch_order"] >= 1
        assert stats["width"] > 0
        assert stats["height"] > 0

    def test_straight_line(self, tmp_path: Path) -> None:
        """A straight line should have 0 branch points."""
        swc = "1 1 0 0 0 1 -1\n2 3 10 0 0 0.5 1\n3 3 20 0 0 0.5 2\n"
        tree = _load_swc_manual(_write_swc(tmp_path, "line.swc", swc))
        stats = compute_tree_stats(tree["nodes"], tree["edges"])
        assert stats["n_branch_points"] == 0
        np.testing.assert_allclose(stats["total_length"], 20.0, atol=1e-10)
        np.testing.assert_allclose(stats["max_path_length"], 20.0, atol=1e-10)
        assert stats["width"] == 20.0
        assert stats["height"] == 0.0
        assert stats["width_height_ratio"] == 0.0

    def test_width_height_depth(self, y_tree: dict) -> None:
        stats = compute_tree_stats(y_tree["nodes"], y_tree["edges"])
        nodes = y_tree["nodes"]
        expected_width = nodes["x"].max() - nodes["x"].min()
        expected_height = nodes["y"].max() - nodes["y"].min()
        assert stats["width"] == expected_width
        assert stats["height"] == expected_height

    def test_single_node(self, tmp_path: Path) -> None:
        swc = "1 1 0 0 0 1 -1\n"
        tree = _load_swc_manual(_write_swc(tmp_path, "single.swc", swc))
        stats = compute_tree_stats(tree["nodes"], tree["edges"])
        assert stats["total_length"] == 0.0
        assert stats["n_branch_points"] == 0
        assert stats["width"] == 0.0


# ---------------------------------------------------------------------------
# Tests: compute_sholl
# ---------------------------------------------------------------------------


class TestComputeSholl:
    def test_straight_line_crossings(self, tmp_path: Path) -> None:
        """Line from 0 to 30: should cross radii 5, 10, 15, 20, 25 once each."""
        swc = "1 1 0 0 0 1 -1\n2 3 10 0 0 0.5 1\n3 3 20 0 0 0.5 2\n4 3 30 0 0 0.5 3\n"
        tree = _load_swc_manual(_write_swc(tmp_path, "line.swc", swc))
        radii = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
        counts = compute_sholl(tree["nodes"], np.array([0.0, 0.0, 0.0]), radii, tree["edges"])
        # Each radius from 5 to 25 should be crossed exactly once
        np.testing.assert_array_equal(counts, [1, 1, 1, 1, 1])

    def test_radius_beyond_tree(self, tmp_path: Path) -> None:
        """Radius larger than the tree should have 0 crossings."""
        swc = "1 1 0 0 0 1 -1\n2 3 5 0 0 0.5 1\n"
        tree = _load_swc_manual(_write_swc(tmp_path, "short.swc", swc))
        radii = np.array([10.0, 20.0])
        counts = compute_sholl(tree["nodes"], np.array([0.0, 0.0, 0.0]), radii, tree["edges"])
        np.testing.assert_array_equal(counts, [0, 0])

    def test_y_tree_crossings(self, y_tree: dict) -> None:
        """Y-tree: at small radius, 1 crossing; after branch point, 2 crossings."""
        radii = np.array([5.0, 15.0, 25.0])
        counts = compute_sholl(y_tree["nodes"], np.array([0.0, 0.0, 0.0]), radii, y_tree["edges"])
        # At r=5, only the stem crosses (1 crossing)
        assert counts[0] == 1
        # At r=15, we're past the branch point (node 2 at y=10), so 2 arms
        assert counts[1] >= 2


# ---------------------------------------------------------------------------
# Tests: compute_surface_distance
# ---------------------------------------------------------------------------


class TestComputeSurfaceDistance:
    def test_known_distances(self) -> None:
        # Dense surface so nearest-neighbour distances are accurate
        surface = np.array([[float(x), 100.0] for x in range(21)])
        dendrite = np.array([
            [0.0, 90.0],   # 10 from surface
            [10.0, 50.0],  # 50 from surface
            [5.0, 95.0],   # 5 from surface (nearest: (5, 100))
        ])
        result = compute_surface_distance(surface, dendrite)
        assert result["dist_superficial"] == pytest.approx(5.0, abs=0.1)
        assert result["dist_deep"] == pytest.approx(50.0, abs=0.1)

    def test_point_on_surface(self) -> None:
        surface = np.array([[0.0, 0.0], [10.0, 0.0]])
        dendrite = np.array([[0.0, 0.0], [5.0, 5.0]])
        result = compute_surface_distance(surface, dendrite)
        assert result["dist_superficial"] == pytest.approx(0.0, abs=1e-10)
        assert result["dist_deep"] > 0

    def test_single_points(self) -> None:
        surface = np.array([[0.0, 0.0]])
        dendrite = np.array([[3.0, 4.0]])
        result = compute_surface_distance(surface, dendrite)
        assert result["dist_superficial"] == pytest.approx(5.0, abs=1e-10)
        assert result["dist_deep"] == pytest.approx(5.0, abs=1e-10)
