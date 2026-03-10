"""Tests for hm2p.anatomy.injection."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import types
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from hm2p.anatomy.injection import (
    _radius_from_volume,
    extract_injection_sites,
    get_injection_coords_for_animal,
    get_rsp_volume,
    load_brainreg_volumes,
    mirror_to_right_hemisphere,
    update_animals_csv,
)


# ── mirror_to_right_hemisphere ──────────────────────────────────────


def test_mirror_left_to_right() -> None:
    """Left-hemisphere point (z > midline) is mirrored."""
    # midline = 11.4 / 2 = 5.7
    x, y, z = mirror_to_right_hemisphere(3.0, 2.0, 8.0)
    assert x == 3.0
    assert y == 2.0
    # z was 8.0, distance from midline = 2.3, mirrored = 5.7 - 2.3 = 3.4
    assert z == pytest.approx(3.4)


def test_mirror_right_stays() -> None:
    """Right-hemisphere point (z <= midline) is not modified."""
    x, y, z = mirror_to_right_hemisphere(3.0, 2.0, 4.0)
    assert (x, y, z) == (3.0, 2.0, 4.0)


def test_mirror_at_midline() -> None:
    """Point exactly at the midline stays unchanged."""
    midline = 11.4 / 2.0
    x, y, z = mirror_to_right_hemisphere(1.0, 1.0, midline)
    assert z == pytest.approx(midline)


def test_mirror_custom_depth() -> None:
    """Custom atlas_depth_mm shifts the midline correctly."""
    # depth=10 → midline=5, z=7 → mirrored to 5-(7-5)=3
    x, y, z = mirror_to_right_hemisphere(0.0, 0.0, 7.0, atlas_depth_mm=10.0)
    assert z == pytest.approx(3.0)


# ── _radius_from_volume ────────────────────────────────────────────


def test_radius_from_volume() -> None:
    """Sphere radius is correctly computed from volume."""
    volume = (4.0 / 3.0) * math.pi * (2.0**3)  # r=2 mm
    r = _radius_from_volume(volume)
    assert r == pytest.approx(2.0)


def test_radius_from_zero_volume() -> None:
    """Zero volume gives zero radius."""
    assert _radius_from_volume(0.0) == pytest.approx(0.0)


def test_radius_from_negative_volume() -> None:
    """Negative volume raises ValueError."""
    with pytest.raises(ValueError, match=">="):
        _radius_from_volume(-1.0)


# ── extract_injection_sites (CSV mode) ─────────────────────────────


def _make_segmentation_csv(brainreg_dir: Path) -> Path:
    """Create a fake summary.csv in the expected brainreg layout."""
    seg_dir = brainreg_dir / "segmentation" / "atlas_space" / "regions"
    seg_dir.mkdir(parents=True)

    csv = seg_dir / "summary.csv"
    df = pd.DataFrame(
        {
            "region": ["RSPd", "RSPv"],
            "volume_mm3": [0.5, 0.3],
            "axis_0_center_um": [5000.0, 5200.0],
            "axis_1_center_um": [3000.0, 3100.0],
            "axis_2_center_um": [4000.0, 4200.0],
        }
    )
    df.to_csv(csv, index=False)
    return csv


def test_extract_injection_sites_csv(tmp_path: Path) -> None:
    """Parses summary.csv and returns site dicts."""
    brainreg_dir = tmp_path / "animal_123"
    _make_segmentation_csv(brainreg_dir)

    sites = extract_injection_sites(brainreg_dir, use_csv=True)

    assert len(sites) == 2
    first = sites[0]
    assert first["region"] == "RSPd"
    assert first["volume_mm3"] == pytest.approx(0.5)
    # 5000 um = 5.0 mm (AP)
    assert first["center_ap_mm"] == pytest.approx(5.0)
    # 3000 um = 3.0 mm (DV)
    assert first["center_dv_mm"] == pytest.approx(3.0)
    # 4000 um = 4.0 mm (ML) — right hemisphere, no mirror needed
    assert first["center_ml_mm"] == pytest.approx(4.0)
    assert first["radius_mm"] > 0.0


def test_extract_injection_sites_csv_mirrors_left(tmp_path: Path) -> None:
    """Left-hemisphere coordinates are mirrored to the right."""
    brainreg_dir = tmp_path / "animal_456"
    seg_dir = brainreg_dir / "segmentation" / "atlas_space" / "regions"
    seg_dir.mkdir(parents=True)

    # z = 8000 um = 8.0 mm > midline (5.7 mm) → left hemisphere
    df = pd.DataFrame(
        {
            "region": ["RSPd"],
            "volume_mm3": [0.1],
            "axis_0_center_um": [5000.0],
            "axis_1_center_um": [3000.0],
            "axis_2_center_um": [8000.0],
        }
    )
    df.to_csv(seg_dir / "summary.csv", index=False)

    sites = extract_injection_sites(brainreg_dir)
    # 8.0 mm → mirrored: 5.7 - (8.0 - 5.7) = 3.4 mm
    assert sites[0]["center_ml_mm"] == pytest.approx(3.4)


def test_extract_injection_sites_no_segmentation(tmp_path: Path) -> None:
    """Returns empty list when segmentation directory is missing."""
    brainreg_dir = tmp_path / "no_seg"
    brainreg_dir.mkdir()

    sites = extract_injection_sites(brainreg_dir)
    assert sites == []


def test_extract_injection_sites_empty_csv(tmp_path: Path) -> None:
    """Returns empty list when summary.csv exists but is empty."""
    brainreg_dir = tmp_path / "empty"
    seg_dir = brainreg_dir / "segmentation" / "atlas_space" / "regions"
    seg_dir.mkdir(parents=True)

    # Write a CSV with headers only.
    df = pd.DataFrame(
        columns=[
            "region",
            "volume_mm3",
            "axis_0_center_um",
            "axis_1_center_um",
            "axis_2_center_um",
        ]
    )
    df.to_csv(seg_dir / "summary.csv", index=False)

    sites = extract_injection_sites(brainreg_dir)
    assert sites == []


def test_extract_injection_sites_no_csv(tmp_path: Path) -> None:
    """Returns empty list when segmentation dir exists but CSV is missing."""
    brainreg_dir = tmp_path / "no_csv"
    seg_dir = brainreg_dir / "segmentation" / "atlas_space" / "regions"
    seg_dir.mkdir(parents=True)

    sites = extract_injection_sites(brainreg_dir, use_csv=True)
    assert sites == []


# ── get_injection_coords_for_animal ─────────────────────────────────


def test_get_injection_coords_for_animal(tmp_path: Path) -> None:
    """Returns inj_ap/ml/dv dict for a matching animal directory."""
    brainreg_base = tmp_path / "brains_reg"
    brainreg_base.mkdir()

    animal_dir = brainreg_base / "green_1117646"
    animal_dir.mkdir()

    fake_sites = [
        {
            "region": "RSPd",
            "volume_mm3": 0.5,
            "center_ap_mm": 5.0,
            "center_dv_mm": 3.0,
            "center_ml_mm": 4.0,
            "radius_mm": 0.49,
        }
    ]

    with patch(
        "hm2p.anatomy.injection.extract_injection_sites",
        return_value=fake_sites,
    ):
        result = get_injection_coords_for_animal(brainreg_base, "1117646")

    assert result is not None
    assert result["inj_ap"] == pytest.approx(5.0)
    assert result["inj_ml"] == pytest.approx(4.0)
    assert result["inj_dv"] == pytest.approx(3.0)


def test_get_injection_coords_no_match(tmp_path: Path) -> None:
    """Returns None when no directory matches the animal ID."""
    brainreg_base = tmp_path / "brains_reg"
    brainreg_base.mkdir()

    result = get_injection_coords_for_animal(brainreg_base, "9999999")
    assert result is None


def test_get_injection_coords_multiple_dirs(tmp_path: Path) -> None:
    """Raises RuntimeError when multiple directories match."""
    brainreg_base = tmp_path / "brains_reg"
    brainreg_base.mkdir()
    (brainreg_base / "green_123_A").mkdir()
    (brainreg_base / "green_123_B").mkdir()

    with pytest.raises(RuntimeError, match="Multiple"):
        get_injection_coords_for_animal(brainreg_base, "123")


def test_get_injection_coords_no_sites(tmp_path: Path) -> None:
    """Returns None when the directory exists but has no sites."""
    brainreg_base = tmp_path / "brains_reg"
    brainreg_base.mkdir()
    (brainreg_base / "green_555").mkdir()

    with patch(
        "hm2p.anatomy.injection.extract_injection_sites",
        return_value=[],
    ):
        result = get_injection_coords_for_animal(brainreg_base, "555")

    assert result is None


# ── update_animals_csv ──────────────────────────────────────────────


def test_update_animals_csv(tmp_path: Path) -> None:
    """Writes injection coordinates back to the CSV."""
    csv_path = tmp_path / "animals.csv"
    df = pd.DataFrame(
        {
            "animal_id": [111, 222],
            "celltype": ["penk", "nonpenk"],
        }
    )
    df.to_csv(csv_path, index=False)

    brainreg_base = tmp_path / "brains_reg"
    brainreg_base.mkdir()

    coords_111 = {"inj_ap": 5.0, "inj_ml": 4.0, "inj_dv": 3.0}

    with patch(
        "hm2p.anatomy.injection.get_injection_coords_for_animal",
        side_effect=lambda base, aid: coords_111 if aid == "111" else None,
    ):
        result = update_animals_csv(csv_path, brainreg_base)

    assert result.loc[result["animal_id"] == 111, "inj_ap"].iloc[0] == pytest.approx(5.0)
    assert result.loc[result["animal_id"] == 111, "inj_ml"].iloc[0] == pytest.approx(4.0)
    assert result.loc[result["animal_id"] == 111, "inj_dv"].iloc[0] == pytest.approx(3.0)
    # Animal 222 should have NaN.
    assert pd.isna(result.loc[result["animal_id"] == 222, "inj_ap"].iloc[0])

    # Verify CSV was actually saved.
    saved = pd.read_csv(csv_path)
    assert "inj_ap" in saved.columns


def test_update_animals_csv_missing_file(tmp_path: Path) -> None:
    """Raises FileNotFoundError when CSV does not exist."""
    with pytest.raises(FileNotFoundError):
        update_animals_csv(
            tmp_path / "nonexistent.csv", tmp_path / "brains"
        )


# ── load_brainreg_volumes ────────────────────────────────────────


def test_load_brainreg_volumes_normal(tmp_path: Path) -> None:
    """Loads a well-formed volumes.csv and returns a DataFrame."""
    df = pd.DataFrame(
        {
            "structure_name": ["Retrosplenial area dorsal", "Primary motor"],
            "left_volume_mm3": [1.2, 0.8],
            "right_volume_mm3": [1.1, 0.9],
            "total_volume_mm3": [2.3, 1.7],
        }
    )
    df.to_csv(tmp_path / "volumes.csv", index=False)

    result = load_brainreg_volumes(tmp_path)
    assert result is not None
    assert len(result) == 2
    assert list(result.columns) == [
        "structure_name",
        "left_volume_mm3",
        "right_volume_mm3",
        "total_volume_mm3",
    ]


def test_load_brainreg_volumes_missing_file(tmp_path: Path) -> None:
    """Returns None when volumes.csv does not exist."""
    result = load_brainreg_volumes(tmp_path)
    assert result is None


def test_load_brainreg_volumes_empty_csv(tmp_path: Path) -> None:
    """Returns None when volumes.csv has headers but no data rows."""
    df = pd.DataFrame(
        columns=[
            "structure_name",
            "left_volume_mm3",
            "right_volume_mm3",
            "total_volume_mm3",
        ]
    )
    df.to_csv(tmp_path / "volumes.csv", index=False)

    result = load_brainreg_volumes(tmp_path)
    assert result is None


# ── get_rsp_volume ───────────────────────────────────────────────


def test_get_rsp_volume_matching_regions() -> None:
    """Sums volumes for rows containing 'Retrosplenial' or 'RSP'."""
    df = pd.DataFrame(
        {
            "structure_name": [
                "Retrosplenial area dorsal",
                "Retrosplenial area ventral",
                "Primary motor",
            ],
            "left_volume_mm3": [1.0, 0.5, 2.0],
            "right_volume_mm3": [1.1, 0.6, 1.9],
            "total_volume_mm3": [2.1, 1.1, 3.9],
        }
    )
    result = get_rsp_volume(df)
    assert result["rsp_total_mm3"] == pytest.approx(3.2)
    assert result["rsp_left_mm3"] == pytest.approx(1.5)
    assert result["rsp_right_mm3"] == pytest.approx(1.7)
    assert len(result["rsp_regions"]) == 2


def test_get_rsp_volume_no_matching_regions() -> None:
    """Returns zeros when no RSP regions are found."""
    df = pd.DataFrame(
        {
            "structure_name": ["Primary motor", "Visual area"],
            "left_volume_mm3": [2.0, 1.0],
            "right_volume_mm3": [1.9, 1.1],
            "total_volume_mm3": [3.9, 2.1],
        }
    )
    result = get_rsp_volume(df)
    assert result["rsp_total_mm3"] == pytest.approx(0.0)
    assert result["rsp_left_mm3"] == pytest.approx(0.0)
    assert result["rsp_right_mm3"] == pytest.approx(0.0)
    assert result["rsp_regions"] == []


def test_get_rsp_volume_case_insensitive() -> None:
    """Matches RSP region names case-insensitively."""
    df = pd.DataFrame(
        {
            "structure_name": ["RETROSPLENIAL area dorsal", "rsp ventral part"],
            "left_volume_mm3": [1.0, 0.5],
            "right_volume_mm3": [1.1, 0.6],
            "total_volume_mm3": [2.1, 1.1],
        }
    )
    result = get_rsp_volume(df)
    assert len(result["rsp_regions"]) == 2
    assert result["rsp_total_mm3"] == pytest.approx(3.2)


# ── extract_injection_sites (TIFF mode) ─────────────────────────


def test_extract_injection_sites_tiff_mode(tmp_path: Path) -> None:
    """TIFF mode computes centroid and volume from non-zero voxels."""
    brainreg_dir = tmp_path / "animal_tiff"
    seg_dir = brainreg_dir / "segmentation" / "atlas_space" / "regions"
    seg_dir.mkdir(parents=True)

    # Create the expected TIFF file path (content loaded via mock).
    tiff_path = seg_dir / "region_0.tiff"
    tiff_path.touch()

    # Synthetic 3D array: 2 non-zero voxels at (10, 20, 30) and (12, 22, 32).
    data = np.zeros((50, 50, 50), dtype=np.uint8)
    data[10, 20, 30] = 1
    data[12, 22, 32] = 1

    fake_tifffile = types.ModuleType("tifffile")
    fake_tifffile.imread = lambda path: data  # type: ignore[attr-defined]

    with patch.dict("sys.modules", {"tifffile": fake_tifffile}):
        sites = extract_injection_sites(brainreg_dir, use_csv=False)

    assert len(sites) == 1
    site = sites[0]
    assert site["region"] == "region_0"

    # Centroid: mean of (10,12)=11, mean of (20,22)=21, mean of (30,32)=31
    # In mm: 11*25/1000=0.275, 21*25/1000=0.525, 31*25/1000=0.775
    assert site["center_ap_mm"] == pytest.approx(0.275)
    assert site["center_dv_mm"] == pytest.approx(0.525)
    # 0.775 < 5.7 midline → no mirror
    assert site["center_ml_mm"] == pytest.approx(0.775)

    # Volume: 2 voxels * (0.025 mm)^3 = 2 * 1.5625e-5
    expected_vol = 2.0 * (25.0 / 1000.0) ** 3
    assert site["volume_mm3"] == pytest.approx(expected_vol)
    assert site["radius_mm"] > 0.0


def test_extract_injection_sites_tiff_missing(tmp_path: Path) -> None:
    """Returns empty list when region_0.tiff does not exist."""
    brainreg_dir = tmp_path / "animal_no_tiff"
    seg_dir = brainreg_dir / "segmentation" / "atlas_space" / "regions"
    seg_dir.mkdir(parents=True)
    # No TIFF file created.

    sites = extract_injection_sites(brainreg_dir, use_csv=False)
    assert sites == []


def test_extract_injection_sites_tiff_all_zeros(tmp_path: Path) -> None:
    """Returns empty list when TIFF contains only zeros."""
    brainreg_dir = tmp_path / "animal_zeros"
    seg_dir = brainreg_dir / "segmentation" / "atlas_space" / "regions"
    seg_dir.mkdir(parents=True)

    tiff_path = seg_dir / "region_0.tiff"
    tiff_path.touch()

    data = np.zeros((20, 20, 20), dtype=np.uint8)

    fake_tifffile = types.ModuleType("tifffile")
    fake_tifffile.imread = lambda path: data  # type: ignore[attr-defined]

    with patch.dict("sys.modules", {"tifffile": fake_tifffile}):
        sites = extract_injection_sites(brainreg_dir, use_csv=False)

    assert sites == []
