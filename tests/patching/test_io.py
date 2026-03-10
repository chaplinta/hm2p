"""Tests for patching.io — WaveSurfer H5 loading and SWC file finding."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hm2p.patching.io import (
    _apply_scaling,
    get_sweep_traces,
    load_swc_files,
    load_wavesurfer,
)


# ---------------------------------------------------------------------------
# Helpers: create minimal WaveSurfer-like HDF5 files
# ---------------------------------------------------------------------------


def _create_ws_h5(
    path: Path,
    *,
    n_scans: int = 100,
    n_channels: int = 1,
    n_sweeps: int = 1,
    raw_value: int = 1000,
    channel_scale: float = 1.0,
    scaling_coeffs: np.ndarray | None = None,
) -> None:
    """Write a minimal WaveSurfer-like HDF5 file for testing.

    The scaling coefficients default to a simple identity-like polynomial:
    ``coeff = [0, 1, 0, 0]`` so that ``scaled = raw / channel_scale``.
    """
    if scaling_coeffs is None:
        # 4 coefficients (cubic polynomial), identity: raw -> raw
        # Horner: c[3]*x + c[2] then *x + c[1] then *x + c[0]
        # For identity (output = raw): c = [0, 1, 0, 0]
        scaling_coeffs = np.zeros((4, n_channels), dtype=np.float64)
        scaling_coeffs[1, :] = 1.0  # linear coefficient = 1

    with h5py.File(path, "w") as f:
        hdr = f.create_group("header")
        hdr.create_dataset(
            "AIChannelScales",
            data=np.full(n_channels, channel_scale, dtype=np.float64),
        )
        hdr.create_dataset("AIScalingCoefficients", data=scaling_coeffs)
        hdr.create_dataset(
            "IsAIChannelActive", data=np.ones(n_channels, dtype=np.uint8)
        )
        hdr.create_dataset(
            "AcquisitionSampleRate", data=np.float64(20000.0)
        )

        for sw in range(1, n_sweeps + 1):
            sweep = f.create_group(f"sweep_{sw:04d}")
            sweep.create_dataset(
                "analogScans",
                data=np.full((n_scans, n_channels), raw_value, dtype=np.int16),
            )


# ---------------------------------------------------------------------------
# _apply_scaling
# ---------------------------------------------------------------------------


class TestApplyScaling:
    """Test the polynomial ADC-to-voltage scaling."""

    def test_identity_scaling(self) -> None:
        """With identity coefficients [0, 1, 0, 0] and scale=1, output == input."""
        raw = np.array([[100], [200], [300]], dtype=np.int16)
        scales = np.array([1.0])
        coeffs = np.array([[0.0], [1.0], [0.0], [0.0]])
        result = _apply_scaling(raw, scales, coeffs)
        np.testing.assert_allclose(result, raw.astype(np.float64))

    def test_channel_scale_divides(self) -> None:
        """Channel scale acts as a divisor."""
        raw = np.array([[100]], dtype=np.int16)
        scales = np.array([10.0])
        coeffs = np.array([[0.0], [1.0], [0.0], [0.0]])
        result = _apply_scaling(raw, scales, coeffs)
        np.testing.assert_allclose(result, [[10.0]])

    def test_quadratic_polynomial(self) -> None:
        """Coefficients [2, 3, 0.5, 0]: scaled = (2 + 3*x + 0.5*x^2) / scale."""
        raw = np.array([[10]], dtype=np.int16)
        scales = np.array([1.0])
        coeffs = np.array([[2.0], [3.0], [0.5], [0.0]])
        # Horner: start with c[3]=0, then 0*10+c[2]=0.5, then 0.5*10+c[1]=8.0,
        # then 8.0*10+c[0]=82.0
        result = _apply_scaling(raw, scales, coeffs)
        np.testing.assert_allclose(result, [[82.0]])

    @given(
        raw_val=st.integers(min_value=-32768, max_value=32767),
        scale=st.floats(min_value=0.01, max_value=1e6),
    )
    @settings(max_examples=50)
    def test_scaling_finite(self, raw_val: int, scale: float) -> None:
        """Scaling always produces finite results for valid inputs."""
        raw = np.array([[raw_val]], dtype=np.int16)
        scales = np.array([scale])
        coeffs = np.array([[0.0], [1.0], [0.0], [0.0]])
        result = _apply_scaling(raw, scales, coeffs)
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# load_wavesurfer
# ---------------------------------------------------------------------------


class TestLoadWavesurfer:
    """Test WaveSurfer H5 file loading."""

    def test_loads_header(self, tmp_path: Path) -> None:
        h5_path = tmp_path / "test.h5"
        _create_ws_h5(h5_path)
        data = load_wavesurfer(h5_path)
        assert "header" in data
        assert "AcquisitionSampleRate" in data["header"]

    def test_sweep_present(self, tmp_path: Path) -> None:
        h5_path = tmp_path / "test.h5"
        _create_ws_h5(h5_path, n_sweeps=3)
        data = load_wavesurfer(h5_path)
        assert "sweep_0001" in data
        assert "sweep_0002" in data
        assert "sweep_0003" in data

    def test_scaling_applied(self, tmp_path: Path) -> None:
        """Scaled data should be float64, not int16."""
        h5_path = tmp_path / "test.h5"
        _create_ws_h5(h5_path, raw_value=500, channel_scale=2.0)
        data = load_wavesurfer(h5_path)
        analog = data["sweep_0001"]["analogScans"]
        assert analog.dtype == np.float64
        # With identity poly and scale=2: 500 / 2 = 250
        np.testing.assert_allclose(analog[:, 0], 250.0)

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_wavesurfer(tmp_path / "missing.h5")

    def test_multichannel(self, tmp_path: Path) -> None:
        h5_path = tmp_path / "test.h5"
        _create_ws_h5(h5_path, n_channels=2, raw_value=100)
        data = load_wavesurfer(h5_path)
        analog = data["sweep_0001"]["analogScans"]
        assert analog.shape[1] == 2


# ---------------------------------------------------------------------------
# get_sweep_traces
# ---------------------------------------------------------------------------


class TestGetSweepTraces:
    """Test sweep trace extraction."""

    def test_returns_1d(self, tmp_path: Path) -> None:
        h5_path = tmp_path / "test.h5"
        _create_ws_h5(h5_path, n_scans=50, n_channels=2)
        data = load_wavesurfer(h5_path)
        trace = get_sweep_traces(data, 1)
        assert trace.ndim == 1
        assert len(trace) == 50

    def test_missing_sweep_raises(self, tmp_path: Path) -> None:
        h5_path = tmp_path / "test.h5"
        _create_ws_h5(h5_path, n_sweeps=1)
        data = load_wavesurfer(h5_path)
        with pytest.raises(KeyError):
            get_sweep_traces(data, 99)


# ---------------------------------------------------------------------------
# load_swc_files
# ---------------------------------------------------------------------------


class TestLoadSwcFiles:
    """Test SWC file discovery."""

    def _make_tracing_dir(self, tmp_path: Path) -> Path:
        """Create a directory with standard SWC files."""
        d = tmp_path / "tracing"
        d.mkdir()
        (d / "Soma.swc").write_text("# SWC\n1 1 0 0 0 1 -1\n")
        (d / "Apical_tree.swc").write_text("# SWC\n1 3 0 0 0 1 -1\n")
        (d / "Basal_tree1.swc").write_text("# SWC\n1 3 0 0 0 1 -1\n")
        (d / "Basal_tree2.swc").write_text("# SWC\n1 3 0 0 0 1 -1\n")
        (d / "Surface.swc").write_text("# SWC\n1 1 0 0 0 1 -1\n")
        (d / "Axon.swc").write_text("# SWC\n1 2 0 0 0 1 -1\n")
        return d

    def test_finds_all_files(self, tmp_path: Path) -> None:
        d = self._make_tracing_dir(tmp_path)
        result = load_swc_files(d)
        assert result["soma"] == d / "Soma.swc"
        assert result["apical"] == d / "Apical_tree.swc"
        assert len(result["basal"]) == 2
        assert "surface" in result
        assert "axon" in result

    def test_optional_surface_absent(self, tmp_path: Path) -> None:
        d = self._make_tracing_dir(tmp_path)
        (d / "Surface.swc").unlink()
        result = load_swc_files(d)
        assert "surface" not in result

    def test_optional_axon_absent(self, tmp_path: Path) -> None:
        d = self._make_tracing_dir(tmp_path)
        (d / "Axon.swc").unlink()
        result = load_swc_files(d)
        assert "axon" not in result

    def test_missing_soma_raises(self, tmp_path: Path) -> None:
        d = self._make_tracing_dir(tmp_path)
        (d / "Soma.swc").unlink()
        with pytest.raises(FileNotFoundError, match="Soma.swc"):
            load_swc_files(d)

    def test_missing_apical_raises(self, tmp_path: Path) -> None:
        d = self._make_tracing_dir(tmp_path)
        (d / "Apical_tree.swc").unlink()
        with pytest.raises(FileNotFoundError, match="Apical_tree.swc"):
            load_swc_files(d)

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_swc_files(tmp_path / "nonexistent")

    def test_no_basal_returns_empty_list(self, tmp_path: Path) -> None:
        d = self._make_tracing_dir(tmp_path)
        for f in d.glob("Basal*.swc"):
            f.unlink()
        result = load_swc_files(d)
        assert result["basal"] == []
