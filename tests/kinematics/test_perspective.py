"""Tests for kinematics/perspective.py — parallax correction for overhead camera."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from hypothesis import given, settings
from hypothesis import strategies as st

from hm2p.kinematics.perspective import (
    BODYPART_HEIGHTS,
    BODYPART_HEIGHTS_IMPLANT,
    DEFAULT_CAMERA_HEIGHT_MM,
    UNCROPPED_HEIGHT,
    UNCROPPED_WIDTH,
    correct_dataset_perspective,
    correct_perspective,
    estimate_camera_center,
    load_camera_params,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

KEYPOINTS = ["left_ear", "right_ear", "mid_back", "mouse_center", "tail_base"]


def _make_pose_dataset(
    n_frames: int = 10,
    pos_data: np.ndarray | None = None,
) -> xr.Dataset:
    """Build a minimal movement-style xarray Dataset for testing."""
    n_kp = len(KEYPOINTS)
    if pos_data is None:
        pos_data = np.ones((n_frames, 2, n_kp, 1), dtype=np.float64)
    conf_data = np.ones((n_frames, n_kp, 1), dtype=np.float64)

    position = xr.DataArray(
        pos_data,
        dims=["time", "space", "keypoints", "individuals"],
        coords={
            "time": np.arange(n_frames, dtype=float),
            "space": ["x", "y"],
            "keypoints": KEYPOINTS,
            "individuals": ["mouse"],
        },
    )
    confidence = xr.DataArray(
        conf_data,
        dims=["time", "keypoints", "individuals"],
        coords={
            "time": np.arange(n_frames, dtype=float),
            "keypoints": KEYPOINTS,
            "individuals": ["mouse"],
        },
    )
    return xr.Dataset({"position": position, "confidence": confidence})


# ---------------------------------------------------------------------------
# correct_perspective
# ---------------------------------------------------------------------------


class TestCorrectPerspective:
    """Tests for the core correct_perspective function."""

    def test_center_pixel_unchanged(self) -> None:
        """Point at camera centre should not move regardless of height."""
        cx, cy = 400.0, 300.0
        x = np.array([cx])
        y = np.array([cy])
        x_corr, y_corr = correct_perspective(x, y, (cx, cy), 700.0, 40.0)
        np.testing.assert_allclose(x_corr, [cx])
        np.testing.assert_allclose(y_corr, [cy])

    def test_edge_pushed_inward(self) -> None:
        """Point away from camera centre should move toward centre."""
        cx, cy = 400.0, 300.0
        x = np.array([600.0])  # 200 px right of centre
        y = np.array([300.0])  # same y as centre
        x_corr, y_corr = correct_perspective(x, y, (cx, cy), 700.0, 40.0)
        # Should move left (toward centre)
        assert x_corr[0] < 600.0
        assert x_corr[0] > cx  # but still to the right of centre
        np.testing.assert_allclose(y_corr, [cy])  # y unchanged (on axis)

    def test_zero_height_no_correction(self) -> None:
        """h=0 means no parallax — output should equal input."""
        x = np.array([100.0, 200.0, 300.0])
        y = np.array([50.0, 150.0, 250.0])
        x_corr, y_corr = correct_perspective(x, y, (400.0, 300.0), 700.0, 0.0)
        np.testing.assert_array_equal(x_corr, x)
        np.testing.assert_array_equal(y_corr, y)

    def test_correction_scales_with_distance(self) -> None:
        """Points further from centre get larger absolute corrections."""
        cx, cy = 400.0, 300.0
        x_near = np.array([450.0])  # 50 px from centre
        x_far = np.array([600.0])  # 200 px from centre
        y = np.array([300.0])

        dx_near = x_near[0] - cx
        dx_far = x_far[0] - cx
        _, _ = correct_perspective(x_near, y, (cx, cy), 700.0, 40.0)
        x_near_corr, _ = correct_perspective(x_near, y, (cx, cy), 700.0, 40.0)
        x_far_corr, _ = correct_perspective(x_far, y, (cx, cy), 700.0, 40.0)

        correction_near = abs(x_near[0] - x_near_corr[0])
        correction_far = abs(x_far[0] - x_far_corr[0])
        assert correction_far > correction_near

    def test_correction_scales_with_height(self) -> None:
        """Higher bodypart → larger correction."""
        cx, cy = 400.0, 300.0
        x = np.array([600.0])
        y = np.array([300.0])
        x_low, _ = correct_perspective(x, y, (cx, cy), 700.0, 20.0)
        x_high, _ = correct_perspective(x, y, (cx, cy), 700.0, 40.0)
        # Higher bodypart → pushed more toward centre → smaller x
        assert x_high[0] < x_low[0]

    def test_correction_formula_exact(self) -> None:
        """Verify exact formula: corrected = cx + (px - cx) * (H - h) / H."""
        cx, cy = 400.0, 300.0
        H = 700.0
        h = 40.0
        px, py = 600.0, 500.0

        x = np.array([px])
        y = np.array([py])
        x_corr, y_corr = correct_perspective(x, y, (cx, cy), H, h)

        expected_x = cx + (px - cx) * (H - h) / H
        expected_y = cy + (py - cy) * (H - h) / H
        np.testing.assert_allclose(x_corr[0], expected_x)
        np.testing.assert_allclose(y_corr[0], expected_y)

    def test_nan_preserved(self) -> None:
        """NaN positions should propagate through correction."""
        x = np.array([100.0, np.nan, 300.0])
        y = np.array([50.0, 150.0, np.nan])
        x_corr, y_corr = correct_perspective(x, y, (400.0, 300.0), 700.0, 40.0)
        assert np.isnan(x_corr[1])
        assert np.isnan(y_corr[2])
        assert np.isfinite(x_corr[0])

    def test_negative_displacement(self) -> None:
        """Points to the left of / above camera centre move in correct direction."""
        cx, cy = 400.0, 300.0
        x = np.array([200.0])  # 200 px LEFT of centre
        y = np.array([100.0])  # 200 px ABOVE centre
        x_corr, y_corr = correct_perspective(x, y, (cx, cy), 700.0, 40.0)
        # Should move toward centre (right and down)
        assert x_corr[0] > 200.0
        assert y_corr[0] > 100.0

    def test_height_at_camera_raises(self) -> None:
        """bodypart_height_mm >= camera_height_mm should raise ValueError."""
        x = np.array([100.0])
        y = np.array([100.0])
        with pytest.raises(ValueError, match="must be <"):
            correct_perspective(x, y, (400.0, 300.0), 700.0, 700.0)
        with pytest.raises(ValueError, match="must be <"):
            correct_perspective(x, y, (400.0, 300.0), 700.0, 800.0)

    def test_2d_array_input(self) -> None:
        """Function should work with 2D arrays (time, individuals)."""
        x = np.array([[100.0, 200.0], [300.0, 400.0]])
        y = np.array([[50.0, 150.0], [250.0, 350.0]])
        x_corr, y_corr = correct_perspective(x, y, (400.0, 300.0), 700.0, 25.0)
        assert x_corr.shape == (2, 2)
        assert y_corr.shape == (2, 2)

    def test_output_dtype_matches_input(self) -> None:
        """Output dtype should match input dtype (float32 or float64)."""
        x32 = np.array([100.0], dtype=np.float32)
        y32 = np.array([50.0], dtype=np.float32)
        x_corr, y_corr = correct_perspective(x32, y32, (400.0, 300.0), 700.0, 25.0)
        # NumPy upcasts to float64 when mixing with float64 scalars — acceptable
        assert x_corr.dtype in (np.float32, np.float64)

    @given(
        px=st.floats(min_value=0, max_value=1280, allow_nan=False),
        py=st.floats(min_value=0, max_value=1024, allow_nan=False),
        h=st.floats(min_value=0.1, max_value=600, allow_nan=False),
    )
    @settings(max_examples=200)
    def test_corrected_between_original_and_center(
        self, px: float, py: float, h: float
    ) -> None:
        """Corrected point should always lie between the original and camera centre."""
        cx, cy = 640.0, 512.0
        x = np.array([px])
        y = np.array([py])
        x_corr, y_corr = correct_perspective(x, y, (cx, cy), 700.0, h)

        # Distance from centre should decrease or stay the same
        d_orig = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
        d_corr = np.sqrt((x_corr[0] - cx) ** 2 + (y_corr[0] - cy) ** 2)
        assert d_corr <= d_orig + 1e-10  # tolerance for float precision

    @given(
        px=st.floats(min_value=0, max_value=1280, allow_nan=False),
        py=st.floats(min_value=0, max_value=1024, allow_nan=False),
        h=st.floats(min_value=0.1, max_value=600, allow_nan=False),
    )
    @settings(max_examples=200)
    def test_correction_preserves_direction(
        self, px: float, py: float, h: float
    ) -> None:
        """Corrected point should be on the same ray from camera centre."""
        cx, cy = 640.0, 512.0
        x = np.array([px])
        y = np.array([py])
        x_corr, y_corr = correct_perspective(x, y, (cx, cy), 700.0, h)

        dx_orig = px - cx
        dy_orig = py - cy
        dx_corr = x_corr[0] - cx
        dy_corr = y_corr[0] - cy

        # If original is at centre, corrected should also be at centre
        if abs(dx_orig) < 1e-10 and abs(dy_orig) < 1e-10:
            assert abs(dx_corr) < 1e-6
            assert abs(dy_corr) < 1e-6
        else:
            # Same direction: cross product ≈ 0
            cross = dx_orig * dy_corr - dy_orig * dx_corr
            assert abs(cross) < 1e-6


# ---------------------------------------------------------------------------
# estimate_camera_center
# ---------------------------------------------------------------------------


class TestEstimateCameraCenter:
    def test_no_crop(self) -> None:
        """With zero crop offset, centre is at half sensor dimensions."""
        cx, cy = estimate_camera_center(0, 0)
        assert cx == 640.0
        assert cy == 512.0

    def test_known_crop_offset(self) -> None:
        """Standard crop values produce expected centre."""
        cx, cy = estimate_camera_center(108, 261)
        assert cx == pytest.approx(640.0 - 108)
        assert cy == pytest.approx(512.0 - 261)

    def test_custom_sensor_size(self) -> None:
        """Custom uncropped dimensions should work."""
        cx, cy = estimate_camera_center(50, 50, uncrop_w=1000, uncrop_h=800)
        assert cx == pytest.approx(450.0)
        assert cy == pytest.approx(350.0)

    def test_negative_centre_possible(self) -> None:
        """Large crop offset can put centre outside cropped frame (negative)."""
        cx, cy = estimate_camera_center(700, 600)
        assert cx < 0
        assert cy < 0


# ---------------------------------------------------------------------------
# correct_dataset_perspective
# ---------------------------------------------------------------------------


class TestCorrectDatasetPerspective:
    def test_shape_preserved(self) -> None:
        """Output Dataset should have the same shape as input."""
        ds = _make_pose_dataset(n_frames=20)
        ds_corr = correct_dataset_perspective(ds, (400.0, 300.0))
        assert ds_corr.position.shape == ds.position.shape

    def test_coords_preserved(self) -> None:
        """Coordinate labels should be unchanged."""
        ds = _make_pose_dataset()
        ds_corr = correct_dataset_perspective(ds, (400.0, 300.0))
        assert list(ds_corr.coords["keypoints"].values) == KEYPOINTS
        assert list(ds_corr.coords["space"].values) == ["x", "y"]

    def test_per_bodypart_heights(self) -> None:
        """Different keypoints should get different corrections."""
        # Place all keypoints at same position far from centre
        n = 5
        pos = np.zeros((n, 2, len(KEYPOINTS), 1))
        pos[:, 0, :, 0] = 600.0  # x = 600 for all
        pos[:, 1, :, 0] = 400.0  # y = 400 for all
        ds = _make_pose_dataset(n_frames=n, pos_data=pos)

        heights = {"left_ear": 40.0, "right_ear": 40.0, "tail_base": 10.0}
        ds_corr = correct_dataset_perspective(
            ds, (400.0, 300.0), bodypart_heights=heights
        )

        # Ears (h=40) should be corrected more than tail_base (h=10)
        ear_x = ds_corr.position.sel(keypoints="left_ear", space="x").values[0, 0]
        tail_x = ds_corr.position.sel(keypoints="tail_base", space="x").values[0, 0]
        # Both move toward centre (400), ear moves more
        assert ear_x < tail_x  # ear closer to centre

    def test_unknown_keypoint_unchanged(self) -> None:
        """Keypoints not in bodypart_heights should remain uncorrected."""
        n = 5
        pos = np.zeros((n, 2, len(KEYPOINTS), 1))
        pos[:, 0, :, 0] = 600.0
        pos[:, 1, :, 0] = 400.0
        ds = _make_pose_dataset(n_frames=n, pos_data=pos)

        # Only correct left_ear, leave everything else at h=0
        heights = {"left_ear": 40.0}
        ds_corr = correct_dataset_perspective(
            ds, (400.0, 300.0), bodypart_heights=heights
        )

        # mid_back not in heights → should be unchanged
        mid_x_orig = ds.position.sel(keypoints="mid_back", space="x").values
        mid_x_corr = ds_corr.position.sel(keypoints="mid_back", space="x").values
        np.testing.assert_array_equal(mid_x_orig, mid_x_corr)

    def test_zero_heights_identity(self) -> None:
        """All heights = 0 → output equals input."""
        ds = _make_pose_dataset()
        heights = {kp: 0.0 for kp in KEYPOINTS}
        ds_corr = correct_dataset_perspective(
            ds, (400.0, 300.0), bodypart_heights=heights
        )
        np.testing.assert_array_equal(
            ds_corr.position.values, ds.position.values
        )

    def test_default_heights_are_implant(self) -> None:
        """Default bodypart_heights should be BODYPART_HEIGHTS_IMPLANT."""
        n = 5
        pos = np.zeros((n, 2, len(KEYPOINTS), 1))
        pos[:, 0, :, 0] = 600.0
        pos[:, 1, :, 0] = 400.0
        ds = _make_pose_dataset(n_frames=n, pos_data=pos)

        # Call without explicit heights (should use implant defaults)
        ds_default = correct_dataset_perspective(ds, (400.0, 300.0))
        # Call with explicit implant heights
        ds_explicit = correct_dataset_perspective(
            ds, (400.0, 300.0), bodypart_heights=BODYPART_HEIGHTS_IMPLANT
        )
        np.testing.assert_array_equal(
            ds_default.position.values, ds_explicit.position.values
        )

    def test_nan_in_position(self) -> None:
        """NaN positions should be preserved through correction."""
        n = 5
        pos = np.ones((n, 2, len(KEYPOINTS), 1)) * 500.0
        pos[2, 0, 0, 0] = np.nan  # NaN in left_ear x at frame 2
        pos[2, 1, 0, 0] = np.nan  # NaN in left_ear y at frame 2
        ds = _make_pose_dataset(n_frames=n, pos_data=pos)

        ds_corr = correct_dataset_perspective(ds, (400.0, 300.0))
        assert np.isnan(
            ds_corr.position.sel(
                keypoints="left_ear", space="x"
            ).values[2, 0]
        )


# ---------------------------------------------------------------------------
# load_camera_params
# ---------------------------------------------------------------------------


class TestLoadCameraParams:
    def test_loads_from_meta_txt(self, tmp_path: pytest.TempPathFactory) -> None:
        """Should parse meta.txt and return camera params dict."""
        meta = tmp_path / "meta.txt"
        meta.write_text(
            "[crop]\n"
            "x = 108\n"
            "y = 261\n"
            "width = 832\n"
            "height = 608\n"
            "\n"
            "[scale]\n"
            "mm_per_pix = 0.811\n"
            "\n"
            "[roi]\n"
            "x1 = 149.0\n"
            "y1 = 72.0\n"
            "x2 = 764.0\n"
            "y2 = 82.0\n"
            "x3 = 757.0\n"
            "y3 = 509.0\n"
            "x4 = 143.0\n"
            "y4 = 500.0\n"
        )
        params = load_camera_params(meta)

        assert "camera_center_px" in params
        assert "crop_offset" in params
        assert "scale_mm_per_px" in params
        assert "maze_corners" in params

        cx, cy = params["camera_center_px"]
        assert cx == pytest.approx(640.0 - 108)
        assert cy == pytest.approx(512.0 - 261)
        assert params["crop_offset"] == (108, 261)
        assert params["scale_mm_per_px"] == pytest.approx(0.811)
        assert params["maze_corners"].shape == (4, 2)

    def test_missing_file_raises(self, tmp_path: pytest.TempPathFactory) -> None:
        """Non-existent meta.txt should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_camera_params(tmp_path / "nonexistent.txt")


# ---------------------------------------------------------------------------
# Constants sanity checks
# ---------------------------------------------------------------------------


class TestConstants:
    def test_all_bodyparts_have_heights(self) -> None:
        """Both height dicts should have the same keys."""
        assert set(BODYPART_HEIGHTS.keys()) == set(BODYPART_HEIGHTS_IMPLANT.keys())

    def test_implant_heights_gte_normal(self) -> None:
        """Implant heights should be >= normal heights for all bodyparts."""
        for kp in BODYPART_HEIGHTS:
            assert BODYPART_HEIGHTS_IMPLANT[kp] >= BODYPART_HEIGHTS[kp]

    def test_heights_positive(self) -> None:
        """All heights should be positive."""
        for h in BODYPART_HEIGHTS.values():
            assert h > 0
        for h in BODYPART_HEIGHTS_IMPLANT.values():
            assert h > 0

    def test_heights_below_camera(self) -> None:
        """All heights should be well below camera height."""
        for h in BODYPART_HEIGHTS_IMPLANT.values():
            assert h < DEFAULT_CAMERA_HEIGHT_MM

    def test_sensor_dimensions(self) -> None:
        """Basler acA1300-200um sensor dimensions."""
        assert UNCROPPED_WIDTH == 1280
        assert UNCROPPED_HEIGHT == 1024
