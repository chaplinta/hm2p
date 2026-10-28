"""Tests for hm2p.qc.syllable_media (small synthetic arrays only)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hm2p.qc import syllable_media as sm


def test_syllable_bouts_skips_unassigned():
    b = sm.syllable_bouts([1, 1, 2, -1, -1, 1, 2, 2, 2])
    assert set(b) == {1, 2}
    np.testing.assert_array_equal(b[1][0], [0, 5])
    np.testing.assert_array_equal(b[1][1], [2, 1])
    np.testing.assert_array_equal(b[2][1], [1, 3])


def test_window_ok_bad_tracking_and_bounds():
    n = 20
    bad = np.zeros(n, bool)
    bad[5] = True
    tracked = np.ones(n, bool)
    tracked[12:15] = False
    ok = sm.window_ok([0, 4, 10, 16, -1, 18], [4, 8, 20, 20, 3, 21], bad, tracked, 0.9)
    # [0,4) clean; [4,8) has bad frame 5; [10,20) 7/10 tracked; [16,20) clean;
    # negative start and end past n are out of range
    assert ok.tolist() == [True, False, False, True, False, False]
    assert sm.window_ok([10], [20], bad, tracked, 0.7).tolist() == [True]
    assert sm.window_ok([0], [4], bad, tracked, n_max=3).tolist() == [False]


def test_select_bouts_prefers_typical_length_and_spreads_sessions():
    cands = [("a", i, 10) for i in range(20)] + [("b", i, 10) for i in range(3)]
    cands += [("c", i, 40) for i in range(5)]  # atypical length
    sel = sm.select_bouts(cands, 10, n=4, rng=np.random.default_rng(1))
    assert len(sel) == 4
    assert all(s[2] == 10 for s in sel)
    assert {s[0] for s in sel[:2]} == {"a", "b"}  # one per session first


def test_select_bouts_falls_back_to_closest_and_is_reproducible():
    cands = [("a", 0, 30), ("b", 1, 12), ("c", 2, 50), ("d", 3, 11)]
    s1 = sm.select_bouts(cands, 10, n=2, rng=np.random.default_rng(5))
    s2 = sm.select_bouts(cands, 10, n=2, rng=np.random.default_rng(5))
    assert s1 == s2
    assert {s[2] for s in s1} == {11, 12}
    assert sm.select_bouts([], 10) == []
    assert len(sm.select_bouts(cands, 10, n=9)) == 4


def test_clip_window_truncates_long_bouts():
    assert sm.clip_window(100, 12, pre=10, post=10, max_frames=75) == (90, 122, 10, 22)
    first, last, b0, b1 = sm.clip_window(100, 200, pre=10, post=10, max_frames=75)
    assert last - first == 75 and (b0, b1) == (10, 65)


def test_heading_angle():
    a = sm.heading_angle([[0, 0], [0, 0]], [[1, 0], [0, -1]])
    np.testing.assert_allclose(a, [0.0, -np.pi / 2])


def test_egocentric_points_forward_and_right():
    # animal facing image-up (heading -pi/2); a point ahead and a point on its right
    centre = np.array([50.0, 50.0])
    pts = np.array([[50.0, 40.0], [60.0, 50.0]])
    ego = sm.egocentric_points(pts, centre, -np.pi / 2)
    np.testing.assert_allclose(ego, [[0.0, 10.0], [10.0, 0.0]], atol=1e-12)
    np.testing.assert_allclose(
        sm.egocentric_points(pts, centre, -np.pi / 2, mirror=True)[1], [-10, 0], atol=1e-12
    )


@settings(max_examples=50, deadline=None)
@given(
    st.floats(-np.pi, np.pi),
    st.floats(-100, 100),
    st.floats(-100, 100),
    st.floats(1, 50),
)
def test_egocentric_heading_point_is_straight_ahead(h, cx, cy, d):
    p = np.array([cx + d * np.cos(h), cy + d * np.sin(h)])
    ego = sm.egocentric_points(p, [cx, cy], h)
    np.testing.assert_allclose(ego, [0.0, d], atol=1e-8)


@settings(max_examples=50, deadline=None)
@given(st.floats(-np.pi, np.pi), st.floats(1, 30))
def test_tile_affine_maps_heading_up_and_right_right(h, d):
    c = np.array([200.0, 150.0])
    m = sm.tile_affine(c, h, src_px=100, tile_px=50)
    f = lambda p: m[:, :2] @ p + m[:, 2]  # noqa: E731
    np.testing.assert_allclose(f(c), [25, 25], atol=1e-9)
    ahead = c + d * np.array([np.cos(h), np.sin(h)])
    np.testing.assert_allclose(f(ahead), [25, 25 - d * 0.5], atol=1e-8)
    right = c + d * np.array([-np.sin(h), np.cos(h)])  # 90 deg clockwise in the image
    np.testing.assert_allclose(f(right), [25 + d * 0.5, 25], atol=1e-8)
    mm = sm.tile_affine(c, h, src_px=100, tile_px=50, mirror=True)
    np.testing.assert_allclose(mm[:, :2] @ right + mm[:, 2], [25 - d * 0.5, 25], atol=1e-8)


def test_tile_affine_matches_egocentric_points():
    rng = np.random.default_rng(0)
    c, h = np.array([10.0, -4.0]), 0.7
    m = sm.tile_affine(c, h, src_px=40, tile_px=40)
    p = rng.normal(size=(5, 2)) * 10 + c
    ego = sm.egocentric_points(p, c, h)
    tile = p @ m[:, :2].T + m[:, 2]
    np.testing.assert_allclose(tile, np.c_[ego[:, 0] + 20, 20 - ego[:, 1]], atol=1e-9)


def test_ego_windows_shape_and_values():
    n, k = 10, 2
    pos = np.zeros((n, k, 2))
    pos[:, 0, 0] = np.arange(n)  # keypoint 0 moves +x (image) one px per frame
    centre = np.zeros((n, 2))
    heading = np.zeros(n)  # facing +x
    w = sm.ego_windows(pos, centre, heading, [2, 5], [-1, 0, 1])
    assert w.shape == (2, 3, 2, 2)
    np.testing.assert_allclose(w[0, :, 0, 1], [1, 2, 3])  # forward = image x
    np.testing.assert_allclose(w[:, :, 1], 0)
    assert sm.ego_windows(pos, centre, heading, [], [0, 1]).shape == (0, 2, 2, 2)


def test_trajectory_sums_and_mean_ignore_nan():
    w = np.ones((3, 2, 1, 2))
    w[0, 0, 0] = np.nan
    w[1] *= 3
    s, c = sm.trajectory_sums(w)
    np.testing.assert_array_equal(c[:, 0], [2, 3])
    mean = sm.mean_trajectory(s, c)
    np.testing.assert_allclose(mean[:, 0, 0], [2.0, 5.0 / 3.0])
    assert np.isnan(sm.mean_trajectory(np.zeros((1, 1, 2)), np.zeros((1, 1)))).all()


def test_left_on_left_frac():
    assert sm.left_on_left_frac([-1, -1, 1], [1, 1, -1]) == pytest.approx(2 / 3)
    assert sm.left_on_left_frac([np.nan], [1]) is None


def test_signed_turn_rate_sign():
    fps = 10.0
    # image angle decreasing = counter-clockwise seen from above = positive
    h = -np.radians(np.arange(5) * 9.0)
    r = sm.signed_turn_rate(h, fps)
    np.testing.assert_allclose(r, 90.0)
    h2 = np.array([0.0, np.nan, 0.2, 0.4])
    assert np.isnan(sm.signed_turn_rate(h2, fps)[1])
    assert np.isnan(sm.signed_turn_rate([np.nan, 1.0], fps)).all()


def test_signed_turn_rate_unwraps():
    h = np.array([np.pi - 0.1, -np.pi + 0.1])  # +0.2 rad across the wrap
    np.testing.assert_allclose(sm.signed_turn_rate(h, 1.0), -np.degrees(0.2))


def test_draw_border_only_on_flagged_frames():
    f = np.zeros((3, 10, 10, 3), np.uint8)
    out = sm.draw_border(f, [False, True, False], width=2, colour=(1, 2, 3))
    assert out[0].sum() == 0 and out[2].sum() == 0
    assert tuple(out[1, 0, 5]) == (1, 2, 3) and tuple(out[1, 5, 9]) == (1, 2, 3)
    assert tuple(out[1, 5, 5]) == (0, 0, 0)
    assert f.sum() == 0  # input untouched
    assert sm.draw_border(f, [False] * 3).sum() == 0


def test_tile_grid_layout_padding_and_errors():
    a = np.full((2, 4, 4, 3), 100, np.uint8)
    b = np.full((4, 4, 4, 3), 200, np.uint8)
    g = sm.tile_grid([a, None, b], rows=2, cols=2, gap=1, dim_after_end=0.5)
    assert g.shape == (4, 9, 9, 3)
    assert g[0, 0, 0, 0] == 100 and g[3, 0, 0, 0] == 50  # held, dimmed
    assert g[0, 0, 5, 0] == 0  # empty slot
    assert g[3, 5, 0, 0] == 200  # third clip on row 2
    with pytest.raises(ValueError):
        sm.tile_grid([None])
    with pytest.raises(ValueError):
        sm.tile_grid([a] * 5, rows=2, cols=2)
    with pytest.raises(ValueError):
        sm.tile_grid([a, np.zeros((1, 3, 3, 3), np.uint8)])


def test_encode_h264_writes_mp4(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    frames = np.random.default_rng(0).integers(0, 255, (5, 15, 17, 3), dtype=np.uint8)
    out = sm.encode_h264(frames, tmp_path / "x.mp4", fps=10)
    data = out.read_bytes()
    assert out.stat().st_size > 0 and b"ftyp" in data[:64]
    with pytest.raises(ValueError):
        sm.encode_h264(np.zeros((0, 4, 4, 3)), tmp_path / "y.mp4")


def test_encode_h264_reports_ffmpeg_failure(tmp_path):
    with pytest.raises(RuntimeError):
        sm.encode_h264(np.zeros((1, 4, 4, 3)), tmp_path / "z.mp4", ffmpeg="false")


def test_hist_median():
    assert sm.hist_median([0, 10, 0, 0], 0, 4) == pytest.approx(1.5)
    assert sm.hist_median([5, 5], 0, 2) == pytest.approx(1.0)
    assert sm.hist_median([0, 0], 0, 1) is None


def test_occupancy_counts():
    h = sm.occupancy_counts(
        [0.5, 1.5, 1.5, np.nan], [0.5, 0.5, 1.5, 0.5], [1, 1, 0, 1], [0, 1, 2], [0, 1, 2]
    )
    np.testing.assert_array_equal(h, [[1, 1], [0, 0]])


def test_syllables_covering():
    u = {1: 50, 2: 30, 3: 15, 4: 5}
    assert sm.syllables_covering(u, 0.8) == [1, 2]
    assert sm.syllables_covering(u, 0.95) == [1, 2, 3]
    assert sm.syllables_covering(u, 1.0, cap=2) == [1, 2]


def _session(sid_frames: dict[int, int], lengths: dict[int, list[int]], n_t=3, n_k=2) -> dict:
    return {
        "n_frames": sum(sid_frames.values()),
        "fps": 30.0,
        "occ_total": [[1, 1], [1, 1]],
        "keypoints": ["nose_tip", "tail_base"][:n_k],
        "traj_offsets": list(range(-1, n_t - 1)),
        "occ_edges": [[0, 1, 2], [0, 1, 2]],
        "speed_range": [0.0, 4.0],
        "syllables": {
            str(k): {
                "n_frames": f,
                "lengths": lengths[k],
                "traj_sum": np.full((n_t, n_k, 2), 2.0 * len(lengths[k])).tolist(),
                "traj_count": np.full((n_t, n_k), len(lengths[k])).tolist(),
                "occ": [[f, 0], [0, 0]],
                "speed_hist": [0, f, 0, 0],
                "turn_sum": 10.0 * f,
                "turn_n": f,
            }
            for k, f in sid_frames.items()
        },
    }


def test_pool_media_combines_sessions():
    a = _session({1: 60, 2: 30, 3: 10}, {1: [10, 20], 2: [15], 3: [5]})
    b = _session({1: 40, 2: 60}, {1: [30], 2: [15, 15]})
    p = sm.pool_media({"a": a, "b": b, "c": {}}, frac=0.8)
    assert p["n_sessions"] == 2 and p["n_frames"] == 200
    ids = [r["id"] for r in p["syllables"]]
    assert ids == [1, 2]
    r1 = p["syllables"][0]
    assert r1["usage"] == pytest.approx(0.5)
    assert r1["n_bouts"] == 3 and r1["median_len"] == 20
    assert r1["median_ms"] == pytest.approx(20 / 30 * 1e3)
    assert r1["n_sessions"] == 2
    assert r1["mean_turn"] == pytest.approx(10.0)
    assert r1["median_speed"] == pytest.approx(1.5)
    assert np.allclose(np.array(r1["traj"]), 2.0)
    assert r1["occ"] == [[100, 0], [0, 0]]
    assert p["occ_total"] == [[2, 2], [2, 2]]
    assert p["covered"] == pytest.approx(0.95)
    assert sm.pool_media({})["syllables"] == []
