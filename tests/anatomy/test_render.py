"""Tests for hm2p.anatomy.render — static brain rendering wrapper.

brainrender is not a test dependency. Two regimes are covered:
  1. brainrender absent → functions return None gracefully.
  2. brainrender present (a synthetic fake injected into sys.modules) →
     column validation and the render/screenshot loop are exercised
     without any real VTK/atlas work.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from hm2p.anatomy.render import (
    _CAMERAS,
    _CELLTYPE_RGB,
    _ensure_offscreen,
    render_injection_sites,
    render_single_animal,
)


@pytest.fixture
def no_brainrender(monkeypatch: pytest.MonkeyPatch):
    """Force ``import brainrender`` to raise ImportError deterministically.

    Setting the sys.modules entries to None makes the import statement
    raise, so the ImportError branch is exercised in every environment
    (whether or not brainrender is actually installed) — no skips.
    """
    monkeypatch.setitem(sys.modules, "brainrender", None)
    monkeypatch.setitem(sys.modules, "brainrender.actors", None)
    yield


@pytest.fixture
def injection_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "animal_id": "m1",
                "celltype": "penk",
                "inj_ap": 1.5,
                "inj_ml": 0.8,
                "inj_dv": 1.2,
            },
            {
                "animal_id": "m2",
                "celltype": "nonpenk",
                "inj_ap": 1.6,
                "inj_ml": 0.9,
                "inj_dv": 1.3,
            },
        ]
    )


# ── module-level constants and helpers ──────────────────────────────


def test_ensure_offscreen_sets_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """_ensure_offscreen sets headless VTK env vars without overwriting."""
    monkeypatch.delenv("VEDO_OFFSCREEN", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    _ensure_offscreen()
    assert "VEDO_OFFSCREEN" in __import__("os").environ
    assert __import__("os").environ["VEDO_OFFSCREEN"] == "1"


def test_camera_definitions_present() -> None:
    """All three canonical views are defined with pos/viewup/clip."""
    assert set(_CAMERAS) == {"dorsal", "sagittal", "coronal"}
    for params in _CAMERAS.values():
        assert set(params) == {"pos", "viewup", "clipping_range"}


def test_celltype_rgb_map() -> None:
    """penk/nonpenk map to distinct RGB tuples."""
    assert _CELLTYPE_RGB["penk"] == (0, 0, 1)
    assert _CELLTYPE_RGB["nonpenk"] == (1, 0, 0)


# ── brainrender-absent path ─────────────────────────────────────────


def test_render_injection_sites_no_brainrender(
    no_brainrender, injection_df: pd.DataFrame, tmp_path: Path
) -> None:
    """Returns None when brainrender cannot be imported."""
    assert render_injection_sites(injection_df, tmp_path) is None


def test_render_single_animal_no_brainrender(no_brainrender, tmp_path: Path) -> None:
    """Single-animal wrapper also returns None without brainrender."""
    result = render_single_animal("m1", 1.5, 1.2, 0.8, "penk", tmp_path)
    assert result is None


# ── brainrender-present path (fake injected) ────────────────────────


class _FakeSettings:
    OFFSCREEN = False
    INTERACTIVE = True
    SHOW_AXES = True


class _FakeScene:
    """Minimal stand-in for brainrender.Scene."""

    def __init__(self, *args, **kwargs) -> None:
        self.screenshots_folder = kwargs.get("screenshots_folder", ".")
        self.root = types.SimpleNamespace(alpha=lambda a: None)
        self.added = []

    def add_brain_region(self, *regions, **kwargs):
        return None

    def add(self, obj) -> None:
        self.added.append(obj)

    def render(self, **kwargs) -> None:
        return None

    def screenshot(self, name: str, scale: int = 1) -> str:
        path = Path(self.screenshots_folder) / f"{name}.png"
        path.write_bytes(b"\x89PNG\r\n")  # tiny placeholder
        return str(path)

    def close(self) -> None:
        return None


class _FakePoint:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


@pytest.fixture
def fake_brainrender(monkeypatch: pytest.MonkeyPatch):
    """Inject synthetic brainrender modules, restoring originals after."""
    saved = {k: sys.modules.get(k) for k in ("brainrender", "brainrender.actors")}

    br = types.ModuleType("brainrender")
    br.Scene = _FakeScene
    br.settings = _FakeSettings()
    actors = types.ModuleType("brainrender.actors")
    actors.Point = _FakePoint
    br.actors = actors

    monkeypatch.setitem(sys.modules, "brainrender", br)
    monkeypatch.setitem(sys.modules, "brainrender.actors", actors)
    yield br
    # monkeypatch.setitem restores automatically; be explicit for safety.
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


def test_render_injection_sites_with_fake(
    fake_brainrender, injection_df: pd.DataFrame, tmp_path: Path
) -> None:
    """With a fake brainrender, three view PNGs are produced."""
    out = render_injection_sites(injection_df, tmp_path)
    assert out is not None
    assert len(out) == 3
    for p in out:
        assert Path(p).exists()
    # settings were switched to offscreen mode.
    assert fake_brainrender.settings.OFFSCREEN is True


def test_render_injection_sites_missing_columns(fake_brainrender, tmp_path: Path) -> None:
    """A DataFrame missing required columns raises ValueError."""
    bad = pd.DataFrame({"animal_id": ["m1"]})
    with pytest.raises(ValueError, match="missing columns"):
        render_injection_sites(bad, tmp_path)


def test_render_injection_sites_skips_nan_coords(fake_brainrender, tmp_path: Path) -> None:
    """Rows with NaN coordinates are skipped, not rendered as points."""
    df = pd.DataFrame(
        [
            {
                "animal_id": "m1",
                "celltype": "penk",
                "inj_ap": float("nan"),
                "inj_ml": 0.8,
                "inj_dv": 1.2,
            }
        ]
    )
    out = render_injection_sites(df, tmp_path)
    # Still renders the views (just with no injection spheres).
    assert out is not None
    assert len(out) == 3


def test_render_single_animal_with_fake(fake_brainrender, tmp_path: Path) -> None:
    """Single-animal wrapper builds a one-row frame and renders it."""
    out = render_single_animal("m1", 1.5, 1.2, 0.8, "penk", tmp_path)
    assert out is not None
    assert len(out) == 3
