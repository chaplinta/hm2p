"""Static brain rendering with brainrender.

Generates publication-quality static images of injection sites on the
Allen Mouse Brain CCFv3 atlas using brainrender's offscreen rendering.

References
----------
Claudi et al. 2021. "Visualizing anatomically registered data with
brainrender." eLife. doi:10.7554/eLife.65751
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from hm2p.constants import CELLTYPE_HEX

log = logging.getLogger(__name__)

# brainrender cell-type colours (RGB 0-1 tuples for vedo/VTK)
_CELLTYPE_RGB = {
    "penk": (0, 0, 1),      # blue
    "nonpenk": (1, 0, 0),   # red
}

# RSP sub-region names in the Allen CCFv3 atlas
_RSP_REGIONS = ("RSPagl", "RSPd", "RSPv")

# Pre-defined camera parameters for the three canonical views.
# These are in brainrender/vedo coordinate space (micrometres).
_CAMERAS = {
    "dorsal": {
        "pos": (7760, -31645, -5943),
        "viewup": (-1, 0, 0),
        "clipping_range": (27262, 45988),
    },
    "sagittal": {
        "pos": (6514, -34, 36854),
        "viewup": (0, -1, 0),
        "clipping_range": (24098, 49971),
    },
    "coronal": {
        "pos": (-19199, -1428, -5763),
        "viewup": (0, -1, 0),
        "clipping_range": (19531, 40903),
    },
}


def _ensure_offscreen() -> None:
    """Configure environment for headless (offscreen) rendering.

    Must be called **before** importing brainrender or vedo so that VTK
    initialises without a display.
    """
    os.environ.setdefault("DISPLAY", "")
    # vedo offscreen flag (respected before vedo import)
    os.environ.setdefault("VEDO_OFFSCREEN", "1")


def render_injection_sites(
    injection_df: pd.DataFrame,
    output_path: str | Path,
    *,
    sphere_radius: float = 200,
    rsp_alpha: float = 0.3,
    root_alpha: float = 0.15,
    screenshot_scale: int = 2,
) -> list[str] | None:
    """Render injection sites on an Allen CCFv3 atlas brain.

    Parameters
    ----------
    injection_df : pd.DataFrame
        Must contain columns: ``animal_id``, ``celltype``,
        ``inj_ap``, ``inj_ml``, ``inj_dv``.  Coordinates are in **mm**
        (Allen CCFv3 space) and will be converted to micrometres for
        brainrender.
    output_path : str or Path
        Directory where PNG screenshots will be saved.
    sphere_radius : float
        Radius of injection-site spheres in micrometres (default 200).
    rsp_alpha : float
        Transparency of the RSP region mesh (0-1, default 0.3).
    root_alpha : float
        Transparency of the whole-brain root mesh (0-1, default 0.15).
    screenshot_scale : int
        Resolution multiplier for screenshots (default 2).

    Returns
    -------
    list[str] | None
        Paths to the generated PNG files (dorsal, sagittal, coronal),
        or ``None`` if brainrender is not available.

    References
    ----------
    Claudi et al. 2021. "Visualizing anatomically registered data with
    brainrender." eLife. doi:10.7554/eLife.65751
    """
    _ensure_offscreen()

    try:
        from brainrender import Scene, settings
        from brainrender.actors import Point
    except ImportError:
        log.warning(
            "brainrender is not installed. Install with: "
            "pip install brainrender"
        )
        return None

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Validate required columns
    required = {"animal_id", "celltype", "inj_ap", "inj_ml", "inj_dv"}
    missing = required - set(injection_df.columns)
    if missing:
        raise ValueError(f"injection_df missing columns: {missing}")

    # Configure brainrender for offscreen rendering
    settings.OFFSCREEN = True
    settings.INTERACTIVE = False
    settings.SHOW_AXES = False

    output_files: list[str] = []

    for view_name, camera_params in _CAMERAS.items():
        scene = Scene(
            atlas_name="allen_mouse_25um",
            inset=False,
            screenshots_folder=str(output_path),
        )

        # Make brain root transparent
        scene.root.alpha(root_alpha)

        # Add RSP regions highlighted in green
        try:
            scene.add_brain_region(
                *_RSP_REGIONS,
                alpha=rsp_alpha,
                color="green",
            )
        except Exception as exc:
            log.warning("Could not add RSP regions: %s", exc)

        # Add injection site spheres
        for _, row in injection_df.iterrows():
            celltype = str(row.get("celltype", ""))
            ap_mm = pd.to_numeric(row.get("inj_ap"), errors="coerce")
            ml_mm = pd.to_numeric(row.get("inj_ml"), errors="coerce")
            dv_mm = pd.to_numeric(row.get("inj_dv"), errors="coerce")

            if pd.isna(ap_mm) or pd.isna(ml_mm) or pd.isna(dv_mm):
                continue

            # Convert mm → micrometres for brainrender
            ap_um = ap_mm * 1000
            dv_um = dv_mm * 1000
            ml_um = ml_mm * 1000

            color = _CELLTYPE_RGB.get(celltype, (0.5, 0.5, 0.5))
            animal_id = str(row.get("animal_id", ""))

            point = Point(
                pos=[ap_um, dv_um, ml_um],
                radius=sphere_radius,
                color=color,
                alpha=0.9,
                name=animal_id,
            )
            scene.add(point)

        # Render and screenshot
        try:
            scene.render(interactive=False, camera=camera_params)
            filepath = scene.screenshot(
                name=f"injection_{view_name}",
                scale=screenshot_scale,
            )
            output_files.append(filepath)
            log.info("Saved %s view to %s", view_name, filepath)
        except Exception as exc:
            log.error("Failed to render %s view: %s", view_name, exc)
        finally:
            try:
                scene.close()
            except Exception:
                pass

    return output_files if output_files else None


def render_single_animal(
    animal_id: str,
    ap_mm: float,
    dv_mm: float,
    ml_mm: float,
    celltype: str,
    output_path: str | Path,
    *,
    sphere_radius: float = 250,
    screenshot_scale: int = 2,
) -> list[str] | None:
    """Render a single animal's injection site.

    Convenience wrapper around :func:`render_injection_sites` for one
    animal.

    Parameters
    ----------
    animal_id : str
        Animal identifier.
    ap_mm, dv_mm, ml_mm : float
        Injection coordinates in mm (Allen CCFv3).
    celltype : str
        ``"penk"`` or ``"nonpenk"``.
    output_path : str or Path
        Directory where PNG screenshots will be saved.
    sphere_radius : float
        Radius of injection-site sphere in micrometres (default 250).
    screenshot_scale : int
        Resolution multiplier for screenshots (default 2).

    Returns
    -------
    list[str] | None
        Paths to generated PNG files, or ``None`` if brainrender is
        unavailable.
    """
    df = pd.DataFrame([{
        "animal_id": animal_id,
        "celltype": celltype,
        "inj_ap": ap_mm,
        "inj_ml": ml_mm,
        "inj_dv": dv_mm,
    }])
    return render_injection_sites(
        df,
        output_path,
        sphere_radius=sphere_radius,
        screenshot_scale=screenshot_scale,
    )
