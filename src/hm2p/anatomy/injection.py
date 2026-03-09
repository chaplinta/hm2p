"""Injection site extraction from brainreg segmentation.

Extracts viral injection site coordinates from brainreg-segmented
brain volumes.  Supports both CSV summary and TIFF segmentation modes.

In the Allen CCFv3 25 um atlas (BrainGlobe convention):
  - Axis 0 (X) = anterior-posterior
  - Axis 1 (Y) = dorsal-ventral
  - Axis 2 (Z) = left-right (medial-lateral)

Coordinates in the brainreg ``summary.csv`` are in micrometres.
This module converts to millimetres for downstream use and mirrors
left-hemisphere injections to the right hemisphere.

References
----------
Tyson et al. 2022. "Accurate determination of marker location within
whole-brain microscopy images." Scientific Reports.
doi:10.1038/s41598-021-04676-9
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

# Allen 25 um atlas depth along the Z (left-right) axis, in mm.
_ATLAS_DEPTH_MM: float = 11.4


def mirror_to_right_hemisphere(
    x: float,
    y: float,
    z: float,
    atlas_depth_mm: float = _ATLAS_DEPTH_MM,
) -> tuple[float, float, float]:
    """Mirror a left-hemisphere coordinate to the right hemisphere.

    In the BrainGlobe Allen 25 um atlas the **Z axis** is the
    left-right (medial-lateral) dimension.  The midline sits at
    ``atlas_depth_mm / 2``.  Points with ``z > midline`` are in the
    left hemisphere and are reflected to the corresponding right-
    hemisphere position.  Right-hemisphere points (``z <= midline``)
    are returned unchanged.

    Args:
        x: Anterior-posterior coordinate (mm).
        y: Dorsal-ventral coordinate (mm).
        z: Medial-lateral coordinate (mm).
        atlas_depth_mm: Full atlas depth along the Z axis in mm.

    Returns:
        ``(x, y, z_mirrored)`` with *z_mirrored* in the right hemisphere.
    """
    midline_mm = atlas_depth_mm / 2.0

    if z > midline_mm:
        z = midline_mm - (z - midline_mm)

    return x, y, z


def _radius_from_volume(volume_mm3: float) -> float:
    """Compute sphere radius (mm) from a volume (mm^3).

    Uses the sphere volume formula  V = (4/3) pi r^3  solved for r.

    Args:
        volume_mm3: Volume in cubic millimetres.  Must be >= 0.

    Returns:
        Radius in millimetres.
    """
    if volume_mm3 < 0.0:
        raise ValueError(f"Volume must be >= 0, got {volume_mm3}")
    return (3.0 * volume_mm3 / (4.0 * math.pi)) ** (1.0 / 3.0)


def extract_injection_sites(
    brainreg_output_dir: Path,
    use_csv: bool = True,
) -> list[dict]:
    """Extract injection site regions from brainreg segmentation output.

    Two modes are supported:

    * **CSV mode** (``use_csv=True``, default): reads
      ``segmentation/atlas_space/regions/summary.csv`` produced by
      brainreg-segment.  Each row becomes one entry.
    * **TIFF mode** (``use_csv=False``): reads ``region_0.tiff`` and
      computes the centroid from non-zero voxels.

    Coordinates are converted from micrometres to millimetres and
    mirrored to the right hemisphere when necessary.

    Args:
        brainreg_output_dir: Root of a single brainreg registration
            output (contains ``segmentation/`` subdirectory).
        use_csv: If ``True`` parse the CSV summary; if ``False`` load
            the TIFF segmentation directly.

    Returns:
        List of dicts with keys:

        * ``region`` — anatomical region name (``str``).
        * ``volume_mm3`` — segmented volume (``float``).
        * ``center_ap_mm`` — anterior-posterior centre (``float``).
        * ``center_dv_mm`` — dorsal-ventral centre (``float``).
        * ``center_ml_mm`` — medial-lateral centre (``float``).
        * ``radius_mm`` — equivalent sphere radius (``float``).

        Returns an empty list when no segmentation data is found.
    """
    seg_dir = (
        brainreg_output_dir / "segmentation" / "atlas_space" / "regions"
    )

    if not seg_dir.exists():
        logger.warning(
            "segmentation_dir_missing",
            path=str(seg_dir),
        )
        return []

    sites: list[dict] = []

    if use_csv:
        csv_path = seg_dir / "summary.csv"
        if not csv_path.exists():
            logger.warning("summary_csv_missing", path=str(csv_path))
            return []

        df = pd.read_csv(csv_path)
        if df.empty:
            logger.warning("summary_csv_empty", path=str(csv_path))
            return []

        for _, row in df.iterrows():
            # Coordinates in the CSV are in micrometres — convert to mm.
            ap_mm = row["axis_0_center_um"] / 1000.0
            dv_mm = row["axis_1_center_um"] / 1000.0
            ml_mm = row["axis_2_center_um"] / 1000.0

            ap_mm, dv_mm, ml_mm = mirror_to_right_hemisphere(
                ap_mm, dv_mm, ml_mm
            )

            volume_mm3 = float(row["volume_mm3"])
            radius_mm = _radius_from_volume(volume_mm3)

            sites.append(
                {
                    "region": str(row["region"]),
                    "volume_mm3": volume_mm3,
                    "center_ap_mm": float(ap_mm),
                    "center_dv_mm": float(dv_mm),
                    "center_ml_mm": float(ml_mm),
                    "radius_mm": float(radius_mm),
                }
            )
    else:
        # TIFF segmentation mode.
        tiff_path = seg_dir / "region_0.tiff"
        if not tiff_path.exists():
            logger.warning("segmentation_tiff_missing", path=str(tiff_path))
            return []

        try:
            import tifffile
        except ImportError as exc:
            raise ImportError(
                "tifffile is required for TIFF segmentation mode. "
                "Install it with: pip install tifffile"
            ) from exc

        data = tifffile.imread(tiff_path)
        coords = np.argwhere(data > 0)

        if coords.shape[0] == 0:
            logger.warning(
                "no_segmented_voxels", path=str(tiff_path)
            )
            return []

        # Convert voxel indices to mm (25 um voxels).
        voxel_um = 25.0
        coords_mm = coords.astype(np.float64) * voxel_um / 1000.0

        centroid = coords_mm.mean(axis=0)
        ap_mm, dv_mm, ml_mm = float(centroid[0]), float(centroid[1]), float(centroid[2])
        ap_mm, dv_mm, ml_mm = mirror_to_right_hemisphere(ap_mm, dv_mm, ml_mm)

        n_voxels = coords.shape[0]
        volume_mm3 = n_voxels * (voxel_um / 1000.0) ** 3
        radius_mm = _radius_from_volume(volume_mm3)

        sites.append(
            {
                "region": "region_0",
                "volume_mm3": float(volume_mm3),
                "center_ap_mm": ap_mm,
                "center_dv_mm": dv_mm,
                "center_ml_mm": ml_mm,
                "radius_mm": float(radius_mm),
            }
        )

    logger.info(
        "extracted_injection_sites",
        n_sites=len(sites),
        output_dir=str(brainreg_output_dir),
    )
    return sites


def get_injection_coords_for_animal(
    brainreg_base_dir: Path,
    animal_id: str,
) -> dict | None:
    """Find and extract injection coordinates for a specific animal.

    Scans subdirectories of *brainreg_base_dir* for one that contains
    *animal_id* in its name, then calls :func:`extract_injection_sites`.

    The first extracted site's coordinates are returned as
    ``{'inj_ap': ..., 'inj_ml': ..., 'inj_dv': ...}`` (all in mm).

    Args:
        brainreg_base_dir: Directory containing per-animal brainreg
            output subdirectories.
        animal_id: Animal identifier to search for.

    Returns:
        Dict with keys ``inj_ap``, ``inj_ml``, ``inj_dv`` (floats in
        mm), or ``None`` if no matching directory or segmentation is
        found.

    Raises:
        RuntimeError: If multiple directories match *animal_id*.
    """
    matching: list[Path] = []
    for subdir in brainreg_base_dir.iterdir():
        if subdir.is_dir() and animal_id in subdir.name:
            matching.append(subdir)

    if len(matching) == 0:
        logger.warning(
            "no_brainreg_dir_for_animal",
            animal_id=animal_id,
            base_dir=str(brainreg_base_dir),
        )
        return None

    if len(matching) > 1:
        raise RuntimeError(
            f"Multiple brainreg directories found for animal {animal_id}: "
            f"{matching}"
        )

    sites = extract_injection_sites(matching[0])
    if not sites:
        logger.warning(
            "no_injection_sites_for_animal",
            animal_id=animal_id,
        )
        return None

    first = sites[0]
    return {
        "inj_ap": first["center_ap_mm"],
        "inj_ml": first["center_ml_mm"],
        "inj_dv": first["center_dv_mm"],
    }


def load_brainreg_volumes(brainreg_output_dir: Path) -> pd.DataFrame | None:
    """Load brain region volumes from brainreg ``volumes.csv``.

    This CSV is produced by brainreg registration (not segmentation) and
    contains left/right/total volumes for every atlas region.

    Args:
        brainreg_output_dir: Root of a single brainreg registration output.

    Returns:
        DataFrame with columns ``structure_name``, ``left_volume_mm3``,
        ``right_volume_mm3``, ``total_volume_mm3``, or ``None`` if the
        file does not exist.
    """
    vol_path = brainreg_output_dir / "volumes.csv"
    if not vol_path.exists():
        logger.warning("volumes_csv_missing", path=str(vol_path))
        return None

    df = pd.read_csv(vol_path)
    if df.empty:
        logger.warning("volumes_csv_empty", path=str(vol_path))
        return None

    logger.info(
        "loaded_brainreg_volumes",
        n_regions=len(df),
        output_dir=str(brainreg_output_dir),
    )
    return df


def get_rsp_volume(volumes_df: pd.DataFrame) -> dict:
    """Extract RSP (retrosplenial cortex) volumes from brainreg volumes.

    Searches for regions containing 'Retrosplenial' in the structure name
    and sums their volumes.

    Args:
        volumes_df: DataFrame from :func:`load_brainreg_volumes`.

    Returns:
        Dict with keys ``rsp_total_mm3``, ``rsp_left_mm3``, ``rsp_right_mm3``,
        ``rsp_regions`` (list of matching region names).
    """
    rsp_mask = volumes_df["structure_name"].str.contains(
        "Retrosplenial|RSP", case=False, na=False
    )
    rsp_df = volumes_df[rsp_mask]

    return {
        "rsp_total_mm3": float(rsp_df["total_volume_mm3"].sum()),
        "rsp_left_mm3": float(rsp_df["left_volume_mm3"].sum()),
        "rsp_right_mm3": float(rsp_df["right_volume_mm3"].sum()),
        "rsp_regions": list(rsp_df["structure_name"]),
    }


def update_animals_csv(
    animals_csv_path: Path,
    brainreg_base_dir: Path,
) -> pd.DataFrame:
    """Update ``animals.csv`` with injection coordinates from brainreg.

    For each unique ``animal_id`` in the CSV, attempts to find brainreg
    segmentation output and writes ``inj_ap``, ``inj_ml``, ``inj_dv``
    columns.  The CSV is saved in-place.

    Args:
        animals_csv_path: Path to ``animals.csv``.
        brainreg_base_dir: Directory containing per-animal brainreg
            output subdirectories.

    Returns:
        The updated :class:`~pandas.DataFrame`.

    Raises:
        FileNotFoundError: If *animals_csv_path* does not exist.
    """
    if not animals_csv_path.exists():
        raise FileNotFoundError(
            f"animals.csv not found at {animals_csv_path}"
        )

    df = pd.read_csv(animals_csv_path)

    # Ensure coordinate columns exist.
    for col in ("inj_ap", "inj_ml", "inj_dv"):
        if col not in df.columns:
            df[col] = np.nan

    updated = 0
    for animal_id in df["animal_id"].unique():
        coords = get_injection_coords_for_animal(
            brainreg_base_dir, str(animal_id)
        )
        if coords is None:
            continue

        mask = df["animal_id"] == animal_id
        df.loc[mask, "inj_ap"] = coords["inj_ap"]
        df.loc[mask, "inj_ml"] = coords["inj_ml"]
        df.loc[mask, "inj_dv"] = coords["inj_dv"]
        updated += 1

        logger.info(
            "updated_injection_coords",
            animal_id=str(animal_id),
            inj_ap=coords["inj_ap"],
            inj_ml=coords["inj_ml"],
            inj_dv=coords["inj_dv"],
        )

    df.to_csv(animals_csv_path, index=False)
    logger.info(
        "animals_csv_saved",
        path=str(animals_csv_path),
        animals_updated=updated,
    )
    return df
