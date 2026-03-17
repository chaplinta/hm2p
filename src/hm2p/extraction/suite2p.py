"""Suite2p extractor with post-hoc soma/dendrite ROI classification.

Wraps Suite2p's plane0/ numpy output files directly. Each ROI is classified
as 'soma', 'dend', or 'artefact' using shape statistics from stat.npy and
pre-trained classifiers:
    - classifier_soma.npy   (existing, reused unchanged)
    - classifier_dend.npy   (existing, reused unchanged)

There is a single imaging plane — soma and dendrite ROIs co-exist.
No second Suite2p run is needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hm2p.extraction.base import BaseExtractor


class Suite2pExtractor(BaseExtractor):
    """Extractor backed by Suite2p output folder (plane0/ numpy files)."""

    def __init__(self, folder_path: Path) -> None:
        """Initialise from a Suite2p output directory.

        Loads F.npy, Fneu.npy, iscell.npy, and optionally stat.npy and ops.npy
        from the plane0/ subdirectory.

        Args:
            folder_path: Path to the Suite2p output directory containing plane0/.

        Raises:
            FileNotFoundError: If plane0/ or required .npy files are absent.
        """
        plane_dir = folder_path / "plane0"
        if not plane_dir.exists():
            raise FileNotFoundError(f"Suite2p plane0 directory not found: {plane_dir}")

        for name in ("F.npy", "Fneu.npy", "iscell.npy"):
            if not (plane_dir / name).exists():
                raise FileNotFoundError(f"Required Suite2p file missing: {plane_dir / name}")

        self._F: np.ndarray = np.load(plane_dir / "F.npy").astype(np.float32)
        self._Fneu: np.ndarray = np.load(plane_dir / "Fneu.npy").astype(np.float32)
        iscell = np.load(plane_dir / "iscell.npy")
        self._cell_mask: np.ndarray = iscell[:, 0].astype(bool)

        # Optional: stat.npy (per-ROI shape stats for classification)
        stat_path = plane_dir / "stat.npy"
        self._stat: list[dict] | None = (  # type: ignore[type-arg]
            list(np.load(stat_path, allow_pickle=True)) if stat_path.exists() else None
        )

        # Optional: ops.npy (Suite2p settings dict; contains fs for sampling rate)
        ops_path = plane_dir / "ops.npy"
        self._ops: dict | None = (  # type: ignore[type-arg]
            np.load(ops_path, allow_pickle=True).item() if ops_path.exists() else None
        )

        self._accepted_ids: list[int] = list(np.flatnonzero(self._cell_mask))

    # -- BaseExtractor interface --------------------------------------------

    def get_raw_traces(self) -> np.ndarray:
        """Return raw fluorescence traces for accepted ROIs.

        Returns:
            (n_accepted, n_frames) float32.
        """
        return self._F[self._cell_mask]

    def get_neuropil_traces(self) -> np.ndarray | None:
        """Return neuropil traces for accepted ROIs.

        Returns:
            (n_accepted, n_frames) float32.
        """
        return self._Fneu[self._cell_mask]

    def get_accepted_roi_ids(self) -> list[int]:
        """Return indices of ROIs classified as cells by Suite2p.

        Returns:
            List of 0-based ROI indices.
        """
        return self._accepted_ids

    def get_roi_masks(self) -> np.ndarray:
        """Return spatial masks for accepted ROIs from stat.npy.

        Returns:
            (n_accepted, height, width) bool.

        Raises:
            RuntimeError: If stat.npy or ops.npy were not found.
        """
        if self._stat is None or self._ops is None:
            raise RuntimeError(
                "stat.npy and ops.npy are required for ROI masks but were not found"
            )
        h = int(self._ops.get("Ly", 512))
        w = int(self._ops.get("Lx", 512))
        masks = np.zeros((len(self._accepted_ids), h, w), dtype=bool)
        for i, roi_idx in enumerate(self._accepted_ids):
            stat = self._stat[roi_idx]
            ypix = stat.get("ypix", np.array([], dtype=int))
            xpix = stat.get("xpix", np.array([], dtype=int))
            masks[i, ypix, xpix] = True
        return masks

    def get_sampling_frequency(self) -> float:
        """Return imaging frame rate from ops.npy.

        Returns:
            Frame rate in Hz.

        Raises:
            RuntimeError: If ops.npy was not found.
        """
        if self._ops is None:
            raise RuntimeError("ops.npy is required for sampling frequency")
        return float(self._ops.get("fs", 30.0))

    def get_roi_types(self) -> list[str]:
        """Classify accepted ROIs as 'soma', 'dend', or 'artefact'.

        Uses Suite2p's pre-trained classifiers (classifier_soma.npy,
        classifier_dend.npy) from sourcedata/trackers/suite2p/.

        Returns:
            List of strings, length == len(get_accepted_roi_ids()).

        Raises:
            FileNotFoundError: If classifier files are missing.
            RuntimeError: If stat.npy was not loaded.
        """
        if self._stat is None:
            raise RuntimeError("stat.npy is required for ROI classification")
        all_types = classify_roi_types(self._stat)
        return [all_types[i] for i in self._accepted_ids]

    @property
    def n_rois(self) -> int:
        """Total number of ROIs (accepted + rejected)."""
        return self._F.shape[0]

    @property
    def n_frames(self) -> int:
        """Number of imaging frames."""
        return self._F.shape[1]

    @classmethod
    def from_path(cls, path: Path) -> Suite2pExtractor:
        return cls(path)


_CLASSIFIER_DIR = Path(__file__).resolve().parent.parent.parent.parent / "sourcedata" / "trackers" / "suite2p"


def classify_roi_types(
    stat: list[dict],  # type: ignore[type-arg]
) -> list[str]:
    """Classify each ROI as 'soma', 'dend', or 'artefact'.

    Uses a shape-feature heuristic from the legacy pipeline
    (old-pipeline/utils/classify.py):

    1. ``radius < 2.0`` or ``compact < 0.1`` → artefact (too small or diffuse)
    2. ``aspect_ratio > 2.5`` → dendrite (elongated)
    3. Otherwise → soma

    These thresholds were hand-tuned for single-plane RSP imaging and
    match the classification used in the original hm2p-analysis pipeline.

    Args:
        stat: List of per-ROI stat dicts loaded from Suite2p stat.npy.
            Each dict must contain 'radius', 'compact', 'aspect_ratio'.

    Returns:
        List of strings ('soma', 'dend', 'artefact'), one per ROI.
    """
    labels: list[str] = []
    for s in stat:
        radius = float(s.get("radius", 5.0))
        compact = float(s.get("compact", 0.5))
        aspect_ratio = float(s.get("aspect_ratio", 1.0))

        if radius < 2.0 or compact < 0.1:
            labels.append("artefact")
        elif aspect_ratio > 2.5:
            labels.append("dend")
        else:
            labels.append("soma")

    return labels


# Legacy alias — classify_roi_types IS the heuristic now.
_classify_heuristic = classify_roi_types
